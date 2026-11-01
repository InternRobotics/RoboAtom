"""Boundary-free feature extraction and isolated seeded inference for PhysExtend."""
from __future__ import annotations

from roboatom.physextend.runtime import DATA, current_run

from contextlib import contextmanager
from roboatom.physextend.runtime import run_cache
import hashlib
import json
import logging
from pathlib import Path
import tempfile
from typing import Any, Iterator, Mapping

import numpy as np
import pyarrow.parquet as pq

from roboatom.physextend.features.records import physical_id, signal_uri, stable_hash, validate_inference
from roboatom.physextend.features.feature_mapping import load_feature_map_config
from roboatom.physextend.features.storage import read_rclone_bytes, is_rclone_remote
from roboatom.physextend.features.signals import extract_features, save_feature_bundle, load_feature_bundle, select_signal_columns
from roboatom.physextend.decoders.sensor_matching import read_column_matrix
from roboatom.physextend.decoders.reference_decoder import SeedTrajectory

LOGGER = logging.getLogger(__name__)
FEATURE_MAP = DATA / 'configs/feature_mapping.json'


def extraction_digest() -> str:
    """Invalidate cache for changes in signal extraction or semantic mapping."""
    return stable_hash({'version': 2, 'signal_source': 'signals', 'mapping': hashlib.sha256(FEATURE_MAP.read_bytes()).hexdigest(), 'schema': 5})


def feature_paths(target: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    """Return cache paths bound to this run and the physical input identity."""
    key = stable_hash([physical_id(target), target['length'], extraction_digest()])
    cache = current_run().workdir / 'features'
    return cache / f'{key}.features.npz', cache / f'{key}.raw.npz', cache / f'{key}.json'


@contextmanager
def materialized_signal(target: Mapping[str, Any]) -> Iterator[tuple[str, str]]:
    """Resolve a boundary-free source and materialize remote parquet briefly."""
    from roboatom.physextend.features.sources import _rewrite_current_uri, LOCAL_ROOTS, ROOT_URIS
    uri = signal_uri(target)
    uri = _rewrite_current_uri(str(target.get('source_dataset', target['dataset'])), uri)
    if not uri.endswith('.parquet'):
        raise ValueError(f'Expected parquet sensor source: {uri}')
    if uri.startswith('/'):
        if not Path(uri).is_file():
            raise FileNotFoundError(uri)
        yield uri, uri
        return
    for dataset, prefix in ROOT_URIS.items():
        if dataset in LOCAL_ROOTS and dataset != '3f' and uri.startswith(prefix.rstrip('/') + '/'):
            local = LOCAL_ROOTS[dataset]/uri[len(prefix.rstrip('/'))+1:]
            if local.is_file():
                yield str(local), uri
                return
    if not is_rclone_remote(uri):
        raise ValueError(f'Unsupported storage URI: {uri}')
    with tempfile.NamedTemporaryFile(suffix='.parquet') as stream:
        stream.write(read_rclone_bytes(uri))
        stream.flush()
        yield stream.name, uri


def extract_one(target: dict[str, Any]) -> dict[str, Any]:
    """Extract safe original Ours features plus raw sensor columns for sensor."""
    feature_path, raw_path, metadata_path = feature_paths(target)
    if all(path.is_file() for path in [feature_path, raw_path, metadata_path]):
        return {'physical_id': physical_id(target), 'status': 'cached'}
    try:
        with materialized_signal(target) as (path, uri):
            parquet = pq.ParquetFile(path)
            columns = select_signal_columns(parquet.schema_arrow)
            if not columns:
                raise ValueError('No annotation-free sensor columns')
            table = parquet.read(columns=columns)
            if target.get('sensor_crop_stop') is not None:
                source_start = int(target['sensor_crop_start'])
                source_stop = int(target['sensor_crop_stop'])
                if source_stop > table.num_rows or source_start < 0 or source_start >= source_stop:
                    raise ValueError('Sensor source does not cover declared benchmark crop')
                table = table.slice(source_start, source_stop-source_start)
                # Source is a temporary copy for remote objects, never production data.
                with tempfile.NamedTemporaryFile(suffix='.parquet') as cropped:
                    pq.write_table(table, cropped.name)
                    features = extract_features(cropped.name, int(target['length']), include_visual=False, dataset=str(target['dataset']), feature_map_config=load_feature_map_config(FEATURE_MAP), source_uri=uri)
            else:
                features = extract_features(path, int(target['length']), include_visual=False, dataset=str(target['dataset']), feature_map_config=load_feature_map_config(FEATURE_MAP), source_uri=uri)
            matrices = {f'column_{index}': read_column_matrix(table, name) for index, name in enumerate(columns)}
            metadata = {'physical_id': physical_id(target), 'uri': uri, 'length': int(target['length']), 'sensor_length': int(table.num_rows), 'columns': list(columns), 'extraction_digest': extraction_digest(), 'alignment': 'normalized-time interpolation when lengths differ', 'raw_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        save_feature_bundle(feature_path, features)
        with raw_path.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream, **matrices)
        raw_path.with_suffix('.tmp').replace(raw_path)
        metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
        return {'physical_id': physical_id(target), 'status': 'extracted', 'columns': list(columns)}
    except Exception as error:
        LOGGER.exception('Feature extraction failed for %s', physical_id(target))
        return {'physical_id': physical_id(target), 'status': 'failed', 'dataset': target['dataset'], 'uri': signal_uri(target), 'error': f'{type(error).__name__}: {error}'}


@run_cache(maxsize=256)
def load_features(path: str) -> Any:
    """Load a manifest-keyed immutable feature cache."""
    return load_feature_bundle(Path(path))


def predict_row(row: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decode a target without loading any scoring reference file."""
    validate_inference(row)
    target = row['target']
    seed = row['seed']
    target_features = load_features(str(feature_paths(target)[0]))
    seed_features = load_features(str(feature_paths(seed)[0]))
    seed_object = SeedTrajectory(episode_key=seed['episode_key'], segments=tuple(seed['segments']), features=seed_features)
    from roboatom.physextend.decoders.seeded_decoder import decode_seeded
    result = decode_seeded(row, target_features, seed_object, config)
    points = [0, *result.boundaries, int(target['length'])]
    if len(points) != len(row['transcript']) + 1 or any(a >= b for a, b in zip(points, points[1:])):
        raise ValueError('Decoder returned invalid full-cover segmentation')
    prediction = {'episode_index': row['record_id'], 'record_id': row['record_id'], 'dataset': target['dataset'], 'task_id': row['canonical_task_id'], 'protocol': row['protocol'], 'fps': float(target['fps']), 'frames': int(target['length']), 'frame_interval': 'open_exclusive', 'segments': [{'start_frame': int(segment['start']), 'end_frame': int(segment['end']), 'skill': segment['skill'], 'action_text': segment['action_text']} for segment in result.segments]}
    diagnostic = {'record_id': row['record_id'], 'task_id': row['canonical_task_id'], 'dataset': target['dataset'], 'seed_record_id': seed['record_id'], 'seed_physical_id': physical_id(seed), 'target_physical_id': physical_id(target), 'boundaries': list(result.boundaries), 'decisions': result.decisions, 'diagnostics': getattr(result, 'diagnostics', {}), 'fallback_reason': result.fallback_reason}
    return prediction, diagnostic

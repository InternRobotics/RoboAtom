"""Seeded Ours/sensor cascade built on the production decoder implementations."""
from __future__ import annotations

from roboatom.physextend.runtime import current_run

from dataclasses import replace
from roboatom.physextend.runtime import run_cache
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from roboatom.physextend.decoders import sensor_matching as sensor
from roboatom.physextend.features.extraction import feature_paths
from roboatom.physextend.decoders.decoder import OursDecodeResult, decode_ours, project_boundaries
from roboatom.physextend.features.feature_mapping import schema_gripper_indices
from roboatom.physextend.features.signals import FeatureBundle
from roboatom.physextend.features.signals import _gripper_series, _read_bytes, _resample, detect_signal_events, robust_unit
from roboatom.physextend.features.records import signal_uri, stable_hash
from roboatom.physextend.decoders.reference_decoder import SeedTrajectory
from roboatom.physextend.decoders.anchor_decoder import anchor_config, _seed_templates

LOGGER = logging.getLogger(__name__)


@run_cache(maxsize=256)
def sensor_schema(uri: str) -> dict[str, Any]:
    """Read only sensor feature names from source metadata, with provenance."""
    if '/data/chunk-' not in uri:
        return {}
    info_uri = uri.split('/data/chunk-')[0] + '/meta/info.json'
    cache = current_run().workdir / 'sensor_schema' / (stable_hash(info_uri) + '.json')
    if cache.exists():
        return json.loads(cache.read_text())['features']
    try:
        # Sensor metadata follows the same storage rules as parquet features.
        # S3 URIs need the configured Petrel client; they are not rclone paths.
        payload = _read_bytes(info_uri)
        features = json.loads(payload)['features']
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        LOGGER.warning('Sensor schema unavailable for %s: %s', info_uri, error)
        return {}
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps({'source_uri': info_uri, 'features': features}, sort_keys=True)+'\n')
    temporary.replace(cache)
    return features


def metadata_gripper_indices(description: Mapping[str, Any], width: int) -> tuple[int, ...]:
    """Recognize named grippers, including scalar left/right effector openings."""
    return schema_gripper_indices(description, width)


def with_metadata_grippers(features: FeatureBundle, target: Mapping[str, Any], preserve_existing: bool = False, event_source: str = 'combined', arm_consistent: bool = False) -> FeatureBundle:
    """Restore gripper evidence from documented vector dimensions."""
    _, raw_path, metadata_path = feature_paths(target)
    raw = raw_sensor_cache(str(raw_path), str(metadata_path))
    schema = sensor_schema(signal_uri(target))
    state, action = [], []
    mapping = []
    for column, matrix in raw.items():
        indexes = metadata_gripper_indices(schema.get(column, {}), matrix.shape[1])
        # AgiBot's effector.position vectors are named left/right grippers in
        # info.json. The older whole-column eef_pose rule predates that schema.
        verified_agibot_gripper = target.get('dataset') == 'AgiBot' and column in {
            'actions.effector.position', 'observation.states.effector.position'
        }
        config_non_gripper = set() if verified_agibot_gripper else {
            int(item['index'])
            for item in features.feature_mapping
            if item['column'] == column and item.get('origin') == 'config' and item.get('family') != 'gripper'
        }
        indexes = tuple(index for index in indexes if index not in config_non_gripper)
        if preserve_existing:
            existing = {int(item['index']) for item in features.feature_mapping if item['column'] == column and item['family'] == 'gripper'}
            indexes = tuple(sorted((set(indexes) | existing) - config_non_gripper))
        if not indexes:
            continue
        values = _resample(matrix[:, indexes], features.length)
        source = 'action' if column.startswith(('action', 'actions')) else 'state'
        (action if source == 'action' else state).append(values)
        mapping.extend({'column': column, 'index': index, 'family': 'gripper', 'source': source, 'side': 'metadata', 'quantity': 'position', 'provenance': 'source_schema'} for index in indexes)
    if not mapping:
        return features
    level, delta = _gripper_series([*state, *action], features.length)
    state_level, state_delta = _gripper_series(state, features.length)
    action_level, action_delta = _gripper_series(action, features.length)
    if event_source == 'state' and state:
        level, delta = state_level, state_delta
    elif event_source == 'action' and action:
        level, delta = action_level, action_delta
    elif event_source not in {'combined', 'state', 'action'}:
        raise ValueError('Unknown gripper event source')
    if arm_consistent:
        matrices = action if event_source == 'action' and action else state or action
        tracks = np.stack([robust_unit(matrix[:, index]) for matrix in matrices for index in range(matrix.shape[1])], axis=1)
        source_kind = 'action' if (event_source == 'action' and action) or not state else 'state'
        keys = [f"{item['column']}:{item['index']}" for item in mapping if item['source'] == source_kind]
        features = replace(features, embodiment={**features.embodiment, 'gripper_tracks': tracks, 'gripper_keys': keys})
    mapped_keys = {(item['column'], item['index']) for item in mapping}
    remaining = [item for item in features.feature_mapping if (item['column'], item['index']) not in mapped_keys]
    return replace(features, gripper_level=level, gripper_delta=delta, state_gripper_level=state_level, state_gripper_delta=state_delta, action_gripper_level=action_level, action_gripper_delta=action_delta, action_state_error=np.abs(action_level-state_level) if action_level is not None and state_level is not None else None, events=detect_signal_events(delta, level, features.activity), feature_mapping=tuple([*remaining, *mapping]))


@run_cache(maxsize=128)
def raw_sensor_cache(raw_path: str, metadata_path: str) -> dict[str, np.ndarray]:
    """Load safe named numeric columns, never annotation fields."""
    metadata = json.loads(Path(metadata_path).read_text())
    with np.load(raw_path, allow_pickle=False) as values:
        return {name: np.asarray(values[f'column_{index}'], dtype=np.float32) for index, name in enumerate(metadata['columns'])}


def sensor_episode(target: Mapping[str, Any], columns: Sequence[str], stride: int = 5, center_state: bool = False) -> sensor.EpisodeFeatures:
    """Apply the original sensor state, derivative and temporal-context features."""
    _, raw_path, metadata_path = feature_paths(target)
    raw = raw_sensor_cache(str(raw_path), str(metadata_path))
    state = np.concatenate([raw[name] for name in columns], axis=1)
    if not np.all(np.isfinite(state)):
        clean = np.where(np.isfinite(state), state, np.nan)
        medians = np.nanmedian(clean, axis=0)
        state = np.where(np.isfinite(state), state, np.where(np.isfinite(medians), medians, 0.0))
    length = int(target['length'])
    if len(state) != length:
        original = np.linspace(0.0, 1.0, len(state))
        current = np.linspace(0.0, 1.0, length)
        state = np.stack([np.interp(current, original, state[:, column]) for column in range(state.shape[1])], axis=1).astype(np.float32)
    first_delta = np.vstack([np.zeros((1, state.shape[1]), dtype=np.float32), np.diff(state, axis=0)])
    second_delta = np.vstack([np.zeros((1, state.shape[1]), dtype=np.float32), np.diff(first_delta, axis=0)])
    if center_state:
        state = state-state[:1]
    motion = sensor.moving_average(np.linalg.norm(first_delta, axis=1), window=17)
    base = np.concatenate([state, first_delta, second_delta], axis=1)
    contextual = sensor.temporal_context(base, offsets=(-20, -10, -5, 0, 5, 10, 20))
    frames = np.arange(0, length, stride, dtype=np.int32)
    # The original solver has a visual field; sensor-only evaluation uses an
    # inert placeholder and zero visual weight, never fabricated visual evidence.
    return sensor.EpisodeFeatures(episode_index=str(target['episode_key']), length=length, sample_frames=frames, state_features=contextual[frames], visual_features=np.zeros((len(frames), 1), dtype=np.float32), combined_motion_full=sensor.robust_01(motion), state_motion_full=motion, visual_motion_full=np.zeros(length, dtype=np.float32))


def sensor_cut(row: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Fit solely on the independent annotated seed and call the original solver."""
    seed, target = row['seed'], row['target']
    seed_paths, target_paths = feature_paths(seed), feature_paths(target)
    seed_raw = raw_sensor_cache(str(seed_paths[1]), str(seed_paths[2]))
    target_raw = raw_sensor_cache(str(target_paths[1]), str(target_paths[2]))
    common = sorted(name for name in set(seed_raw) & set(target_raw) if seed_raw[name].shape[1] == target_raw[name].shape[1])
    if config.get('state_only', False):
        measured = [name for name in common if 'action' not in name.lower() and 'command' not in name.lower()]
        common = measured or common
    if not common:
        raise ValueError('Seed and target have no compatible safe sensor columns')
    seed_features = sensor_episode(seed, common, center_state=bool(config.get('center_state', False)))
    target_features = sensor_episode(target, common, center_state=bool(config.get('center_state', False)))
    label = sensor.ManualLabel(episode_index=seed['episode_key'], path=str(seed['data_uri']), length=int(seed['length']), tasks=[seed['canonical_task_id']], action_config=[{'start_frame': int(segment['start']), 'end_frame': int(segment['end']), 'skill': segment['skill'], 'action_text': segment['action_text']} for segment in seed['segments']])
    model = sensor.fit_model([label], {seed['episode_key']: seed_features})
    costs = sensor.episode_costs(target_features, model, state_weight=1.0, visual_weight=0.0)
    minimum_requested = round(float(config['min_segment_seconds'])*float(target['fps'])) if 'min_segment_seconds' in config else int(config.get('min_segment_frames', 30))
    minimum = min(max(1, minimum_requested), max(1, int(target['length'])//(2*len(row['transcript']))))
    diagnostics: dict[str, Any] = {'sensor_columns': common, 'visual_weight': 0.0}
    boundaries, confidence, expected = sensor.solve_boundaries(features=target_features, cost=costs, model=model, duration_weight=float(config.get('duration_weight', 0.85)), boundary_motion_weight=float(config.get('motion_weight', 0.25)), min_segment_frames=minimum, candidate_keep=int(config.get('candidate_keep', 45)), boundary_window_ratio=float(config.get('window_ratio', 0.55)), refine_window=int(config.get('refine_window', 55)), diagnostics=diagnostics)
    diagnostics.update({'confidence': confidence, 'expected': expected, 'solver': 'production_dynamic_matching.solve_boundaries', 'minimum_segment_frames': minimum})
    return project_boundaries(np.asarray(boundaries), int(target['length']), len(row['transcript'])), diagnostics


def project_with_anchors(values: Sequence[int], anchors: Mapping[int, int], length: int) -> tuple[int, ...]:
    """Restore monotonicity inside anchor intervals without moving an anchor."""
    result = np.asarray(values, dtype=np.int64).copy()
    fixed = [(-1, 0), *sorted(anchors.items()), (len(values), length)]
    for (left_index, left_frame), (right_index, right_frame) in zip(fixed, fixed[1:]):
        slots = right_index - left_index
        if right_frame - left_frame < slots:
            raise ValueError('Conflicting anchors cannot form positive segments')
        minimum = min(max(1, round(length * 0.012)), (right_frame-left_frame)//slots)
        for index in range(left_index+1, right_index):
            lower = left_frame + minimum*(index-left_index)
            if index > left_index+1:
                lower = max(lower, int(result[index-1])+minimum)
            upper = right_frame-minimum*(right_index-index)
            result[index] = min(max(int(result[index]), lower), upper)
        for index in range(right_index-1, left_index, -1):
            upper = right_frame-minimum*(right_index-index)
            if index < right_index-1:
                upper = min(upper, int(result[index+1])-minimum)
            result[index] = min(result[index], upper)
        if right_index < len(values):
            result[right_index] = right_frame
    return tuple(int(value) for value in result)


def decode_seeded(row: Mapping[str, Any], features: FeatureBundle, seed: SeedTrajectory, config: Mapping[str, Any]) -> OursDecodeResult:
    """Preserve Ours high-confidence anchors; fill other boundaries with sensor."""
    if config.get('metadata_grippers', False):
        preserve_existing = bool(config.get('preserve_existing_grippers', False))
        event_source = str(config.get('gripper_event_source', 'combined'))
        arm_consistent = bool(config.get('arm_consistent_quality', False))
        if event_source == 'seed_best':
            calibration = replace(anchor_config(), **config.get('anchor', {}))
            scores = {}
            for candidate in ('state', 'action'):
                calibrated = replace(seed, features=with_metadata_grippers(seed.features, row['seed'], preserve_existing, candidate, arm_consistent))
                if config.get('ignore_command_state_error', False):
                    calibrated = replace(calibrated, features=replace(calibrated.features, action_state_error=None))
                templates, _ = _seed_templates(row['transcript'], calibrated, calibration)
                scores[candidate] = (len(templates), sum(template.confidence for template in templates))
            event_source = max(scores, key=scores.get)
        features = with_metadata_grippers(features, row['target'], preserve_existing, event_source, arm_consistent)
        seed = replace(seed, features=with_metadata_grippers(seed.features, row['seed'], preserve_existing, event_source, arm_consistent))
    if config.get('ignore_command_state_error', False):
        features = replace(features, action_state_error=None)
        seed = replace(seed, features=replace(seed.features, action_state_error=None))
    if config.get('drop_zero_gripper_events', False):
        def nonzero_events(bundle: FeatureBundle) -> FeatureBundle:
            events = tuple(event for event in bundle.events if event.channel != 'gripper' or abs(float(bundle.gripper_delta[event.frame])) > 1e-6)
            return replace(bundle, events=events)
        features = nonzero_events(features)
        seed = replace(seed, features=nonzero_events(seed.features))
    if config.get('anchor'):
        base_config = anchor_config()
        ours = decode_ours(row['transcript'], features, [seed], anchor_config_override=replace(base_config, **config['anchor']))
    else:
        ours = decode_ours(row['transcript'], features, [seed])
    if not ours.boundaries:
        return OursDecodeResult(ours.boundaries, ours.segments, ours.total_cost, ours.decisions, getattr(ours, 'diagnostics', {}), ours.fallback_reason)
    diagnostics = dict(getattr(ours, 'diagnostics', {}))
    if not config.get('cascade', False):
        return OursDecodeResult(ours.boundaries, ours.segments, ours.total_cost, ours.decisions, diagnostics, ours.fallback_reason)
    sensor_boundaries, sensor_diagnostics = sensor_cut(row, config.get('sensor', {}))
    raw = []
    decisions = []
    for index, (ours_frame, sensor_frame) in enumerate(zip(ours.boundaries, sensor_boundaries, strict=True)):
        original_decision = dict(ours.decisions[index])
        anchored = bool(original_decision.get('anchor_applied'))
        sensor_weight = float(config.get('sensor_weight', 1.0))
        if config.get('sensor_confidence_weight', False):
            sensor_weight *= max(0.05, float(sensor_diagnostics['confidence'][index])**2)
        if not 0.0 <= sensor_weight <= 1.0:
            raise ValueError('sensor_weight must be in [0, 1]')
        candidate_sensor_frame = sensor_frame
        if 'sensor_correction_fraction' in config:
            points = (0, *ours.boundaries, features.length)
            limit = max(1, round(float(config['sensor_correction_fraction'])*min(ours_frame-points[index], points[index+2]-ours_frame)))
            candidate_sensor_frame = max(ours_frame-limit, min(ours_frame+limit, sensor_frame))
        selected = ours_frame if anchored else int(round((1.0-sensor_weight)*ours_frame + sensor_weight*candidate_sensor_frame))
        raw.append(selected)
        decisions.append({**original_decision, 'source': 'ours_anchor' if anchored else 'sensor', 'ours_frame': ours_frame, 'sensor_frame': sensor_frame, 'sensor_weight': sensor_weight})
    anchors = {index: int(ours.boundaries[index]) for index, decision in enumerate(ours.decisions) if decision.get('anchor_applied')}
    boundaries = project_with_anchors(raw, anchors, features.length)
    points = [0, *boundaries, features.length]
    segments = tuple({'start': points[index], 'end': points[index+1], 'skill': step['skill'].strip().lower().replace('-', '_').replace(' ', '_'), 'action_text': step.get('action_text', '')} for index, step in enumerate(row['transcript']))
    diagnostics['sensor'] = sensor_diagnostics
    for index, frame in enumerate(boundaries):
        decisions[index]['frame'] = frame
        decisions[index]['monotone_adjustment'] = int(frame) - int(raw[index])
    return OursDecodeResult(boundaries, segments, ours.total_cost, tuple(decisions), diagnostics, 'sensor_duration_recovery' if sensor_diagnostics['used_fallback'] else ours.fallback_reason)

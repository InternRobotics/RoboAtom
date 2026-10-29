"""Reference identity and target-label isolation for physical inference."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import re
from typing import Any

FORBIDDEN = {'segments', 'action_config', 'start_frame', 'end_frame', 'boundaries', 'subtask_index'}


def stable_hash(value: Any) -> str:
    """Hash a JSON-serializable value deterministically."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def signal_uri(target: Mapping[str, Any]) -> str:
    """Normalize storage aliases and LeRobot video paths without accessing data."""
    value = str(target.get('resolved_sensor_uri') or target.get('data_uri') or target.get('signals_path') or '').strip()
    value = value.replace('cluster_hoss:s3://', 'hoss:').replace(':s3://', ':')
    from roboatom.physextend.features.sources import _rewrite_current_uri
    source_dataset = str(target.get('source_dataset', target.get('dataset', '')))
    value = _rewrite_current_uri(source_dataset, value)
    if value.endswith('.mp4'):
        value = re.sub(r'/videos/(chunk-\d+)/[^/]+/(episode_\d+)\.mp4$', r'/data/\1/\2.parquet', value)
    if not value:
        raise ValueError('Missing signal URI')
    return value


def physical_id(target: Mapping[str, Any]) -> str:
    """Identify a physical episode independently of annotation source and row."""
    return stable_hash(signal_uri(target))


def signature(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the boundary-free exact ordered skill sequence."""
    return tuple(str(step['skill']).strip().lower().replace('-', '_').replace(' ', '_') for step in row['transcript'])


def reject_target_boundaries(value: Any) -> None:
    """Reject recursively embedded target annotations."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in FORBIDDEN:
                raise ValueError(f'Target boundary field is forbidden: {key}')
            reject_target_boundaries(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            reject_target_boundaries(child)


def validate_inference(row: Mapping[str, Any]) -> None:
    """Validate seed identity and strict target/seed field separation."""
    reject_target_boundaries(row['target'])
    reject_target_boundaries(row['transcript'])
    seed, target = row['seed'], row['target']
    if physical_id(seed) == physical_id(target):
        raise ValueError('Seed/target physical identity violates evaluation protocol')
    if seed['canonical_task_id'] != target['canonical_task_id']:
        raise ValueError('Seed and target belong to different tasks')
    if signature(row) != tuple(str(segment['skill']) for segment in seed['segments']):
        raise ValueError('Seed skill sequence mismatch')
    points = [int(segment['start']) for segment in seed['segments']] + [int(seed['length'])]
    if points[0] != 0 or any(a >= b for a, b in zip(points, points[1:])):
        raise ValueError('Seed must be a positive full cover')
    if any(int(segment['end']) != points[index+1] for index, segment in enumerate(seed['segments'])):
        raise ValueError('Seed intervals are not contiguous')
    if float(target['fps']) <= 0 or int(target['length']) <= 0:
        raise ValueError('Invalid target time base')

"""Public input adapters and strict prediction coverage checks."""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding='utf-8').strip()
    if not text:
        raise ValueError(f'Empty input: {path}')
    try:
        data = json.loads(text)
        rows = data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    result = []
    for row in rows:
        if isinstance(row, list):
            row = row[0]
        if isinstance(row, dict) and isinstance(row.get('result'), dict):
            row = row['result']
        if not isinstance(row, dict):
            raise ValueError('Each episode must be an object')
        result.append(row)
    return result


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows), encoding='utf-8')
    temporary.replace(path)


def canonical_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result['episode_id'] = next((str(row[k]) for k in ('episode_id', 'benchmark_id', 'episode_index', 'sample_id', 'id', 'path') if row.get(k) is not None), '')
    result['num_frames'] = row.get('num_frames', row.get('frames', row.get('length')))
    result['task_id'] = str(row.get('task_id', row.get('canonical_task_id', row.get('task', ''))))
    raw_segments = row.get('segments', row.get('action_config', []))
    interval = row.get('frame_interval', '')
    if not interval and raw_segments:
        interval = 'half_open' if raw_segments[-1].get('end_frame') == result['num_frames'] else 'closed'
    if interval not in ('', 'closed', 'closed_inclusive', 'half_open', 'open', 'open_exclusive'):
        raise ValueError(f'Unsupported frame_interval: {interval}')
    half_open = interval in ('half_open', 'open', 'open_exclusive')
    result['segments'] = [{**s, 'start_frame': s.get('start_frame'),
                           'end_frame': s.get('end_frame') - (1 if half_open else 0) if s.get('end_frame') is not None else None,
                           'skill': s.get('skill', s.get('primitive_skill', '')),
                           'subtask': s.get('subtask', s.get('action_text', ''))} for s in raw_segments]
    result['frame_interval'] = 'closed'
    return result


def validate_rows(rows: list[dict[str, Any]], require_segments: bool = True) -> set[str]:
    ids = set()
    for raw in rows:
        row = canonical_row(raw)
        identifier = row['episode_id']
        if not identifier or identifier in ids:
            raise ValueError(f'Missing or duplicate episode ID: {identifier!r}')
        ids.add(identifier)
        length, fps = row['num_frames'], row.get('fps')
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise ValueError(f'{identifier}: num_frames must be a positive integer')
        if not isinstance(fps, (float, int)) or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f'{identifier}: fps must be positive and finite')
        if require_segments and not row['segments']:
            raise ValueError(f'{identifier}: no segments')
        previous = -1
        for segment in row['segments']:
            start, end = segment['start_frame'], segment['end_frame']
            if not all(isinstance(x, int) and not isinstance(x, bool) for x in (start, end)):
                raise ValueError(f'{identifier}: frame boundaries must be integers')
            if not 0 <= start <= end < length or start <= previous:
                raise ValueError(f'{identifier}: invalid, overlapping, or unordered segments')
            previous = end
    return ids


def check_coverage(gt: list[dict[str, Any]], prediction: list[dict[str, Any]], expected: int | None = None) -> dict[str, Any]:
    gold_ids = validate_rows(gt)
    pred_ids = validate_rows(prediction)
    missing, extra = sorted(gold_ids - pred_ids), sorted(pred_ids - gold_ids)
    if expected is not None and len(gold_ids) != expected:
        raise ValueError(f'Expected {expected} GT episodes; got {len(gold_ids)}')
    if missing or extra:
        raise ValueError(f'Prediction coverage mismatch: missing={missing}; extra={extra}')
    return {'coverage': len(pred_ids), 'expected': len(gold_ids), 'missing': missing, 'extra': extra}


def check_evaluator_coverage(gold: dict, predicted: dict, expected: int | None = None) -> dict:
    """Validate indexed evaluator inputs, including seconds and generated outputs."""
    from roboatom.evaluation import action
    gold_ids, pred_ids = set(gold), set(predicted)
    if expected is not None and len(gold) != expected:
        raise ValueError(f'Expected {expected} GT episodes; got {len(gold)}')
    missing, extra = sorted(gold_ids - pred_ids), sorted(pred_ids - gold_ids)
    if missing or extra:
        raise ValueError(f'Prediction coverage mismatch: missing={missing}; extra={extra}')
    for identifier in gold:
        reference = gold[identifier]
        for role, row in [('GT', reference), ('Prediction', predicted[identifier])]:
            raw = action.base_evaluator.raw_segment_values(row, identifier, role)
            parsed = action.base_evaluator.parse_segments(row, identifier, role)
            if not parsed or len(raw) != len(parsed):
                raise ValueError(f'{role} {identifier}: missing or zero-length segments')
            action.validate_non_overlapping(parsed, context=f'{role} {identifier}')
            length = reference.get('num_frames', reference.get('frames', reference.get('length')))
            if length is not None:
                duration = float(length) / action.base_evaluator.positive_fps(reference, identifier, 'GT')
                if any(segment['end'] > duration + 1e-8 for segment in parsed):
                    raise ValueError(f'{role} {identifier}: segment exceeds video duration')
    return {'coverage':len(gold), 'expected':len(gold), 'missing':missing, 'extra':extra}

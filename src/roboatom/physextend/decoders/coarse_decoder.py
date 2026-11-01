"""Rule-based trajectory extension with transcript-constrained global decoding."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from roboatom.physextend.features.signals import FeatureBundle, SignalEvent, robust_unit


PICK_LIKE = {"pick", "plug", "stack", "grasp"}
PLACE_LIKE = {"place", "release", "insert"}
CONTACT_LIKE = {"press", "push", "pull", "open", "close", "screw", "unscrew"}
HOLDING_LIKE = {"move", "pour", "wipe", "sweep", "fold", "handover"}


@dataclass(frozen=True)
class DecoderConfig:
    """Weights and switches for one rule hypothesis round."""

    name: str
    hypothesis: str
    use_seed_duration: bool = True
    use_skill_duration_priors: bool = True
    use_activity_warp: bool = False
    use_gripper_events: bool = False
    use_semantic_events: bool = False
    use_persistence: bool = False
    use_kinematics: bool = False
    use_holding_state: bool = False
    use_visual: bool = False
    use_global_dp: bool = True
    use_conditional_persistence: bool = False
    use_calibrated_holding: bool = False
    use_seed_event_lag: bool = False
    use_retry_filter: bool = False
    use_dtw_anchor: bool = False
    position_weight: float = 1.0
    activity_weight: float = 0.0
    signature_weight: float = 0.0
    gripper_weight: float = 0.0
    persistence_weight: float = 0.0
    pause_weight: float = 0.0
    turn_weight: float = 0.0
    holding_weight: float = 0.0
    visual_weight: float = 0.0
    duration_weight: float = 0.6
    minimum_segment_ratio: float = 0.012
    candidate_keep: int = 96
    local_grid_radius_ratio: float = 0.12
    retain_strong_events: bool = False
    strong_event_threshold: float = 0.8
    seed_activity_progress: bool = False
    relative_event_lag: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.hypothesis:
            raise ValueError("Decoder config name and hypothesis must be non-empty")
        if self.minimum_segment_ratio <= 0.0:
            raise ValueError("minimum_segment_ratio must be positive")
        if self.candidate_keep < 8:
            raise ValueError("candidate_keep must be at least 8")


@dataclass(frozen=True)
class DecodeResult:
    """Globally consistent boundaries and per-boundary diagnostics."""

    boundaries: tuple[int, ...]
    segments: tuple[dict[str, Any], ...]
    total_cost: float
    decisions: tuple[dict[str, Any], ...]
    fallback_reason: str | None = None


@dataclass(frozen=True)
class _EventIndex:
    """Sorted event arrays supporting logarithmic nearest-neighbor lookup."""

    frames_by_kind: Mapping[str, np.ndarray]
    strengths_by_kind: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class _HoldingCalibration:
    """Seed-derived gripper levels for holding and non-holding phases."""

    holding_level: float
    nonholding_level: float
    confidence: float


def semantic_event_config() -> DecoderConfig:
    """Return the coarse decoder configuration used as the anchor baseline."""
    return DecoderConfig(
        name="semantic_events",
        hypothesis="Skill-aware event direction and seed-derived lag reject incompatible transitions.",
        use_activity_warp=True,
        use_gripper_events=True,
        use_semantic_events=True,
        activity_weight=1.0,
        signature_weight=0.5,
        gripper_weight=2.2,
    )


def _canonical_skill(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _skill_duration_weight(skill: str) -> float:
    if skill in PICK_LIKE or skill in PLACE_LIKE:
        return 1.0
    if skill in CONTACT_LIKE:
        return 0.8
    if skill in HOLDING_LIKE:
        return 1.35
    return 1.0


def _expected_durations(
    transcript: Sequence[Mapping[str, Any]],
    target_length: int,
    seed_segments: Sequence[Mapping[str, Any]] | None,
    config: DecoderConfig,
) -> np.ndarray:
    if config.use_seed_duration and seed_segments and len(seed_segments) == len(transcript):
        raw = np.asarray(
            [max(1, int(segment["end"]) - int(segment["start"])) for segment in seed_segments],
            dtype=np.float64,
        )
    elif config.use_skill_duration_priors:
        raw = np.asarray(
            [_skill_duration_weight(_canonical_skill(step.get("skill"))) for step in transcript],
            dtype=np.float64,
        )
    else:
        raw = np.ones(len(transcript), dtype=np.float64)
    return (raw / raw.sum() * target_length).astype(np.float64)


def _cumulative_progress(values: np.ndarray) -> np.ndarray:
    energy = np.maximum(np.asarray(values, dtype=np.float64), 0.0) + 1e-4
    cumulative = np.cumsum(energy)
    return cumulative / cumulative[-1]


def _activity_warp_anchors(
    ratio_anchors: np.ndarray,
    seed_segments: Sequence[Mapping[str, Any]] | None,
    seed_features: FeatureBundle | None,
    target_features: FeatureBundle,
) -> np.ndarray:
    if seed_segments is None or seed_features is None:
        return ratio_anchors
    seed_progress = _cumulative_progress(seed_features.activity)
    target_progress = _cumulative_progress(target_features.activity)
    warped: list[int] = []
    for segment in seed_segments[:-1]:
        seed_frame = min(max(int(segment["end"]) - 1, 0), seed_features.length - 1)
        target_frame = int(np.searchsorted(target_progress, seed_progress[seed_frame], side="left"))
        warped.append(min(max(target_frame + 1, 1), target_features.length - 1))
    if len(warped) != len(ratio_anchors):
        return ratio_anchors
    return np.asarray(warped, dtype=np.int32)


def _resample_signal(values: np.ndarray, count: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == count:
        return array
    source = np.linspace(0.0, 1.0, num=max(array.size, 1), dtype=np.float32)
    target = np.linspace(0.0, 1.0, num=count, dtype=np.float32)
    if not array.size:
        return np.zeros(count, dtype=np.float32)
    return np.interp(target, source, array).astype(np.float32)


def _alignment_matrix(features: FeatureBundle, count: int = 192) -> np.ndarray:
    delta = np.asarray(features.gripper_delta, dtype=np.float32)
    delta_scale = max(float(np.percentile(np.abs(delta), 95.0)), 1e-6)
    return np.stack(
        (
            _resample_signal(robust_unit(features.activity), count),
            _resample_signal(robust_unit(features.pause), count),
            _resample_signal(np.clip(delta / delta_scale, -1.0, 1.0), count),
        ),
        axis=1,
    )


def _dtw_anchors(
    seed_segments: Sequence[Mapping[str, Any]] | None,
    seed_features: FeatureBundle | None,
    target_features: FeatureBundle,
) -> np.ndarray | None:
    """Map seed boundaries through a fixed-band, model-free DTW path."""
    if seed_segments is None or seed_features is None or len(seed_segments) < 2:
        return None
    count = 192
    seed_matrix = _alignment_matrix(seed_features, count)
    target_matrix = _alignment_matrix(target_features, count)
    band = int(math.ceil(0.15 * count))
    costs = np.full((count, count), np.inf, dtype=np.float64)
    back = np.full((count, count, 2), -1, dtype=np.int16)
    for seed_index in range(count):
        lower = max(0, seed_index - band)
        upper = min(count, seed_index + band + 1)
        for target_index in range(lower, upper):
            local = float(
                np.mean(
                    np.abs(seed_matrix[seed_index] - target_matrix[target_index])
                )
            )
            if seed_index == 0 and target_index == 0:
                costs[seed_index, target_index] = local
                continue
            predecessors = (
                (seed_index - 1, target_index - 1),
                (seed_index - 1, target_index),
                (seed_index, target_index - 1),
            )
            feasible = [
                (costs[left, right], left, right)
                for left, right in predecessors
                if left >= 0 and right >= 0 and math.isfinite(costs[left, right])
            ]
            if not feasible:
                continue
            _, previous_seed, previous_target = min(feasible)
            costs[seed_index, target_index] = (
                costs[previous_seed, previous_target] + local
            )
            back[seed_index, target_index] = (previous_seed, previous_target)
    if not math.isfinite(costs[-1, -1]):
        return None
    mapping: dict[int, list[int]] = {}
    seed_index = count - 1
    target_index = count - 1
    while seed_index >= 0 and target_index >= 0:
        mapping.setdefault(seed_index, []).append(target_index)
        previous_seed, previous_target = back[seed_index, target_index]
        if previous_seed < 0 or previous_target < 0:
            break
        seed_index, target_index = int(previous_seed), int(previous_target)
    anchors: list[int] = []
    for segment in seed_segments[:-1]:
        seed_ratio = int(segment["end"]) / max(seed_features.length, 1)
        mapped_seed = min(count - 1, max(0, int(round(seed_ratio * (count - 1)))))
        available = mapping.get(mapped_seed)
        if not available:
            nearest = min(mapping, key=lambda index: abs(index - mapped_seed))
            available = mapping[nearest]
        target_ratio = float(np.median(available)) / (count - 1)
        anchors.append(
            min(
                target_features.length - 1,
                max(1, int(round(target_ratio * target_features.length))),
            )
        )
    return np.asarray(anchors, dtype=np.int32)


def _boundary_signature(features: FeatureBundle, frame: int, radius: int = 4) -> np.ndarray:
    start = max(0, frame - radius)
    end = min(features.length, frame + radius + 1)
    return np.asarray(
        [
            np.mean(features.activity[start:end]),
            np.mean(features.pause[start:end]),
            np.mean(features.turn[start:end]),
            np.max(np.abs(features.gripper_delta[start:end])),
            np.mean(features.gripper_level[start:end]),
            np.mean(features.visual_motion[start:end]),
        ],
        dtype=np.float32,
    )


def _seed_signatures(
    seed_segments: Sequence[Mapping[str, Any]] | None,
    seed_features: FeatureBundle | None,
) -> list[np.ndarray | None]:
    if seed_segments is None or seed_features is None:
        return [None] * max(len(seed_segments or ()) - 1, 0)
    return [_boundary_signature(seed_features, int(segment["end"])) for segment in seed_segments[:-1]]


def _build_event_index(events: Sequence[SignalEvent]) -> _EventIndex:
    grouped: dict[str, list[SignalEvent]] = {}
    for event in events:
        grouped.setdefault(event.kind, []).append(event)
    frames_by_kind: dict[str, np.ndarray] = {}
    strengths_by_kind: dict[str, np.ndarray] = {}
    for kind, kind_events in grouped.items():
        ordered = sorted(kind_events, key=lambda event: event.frame)
        frames_by_kind[kind] = np.asarray([event.frame for event in ordered], dtype=np.int32)
        strengths_by_kind[kind] = np.asarray([event.strength for event in ordered], dtype=np.float32)
    return _EventIndex(frames_by_kind, strengths_by_kind)


def _event_distance(
    event_index: _EventIndex, frame: int, kinds: set[str], lag: int = 0
) -> tuple[float, SignalEvent | None]:
    candidates: list[SignalEvent] = []
    for kind in kinds:
        frames = event_index.frames_by_kind.get(kind)
        strengths = event_index.strengths_by_kind.get(kind)
        if frames is None or strengths is None or not frames.size:
            continue
        event_target = frame - lag
        insertion = int(np.searchsorted(frames, event_target))
        for index in {max(0, insertion - 1), min(insertion, frames.size - 1)}:
            candidates.append(
                SignalEvent(int(frames[index]), kind, float(strengths[index]), "gripper")
            )
    if not candidates:
        return 1.0, None
    best = min(
        candidates,
        key=lambda event: (
            abs(event.frame + lag - frame),
            -event.strength,
            event.frame,
        ),
    )
    return abs(best.frame + lag - frame), best


def _semantic_event_kinds(
    previous_skill: str,
    seed_segment: Mapping[str, Any] | None,
    seed_features: FeatureBundle | None,
) -> set[str]:
    if previous_skill not in PICK_LIKE | PLACE_LIKE | CONTACT_LIKE:
        return {"gripper_increase", "gripper_decrease"}
    if seed_segment is None or seed_features is None:
        return {"gripper_increase", "gripper_decrease"}
    frame = min(max(int(seed_segment["end"]), 1), seed_features.length - 1)
    start = max(0, frame - 8)
    end = min(seed_features.length, frame + 9)
    delta = float(np.sum(seed_features.gripper_delta[start:end]))
    if abs(delta) < 0.02:
        return {"gripper_increase", "gripper_decrease"}
    return {"gripper_increase" if delta > 0 else "gripper_decrease"}


def _persistence_score(features: FeatureBundle, frame: int, radius: int = 9) -> float:
    before = features.gripper_level[max(0, frame - radius) : frame]
    after = features.gripper_level[frame : min(features.length, frame + radius)]
    if before.size < 2 or after.size < 2:
        return 0.0
    shift = abs(float(np.median(after) - np.median(before)))
    instability = float(np.median(np.abs(after - np.median(after))))
    return float(np.clip(shift / (0.05 + instability), 0.0, 1.0))


def _seed_event_lags(
    transcript: Sequence[Mapping[str, Any]],
    seed_segments: Sequence[Mapping[str, Any]] | None,
    seed_features: FeatureBundle | None,
) -> list[int]:
    """Estimate semantic-boundary lag relative to seed gripper events."""
    if seed_segments is None or seed_features is None:
        return [0] * max(len(transcript) - 1, 0)
    event_index = _build_event_index(seed_features.events)
    maximum_lag = max(6, int(round(0.05 * seed_features.length)))
    lags: list[int] = []
    for boundary_index, segment in enumerate(seed_segments[:-1]):
        skill = _canonical_skill(transcript[boundary_index].get("skill"))
        kinds = _semantic_event_kinds(skill, segment, seed_features)
        boundary = int(segment["end"])
        distance, event = _event_distance(event_index, boundary, kinds)
        if event is None or distance > maximum_lag:
            lags.append(0)
        else:
            lags.append(int(boundary - event.frame))
    return lags


def _holding_calibration(
    transcript: Sequence[Mapping[str, Any]],
    seed_segments: Sequence[Mapping[str, Any]] | None,
    seed_features: FeatureBundle | None,
) -> _HoldingCalibration | None:
    """Infer holding polarity from stable tails of annotated seed segments."""
    if seed_segments is None or seed_features is None:
        return None
    level = (
        seed_features.state_gripper_level
        if seed_features.state_gripper_level is not None
        else seed_features.gripper_level
    )
    states = _holding_expectation(transcript)
    holding_values: list[np.ndarray] = []
    nonholding_values: list[np.ndarray] = []
    for segment, is_holding in zip(seed_segments, states, strict=True):
        start = int(segment["start"])
        end = int(segment["end"])
        tail_start = start + int(round(0.65 * max(end - start, 1)))
        values = np.asarray(level[tail_start:end], dtype=np.float32)
        if not values.size:
            continue
        (holding_values if is_holding else nonholding_values).append(values)
    if not holding_values or not nonholding_values:
        return None
    holding = np.concatenate(holding_values)
    nonholding = np.concatenate(nonholding_values)
    holding_median = float(np.median(holding))
    nonholding_median = float(np.median(nonholding))
    holding_mad = float(np.median(np.abs(holding - holding_median)))
    nonholding_mad = float(np.median(np.abs(nonholding - nonholding_median)))
    separation = abs(holding_median - nonholding_median)
    confidence = separation / (separation + holding_mad + nonholding_mad + 0.05)
    return _HoldingCalibration(
        holding_level=holding_median,
        nonholding_level=nonholding_median,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
    )


def _holding_expectation(transcript: Sequence[Mapping[str, Any]]) -> list[bool]:
    is_holding = False
    states: list[bool] = []
    for step in transcript:
        skill = _canonical_skill(step.get("skill"))
        if skill in PICK_LIKE:
            is_holding = True
        elif skill in PLACE_LIKE:
            is_holding = False
        states.append(is_holding)
    return states


def _candidate_frames(
    anchor: int,
    features: FeatureBundle,
    minimum_frame: int,
    maximum_frame: int,
    config: DecoderConfig,
) -> np.ndarray:
    radius = max(8, int(round(features.length * config.local_grid_radius_ratio)))
    local_step = max(1, radius // 18)
    values = set(
        range(
            max(minimum_frame, anchor - radius),
            min(maximum_frame, anchor + radius) + 1,
            local_step,
        )
    )
    values.add(min(max(anchor, minimum_frame), maximum_frame))
    if config.use_gripper_events:
        values.update(event.frame for event in features.events if minimum_frame <= event.frame <= maximum_frame)
    if config.use_kinematics:
        evidence = np.maximum(features.turn, features.pause)
        keep_count = min(config.candidate_keep, evidence.size)
        strongest = np.argpartition(evidence, -keep_count)[-keep_count:]
        values.update(int(frame) for frame in strongest if minimum_frame <= frame <= maximum_frame)
    if config.use_visual and np.any(features.visual_motion):
        visual_count = min(32, features.length)
        strongest = np.argpartition(features.visual_motion, -visual_count)[-visual_count:]
        values.update(int(frame) for frame in strongest if minimum_frame <= frame <= maximum_frame)
    ordered = sorted(values, key=lambda frame: (abs(frame - anchor), frame))[: config.candidate_keep]
    if config.retain_strong_events and config.use_gripper_events:
        # Strong physical evidence survives template-distance truncation. The
        # global decoder still enforces transcript order and semantic costs.
        ordered = list(set(ordered) | {
            event.frame for event in features.events
            if event.strength >= config.strong_event_threshold
            and minimum_frame <= event.frame <= maximum_frame
        })
    return np.asarray(sorted(ordered), dtype=np.int32)


def _local_boundary_cost(
    frame: int,
    boundary_index: int,
    anchor: int,
    activity_anchor: int,
    expected_progress: float,
    transcript: Sequence[Mapping[str, Any]],
    target_features: FeatureBundle,
    target_progress: np.ndarray,
    visual_unit: np.ndarray,
    event_index: _EventIndex,
    seed_segments: Sequence[Mapping[str, Any]] | None,
    seed_features: FeatureBundle | None,
    seed_signature: np.ndarray | None,
    holding_states: Sequence[bool],
    holding_calibration: _HoldingCalibration | None,
    seed_event_lag: int,
    config: DecoderConfig,
) -> tuple[float, dict[str, Any]]:
    length = target_features.length
    previous_skill = _canonical_skill(transcript[boundary_index].get("skill"))
    position_cost = abs(frame - anchor) / length
    activity_cost = abs(float(target_progress[frame - 1]) - expected_progress)
    signature_cost = 0.0
    if seed_signature is not None:
        signature_cost = float(np.mean(np.abs(_boundary_signature(target_features, frame) - seed_signature)))
    event_cost = 0.0
    nearest_event: SignalEvent | None = None
    if config.use_gripper_events and previous_skill in PICK_LIKE | PLACE_LIKE | CONTACT_LIKE:
        seed_segment = seed_segments[boundary_index] if seed_segments is not None else None
        kinds = (
            _semantic_event_kinds(previous_skill, seed_segment, seed_features)
            if config.use_semantic_events
            else {"gripper_increase", "gripper_decrease"}
        )
        distance, nearest_event = _event_distance(
            event_index,
            frame,
            kinds,
            lag=seed_event_lag if config.use_seed_event_lag else 0,
        )
        event_cost = min(float(distance) / max(0.12 * length, 1.0), 1.0)
        if nearest_event is not None:
            event_cost *= 1.0 - 0.35 * nearest_event.strength
            if config.use_retry_filter:
                event_cost += 0.8 * (
                    1.0 - _persistence_score(target_features, nearest_event.frame)
                )
    is_gripper_boundary = previous_skill in PICK_LIKE | PLACE_LIKE | CONTACT_LIKE
    persistence_cost = 1.0 - _persistence_score(target_features, frame)
    if config.use_conditional_persistence and not is_gripper_boundary:
        persistence_cost = 0.0
    pause_cost = 1.0 - float(target_features.pause[frame])
    turn_cost = 1.0 - float(target_features.turn[frame])
    visual_cost = 1.0 - float(visual_unit[frame])
    holding_cost = 0.0
    if config.use_holding_state:
        target_level = (
            target_features.state_gripper_level
            if target_features.state_gripper_level is not None
            else target_features.gripper_level
        )
        level = float(target_level[frame])
        expected_holding = holding_states[boundary_index]
        if config.use_calibrated_holding and holding_calibration is not None:
            expected_level = (
                holding_calibration.holding_level
                if expected_holding
                else holding_calibration.nonholding_level
            )
            holding_cost = holding_calibration.confidence * abs(
                level - expected_level
            )
        else:
            holding_cost = (
                min(level, 1.0 - level)
                if expected_holding
                else 0.5 * min(level, 1.0 - level)
            )
    total = (
        config.position_weight * position_cost
        + config.activity_weight * activity_cost
        + config.signature_weight * signature_cost
        + config.gripper_weight * event_cost
        + config.persistence_weight * persistence_cost
        + config.pause_weight * pause_cost
        + config.turn_weight * turn_cost
        + config.holding_weight * holding_cost
        + config.visual_weight * visual_cost
    )
    detail = {
        "boundary_index": boundary_index,
        "frame": frame,
        "ratio_anchor": anchor,
        "activity_anchor": activity_anchor,
        "previous_skill": previous_skill,
        "next_skill": _canonical_skill(transcript[boundary_index + 1].get("skill")),
        "nearest_event": nearest_event.kind if nearest_event is not None else None,
        "nearest_event_frame": nearest_event.frame if nearest_event is not None else None,
        "seed_event_lag": seed_event_lag,
        "position_cost": position_cost,
        "activity_cost": activity_cost,
        "signature_cost": signature_cost,
        "event_cost": event_cost,
        "persistence_cost": persistence_cost,
        "holding_cost": holding_cost,
        "holding_confidence": (
            holding_calibration.confidence
            if holding_calibration is not None
            else 0.0
        ),
        "total_local_cost": total,
    }
    return total, detail


def _duration_cost(actual: int, expected: float) -> float:
    relative = (actual - expected) / max(expected, 1.0)
    absolute = abs(relative)
    return 0.5 * relative * relative if absolute <= 1.0 else absolute - 0.5


def _decode_dynamic_program(
    candidates: Sequence[np.ndarray],
    local_costs: Sequence[dict[int, tuple[float, dict[str, Any]]]],
    expected_durations: np.ndarray,
    length: int,
    minimum_frames: int,
    config: DecoderConfig,
) -> tuple[tuple[int, ...], float, tuple[dict[str, Any], ...]] | None:
    previous_costs: dict[int, float] = {0: 0.0}
    backpointers: list[dict[int, int]] = []
    for boundary_index, boundary_candidates in enumerate(candidates):
        current_costs: dict[int, float] = {}
        current_backpointers: dict[int, int] = {}
        for frame in boundary_candidates:
            best_cost = math.inf
            best_previous = -1
            for previous_frame, previous_cost in previous_costs.items():
                if int(frame) - previous_frame < minimum_frames:
                    continue
                interval_cost = config.duration_weight * _duration_cost(
                    int(frame) - previous_frame, expected_durations[boundary_index]
                )
                candidate_cost = previous_cost + local_costs[boundary_index][int(frame)][0] + interval_cost
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_previous = previous_frame
            if best_previous >= 0:
                current_costs[int(frame)] = best_cost
                current_backpointers[int(frame)] = best_previous
        if not current_costs:
            return None
        previous_costs = current_costs
        backpointers.append(current_backpointers)
    best_last = -1
    best_total = math.inf
    for frame, cost in previous_costs.items():
        if length - frame < minimum_frames:
            continue
        total = cost + config.duration_weight * _duration_cost(length - frame, expected_durations[-1])
        if total < best_total:
            best_last = frame
            best_total = total
    if best_last < 0:
        return None
    reversed_boundaries: list[int] = []
    frame = best_last
    for boundary_index in reversed(range(len(candidates))):
        reversed_boundaries.append(frame)
        frame = backpointers[boundary_index][frame]
    boundaries = tuple(reversed(reversed_boundaries))
    decisions = tuple(local_costs[index][frame][1] for index, frame in enumerate(boundaries))
    return boundaries, best_total, decisions


def _segments_from_boundaries(
    transcript: Sequence[Mapping[str, Any]], boundaries: Sequence[int], length: int
) -> tuple[dict[str, Any], ...]:
    points = (0, *boundaries, length)
    return tuple(
        {
            "skill": _canonical_skill(step.get("skill")),
            "action_text": str(step.get("action_text", "")),
            "start": int(points[index]),
            "end": int(points[index + 1]),
        }
        for index, step in enumerate(transcript)
    )


def decode_trajectory(
    transcript: Sequence[Mapping[str, Any]],
    target_features: FeatureBundle,
    config: DecoderConfig,
    seed_segments: Sequence[Mapping[str, Any]] | None = None,
    seed_features: FeatureBundle | None = None,
) -> DecodeResult:
    """Extend one annotated transcript or blindly self-cut one trajectory.

    Target GT boundaries are not accepted by this API. Seed boundaries are
    used only when ``seed_segments`` belongs to a separate extension seed.
    """
    segment_count = len(transcript)
    if segment_count < 1:
        raise ValueError("transcript must contain at least one step")
    if seed_segments is not None and len(seed_segments) != segment_count:
        raise ValueError("seed segment count must match transcript")
    if segment_count == 1:
        segments = _segments_from_boundaries(transcript, (), target_features.length)
        return DecodeResult((), segments, 0.0, ())
    expected_durations = _expected_durations(transcript, target_features.length, seed_segments, config)
    ratio_anchors = np.rint(np.cumsum(expected_durations)[:-1]).astype(np.int32)
    activity_anchors = (
        _activity_warp_anchors(ratio_anchors, seed_segments, seed_features, target_features)
        if config.use_activity_warp
        else ratio_anchors
    )
    dtw_anchors = (
        _dtw_anchors(seed_segments, seed_features, target_features)
        if config.use_dtw_anchor
        else None
    )
    if dtw_anchors is not None and len(dtw_anchors) == len(ratio_anchors):
        anchors = np.rint(
            np.median(
                np.stack((ratio_anchors, activity_anchors, dtw_anchors), axis=0),
                axis=0,
            )
        ).astype(np.int32)
    else:
        anchors = np.rint(0.45 * ratio_anchors + 0.55 * activity_anchors).astype(
            np.int32
        )
    minimum_frames = max(2, int(round(target_features.length * config.minimum_segment_ratio)))
    signatures = (
        _seed_signatures(seed_segments, seed_features)
        if seed_segments is not None
        else [None] * (segment_count - 1)
    )
    holding_states = _holding_expectation(transcript)
    holding_calibration = (
        _holding_calibration(transcript, seed_segments, seed_features)
        if config.use_calibrated_holding
        else None
    )
    event_lags = (
        _seed_event_lags(transcript, seed_segments, seed_features)
        if config.use_seed_event_lag
        else [0] * (segment_count - 1)
    )
    if config.relative_event_lag and seed_features is not None:
        event_lags = [round(lag*target_features.length/max(seed_features.length, 1)) for lag in event_lags]
    expected_progress = np.cumsum(expected_durations)[:-1] / target_features.length
    if config.seed_activity_progress and seed_features is not None and seed_segments:
        seed_progress = _cumulative_progress(seed_features.activity)
        expected_progress = np.asarray([seed_progress[min(seed_features.length-1, max(0, int(segment['end'])-1))] for segment in seed_segments[:-1]])
    target_progress = _cumulative_progress(target_features.activity)
    visual_unit = robust_unit(target_features.visual_motion)
    event_index = _build_event_index(target_features.events)
    all_candidates: list[np.ndarray] = []
    local_costs: list[dict[int, tuple[float, dict[str, Any]]]] = []
    for boundary_index, anchor in enumerate(anchors):
        minimum_frame = minimum_frames * (boundary_index + 1)
        maximum_frame = target_features.length - minimum_frames * (segment_count - boundary_index - 1)
        candidates = _candidate_frames(int(anchor), target_features, minimum_frame, maximum_frame, config)
        costs = {
            int(frame): _local_boundary_cost(
                int(frame),
                boundary_index,
                int(ratio_anchors[boundary_index]),
                int(activity_anchors[boundary_index]),
                float(expected_progress[boundary_index]),
                transcript,
                target_features,
                target_progress,
                visual_unit,
                event_index,
                seed_segments,
                seed_features,
                signatures[boundary_index],
                holding_states,
                holding_calibration,
                event_lags[boundary_index],
                config,
            )
            for frame in candidates
        }
        all_candidates.append(candidates)
        local_costs.append(costs)
    decoded = _decode_dynamic_program(
        all_candidates,
        local_costs,
        expected_durations,
        target_features.length,
        minimum_frames,
        config,
    )
    fallback_reason: str | None = None
    if decoded is None:
        boundaries = tuple(int(value) for value in ratio_anchors)
        total_cost = math.inf
        decisions = tuple(
            {
                "boundary_index": index,
                "frame": frame,
                "ratio_anchor": frame,
                "reason": "duration_ratio_fallback",
            }
            for index, frame in enumerate(boundaries)
        )
        fallback_reason = "no_feasible_dynamic_program_path"
    else:
        boundaries, total_cost, decisions = decoded
    return DecodeResult(
        boundaries=boundaries,
        segments=_segments_from_boundaries(transcript, boundaries, target_features.length),
        total_cost=total_cost,
        decisions=decisions,
        fallback_reason=fallback_reason,
    )

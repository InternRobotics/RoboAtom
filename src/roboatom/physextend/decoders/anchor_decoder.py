"""Retry-aware semantic anchors for strict single-seed trajectory extension.

The decoder first computes the frozen ``semantic_events`` path.  It
then discovers high-precision Pick/Place gripper-event anchors from exactly one
annotated seed.  Nearby gripper transitions are consolidated before matching,
and unresolved repeated attempts abstain instead of forcing a boundary.

No target annotation, learned representation, dataset name, or additional
annotated seed is accepted by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from roboatom.physextend.decoders.coarse_decoder import (
    PICK_LIKE,
    PLACE_LIKE,
    DecodeResult,
    semantic_event_config,
    decode_trajectory,
    _decode_dynamic_program,
)
from roboatom.physextend.features.signals import FeatureBundle, SignalEvent
from roboatom.physextend.decoders.reference_decoder import SeedTrajectory


RetryPolicy = Literal["reject", "group", "seed_group"]


@dataclass(frozen=True)
class AnchorDecoderConfig:
    """Configuration for one pre-registered single-seed anchor hypothesis.

    Args:
        name: Stable experiment identifier.
        hypothesis: Human-readable falsifiable claim.
        anchor_weight: Fraction of an accepted anchor correction to apply.
        cluster_radius_ratio: Maximum normalized gap inside one event cluster.
        search_radius_ratio: Target-local bidirectional search radius.
        minimum_cluster_quality: Minimum transition quality to consider.
        minimum_seed_confidence: Minimum seed template confidence.
        minimum_target_confidence: Minimum target match confidence.
        uniqueness_margin: Required score gap over the second candidate.
        retry_policy: Whether separated close/open/close attempts are rejected
            or grouped into one semantic action.
        use_interval_warp: Whether uncertain boundaries between accepted
            anchors inherit a piecewise-linear correction.
    """

    name: str
    hypothesis: str
    anchor_weight: float
    cluster_radius_ratio: float = 0.012
    search_radius_ratio: float = 0.10
    minimum_cluster_quality: float = 0.0
    minimum_seed_confidence: float = 0.0
    minimum_target_confidence: float = 0.0
    uniqueness_margin: float = 0.0
    retry_policy: RetryPolicy = "reject"
    use_interval_warp: bool = False
    coarse_overrides: Mapping[str, Any] = field(default_factory=dict)
    allow_obvious_ambiguous: bool = False
    allow_obvious_seed: bool = False
    confidence_weighted_anchors: bool = False
    global_proposals: bool = False
    global_conflicts_only: bool = False
    match_gripper_channel: bool = False
    channel_mismatch_penalty: float = 0.0
    separate_polarity_clusters: bool = False
    seed_event_warp: bool = False
    event_warp_interpolation_only: bool = False
    hard_high_confidence_anchors: bool = False
    event_warp_adjacent_only: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.hypothesis:
            raise ValueError("Anchor config name and hypothesis must be non-empty")
        for field_name, value in (
            ("anchor_weight", self.anchor_weight),
            ("cluster_radius_ratio", self.cluster_radius_ratio),
            ("search_radius_ratio", self.search_radius_ratio),
            ("minimum_cluster_quality", self.minimum_cluster_quality),
            ("minimum_seed_confidence", self.minimum_seed_confidence),
            ("minimum_target_confidence", self.minimum_target_confidence),
            ("uniqueness_margin", self.uniqueness_margin),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be within [0, 1]")


@dataclass(frozen=True)
class EventCluster:
    """One consolidated physical gripper transition."""

    start: int
    end: int
    frame: int
    kind: str
    quality: float
    persistence: float
    purity: float
    after_level: float
    event_count: int
    channel_key: str | None = None


@dataclass(frozen=True)
class _AnchorTemplate:
    """Seed-derived semantic-event location inside one labeled segment."""

    segment_index: int
    skill: str
    kind: str
    event_fraction: float
    confidence: float
    seed_event_frame: int
    has_seed_retry: bool
    channel_key: str | None = None


@dataclass(frozen=True)
class _TargetMatch:
    """One target event or retry group assigned to a semantic template."""

    template: _AnchorTemplate
    selected: EventCluster
    first_attempt: EventCluster
    confidence: float
    candidate_count: int
    is_retry_group: bool


@dataclass(frozen=True)
class AnchorDecodeResult:
    """Single-seed anchored segmentation and abstention diagnostics."""

    boundaries: tuple[int, ...]
    segments: tuple[dict[str, Any], ...]
    total_cost: float
    decisions: tuple[dict[str, Any], ...]
    diagnostics: Mapping[str, Any]
    fallback_reason: str | None = None


def anchor_config() -> AnchorDecoderConfig:
    """Return the semantic-anchor configuration used by PhysExtend."""
    return AnchorDecoderConfig(
        name="unique_anchor_soft75",
        hypothesis="A unique seed-calibrated event supports a stronger bounded correction.",
        anchor_weight=0.75,
        minimum_cluster_quality=0.55,
        minimum_seed_confidence=0.60,
        minimum_target_confidence=0.60,
        uniqueness_margin=0.12,
    )


def _canonical_skill(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _gripper_level(features: FeatureBundle) -> np.ndarray:
    return np.asarray(
        features.state_gripper_level
        if features.state_gripper_level is not None
        else features.gripper_level,
        dtype=np.float64,
    )


def _weighted_median_frame(events: Sequence[SignalEvent], kind: str) -> int:
    matching = [event for event in events if event.kind == kind]
    selected = matching or list(events)
    ordered = sorted(selected, key=lambda event: event.frame)
    total = sum(max(event.strength, 1e-6) for event in ordered)
    cumulative = 0.0
    for event in ordered:
        cumulative += max(event.strength, 1e-6)
        if cumulative >= 0.5 * total:
            return event.frame
    return ordered[-1].frame


def _cluster_from_events(
    events: Sequence[SignalEvent], features: FeatureBundle, radius: int
) -> EventCluster:
    level = _gripper_level(features)
    start = min(event.frame for event in events)
    end = max(event.frame for event in events)
    tracks = features.embodiment.get('gripper_tracks')
    channel_key = None
    if tracks is not None and start > 0 and end+1 < features.length:
        before_tracks = np.median(tracks[max(0, start-radius):start], axis=0)
        after_tracks = np.median(tracks[end+1:min(features.length, end+radius+1)], axis=0)
        channel = int(np.argmax(np.abs(after_tracks-before_tracks)))
        level = tracks[:, channel]
        keys = features.embodiment.get('gripper_keys', [])
        channel_key = keys[channel] if channel < len(keys) else None
    before = level[max(0, start - radius) : start]
    after = level[end + 1 : min(features.length, end + radius + 1)]
    before_level = float(np.median(before)) if before.size else float(level[start])
    after_level = float(np.median(after)) if after.size else float(level[end])
    shift = after_level - before_level
    strengths = {
        kind: sum(event.strength for event in events if event.kind == kind)
        for kind in ("gripper_increase", "gripper_decrease")
    }
    dominant_kind = max(strengths, key=lambda kind: (strengths[kind], kind))
    if abs(shift) >= 0.03:
        kind = "gripper_increase" if shift > 0.0 else "gripper_decrease"
    else:
        kind = dominant_kind
    total_strength = max(sum(strengths.values()), 1e-6)
    purity = strengths[kind] / total_strength
    instability = float(np.median(np.abs(after - after_level))) if after.size else 1.0
    persistence = abs(shift) / (abs(shift) + instability + 0.05)
    peak_strength = max(event.strength for event in events if event.kind == kind) if any(
        event.kind == kind for event in events
    ) else max(event.strength for event in events)
    quality = 0.40 * peak_strength + 0.35 * persistence + 0.25 * purity
    if features.action_state_error is not None and after.size:
        error_slice = features.action_state_error[
            end + 1 : min(features.length, end + radius + 1)
        ]
        if error_slice.size:
            quality *= 1.0 - 0.25 * float(np.clip(np.median(error_slice), 0.0, 1.0))
    return EventCluster(
        start=start,
        end=end,
        frame=_weighted_median_frame(events, kind),
        kind=kind,
        quality=float(np.clip(quality, 0.0, 1.0)),
        persistence=float(np.clip(persistence, 0.0, 1.0)),
        purity=float(np.clip(purity, 0.0, 1.0)),
        after_level=after_level,
        event_count=len(events),
        channel_key=channel_key,
    )


def cluster_gripper_events(
    features: FeatureBundle, radius_ratio: float, separate_polarity: bool = False
) -> tuple[EventCluster, ...]:
    """Consolidate temporally adjacent gripper events into physical changes.

    Adjacent opposite-polarity peaks are intentionally merged too: dual-arm
    transitions and controller chatter often produce several peaks for one
    physical manipulation event.  A later event is independent only after a
    gap larger than the scale-normalized cluster radius.

    Args:
        features: Boundary-free robot signal features.
        radius_ratio: Maximum normalized gap within a cluster.

    Returns:
        Ordered consolidated transition clusters.
    """
    radius = max(6, int(round(features.length * radius_ratio)))
    events = sorted(
        (
            event
            for event in features.events
            if event.kind in {"gripper_increase", "gripper_decrease"}
            and event.strength >= 0.08
        ),
        key=lambda event: (event.frame, event.kind),
    )
    if not events:
        return ()
    groups: list[list[SignalEvent]] = [[events[0]]]
    for event in events[1:]:
        if event.frame - groups[-1][-1].frame <= radius and (not separate_polarity or event.kind == groups[-1][-1].kind):
            groups[-1].append(event)
        else:
            groups.append([event])
    return tuple(_cluster_from_events(group, features, radius) for group in groups)


def _candidate_score(
    cluster: EventCluster, expected_frame: float, radius: int, template_confidence: float
) -> float:
    location = 1.0 - min(abs(cluster.frame - expected_frame) / max(radius, 1), 1.0)
    return float(
        np.clip(
            0.50 * cluster.quality + 0.35 * location + 0.15 * template_confidence,
            0.0,
            1.0,
        )
    )


def _seed_templates(
    transcript: Sequence[Mapping[str, Any]],
    seed: SeedTrajectory,
    config: AnchorDecoderConfig,
) -> tuple[list[_AnchorTemplate], list[dict[str, Any]]]:
    clusters = cluster_gripper_events(seed.features, config.cluster_radius_ratio, config.separate_polarity_clusters)
    templates: list[_AnchorTemplate] = []
    diagnostics: list[dict[str, Any]] = []
    level = _gripper_level(seed.features)
    for segment_index, (step, segment) in enumerate(
        zip(transcript, seed.segments, strict=True)
    ):
        skill = _canonical_skill(step.get("skill"))
        if skill not in PICK_LIKE | PLACE_LIKE:
            continue
        start = int(segment["start"])
        end = int(segment["end"])
        duration = max(end - start, 1)
        tail = level[max(start, end - max(4, duration // 8)) : end]
        tail_level = float(np.median(tail)) if tail.size else float(level[end - 1])
        candidates = [cluster for cluster in clusters if start <= cluster.frame < end]
        ranked = sorted(
            (
                (
                    0.70 * cluster.quality
                    + 0.30 * (1.0 - min(abs(cluster.after_level - tail_level), 1.0)),
                    cluster,
                )
                for cluster in candidates
            ),
            key=lambda item: (item[0], item[1].frame),
            reverse=True,
        )
        if not ranked:
            diagnostics.append({"segment_index": segment_index, "reason": "no_seed_event"})
            continue
        best_score, selected = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        uniqueness = min(max(best_score - second_score, 0.0) / 0.25, 1.0)
        confidence = float(np.clip(0.75 * best_score + 0.25 * uniqueness, 0.0, 1.0))
        same_kind = [cluster for cluster in candidates if cluster.kind == selected.kind]
        has_retry = _has_reversal_between(candidates, same_kind[0], same_kind[-1]) if len(same_kind) > 1 else False
        reason = "accepted"
        if selected.quality < config.minimum_cluster_quality:
            reason = "low_seed_cluster_quality"
        elif confidence < config.minimum_seed_confidence:
            reason = "low_seed_confidence"
        elif config.uniqueness_margin and best_score - second_score < config.uniqueness_margin and not (
            config.allow_obvious_seed and selected.quality >= 0.80 and confidence >= 0.60
        ):
            reason = "ambiguous_seed_event"
        if reason == "accepted":
            templates.append(
                _AnchorTemplate(
                    segment_index=segment_index,
                    skill=skill,
                    kind=selected.kind,
                    event_fraction=float(np.clip((selected.frame - start) / duration, 0.0, 1.0)),
                    confidence=confidence,
                    seed_event_frame=selected.frame,
                    has_seed_retry=has_retry,
                    channel_key=selected.channel_key,
                )
            )
        diagnostics.append(
            {
                "segment_index": segment_index,
                "skill": skill,
                "reason": reason,
                "candidate_count": len(candidates),
                "selected_frame": selected.frame,
                "selected_kind": selected.kind,
                "selected_quality": selected.quality,
                "confidence": confidence,
                "score_margin": best_score - second_score,
                "has_retry": has_retry,
            }
        )
    return templates, diagnostics


def _has_reversal_between(
    clusters: Sequence[EventCluster], first: EventCluster, last: EventCluster
) -> bool:
    return any(
        first.frame < cluster.frame < last.frame and cluster.kind != first.kind
        for cluster in clusters
    )


def _expected_template_frames(
    templates: Sequence[_AnchorTemplate], base_result: DecodeResult, length: int
) -> list[float]:
    points = (0, *base_result.boundaries, length)
    return [
        points[template.segment_index]
        + template.event_fraction
        * (points[template.segment_index + 1] - points[template.segment_index])
        for template in templates
    ]


def _match_target_template(
    template: _AnchorTemplate,
    expected_frame: float,
    lower: int,
    upper: int,
    clusters: Sequence[EventCluster],
    config: AnchorDecoderConfig,
    radius: int,
) -> tuple[_TargetMatch | None, dict[str, Any]]:
    local = [cluster for cluster in clusters if lower <= cluster.frame <= upper]
    matching = [cluster for cluster in local if cluster.kind == template.kind and (
        not config.match_gripper_channel or template.channel_key is None or cluster.channel_key == template.channel_key
    )]
    ranked = sorted(
        ((_candidate_score(cluster, expected_frame, radius, template.confidence) - (
            config.channel_mismatch_penalty if template.channel_key is not None and cluster.channel_key != template.channel_key else 0.0
        ), cluster) for cluster in matching),
        key=lambda item: (item[0], -abs(item[1].frame - expected_frame), item[1].frame),
        reverse=True,
    )
    if not ranked:
        return None, {"segment_index": template.segment_index, "reason": "no_target_event"}
    selected_score, selected = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    first_attempt = selected
    is_retry_group = False
    has_unresolved_repeat = False
    if len(matching) > 1:
        ordered = sorted(matching, key=lambda cluster: cluster.frame)
        reversal_candidates = [
            cluster
            for cluster in ordered
            if _has_reversal_between(local, ordered[0], cluster)
        ]
        if reversal_candidates and (config.retry_policy == "group" or (config.retry_policy == "seed_group" and template.has_seed_retry)):
            persistent = [
                cluster
                for cluster in reversal_candidates
                if cluster.quality >= config.minimum_cluster_quality
            ]
            if persistent:
                selected = persistent[-1]
                selected_score = _candidate_score(
                    selected, expected_frame, radius, template.confidence
                )
                first_attempt = ordered[0]
                is_retry_group = True
            else:
                has_unresolved_repeat = True
        elif reversal_candidates:
            has_unresolved_repeat = True
    confidence = float(np.clip(0.75 * selected_score + 0.25 * template.confidence, 0.0, 1.0))
    score_margin = selected_score - second_score
    reason = "accepted"
    if selected.quality < config.minimum_cluster_quality:
        reason = "low_target_cluster_quality"
    elif confidence < config.minimum_target_confidence:
        reason = "low_target_confidence"
    elif has_unresolved_repeat:
        reason = "unresolved_retry"
    elif config.uniqueness_margin and not is_retry_group and score_margin < config.uniqueness_margin and not (
        config.allow_obvious_ambiguous and confidence >= 0.85 and selected.quality >= 0.80
    ):
        reason = "ambiguous_target_event"
    detail = {
        "segment_index": template.segment_index,
        "skill": template.skill,
        "reason": reason,
        "expected_frame": expected_frame,
        "search_window": [lower, upper],
        "candidate_count": len(matching),
        "selected_frame": selected.frame,
        "first_attempt_frame": first_attempt.frame,
        "selected_kind": selected.kind,
        "selected_quality": selected.quality,
        "confidence": confidence,
        "score_margin": score_margin,
        "is_retry_group": is_retry_group,
        "obvious_ambiguity_override": bool(config.allow_obvious_ambiguous and reason == 'accepted' and score_margin < config.uniqueness_margin),
    }
    if reason != "accepted":
        return None, detail
    return (
        _TargetMatch(
            template=template,
            selected=selected,
            first_attempt=first_attempt,
            confidence=confidence,
            candidate_count=len(matching),
            is_retry_group=is_retry_group,
        ),
        detail,
    )


def _target_matches(
    templates: Sequence[_AnchorTemplate],
    base_result: DecodeResult,
    target_features: FeatureBundle,
    config: AnchorDecoderConfig,
) -> tuple[list[_TargetMatch], list[dict[str, Any]]]:
    clusters = cluster_gripper_events(target_features, config.cluster_radius_ratio, config.separate_polarity_clusters)
    expected = _expected_template_frames(templates, base_result, target_features.length)
    radius = max(8, int(round(target_features.length * config.search_radius_ratio)))
    matches: list[_TargetMatch] = []
    diagnostics: list[dict[str, Any]] = []
    for index, template in enumerate(templates):
        midpoint_before = 0 if index == 0 else int(math.floor(0.5 * (expected[index - 1] + expected[index])))
        midpoint_after = (
            target_features.length - 1
            if index + 1 == len(templates)
            else int(math.ceil(0.5 * (expected[index] + expected[index + 1])))
        )
        lower = max(0, int(math.floor(expected[index] - radius)), midpoint_before)
        upper = min(target_features.length - 1, int(math.ceil(expected[index] + radius)), midpoint_after)
        match, detail = _match_target_template(
            template,
            expected[index],
            lower,
            upper,
            clusters,
            config,
            radius,
        )
        diagnostics.append(detail)
        if match is not None:
            matches.append(match)
    return matches, diagnostics


def _boundary_proposals(
    matches: Sequence[_TargetMatch],
    base_result: DecodeResult,
    segment_count: int,
    length: int,
    global_constraints: bool = False,
    conflicts_only: bool = False,
) -> tuple[dict[int, float], dict[int, list[dict[str, Any]]]]:
    points = (0, *base_result.boundaries, length)
    raw: dict[int, list[tuple[float, float, dict[str, Any]]]] = {}
    for match in matches:
        segment_index = match.template.segment_index
        duration = max(points[segment_index + 1] - points[segment_index], 1)
        fraction = match.template.event_fraction
        start = match.first_attempt.frame - fraction * duration
        end = match.selected.frame + (1.0 - fraction) * duration
        detail = {
            "segment_index": segment_index,
            "skill": match.template.skill,
            "confidence": match.confidence,
            "is_retry_group": match.is_retry_group,
        }
        if segment_index > 0:
            raw.setdefault(segment_index - 1, []).append((start, match.confidence, detail))
        if segment_index < segment_count - 1:
            raw.setdefault(segment_index, []).append((end, match.confidence, detail))
    proposals: dict[int, float] = {}
    diagnostics: dict[int, list[dict[str, Any]]] = {}
    conflict_limit = max(6.0, 0.06 * length)
    for boundary_index, values in raw.items():
        diagnostics[boundary_index] = [
            {**detail, "proposed_frame": frame} for frame, _, detail in values
        ]
        frames = np.asarray([value[0] for value in values], dtype=np.float64)
        weights = np.asarray([value[1] for value in values], dtype=np.float64)
        if frames.size > 1 and float(np.ptp(frames)) > conflict_limit:
            continue
        proposals[boundary_index] = float(np.average(frames, weights=weights))
    if global_constraints and raw:
        candidates, costs = [], []
        durations = np.diff(points).astype(np.float64)
        for index, baseline in enumerate(base_result.boundaries):
            values = raw.get(index, [])
            options = {int(baseline)}
            options.update(max(1, min(length-1, round(value[0]))) for value in values)
            if values:
                options.add(max(1, min(length-1, round(float(np.average([v[0] for v in values], weights=[v[1] for v in values]))))))
            candidates.append(np.asarray(sorted(options), dtype=np.int32))
            scale = max(1.0, min(durations[index:index+2]))
            costs.append({frame: (sum(confidence*abs(frame-proposal)/scale for proposal, confidence, _ in values)/max(sum(v[1] for v in values), 1e-8) + 0.15*abs(frame-baseline)/scale, {}) for frame in options})
        solved = _decode_dynamic_program(candidates, costs, durations, length, 1, semantic_event_config())
        if solved is not None:
            proposals.update({index: float(solved[0][index]) for index in raw if not conflicts_only or index not in proposals})
    return proposals, diagnostics


def _piecewise_warp(
    baseline: np.ndarray, corrected: np.ndarray, anchored: set[int], length: int
) -> np.ndarray:
    base_points = np.asarray([0.0, *baseline.tolist(), float(length)], dtype=np.float64)
    target_points = np.asarray([0.0, *corrected.tolist(), float(length)], dtype=np.float64)
    anchor_points = [0, *(index + 1 for index in sorted(anchored)), baseline.size + 1]
    warped = corrected.copy()
    for left, right in zip(anchor_points, anchor_points[1:]):
        base_span = base_points[right] - base_points[left]
        target_span = target_points[right] - target_points[left]
        if base_span <= 0.0:
            continue
        for point_index in range(left + 1, right):
            ratio = (base_points[point_index] - base_points[left]) / base_span
            warped[point_index - 1] = target_points[left] + ratio * target_span
    return warped


def _project_boundaries(values: np.ndarray, length: int, segment_count: int) -> tuple[int, ...]:
    if values.size == 0:
        return ()
    minimum = max(2, int(round(length * 0.012)))
    if segment_count * minimum > length:
        minimum = 1
    projected = np.rint(values).astype(np.int64)
    for index in range(projected.size):
        lower = minimum * (index + 1)
        upper = length - minimum * (segment_count - index - 1)
        if index:
            lower = max(lower, int(projected[index - 1]) + minimum)
        projected[index] = min(max(projected[index], lower), upper)
    for index in range(projected.size - 1, -1, -1):
        upper = length - minimum * (segment_count - index - 1)
        if index + 1 < projected.size:
            upper = min(upper, int(projected[index + 1]) - minimum)
        projected[index] = min(projected[index], upper)
    return tuple(int(value) for value in projected)


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


def decode_anchor(
    transcript: Sequence[Mapping[str, Any]],
    target_features: FeatureBundle,
    seed: SeedTrajectory,
    config: AnchorDecoderConfig,
) -> AnchorDecodeResult:
    """Decode one target from exactly one annotated seed.

    Args:
        transcript: Ordered skill sequence copied from the single seed.
        target_features: Boundary-free target robot signals.
        seed: The only annotated source trajectory available to the decoder.
        config: Pre-registered semantic-anchor hypothesis.

    Returns:
        Monotone full-cover segmentation with anchor and abstention provenance.

    Raises:
        ValueError: If the seed and transcript are empty or incompatible.
    """
    if not transcript:
        raise ValueError("Transcript must contain at least one step")
    if len(seed.segments) != len(transcript):
        raise ValueError("Seed segment count does not match transcript")
    transcript_skills = tuple(_canonical_skill(step.get("skill")) for step in transcript)
    seed_skills = tuple(_canonical_skill(segment.get("skill")) for segment in seed.segments)
    if transcript_skills != seed_skills:
        raise ValueError("Seed skill sequence does not match transcript")
    round4 = semantic_event_config()
    if config.coarse_overrides:
        round4 = replace(round4, **config.coarse_overrides)
    baseline = decode_trajectory(
        transcript,
        target_features,
        round4,
        seed_segments=seed.segments,
        seed_features=seed.features,
    )
    if not baseline.boundaries:
        return AnchorDecodeResult(
            boundaries=baseline.boundaries,
            segments=baseline.segments,
            total_cost=baseline.total_cost,
            decisions=baseline.decisions,
            diagnostics={"seed_templates": [], "target_matches": [], "proposals": {}},
            fallback_reason=baseline.fallback_reason,
        )
    templates, seed_diagnostics = _seed_templates(transcript, seed, config)
    matches, target_diagnostics = _target_matches(
        templates, baseline, target_features, config
    )
    proposals, proposal_diagnostics = _boundary_proposals(
        matches, baseline, len(transcript), target_features.length, config.global_proposals, config.global_conflicts_only
    )
    event_warp_applied = False
    if config.seed_event_warp and len(matches) >= 2:
        pairs = sorted((match.template.seed_event_frame, match.selected.frame, match.template.segment_index) for match in matches
                       if 0 < match.template.seed_event_frame < seed.features.length)
        source_points = [0, *(pair[0] for pair in pairs), seed.features.length]
        target_points = [0, *(pair[1] for pair in pairs), target_features.length]
        if np.all(np.diff(source_points) > 0) and np.all(np.diff(target_points) > 0):
            def supported(index: int) -> bool:
                boundary = int(seed.segments[index]['end'])
                if config.event_warp_interpolation_only and not pairs[0][0] <= boundary <= pairs[-1][0]:
                    return False
                return not config.event_warp_adjacent_only or any(left[0] <= boundary <= right[0] and right[2]-left[2] == 1 for left, right in zip(pairs, pairs[1:]))
            warped = {index: float(np.interp(int(seed.segments[index]['end']), source_points, target_points)) for index in proposal_diagnostics if supported(index)}
            proposals.update(warped)
            event_warp_applied = bool(warped)
    baseline_values = np.asarray(baseline.boundaries, dtype=np.float64)
    corrected = baseline_values.copy()
    for boundary_index, proposal in proposals.items():
        weight = config.anchor_weight
        if config.hard_high_confidence_anchors and min(detail['confidence'] for detail in proposal_diagnostics[boundary_index]) >= 0.90:
            weight = 1.0
        if config.confidence_weighted_anchors:
            weight *= max(detail['confidence'] for detail in proposal_diagnostics[boundary_index])
        corrected[boundary_index] = (
            (1.0 - weight) * baseline_values[boundary_index]
            + weight * proposal
        )
    if config.use_interval_warp and proposals:
        corrected = _piecewise_warp(
            baseline_values, corrected, set(proposals), target_features.length
        )
    boundaries = _project_boundaries(corrected, target_features.length, len(transcript))
    decisions = tuple(
        {
            **dict(baseline.decisions[index]),
            "frame": boundary,
            "coarse_frame": baseline.boundaries[index],
            "anchor_config": config.name,
            "anchor_applied": index in proposals,
            "anchor_proposals": proposal_diagnostics.get(index, []),
        }
        for index, boundary in enumerate(boundaries)
    )
    diagnostics = {
        "single_seed_episode_key": seed.episode_key,
        "seed_count": 1,
        "seed_templates": seed_diagnostics,
        "target_matches": target_diagnostics,
        "proposals": {str(index): value for index, value in proposals.items()},
        "accepted_template_count": len(templates),
        "accepted_match_count": len(matches),
        "anchored_boundary_count": len(proposals),
        "retry_group_count": sum(match.is_retry_group for match in matches),
        "event_seed_warp_applied": event_warp_applied,
    }
    return AnchorDecodeResult(
        boundaries=boundaries,
        segments=_segments_from_boundaries(transcript, boundaries, target_features.length),
        total_cost=baseline.total_cost,
        decisions=decisions,
        diagnostics=diagnostics,
        fallback_reason=baseline.fallback_reason,
    )

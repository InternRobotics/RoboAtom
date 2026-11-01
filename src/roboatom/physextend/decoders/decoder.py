"""Unified single- and multi-seed decoder for optimized trajectory rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from roboatom.physextend.decoders.coarse_decoder import semantic_event_config, decode_trajectory
from roboatom.physextend.features.signals import FeatureBundle
from roboatom.physextend.decoders.reference_decoder import SeedTrajectory
from roboatom.physextend.decoders.anchor_decoder import (
    AnchorDecodeResult,
    AnchorDecoderConfig,
    decode_anchor,
    anchor_config,
)

BaseDecoder = Literal["coarse", "anchor"]
Aggregation = Literal["mean", "guarded_mean"]
CANONICAL_OURS_CONFIG_NAME = "ours_semantic_anchor_mean"


@dataclass(frozen=True)
class OursDecoderConfig:
    """Configuration for independent per-seed decoding and aggregation."""

    name: str
    hypothesis: str
    base_decoder: BaseDecoder = "anchor"
    aggregation: Aggregation = "mean"
    consensus_weight: float = 1.0
    disagreement_ratio: float = 0.08

    def __post_init__(self) -> None:
        if not self.name or not self.hypothesis:
            raise ValueError("Ours config name and hypothesis must be non-empty")
        if not 0.0 <= self.consensus_weight <= 1.0:
            raise ValueError("consensus_weight must be within [0, 1]")
        if not 0.0 < self.disagreement_ratio <= 1.0:
            raise ValueError("disagreement_ratio must be within (0, 1]")


@dataclass(frozen=True)
class OursDecodeResult:
    """Full-cover segmentation and single/multi-seed diagnostics."""

    boundaries: tuple[int, ...]
    segments: tuple[dict[str, Any], ...]
    total_cost: float
    decisions: tuple[dict[str, Any], ...]
    diagnostics: Mapping[str, Any]
    fallback_reason: str | None = None


def canonical_ours_config() -> OursDecoderConfig:
    """Return the stable production config for any positive seed count."""
    return OursDecoderConfig(
        name=CANONICAL_OURS_CONFIG_NAME,
        hypothesis=(
            "Semantic anchors decode every seed independently; corresponding "
            "boundaries are averaged and projected monotonically."
        ),
        disagreement_ratio=0.08,
    )


def _canonical_skill(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _validate_seeds(
    transcript: Sequence[Mapping[str, Any]], seeds: Sequence[SeedTrajectory]
) -> None:
    if not transcript:
        raise ValueError("Transcript must contain at least one step")
    if not seeds:
        raise ValueError("At least one annotated seed is required")
    transcript_skills = tuple(_canonical_skill(step.get("skill")) for step in transcript)
    seen: set[str] = set()
    for seed in seeds:
        if seed.episode_key in seen:
            raise ValueError(f"Duplicate seed episode: {seed.episode_key}")
        seen.add(seed.episode_key)
        seed_skills = tuple(
            _canonical_skill(segment.get("skill")) for segment in seed.segments
        )
        if seed_skills != transcript_skills:
            raise ValueError(
                f"Seed {seed.episode_key} skill sequence does not match transcript"
            )


def _decode_single_seed(
    transcript: Sequence[Mapping[str, Any]],
    target_features: FeatureBundle,
    seed: SeedTrajectory,
    base_decoder: BaseDecoder,
    anchor_config_override: AnchorDecoderConfig | None = None,
) -> AnchorDecodeResult:
    if base_decoder == "anchor":
        return decode_anchor(
            transcript,
            target_features,
            seed,
            anchor_config_override or anchor_config(),
        )
    result = decode_trajectory(
        transcript,
        target_features,
        semantic_event_config(),
        seed_segments=seed.segments,
        seed_features=seed.features,
    )
    return AnchorDecodeResult(
        boundaries=result.boundaries,
        segments=result.segments,
        total_cost=result.total_cost,
        decisions=result.decisions,
        diagnostics={"single_seed_episode_key": seed.episode_key, "seed_count": 1},
        fallback_reason=result.fallback_reason,
    )


def project_boundaries(
    values: np.ndarray, length: int, segment_count: int
) -> tuple[int, ...]:
    """Round and project internal boundaries to a monotone full cover."""
    if values.size == 0:
        return ()
    minimum = max(2, round(length * 0.012))
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


def posthoc_boundary_ablations(
    paths: np.ndarray,
    length: int,
    segment_count: int,
    guarded_ratio: float = 0.08,
    inlier_ratio: float = 0.05,
) -> dict[str, list[int]]:
    """Compute frozen non-primary aggregation alternatives without labels."""
    array = np.asarray(paths, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1:
        raise ValueError("paths must be a non-empty seed-by-boundary matrix")
    primary = array[0]
    mean_path = np.mean(array, axis=0)
    median_path = np.median(array, axis=0)
    spread = np.ptp(array, axis=0)
    guarded_path = np.where(spread <= guarded_ratio * length, mean_path, primary)
    inlier_values: list[float] = []
    maximum_inlier_distance = inlier_ratio * length
    for boundary_index, median_value in enumerate(median_path):
        values = array[:, boundary_index]
        keep = np.abs(values - median_value) <= maximum_inlier_distance
        inlier_values.append(
            float(np.mean(values[keep])) if np.any(keep) else float(median_value)
        )
    candidates = {
        "mean": mean_path,
        "median": median_path,
        "guarded_primary_8pct": guarded_path,
        "median_centered_inlier_mean_5pct": np.asarray(inlier_values),
    }
    return {
        name: list(project_boundaries(values, length, segment_count))
        for name, values in candidates.items()
    }


def decode_ours(
    transcript: Sequence[Mapping[str, Any]],
    target_features: FeatureBundle,
    seeds: Sequence[SeedTrajectory],
    config: OursDecoderConfig | None = None,
    *,
    anchor_config_override: AnchorDecoderConfig | None = None,
) -> OursDecodeResult:
    """Decode a target with one or more compatible annotated seeds.

    Each seed is decoded independently. The canonical configuration uses the
    arithmetic mean for every seed count; disagreement beyond eight percent of
    the target length is reported as uncertainty and never changes that mean.
    """
    resolved_config = config or canonical_ours_config()
    _validate_seeds(transcript, seeds)
    results = [
        _decode_single_seed(
            transcript, target_features, seed, resolved_config.base_decoder
        ) if anchor_config_override is None else _decode_single_seed(
            transcript, target_features, seed, resolved_config.base_decoder, anchor_config_override
        )
        for seed in seeds
    ]
    primary = np.asarray(results[0].boundaries, dtype=np.float64)
    paths = np.asarray([result.boundaries for result in results], dtype=np.float64)
    mean_path = np.mean(paths, axis=0) if paths.size else primary
    disagreement = (
        np.ptp(paths, axis=0) if len(results) > 1 else np.zeros_like(primary)
    )
    normalized_disagreement = disagreement / max(target_features.length, 1)
    high_uncertainty = normalized_disagreement > resolved_config.disagreement_ratio
    if resolved_config.aggregation == "guarded_mean":
        proposal = np.where(high_uncertainty, primary, mean_path)
        consensus_applied = np.logical_not(high_uncertainty)
    else:
        proposal = mean_path
        consensus_applied = np.ones(primary.shape, dtype=np.bool_)
    combined = (
        (1.0 - resolved_config.consensus_weight) * primary
        + resolved_config.consensus_weight * proposal
    )
    boundaries = project_boundaries(combined, target_features.length, len(transcript))
    decisions = tuple(
        {
            **dict(results[0].decisions[index]),
            "frame": boundary,
            "primary_frame": int(primary[index]),
            "seed_frames": [int(path[index]) for path in paths],
            "normalized_seed_range": float(normalized_disagreement[index]),
            "high_uncertainty": bool(high_uncertainty[index]),
            "consensus_applied": bool(consensus_applied[index]),
            "ours_config": resolved_config.name,
            "decoder_config": resolved_config.name,
        }
        for index, boundary in enumerate(boundaries)
    )
    fallback_reasons = [
        result.fallback_reason
        for result in results
        if result.fallback_reason is not None
    ]
    return OursDecodeResult(
        boundaries=boundaries,
        segments=_segments_from_boundaries(
            transcript, boundaries, target_features.length
        ),
        total_cost=float(np.mean([result.total_cost for result in results])),
        decisions=decisions,
        diagnostics={
            "aggregation": resolved_config.aggregation,
            "seed_count": len(seeds),
            "seed_episode_keys": [seed.episode_key for seed in seeds],
            "seed_paths": paths.astype(int).tolist(),
            "boundary_disagreement": disagreement.tolist(),
            "normalized_boundary_disagreement": normalized_disagreement.tolist(),
            "high_uncertainty": high_uncertainty.tolist(),
            "high_uncertainty_count": int(np.count_nonzero(high_uncertainty)),
            "uncertainty_ratio": resolved_config.disagreement_ratio,
            "consensus_applied": consensus_applied.tolist(),
            "single_seed_diagnostics": [result.diagnostics for result in results],
            "posthoc_boundary_ablations": posthoc_boundary_ablations(
                paths, target_features.length, len(transcript)
            ),
        },
        fallback_reason=(";".join(fallback_reasons) if fallback_reasons else None),
    )

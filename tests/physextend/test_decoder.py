"""Tests for the unified decoder and single-seed equivalence."""

from __future__ import annotations

import numpy as np

from roboatom.physextend.decoders.decoder import (
    CANONICAL_OURS_CONFIG_NAME,
    canonical_ours_config,
    decode_ours,
    project_boundaries,
)
from roboatom.physextend.features.signals import FeatureBundle
from roboatom.physextend.decoders.reference_decoder import SeedTrajectory
from roboatom.physextend.decoders.anchor_decoder import decode_anchor, anchor_config


def _features(length: int) -> FeatureBundle:
    zeros = np.zeros(length, dtype=np.float32)
    return FeatureBundle(
        length=length,
        activity=np.ones(length, dtype=np.float32),
        pause=zeros,
        turn=zeros,
        gripper_delta=zeros,
        gripper_level=zeros,
        visual_motion=zeros,
        events=(),
        source_columns=("synthetic",),
        state_gripper_level=zeros,
    )


def _transcript() -> list[dict[str, str]]:
    return [
        {"skill": "pick", "action_text": "pick"},
        {"skill": "move", "action_text": "move"},
        {"skill": "place", "action_text": "place"},
    ]


def _seed(key: str, boundaries: tuple[int, int]) -> SeedTrajectory:
    points = (0, *boundaries, 100)
    return SeedTrajectory(
        episode_key=key,
        segments=tuple(
            {
                "skill": skill,
                "action_text": skill,
                "start": points[index],
                "end": points[index + 1],
            }
            for index, skill in enumerate(("pick", "move", "place"))
        ),
        features=_features(100),
    )


def test_one_seed_matches_single_seed_anchor_decoder() -> None:
    target = _features(120)
    seed = _seed("only", (25, 75))

    legacy = decode_anchor(
        _transcript(),
        target,
        seed,
        anchor_config(),
    )
    unified = decode_ours(_transcript(), target, [seed])

    assert unified.boundaries == legacy.boundaries
    assert unified.segments == legacy.segments
    assert unified.diagnostics["seed_count"] == 1
    assert unified.diagnostics["high_uncertainty_count"] == 0




def test_canonical_mean_keeps_large_disagreement_and_flags_uncertainty() -> None:
    target = _features(120)
    seeds = [_seed("primary", (10, 90)), _seed("peer", (45, 55))]
    singles = [decode_ours(_transcript(), target, [seed]) for seed in seeds]
    paths = np.asarray([result.boundaries for result in singles], dtype=np.float64)
    expected = project_boundaries(np.mean(paths, axis=0), 120, 3)

    combined = decode_ours(_transcript(), target, seeds, canonical_ours_config())

    assert combined.boundaries == expected
    assert any(combined.diagnostics["high_uncertainty"])
    assert all(combined.diagnostics["consensus_applied"])
    assert combined.diagnostics["uncertainty_ratio"] == 0.08
    ablations = combined.diagnostics["posthoc_boundary_ablations"]
    assert tuple(ablations["mean"]) == combined.boundaries
    assert ablations["median"] == ablations["mean"]


def test_production_config_uses_semantic_anchor_mean() -> None:
    config = canonical_ours_config()
    assert config.name == CANONICAL_OURS_CONFIG_NAME
    assert config.base_decoder == 'anchor'
    assert config.aggregation == 'mean'

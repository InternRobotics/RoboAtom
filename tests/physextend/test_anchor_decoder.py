"""Tests for strict single-seed retry-aware semantic anchors."""

from __future__ import annotations

import numpy as np
import pytest

from roboatom.physextend.decoders.coarse_decoder import semantic_event_config, decode_trajectory
from roboatom.physextend.features.signals import FeatureBundle, SignalEvent
from roboatom.physextend.decoders.reference_decoder import SeedTrajectory
from roboatom.physextend.decoders.anchor_decoder import (
    AnchorDecoderConfig,
    cluster_gripper_events,
    decode_anchor,
    anchor_config,
)


def _features(level: np.ndarray, event_frames: tuple[int, ...]) -> FeatureBundle:
    length = level.size
    delta = np.diff(level, prepend=level[0]).astype(np.float32)
    events = tuple(
        SignalEvent(
            frame=frame,
            kind="gripper_increase" if delta[frame] >= 0.0 else "gripper_decrease",
            strength=1.0,
            channel="gripper",
        )
        for frame in event_frames
    )
    return FeatureBundle(
        length=length,
        activity=np.ones(length, dtype=np.float32),
        pause=np.zeros(length, dtype=np.float32),
        turn=np.zeros(length, dtype=np.float32),
        gripper_delta=delta,
        gripper_level=level.astype(np.float32),
        visual_motion=np.zeros(length, dtype=np.float32),
        events=events,
        source_columns=("synthetic",),
        state_gripper_level=level.astype(np.float32),
    )


def _transcript(*skills: str) -> list[dict[str, str]]:
    return [{"skill": skill, "action_text": skill} for skill in skills]


def _seed(features: FeatureBundle) -> SeedTrajectory:
    return SeedTrajectory(
        episode_key="seed",
        segments=(
            {"skill": "move", "action_text": "move", "start": 0, "end": 25},
            {"skill": "pick", "action_text": "pick", "start": 25, "end": 60},
            {"skill": "move", "action_text": "move", "start": 60, "end": 100},
        ),
        features=features,
    )


def _retry_config(policy: str) -> AnchorDecoderConfig:
    return AnchorDecoderConfig(
        name=f"test_{policy}",
        hypothesis="Synthetic retry behavior is deterministic.",
        anchor_weight=1.0,
        cluster_radius_ratio=0.03,
        search_radius_ratio=0.30,
        minimum_cluster_quality=0.40,
        minimum_seed_confidence=0.40,
        minimum_target_confidence=0.40,
        uniqueness_margin=0.05,
        retry_policy=policy,  # type: ignore[arg-type]
    )


def test_anchor_config_requires_confident_unique_matches() -> None:
    config = anchor_config()
    assert config.anchor_weight == 0.75
    assert config.minimum_target_confidence == 0.60
    assert config.uniqueness_margin == 0.12


def test_nearby_opposite_events_form_one_physical_cluster() -> None:
    level = np.zeros(100, dtype=np.float32)
    level[20:] = 1.0
    level[23:] = 0.8
    level[26:] = 1.0
    features = _features(level, (20, 23, 26))

    clusters = cluster_gripper_events(features, radius_ratio=0.06)

    assert len(clusters) == 1
    assert clusters[0].start == 20
    assert clusters[0].end == 26
    assert clusters[0].event_count == 3


def test_resolved_retry_stays_inside_one_pick_segment() -> None:
    seed_level = np.zeros(100, dtype=np.float32)
    seed_level[45:] = 1.0
    seed = _seed(_features(seed_level, (45,)))
    target_level = np.zeros(120, dtype=np.float32)
    target_level[42:] = 1.0
    target_level[50:] = 0.0
    target_level[62:] = 1.0
    target = _features(target_level, (42, 50, 62))

    result = decode_anchor(
        _transcript("move", "pick", "move"),
        target,
        seed,
        _retry_config("group"),
    )

    pick_segment = result.segments[1]
    assert len(result.segments) == 3
    assert [segment["skill"] for segment in result.segments].count("pick") == 1
    assert pick_segment["start"] <= 42
    assert pick_segment["end"] > 62
    assert result.diagnostics["retry_group_count"] == 1


def test_unresolved_retry_abstains_to_coarse_decoder() -> None:
    seed_level = np.zeros(100, dtype=np.float32)
    seed_level[45:] = 1.0
    seed = _seed(_features(seed_level, (45,)))
    target_level = np.zeros(120, dtype=np.float32)
    target_level[42:] = 1.0
    target_level[50:] = 0.0
    target_level[62:] = 1.0
    target = _features(target_level, (42, 50, 62))
    transcript = _transcript("move", "pick", "move")
    baseline = decode_trajectory(
        transcript,
        target,
        semantic_event_config(),
        seed_segments=seed.segments,
        seed_features=seed.features,
    )

    result = decode_anchor(transcript, target, seed, _retry_config("reject"))

    assert result.boundaries == baseline.boundaries
    assert result.diagnostics["accepted_match_count"] == 0
    assert result.diagnostics["target_matches"][0]["reason"] == "unresolved_retry"


def test_no_semantic_anchor_is_coarse_decoder_fallback() -> None:
    level = np.zeros(100, dtype=np.float32)
    features = _features(level, ())
    seed = _seed(features)
    transcript = _transcript("move", "pick", "move")
    baseline = decode_trajectory(
        transcript,
        features,
        semantic_event_config(),
        seed_segments=seed.segments,
        seed_features=seed.features,
    )

    result = decode_anchor(
        transcript,
        features,
        seed,
        anchor_config(),
    )

    assert result.boundaries == baseline.boundaries
    assert result.diagnostics["anchored_boundary_count"] == 0


def test_incompatible_single_seed_fails_fast() -> None:
    level = np.zeros(100, dtype=np.float32)
    seed = _seed(_features(level, ()))

    with pytest.raises(ValueError, match="skill sequence"):
        decode_anchor(
            _transcript("move", "place", "move"),
            seed.features,
            seed,
            anchor_config(),
        )

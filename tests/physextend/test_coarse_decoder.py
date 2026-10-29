"""Tests for rule-based transcript-constrained trajectory decoding."""

from __future__ import annotations

import numpy as np
import pytest

from roboatom.physextend.decoders.coarse_decoder import (
    DecoderConfig,
    semantic_event_config,
    decode_trajectory,
)
from roboatom.physextend.features.signals import FeatureBundle, SignalEvent


def _features(
    length: int,
    activity: np.ndarray | None = None,
    gripper_level: np.ndarray | None = None,
) -> FeatureBundle:
    actual_activity = np.ones(length, dtype=np.float32) if activity is None else activity.astype(np.float32)
    level = np.zeros(length, dtype=np.float32) if gripper_level is None else gripper_level.astype(np.float32)
    delta = np.diff(level, prepend=level[0])
    events = tuple(
        SignalEvent(int(frame), "gripper_increase" if delta[frame] > 0 else "gripper_decrease", 1.0, "gripper")
        for frame in np.flatnonzero(np.abs(delta) > 0.2)
    )
    return FeatureBundle(
        length=length,
        activity=actual_activity,
        pause=1.0 - actual_activity / max(float(actual_activity.max()), 1.0),
        turn=np.zeros(length, dtype=np.float32),
        gripper_delta=delta.astype(np.float32),
        gripper_level=level,
        visual_motion=np.zeros(length, dtype=np.float32),
        events=events,
        source_columns=("synthetic",),
    )


def _transcript(*skills: str) -> list[dict[str, str]]:
    return [{"skill": skill, "action_text": f"{skill} object"} for skill in skills]


def test_semantic_config_enables_directional_gripper_evidence() -> None:
    config = semantic_event_config()
    assert config.use_activity_warp
    assert config.use_gripper_events
    assert config.use_semantic_events
    assert not config.use_visual





def test_uniform_baseline_produces_equal_monotonic_segments() -> None:
    result = decode_trajectory(
        _transcript("pick", "move", "place"),
        _features(90),
        DecoderConfig(name="uniform", hypothesis="Equal duration prior.", use_seed_duration=False, use_skill_duration_priors=False, use_global_dp=False),
    )

    assert result.boundaries == (30, 60)
    assert [(segment["start"], segment["end"]) for segment in result.segments] == [(0, 30), (30, 60), (60, 90)]
    assert result.fallback_reason is None


def test_seed_duration_ratio_transfers_to_longer_target() -> None:
    seed_segments = [
        {"skill": "pick", "start": 0, "end": 20},
        {"skill": "move", "start": 20, "end": 80},
        {"skill": "place", "start": 80, "end": 100},
    ]

    result = decode_trajectory(
        _transcript("pick", "move", "place"),
        _features(200),
        DecoderConfig(name="duration", hypothesis="Transfer reference durations."),
        seed_segments=seed_segments,
        seed_features=_features(100),
    )

    assert result.boundaries == (40, 160)


def test_activity_warp_moves_boundary_toward_matching_progress() -> None:
    seed_activity = np.ones(100, dtype=np.float32)
    target_activity = np.concatenate(
        [np.full(100, 0.15, dtype=np.float32), np.full(100, 1.0, dtype=np.float32)]
    )
    seed_segments = [
        {"skill": "pick", "start": 0, "end": 50},
        {"skill": "place", "start": 50, "end": 100},
    ]

    ratio_result = decode_trajectory(
        _transcript("pick", "place"),
        _features(200, target_activity),
        DecoderConfig(name="duration", hypothesis="Transfer reference durations."),
        seed_segments=seed_segments,
        seed_features=_features(100, seed_activity),
    )
    warped_result = decode_trajectory(
        _transcript("pick", "place"),
        _features(200, target_activity),
        DecoderConfig(name="activity", hypothesis="Match activity progress.", use_activity_warp=True, activity_weight=1.0),
        seed_segments=seed_segments,
        seed_features=_features(100, seed_activity),
    )

    assert ratio_result.boundaries == (100,)
    assert warped_result.boundaries[0] > ratio_result.boundaries[0]


def test_semantic_event_rule_uses_seed_derived_direction() -> None:
    seed_level = np.zeros(100, dtype=np.float32)
    seed_level[45:] = 1.0
    target_level = np.zeros(140, dtype=np.float32)
    target_level[35:55] = -1.0
    target_level[90:] = 1.0
    seed_segments = [
        {"skill": "pick", "start": 0, "end": 45},
        {"skill": "place", "start": 45, "end": 100},
    ]

    result = decode_trajectory(
        _transcript("pick", "place"),
        _features(140, gripper_level=target_level),
        semantic_event_config(),
        seed_segments=seed_segments,
        seed_features=_features(100, gripper_level=seed_level),
    )

    assert result.decisions[0]["nearest_event"] == "gripper_increase"
    assert result.decisions[0]["nearest_event_frame"] in {55, 90}
    assert result.decisions[0]["nearest_event_frame"] != 35


def test_single_step_self_cut_needs_no_seed() -> None:
    result = decode_trajectory(_transcript("close"), _features(55), semantic_event_config())

    assert result.boundaries == ()
    assert result.segments[0]["start"] == 0
    assert result.segments[0]["end"] == 55


def test_seed_count_mismatch_fails_fast() -> None:
    with pytest.raises(ValueError, match="seed segment count"):
        decode_trajectory(
            _transcript("pick", "place"),
            _features(100),
            DecoderConfig(name="duration", hypothesis="Transfer reference durations."),
            seed_segments=[{"skill": "pick", "start": 0, "end": 100}],
        )

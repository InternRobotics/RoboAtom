"""Regression checks for the corrected PhysExtend short-segment step."""

from copy import deepcopy

import pytest

from roboatom.physextend.refine import group_methods as methods


@pytest.fixture(autouse=True)
def route_to_robodojo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(methods.base, "target_groups", lambda target: {"RoboDojo"})


def test_short_distinct_action_is_not_deleted() -> None:
    row = {"target": {"fps": 30, "dataset": "RoboDojo"}}
    parent = {
        "segments": [
            {"start_frame": 0, "end_frame": 60, "skill": "pick", "action_text": "Pick the block"},
            {"start_frame": 60, "end_frame": 75, "skill": "place", "action_text": "Place the block"},
            {"start_frame": 75, "end_frame": 150, "skill": "pick", "action_text": "Pick another block"},
        ]
    }
    original = deepcopy(parent)
    result, diagnostic = methods.merge_short_segments(
        row, parent, [], {}, {"params": {"RoboDojo": [1.0, "longer"]}}
    )
    assert result == original
    assert parent == original
    assert diagnostic["status"] == "unsafe_short_segments_retained"
    assert diagnostic["retained"][0]["skill"] == "place"


def test_repeated_short_action_is_not_deleted() -> None:
    row = {"target": {"fps": 30, "dataset": "RoboDojo"}}
    parent = {
        "segments": [
            {"start_frame": 0, "end_frame": 60, "skill": "pick", "action_text": "Pick the block"},
            {"start_frame": 60, "end_frame": 75, "skill": "pick", "action_text": "Pick the block"},
        ]
    }
    result, diagnostic = methods.merge_short_segments(
        row, parent, [], {}, {"params": {"RoboDojo": [1.0, "longer"]}}
    )
    assert len(result["segments"]) == 2
    assert diagnostic["status"] == "unsafe_short_segments_retained"


def test_no_short_segment_keeps_prediction() -> None:
    row = {"target": {"fps": 30, "dataset": "RoboDojo"}}
    parent = {"segments": [{"start_frame": 0, "end_frame": 60, "skill": "pick"}]}
    result, diagnostic = methods.merge_short_segments(
        row, parent, [], {}, {"params": {"RoboDojo": [1.0, "longer"]}}
    )
    assert result is parent
    assert diagnostic["status"] == "no_short_segment"

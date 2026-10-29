"""Behavioral checks for reference-only sequence choice and fusion."""

import copy

import pytest

from roboatom.physextend.decoders.fusion import decode_candidates, refined_path


def test_signature_fusion_cost_can_be_separated_from_sequence_choice(row: dict) -> None:
    first = candidate("first", ["pick", "place"], [40], 0.1)
    first["diagnostic"]["decisions"][0]["signature_cost"] = 10
    second = candidate("second", ["move", "push"], [70], 1)
    config = {"decoder": "multi", "signature_weight": 1}
    _, original = decode_candidates(row, [first, second], config, "multi")
    _, separated = decode_candidates(
        row, [first, second], dict(config, sequence_signature_weight=0), "multi"
    )
    assert original["selected_signature"] == ["move", "push"]
    assert separated["selected_signature"] == ["pick", "place"]
    assert (
        separated["reference_scores"]["first"] > separated["reference_scores"]["second"]
    )


@pytest.fixture
def row() -> dict:
    return {
        "record_id": "query",
        "canonical_task_id": "task",
        "protocol": "multi",
        "support_mode": "mixed",
        "target": {"length": 100, "dataset": "mock"},
    }


def candidate(identifier: str, skills: list[str], cuts: list[int], cost: float) -> dict:
    points = [0, *cuts, 100]
    segments = [
        {"start_frame": points[i], "end_frame": points[i + 1], "skill": skill}
        for i, skill in enumerate(skills)
    ]
    evidence = [
        dict(
            phase=cost,
            centered=cost,
            state_mean=cost,
            motion=cost,
            duration_log=0,
            target_ratio=1 / len(skills),
        )
        for _ in skills
    ]
    decisions = [
        dict(
            ours_frame=frame,
            sensor_frame=frame + 10,
            sensor_weight=0.25,
            anchor_applied=False,
        )
        for frame in cuts
    ]
    return {
        "status": "ok",
        "seed_record_id": identifier,
        "seed_signature": skills,
        "prediction": {"frames": 100, "fps": 30, "segments": segments},
        "segment_evidence": evidence,
        "diagnostic": {"decisions": decisions},
    }


def test_single_preserves_original_even_with_short_anchored_segment(row: dict) -> None:
    original = candidate("a", ["pick", "place", "move"], [1, 70], 2)
    pred, _ = decode_candidates(row, [original], {}, "single")
    assert pred["segments"] == original["prediction"]["segments"]
    assert pred["fps"] == 30


def test_single_does_not_select_a_better_second_reference(row: dict) -> None:
    first = candidate("a", ["pick", "place"], [25], 10)
    second = candidate("b", ["move", "push", "stop"], [30, 60], 0)
    prediction, diagnostic = decode_candidates(
        row, [first, second], {"decoder": "multi"}, "single"
    )
    assert diagnostic["selected_signature"] == first["seed_signature"]
    assert prediction["segments"] == first["prediction"]["segments"]


def test_mixed_selects_observed_sequence_then_fuses_only_compatible_paths(
    row: dict,
) -> None:
    first = candidate("a", ["pick", "place"], [20], 5)
    second = candidate("b", ["move", "push", "stop"], [30, 60], 1)
    third = candidate("c", ["move", "push", "stop"], [40, 70], 2)
    prediction, diagnostic = decode_candidates(
        row, [first, second, third], {"decoder": "multi"}, "multi"
    )
    assert diagnostic["selected_signature"] == second["seed_signature"]
    assert diagnostic["boundaries"] == [35, 65]
    assert len(prediction["segments"]) == 3
    assert diagnostic["used_references"] == ["b", "c"]


def test_single_failure_is_explicit_and_cannot_switch_references(row: dict) -> None:
    with pytest.raises(ValueError, match="No usable"):
        decode_candidates(
            row,
            [{"status": "failed"}, candidate("b", ["a", "b"], [50], 0)],
            {},
            "single",
        )


def test_low_confidence_correction_preserves_accepted_anchor() -> None:
    source = candidate("a", ["a", "b", "c"], [20, 65], 1)
    source["diagnostic"]["decisions"][0]["anchor_applied"] = True
    assert refined_path(source, {"sensor_scale": 2}) == (20, 70)


def test_failed_candidate_does_not_remove_query(row: dict) -> None:
    pred, diagnostic = decode_candidates(
        row,
        [{"status": "failed"}, candidate("b", ["a", "b"], [50], 0)],
        {"decoder": "multi"},
        "multi",
    )
    assert len(pred["segments"]) == 2
    assert diagnostic["failed_candidates"] == 1


def test_refinement_does_not_mutate_cached_predictions(row: dict) -> None:
    source = candidate("a", ["a", "b"], [50], 1)
    before = copy.deepcopy(source)
    decode_candidates(row, [source], {"sensor_scale": 2}, "single")
    assert source == before








def test_robust_fusion_resists_single_outlier_and_keeps_full_cover(row: dict) -> None:
    sources = [
        candidate("a", ["pick", "place"], [40], 1),
        candidate("b", ["pick", "place"], [42], 1),
        candidate("c", ["pick", "place"], [90], 20),
    ]
    pred, diagnostics = decode_candidates(
        row, sources, {"decoder": "multi", "aggregation": "median"}, "multi"
    )
    assert diagnostics["boundaries"] == [42]
    assert pred["segments"][0]["start_frame"] == 0
    assert pred["segments"][-1]["end_frame"] == 100


def test_weighted_fusion_responds_to_reference_reliability(row: dict) -> None:
    sources = [
        candidate("a", ["pick", "place"], [40], 0),
        candidate("b", ["pick", "place"], [90], 20),
    ]
    _, diagnostics = decode_candidates(
        row, sources, {"decoder": "multi", "aggregation": "weighted_mean"}, "multi"
    )
    assert 40 <= diagnostics["boundaries"][0] < 45


def test_disagreement_guard_uses_an_observed_boundary(row: dict) -> None:
    sources = [
        candidate("a", ["pick", "place"], [40], 1),
        candidate("b", ["pick", "place"], [90], 2),
    ]
    _, diagnostics = decode_candidates(
        row, sources, {"decoder": "multi", "disagreement_guard": 0.1}, "multi"
    )
    assert diagnostics["boundaries"] == [40]


def test_two_reference_mean_avoids_discrete_selection_when_requested(row: dict) -> None:
    sources = [
        candidate("a", ["pick", "place"], [40], 0),
        candidate("b", ["pick", "place"], [90], 20),
    ]
    _, diagnostic = decode_candidates(
        row,
        sources,
        {
            "decoder": "multi",
            "aggregation": "weighted_median",
            "two_reference_mean": True,
        },
        "multi",
    )
    assert diagnostic["boundaries"] == [65]


def test_whole_path_selection_preserves_a_valid_short_anchored_segment(
    row: dict,
) -> None:
    sources = [
        candidate("a", ["pick", "place", "move"], [1, 70], 0),
        candidate("b", ["pick", "place", "move"], [40, 90], 2),
    ]
    _, diagnostic = decode_candidates(
        row, sources, {"decoder": "multi", "aggregation": "best"}, "multi"
    )
    assert diagnostic["boundaries"] == [1, 70]


def test_duration_guard_repairs_novel_collapse_without_dropping_a_cut(
    row: dict,
) -> None:
    first = candidate("a", ["pick", "place", "move"], [10, 30], 0)
    second = candidate("b", ["pick", "place", "move"], [60, 90], 1)
    first["diagnostic"]["decisions"][1]["anchor_applied"] = True
    second["diagnostic"]["decisions"][0]["anchor_applied"] = True
    _, diagnostic = decode_candidates(
        row,
        [first, second],
        {
            "decoder": "multi",
            "aggregation": "weighted_median",
            "anchor_weight": 100,
            "duration_consistency_guard": 0.5,
        },
        "multi",
    )
    assert diagnostic["boundaries"] == [10, 30]
    assert diagnostic["decisions"][-1]["duration_guard_applied"]


def test_semantic_anchor_calibration_requires_confident_accepted_proposal() -> None:
    source = candidate("a", ["a", "b", "c"], [20, 65], 1)
    source["diagnostic"]["decisions"][0].update(
        anchor_applied=True,
        anchor_proposals=[{"proposed_frame": 40, "confidence": 0.95}],
    )
    source["diagnostic"]["decisions"][1].update(
        anchor_applied=True,
        anchor_proposals=[{"proposed_frame": 80, "confidence": 0.70}],
    )
    assert refined_path(source, {"anchor_blend": 0.25}) == (25, 65)


def test_conflicting_semantic_proposals_keep_all_cuts_ordered() -> None:
    source = candidate("a", ["a", "b", "c"], [20, 21], 1)
    for decision, proposal in zip(source["diagnostic"]["decisions"], [100, 0]):
        decision.update(
            anchor_applied=True,
            anchor_proposals=[{"proposed_frame": proposal, "confidence": 1}],
        )
    boundaries = refined_path(source, {"anchor_blend": 1})
    assert boundaries == (20, 21)


def test_anchor_agreement_softens_conflicting_confident_proposals() -> None:
    source = candidate("a", ["a", "b"], [20], 1)
    source["diagnostic"]["decisions"][0].update(
        anchor_applied=True,
        anchor_proposals=[
            {"proposed_frame": 30, "confidence": 0.95},
            {"proposed_frame": 70, "confidence": 0.95},
        ],
    )
    assert refined_path(source, {"anchor_blend": 0.5}) == (35,)
    assert refined_path(source, {"anchor_blend": 0.5, "anchor_agreement_scale": 2}) == (
        23,
    )


def test_sensor_activity_conflict_reduces_correction_without_removing_cut() -> None:
    source = candidate("a", ["a", "b"], [50], 1)
    source["diagnostic"]["decisions"][0]["activity_anchor"] = 30
    assert refined_path(source, {"sensor_activity_agreement": True}) == (51,)
    source["diagnostic"]["decisions"][0]["activity_anchor"] = 70
    assert refined_path(source, {"sensor_activity_agreement": True}) == (52,)


def test_sensor_boost_requires_agreement_and_remains_bounded() -> None:
    source = candidate("a", ["a", "b"], [50], 1)
    source["diagnostic"]["decisions"][0]["activity_anchor"] = 70
    assert refined_path(
        source, {"sensor_activity_agreement": True, "sensor_agreement_boost": 1.5}
    ) == (54,)
    source["diagnostic"]["decisions"][0]["activity_anchor"] = 30
    assert refined_path(
        source, {"sensor_activity_agreement": True, "sensor_agreement_boost": 1.5}
    ) == (51,)


def test_anchor_can_follow_the_strongest_observed_proposal() -> None:
    source = candidate("a", ["a", "b"], [20], 1)
    source["diagnostic"]["decisions"][0].update(
        anchor_applied=True,
        anchor_proposals=[
            {"proposed_frame": 30, "confidence": 0.90},
            {"proposed_frame": 50, "confidence": 0.95},
        ],
    )
    assert refined_path(
        source, {"anchor_blend": 0.5, "anchor_proposal_reduction": "highest_confidence"}
    ) == (35,)


def test_joint_projection_balances_conflicts_and_preserves_reference_shortcuts() -> (
    None
):
    import numpy as np
    from roboatom.physextend.decoders.fusion import reference_ordered_projection

    assert reference_ordered_projection(
        [80, 20], np.array([[10, 30], [60, 90]]), 100
    ) == (49, 51)
    assert reference_ordered_projection([1, 70], np.array([[1, 70], [1, 80]]), 100) == (
        1,
        70,
    )


def test_agreement_mean_smooths_nearby_cuts_but_not_distant_events(row: dict) -> None:
    near = [
        candidate("a", ["pick", "place"], [40], 0),
        candidate("b", ["pick", "place"], [44], 0),
    ]
    _, close_diagnostic = decode_candidates(
        row, near, {"decoder": "multi", "aggregation": "agreement_mean"}, "multi"
    )
    assert close_diagnostic["boundaries"] == [42]
    far = [
        candidate("a", ["pick", "place"], [10], 0),
        candidate("b", ["pick", "place"], [90], 0),
    ]
    _, far_diagnostic = decode_candidates(
        row, far, {"decoder": "multi", "aggregation": "agreement_mean"}, "multi"
    )
    assert far_diagnostic["boundaries"] == [10]


def test_sensor_confidence_can_disambiguate_equal_shape_costs(row: dict) -> None:
    first = candidate("a", ["pick", "place"], [50], 1)
    second = candidate("b", ["pick", "move"], [50], 1)
    first["diagnostic"]["diagnostics"] = {"sensor": {"confidence": [0.1]}}
    second["diagnostic"]["diagnostics"] = {"sensor": {"confidence": [0.9]}}
    _, diagnostic = decode_candidates(
        row,
        [first, second],
        {"decoder": "multi", "sensor_confidence_bonus": 0.5},
        "multi",
    )
    assert diagnostic["selected_signature"] == ["pick", "move"]


def test_rank_reference_weights_are_invariant_to_cost_scale(row: dict) -> None:
    sources = [
        candidate("a", ["pick", "place"], [20], 1),
        candidate("b", ["pick", "place"], [40], 2),
        candidate("c", ["pick", "place"], [80], 3),
    ]
    config = {
        "decoder": "multi",
        "aggregation": "weighted_mean",
        "reference_weighting": "rank",
    }
    _, first = decode_candidates(row, sources, config, "multi")
    for source in sources:
        for evidence in source["segment_evidence"]:
            evidence["phase"] *= 100
    _, scaled = decode_candidates(row, sources, config, "multi")
    assert first["boundaries"] == scaled["boundaries"] == [40]
    assert first["decisions"][0]["weights"] == [4, 3, 2]




def test_unique_anchor_is_an_explicit_option_and_never_removes_a_cut(row: dict) -> None:
    first = candidate("anchor", ["pick", "place"], [25], 10)
    first["diagnostic"]["decisions"][0]["anchor_applied"] = True
    second = candidate("unanchored", ["pick", "place"], [70], 0)
    config = {"decoder": "multi", "aggregation": "weighted_median"}
    _, ordinary = decode_candidates(row, [first, second], config, "multi")
    prediction, explicit = decode_candidates(
        row, [first, second], dict(config, preserve_unique_anchor=True), "multi"
    )
    assert ordinary["boundaries"] == [70]
    assert explicit["boundaries"] == [25]
    assert len(prediction["segments"]) == 2



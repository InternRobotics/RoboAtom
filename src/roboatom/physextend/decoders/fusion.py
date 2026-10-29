"""Select an observed reference sequence and fuse compatible reference paths.

All selection features come from reference labels and target sensors. Query
annotations are unavailable; unknown target sequences remain evaluation errors.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import copy
from typing import Any

import numpy as np

from roboatom.physextend.decoders.decoder import project_boundaries
from roboatom.physextend.decoders.seeded_decoder import project_with_anchors


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Return a deterministic weighted median of boundary proposals."""
    order = np.argsort(values, kind="stable")
    index = np.searchsorted(
        np.cumsum(weights[order]), 0.5 * np.sum(weights), side="left"
    )
    return float(values[order[min(int(index), len(order) - 1)]])


def candidate_score(candidate: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    """Compare a reference with the query using allowed segment evidence."""
    evidence = candidate["segment_evidence"]
    weights = np.asarray([item["target_ratio"] for item in evidence], dtype=float)
    if config.get("equal_segment_weight", False):
        weights = np.ones(len(evidence))
    weights = weights / np.sum(weights)
    score = 0.0
    defaults = {
        "phase": 1.0,
        "centered": 0.0,
        "state_mean": 0.0,
        "motion": 0.0,
        "duration_log": 0.0,
    }
    for key, default in defaults.items():
        values = np.asarray([item[key] for item in evidence], dtype=float)
        if config.get("log_cost", False):
            values = np.log1p(values)
        score += float(config.get(key + "_weight", default)) * float(
            np.dot(weights, values)
        )
    decisions = candidate["diagnostic"]["decisions"]
    anchor_ratio = sum(bool(item.get("anchor_applied")) for item in decisions) / max(
        len(decisions), 1
    )
    score -= float(config.get("anchor_bonus", 0.0)) * anchor_ratio
    sensor = candidate["diagnostic"].get("diagnostics", {}).get("sensor", {})
    confidence = sensor.get("confidence", [])
    if confidence:
        score -= float(config.get("sensor_confidence_bonus", 0.0)) * float(
            np.mean(confidence)
        )
    if decisions:
        score += float(config.get("signature_weight", 0.0)) * float(
            np.mean([item.get("signature_cost", 0.0) for item in decisions])
        )
        score += float(config.get("holding_weight", 0.0)) * float(
            np.mean([item.get("holding_cost", 0.0) for item in decisions])
        )
    return score


def calibrate_semantic_anchors(
    decisions: Sequence[dict[str, Any]],
    boundaries: Sequence[int],
    length: int,
    config: Mapping[str, Any],
) -> list[int]:
    """Adjust accepted anchors toward their own high-confidence event proposals.

    Each cut stays in a disjoint neighborhood around the original cut. This
    preserves order and every segment without borrowing query annotations.
    """
    blend = float(config.get("anchor_blend", 0.0))
    adjusted = list(boundaries)
    if blend == 0:
        return adjusted
    points = [0, *boundaries, length]
    threshold = float(config.get("anchor_min_confidence", 0.85))
    for index, decision in enumerate(decisions):
        proposals = [
            item
            for item in decision.get("anchor_proposals", [])
            if item["confidence"] >= threshold
        ]
        if not decision.get("anchor_applied") or not proposals:
            continue
        proposal = float(
            np.average(
                [item["proposed_frame"] for item in proposals],
                weights=[item["confidence"] for item in proposals],
            )
        )
        if config.get("anchor_proposal_reduction") == "highest_confidence":
            proposal = float(
                max(proposals, key=lambda item: item["confidence"])["proposed_frame"]
            )
        local_blend = blend
        agreement_scale = float(config.get("anchor_agreement_scale", 0.0))
        if agreement_scale > 0 and len(proposals) > 1:
            spread = np.ptp([item["proposed_frame"] for item in proposals])
            local_duration = max(
                1,
                min(
                    boundaries[index] - points[index],
                    points[index + 2] - boundaries[index],
                ),
            )
            local_blend /= 1 + agreement_scale * spread / local_duration
        frame = round((1 - local_blend) * boundaries[index] + local_blend * proposal)
        lower = (points[index] + boundaries[index] + 1) // 2
        upper = (boundaries[index] + points[index + 2] + 1) // 2 - 1
        adjusted[index] = max(lower, min(upper, frame))
    return adjusted


def refined_path(
    candidate: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[int, ...]:
    """Retune bounded sensor correction while keeping accepted semantic anchors."""
    prediction = candidate["prediction"]
    original = tuple(segment["end_frame"] for segment in prediction["segments"][:-1])
    if (
        config.get("sensor_scale", 1.0) == 1.0
        and config.get("sensor_limit", 0.5) == 0.5
        and config.get("sensor_power", 1.0) == 1.0
        and config.get("anchor_blend", 0.0) == 0.0
        and not config.get("sensor_activity_agreement", False)
    ):
        return original
    decisions = candidate["diagnostic"]["decisions"]
    ours = [
        int(item.get("ours_frame", frame))
        for item, frame in zip(decisions, original, strict=True)
    ]
    ours = calibrate_semantic_anchors(
        decisions, ours, int(prediction["frames"]), config
    )
    points = [0, *ours, int(prediction["frames"])]
    anchors = {
        index: ours[index]
        for index, item in enumerate(decisions)
        if item.get("anchor_applied")
    }
    proposed = []
    for index, item in enumerate(decisions):
        if index in anchors:
            proposed.append(anchors[index])
            continue
        limit = max(
            1,
            round(
                float(config.get("sensor_limit", 0.5))
                * min(ours[index] - points[index], points[index + 2] - ours[index])
            ),
        )
        sensor = int(item.get("sensor_frame", ours[index]))
        sensor = max(ours[index] - limit, min(ours[index] + limit, sensor))
        weight = float(item.get("sensor_weight", 0.0))
        weight = min(
            1.0,
            max(
                0.0,
                float(config.get("sensor_scale", 1.0))
                * weight ** float(config.get("sensor_power", 1.0)),
            ),
        )
        if config.get("sensor_activity_agreement", False):
            activity = int(item.get("activity_anchor", ours[index]))
            direction_product = (sensor - ours[index]) * (activity - ours[index])
            if direction_product < 0:
                weight *= 0.5
            elif direction_product > 0:
                weight = min(
                    1.0, weight * float(config.get("sensor_agreement_boost", 1.0))
                )
        proposed.append(round((1 - weight) * ours[index] + weight * sensor))
    return project_with_anchors(proposed, anchors, int(prediction["frames"]))


def choose_sequence(
    groups: Mapping[tuple[str, ...], list[dict[str, Any]]],
    scores: Mapping[str, float],
    config: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, float]]:
    """Rank observed sequences only; never synthesize a new transcript."""
    ranked = []
    for signature, candidates in groups.items():
        values = sorted(scores[candidate["seed_record_id"]] for candidate in candidates)
        method = config.get("sequence_reduction", "min")
        if method == "mean":
            score = float(np.mean(values))
        elif method == "median":
            score = float(np.median(values))
        elif method == "softmin":
            temperature = max(0.01, float(config.get("sequence_temperature", 1.0)))
            minimum = values[0]
            score = minimum - temperature * np.log(
                np.mean(np.exp(-(np.asarray(values) - minimum) / temperature))
            )
        else:
            score = values[0]
        score -= float(config.get("support_prior", 0.0)) * np.log(len(candidates))
        ranked.append((score, signature))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked[0][1], {
        "|".join(signature): float(score) for score, signature in ranked
    }


def reference_ordered_projection(
    values: Sequence[float],
    paths: np.ndarray,
    length: int,
) -> tuple[int, ...]:
    """Project all cuts jointly with feasible reference-derived segment minima."""
    durations = np.diff(
        np.column_stack([np.zeros(len(paths)), paths, np.full(len(paths), length)]),
        axis=1,
    )
    original_minimum = max(2, round(length * 0.012))
    minima = np.maximum(
        1, np.minimum(original_minimum, np.floor(0.5 * np.min(durations, axis=0)))
    ).astype(int)
    offsets = np.cumsum(minima)[:-1]
    slack = length - int(np.sum(minima))
    if slack < 0:
        raise ValueError("Reference segment constraints are infeasible")
    blocks: list[tuple[float, int]] = []
    for value in np.asarray(values) - offsets:
        blocks.append((float(value), 1))
        while (
            len(blocks) > 1
            and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]
        ):
            right_sum, right_count = blocks.pop()
            left_sum, left_count = blocks.pop()
            blocks.append((left_sum + right_sum, left_count + right_count))
    projected = np.asarray(
        [total / count for total, count in blocks for _ in range(count)]
    )
    projected = np.rint(np.clip(projected, 0, slack)).astype(int) + offsets
    return tuple(int(frame) for frame in projected)


def agreement_mean_or_median(
    paths: np.ndarray,
    index: int,
    weights: np.ndarray,
    length: int,
    agreement_ratio: float,
) -> float:
    """Average nearby boundaries; retain robust selection for distant events."""
    selected = paths[:, index]
    left = paths[:, index - 1] if index else np.zeros(len(paths))
    right = (
        paths[:, index + 1]
        if index + 1 < paths.shape[1]
        else np.full(len(paths), length)
    )
    local_duration = max(
        1.0, float(np.median(np.minimum(selected - left, right - selected)))
    )
    if np.ptp(selected) <= agreement_ratio * local_duration:
        return float(np.dot(weights, selected) / np.sum(weights))
    return weighted_median(selected, weights)


def fuse_paths(
    candidates: Sequence[dict[str, Any]],
    scores: Mapping[str, float],
    config: Mapping[str, Any],
) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    """Fuse compatible semantic boundaries, retaining per-reference evidence."""
    paths = np.asarray(
        [refined_path(candidate, config) for candidate in candidates], dtype=float
    )
    if len(candidates) == 1:
        return tuple(int(frame) for frame in paths[0]), []
    count = len(candidates[0]["seed_signature"])
    length = int(candidates[0]["prediction"]["frames"])
    if count == 1:
        return (), []
    costs = np.asarray(
        [scores[candidate["seed_record_id"]] for candidate in candidates]
    )
    temperature = max(0.01, float(config.get("reference_temperature", 1.0)))
    seed_weights = np.exp(-(costs - np.min(costs)) / temperature)
    if config.get("reference_weighting") == "rank":
        seed_weights = np.asarray(
            [len(costs) + 1 - int(np.sum(costs < cost)) for cost in costs], dtype=float
        )
    method = config.get("aggregation", "mean")
    if method == "best":
        selected_index = int(np.argmin(costs))
        return tuple(int(frame) for frame in paths[selected_index]), [
            {"whole_path_reference": candidates[selected_index]["seed_record_id"]}
        ]
    if len(candidates) == 2 and config.get("two_reference_mean", False):
        method = "mean"
    boundary_diagnostics = []
    values = []
    for index in range(paths.shape[1]):
        weights = (
            seed_weights.copy()
            if method in {"weighted_mean", "weighted_median", "agreement_mean"}
            else np.ones(len(candidates))
        )
        for j, candidate in enumerate(candidates):
            decision = candidate["diagnostic"]["decisions"][index]
            if decision.get("anchor_applied"):
                weights[j] *= float(config.get("anchor_weight", 1.0))
            weights[j] /= 1 + float(config.get("local_signature_weight", 0.0)) * float(
                decision.get("signature_cost", 0.0)
            )
        selected = paths[:, index]
        if method == "first":
            value = selected[0]
        elif method == "best":
            value = selected[int(np.argmin(costs))]
        elif method == "median":
            value = float(np.median(selected))
        elif method == "weighted_median":
            value = weighted_median(selected, weights)
        elif method == "agreement_mean":
            value = agreement_mean_or_median(
                paths,
                index,
                weights,
                length,
                float(config.get("mean_agreement_ratio", 0.2)),
            )
        elif method == "medoid":
            value = selected[
                int(
                    np.argmin(
                        np.sum(np.abs(selected[:, None] - selected[None, :]), axis=1)
                    )
                )
            ]
        else:
            value = float(np.dot(weights, selected) / np.sum(weights))
        spread = float(np.ptp(selected)) / max(length, 1)
        guard = config.get("disagreement_guard")
        if guard is not None and spread > float(guard):
            value = selected[int(np.argmin(costs))]
        if config.get("preserve_unique_anchor", False):
            anchored = [
                j
                for j, candidate in enumerate(candidates)
                if candidate["diagnostic"]["decisions"][index].get("anchor_applied")
            ]
            if len(anchored) == 1:
                value = selected[anchored[0]]
        boundary_diagnostics.append(
            {
                "boundary_index": index,
                "reference_frames": selected.tolist(),
                "weights": weights.tolist(),
                "spread_ratio": spread,
                "proposal": float(value),
            }
        )
        values.append(value)
    if config.get("projection") == "reference_isotonic":
        projected = reference_ordered_projection(values, paths, length)
    else:
        projected = project_boundaries(np.asarray(values), length, count)
    guard = config.get("duration_consistency_guard")
    if guard is not None:
        reference_durations = np.diff(
            np.column_stack([np.zeros(len(paths)), paths, np.full(len(paths), length)]),
            axis=1,
        )
        fused_durations = np.diff([0, *projected, length])
        collapsed = (fused_durations < np.min(reference_durations, axis=0)) & (
            fused_durations < float(guard) * np.median(reference_durations, axis=0)
        )
        if np.any(collapsed):
            selected_index = int(np.argmin(costs))
            projected = tuple(int(frame) for frame in paths[selected_index])
            boundary_diagnostics.append(
                {
                    "duration_guard_applied": True,
                    "collapsed_segments": np.flatnonzero(collapsed).tolist(),
                    "whole_path_reference": candidates[selected_index][
                        "seed_record_id"
                    ],
                }
            )
    return projected, boundary_diagnostics


def decode_candidates(
    row: dict[str, Any],
    all_candidates: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
    route: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decode fixed-single or multi-reference inputs under the same query set."""
    source = (
        all_candidates[:1]
        if route == "single" or config.get("decoder", "first") == "first"
        else all_candidates
    )
    valid = [candidate for candidate in source if candidate["status"] == "ok"]
    if not valid:
        raise ValueError("No usable independent reference prediction")
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for candidate in valid:
        groups[tuple(candidate["seed_signature"])].append(candidate)
    scores = {
        candidate["seed_record_id"]: candidate_score(candidate, config)
        for candidate in valid
    }
    if route == "single" or config.get("decoder", "first") == "first":
        signature = tuple(valid[0]["seed_signature"])
        rankings = {"|".join(signature): scores[valid[0]["seed_record_id"]]}
    else:
        selection_scores = scores
        if "sequence_signature_weight" in config:
            selection_config = dict(
                config, signature_weight=config["sequence_signature_weight"]
            )
            selection_scores = {
                candidate["seed_record_id"]: candidate_score(
                    candidate, selection_config
                )
                for candidate in valid
            }
        signature, rankings = choose_sequence(groups, selection_scores, config)
    chosen = groups[signature]
    boundaries, decisions = fuse_paths(chosen, scores, config)
    prediction = copy.deepcopy(chosen[0]["prediction"])
    prediction.update(
        record_id=row["record_id"],
        episode_index=row["record_id"],
        task_id=row["canonical_task_id"],
        protocol="fixed_single_reference" if route == "single" else row["protocol"],
    )
    points = [0, *boundaries, int(row["target"]["length"])]
    for index, segment in enumerate(prediction["segments"]):
        segment["start_frame"] = points[index]
        segment["end_frame"] = points[index + 1]
    diagnostic = {
        "record_id": row["record_id"],
        "task_id": row["canonical_task_id"],
        "dataset": row["target"]["dataset"],
        "support_mode": row["support_mode"],
        "route": route,
        "selected_signature": list(signature),
        "sequence_scores": rankings,
        "reference_scores": scores,
        "used_references": [candidate["seed_record_id"] for candidate in chosen],
        "failed_candidates": len(source) - len(valid),
        "boundaries": list(boundaries),
        "decisions": decisions,
    }
    return prediction, diagnostic

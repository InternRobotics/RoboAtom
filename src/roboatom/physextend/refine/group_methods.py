"""Dataset- and embodiment-specific boundary refinements."""

from __future__ import annotations

from roboatom.physextend.runtime import current_run

from roboatom.physextend.runtime import run_cache

from collections import Counter
from typing import Any

import numpy as np

from roboatom.physextend.refine import iterative as previous
from roboatom.physextend.refine import methods as base


def signature(segments: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(s["skill"] for s in segments)


def physical_seconds(episode: dict[str, Any]) -> float:
    return episode["length"] / episode["fps"]


def vote_sequence(row: dict[str, Any], parent: dict[str, Any], rule: str) -> tuple[str, ...] | None:
    """Pick a complete observed sequence from reference metadata only (no target GT)."""
    references = row["references"]
    counts = Counter(signature(seed["segments"]) for seed in references)
    if len(counts) < 2:
        return None
    current = signature(parent["segments"])
    top = max(counts.values())
    modes = sorted((s for s in counts if counts[s] == top), key=lambda s: (len(s), s))
    if rule == "majority_keep":
        # Majority only; ties keep the decoder's choice when it is one of the modes.
        if current in modes:
            return current
        return modes[0] if len(modes) == 1 else None
    if rule == "majority_shortest":
        if len(modes) == 1:
            return modes[0]
        return current if current in modes and len(current) == len(modes[0]) else modes[0]
    if rule == "shortest":
        # Fewer-segment transcript among references (RoboCOIN: extra-detail annotations are the minority style).
        return min(counts, key=lambda s: (len(s), -counts[s], s))
    if rule == "majority_then_event":
        penalties = sequence_event_penalty(row) or {}
        return min(counts, key=lambda s: (-counts[s], penalties.get(s, 9.0), len(s), s))
    if rule == "duration_nearest":
        target = physical_seconds(row["target"])
        best = min(references, key=lambda seed: abs(np.log(physical_seconds(seed) / target)))
        return signature(best["segments"])
    raise ValueError("Unknown sequence rule")


def sequence_event_penalty(row: dict[str, Any]) -> dict[tuple[str, ...], float] | None:
    """Mean relative gripper-transition mismatch per observed reference sequence."""
    target = row["target"]
    count = base.episode_transitions(target["data_uri"], target["dataset"])
    if count is None:
        return None
    groups: dict[tuple[str, ...], list[float]] = {}
    for seed in row["references"]:
        other = base.episode_transitions(seed["data_uri"], seed["dataset"])
        if other is None:
            continue
        groups.setdefault(signature(seed["segments"]), []).append(abs(other - count) / max(other, count, 1))
    return {key: float(np.mean(v)) for key, v in groups.items()}


def decode_forced(
    row: dict[str, Any], pairs: list[Any], config: dict[str, Any], selected: tuple[str, ...]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decode a fixed sequence, then refine joint-state and opening boundaries."""
    from roboatom.physextend.decoders import fusion as decoder

    original = decoder.choose_sequence

    def forced(groups: Any, scores: Any, cfg: Any) -> Any:
        if selected not in groups:
            raise ValueError("Forced sequence absent from usable candidates")
        return selected, {"forced": 1.0}

    decoder.choose_sequence = forced

    try:
        output, pipeline = previous.integrated_prediction(row, pairs, config)
    finally:
        decoder.choose_sequence = original
    output, joint = base.joint_state_refine(row, output, pairs, {"opening_nonworsening": True})
    output, local = base.local_opening_search(row, output, pairs, {"search_seconds": 0.5, "minimum_gain": 0.01})
    return output, {"pipeline_sequence": pipeline.get("sequence"), "joint": joint.get("changed"), "local": local.get("changed")}


def sequence_vote(
    row: dict[str, Any], parent: dict[str, Any], pairs: list[Any], config: dict[str, Any], proposal: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = vote_sequence(row, parent, proposal["params"]["rule"])
    if selected is None or selected == signature(parent["segments"]):
        return parent, {"status": "sequence_kept"}
    tolerance = proposal["params"].get("event_tolerance")
    if tolerance is not None:
        penalties = sequence_event_penalty(row)
        current = signature(parent["segments"])
        if penalties is None or selected not in penalties or current not in penalties:
            return parent, {"status": "no_event_evidence_kept"}
        if penalties[selected] > penalties[current] + tolerance:
            return parent, {"status": "event_count_contradicts_majority", "penalties": [penalties[selected], penalties[current]]}
    output, diagnostic = decode_forced(row, pairs, config, selected)
    if signature(output["segments"]) != selected:
        return parent, {"status": "forced_decode_mismatch"}
    return output, {"status": "sequence_vote_changed", "selected": list(selected), **diagnostic}


def merge_short_segments(
    row: dict[str, Any], parent: dict[str, Any], pairs: list[Any], config: dict[str, Any], proposal: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Flag suspiciously short actions without deleting an observed action.

    Args:
        row: Inference item containing the target trajectory.
        parent: Prediction from the previous PhysExtend step.
        pairs: Reference and candidate pairs (unused by this method).
        config: Decoder configuration (unused by this method).
        proposal: Dataset-specific duration threshold and neighbour preference.

    Returns:
        The unchanged prediction and an audit record of short segments.
    """
    target = row["target"]
    groups = base.target_groups(target)
    rule = next((proposal["params"][g] for g in sorted(groups, key=len, reverse=True) if g in proposal["params"]), None)
    if rule is None:
        return parent, {"status": "no_rule"}
    seconds, _into = rule
    fps = target["fps"]
    retained = [
        {"index": index, "skill": segment["skill"], "duration_seconds": (segment["end_frame"] - segment["start_frame"]) / fps}
        for index, segment in enumerate(parent["segments"])
        if (segment["end_frame"] - segment["start_frame"]) / fps < seconds
    ]
    return parent, {
        "status": "unsafe_short_segments_retained" if retained else "no_short_segment",
        "retained": retained,
        "rule": rule,
    }


def event_anchor(
    row: dict[str, Any], parent: dict[str, Any], pairs: list[Any], config: dict[str, Any], proposal: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reference-calibrated gripper-event anchoring with per-group (window, attach) chosen on dev."""
    from roboatom.physextend.refine import event_transfer as transfer
    from roboatom.physextend.refine import rounds as rounds

    groups = base.target_groups(row["target"])
    key = next((g for g in sorted(groups, key=len, reverse=True) if g in proposal["params"]), None)
    if key is None:
        return parent, {"status": "no_rule"}
    window, attach = proposal["params"][key]
    labels = [s["skill"] for s in parent["segments"]]
    cuts = np.asarray([s["end_frame"] for s in parent["segments"][:-1]], dtype=float)
    if not len(cuts):
        return parent, {"status": "no_internal_boundary"}
    group = row["target"]["dataset"] + "|" + base.router_profiles()[row["target"]["data_uri"]]["robot_type"]
    final, moved = transfer.anchored(row, labels, cuts, group, window, attach, 0.7)
    if not moved:
        return parent, {"status": "no_anchor"}
    return rounds.set_boundaries(parent, final.astype(int)), {"status": "event_anchored", "moved_seconds": moved}


@run_cache(maxsize=3000)
def combined_channels(uri: str, dataset: str, mode: str) -> dict[str, np.ndarray]:
    """Arm joints (j), opening (o) and observed chassis state (c) for one trajectory."""
    from roboatom.physextend.refine import evidence as evidence

    out: dict[str, np.ndarray] = {}
    if "j" in mode:
        out.update(base.joint_channels(uri, dataset))
    if "o" in mode:
        out.update(base.clean_channels(uri, dataset))
    if "c" in mode:
        raw, _ = evidence.raw_episode(uri, dataset, current_run().episodes[uri]["length"])
        for column, matrix in raw.items():
            if ("chassis" in column and column.startswith("observation")) or column == "state.base":
                for index in range(matrix.shape[1]):
                    out[f"{column}[{index}]/base"] = matrix[:, index]
    return out


@run_cache(maxsize=6000)
def combined_alignment(source: str, target: str, dataset: str, mode: str) -> Any:
    from roboatom.physextend.decoders.gripper_cycle import align_channels

    return align_channels(combined_channels(source, dataset, mode), combined_channels(target, dataset, mode), cycle=True)


def joint_dtw_transfer(
    row: dict[str, Any], parent: dict[str, Any], pairs: list[Any], config: dict[str, Any], proposal: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace cuts by reference boundaries warped through arm-joint DTW (per-group cost cap and blend)."""
    from roboatom.physextend.decoders.gripper_cycle import transfer_boundaries
    from roboatom.physextend.refine import rounds as rounds

    groups = base.target_groups(row["target"])
    key = next((g for g in sorted(groups, key=len, reverse=True) if g in proposal["params"]), None)
    if key is None:
        return parent, {"status": "no_rule"}
    cap, alpha, *rest = proposal["params"][key]
    shift = rest[0] if rest else None
    mode = rest[1] if len(rest) > 1 else "j"
    target = row["target"]
    labels = [s["skill"] for s in parent["segments"]]
    cuts = np.asarray([s["end_frame"] for s in parent["segments"][:-1]], dtype=float)
    if not len(cuts):
        return parent, {"status": "no_internal_boundary"}
    warps, used = [], []
    for seed in row["references"]:
        if [s["skill"] for s in seed["segments"]] != labels:
            continue
        fit = (
            base.joint_alignment(seed["data_uri"], target["data_uri"], target["dataset"])
            if mode == "j"
            else combined_alignment(seed["data_uri"], target["data_uri"], target["dataset"], mode)
        )
        if fit is None or fit.cost >= cap:
            continue
        warps.append(np.asarray(transfer_boundaries(fit, seed["segments"]), dtype=float))
        used.append({"seed": seed["benchmark_id"], "cost": float(fit.cost)})
    if not warps:
        return parent, {"status": "no_compatible_joint_fit"}
    proposal_cuts = alpha * np.median(warps, axis=0) + (1 - alpha) * cuts
    if shift is not None:
        proposal_cuts = np.where(np.abs(proposal_cuts - cuts) <= shift * target["fps"], proposal_cuts, cuts)
    final = np.clip(np.rint(proposal_cuts), 1, target["length"] - 1)
    points = [0, *final, target["length"]]
    if any(b <= a for a, b in zip(points, points[1:])):
        return parent, {"status": "collapsing_partition"}
    return rounds.set_boundaries(parent, final.astype(int)), {"status": "joint_dtw_transfer", "fits": used}


def grasp_events(uri: str, dataset: str) -> list[tuple[int, str, str]]:
    """Per-side opening events mapped to grasp (pick) / release (place) from the resting state."""
    from roboatom.physextend.refine import events as events

    fps = current_run().episodes[uri]["fps"]
    out = []
    for key, value in base.clean_channels(uri, dataset).items():
        if key.startswith("hand_synergy/"):
            continue
        unit = base.normalize_channel(value)
        if unit is None:
            continue
        side = key.rsplit("/", 1)[-1]
        rest_high = float(np.median(unit[: max(3, int(0.5 * fps))])) > 0.5
        for frame, kind in events.channel_events(unit, fps):
            out.append((frame, "pick" if (kind == "fall") == rest_high else "place", side))
    return sorted(out)


def event_transcript(
    row: dict[str, Any], parent: dict[str, Any], pairs: list[Any], config: dict[str, Any], proposal: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild a pick/place transcript from target grasp/release events, gated by the reference.

    Fires only when (a) the single reference is pure pick/place, (b) the reference's own events
    reproduce its annotation exactly (validating the event->label convention for this task), and
    (c) the target's events give a different transcript. Cuts = target event + reference offset.
    """
    import copy

    references = row["references"]
    if len(references) != 1:
        return parent, {"status": "not_single_reference"}
    reference = references[0]
    labels = [s["skill"] for s in reference["segments"]]
    if not set(labels) <= {"pick", "place"} or [s["skill"] for s in parent["segments"]] != labels:
        return parent, {"status": "not_pick_place"}
    r_events = grasp_events(reference["data_uri"], reference["dataset"])
    if [e[1] for e in r_events] != labels:
        return parent, {"status": "reference_events_disagree"}
    target = row["target"]
    t_events = grasp_events(target["data_uri"], target["dataset"])
    sequence = [e[1] for e in t_events]
    if sequence == labels or len(sequence) < 2:
        return parent, {"status": "same_transcript"}
    if abs(len(sequence) - len(labels)) > proposal["params"].get("max_count_change", 99):
        return parent, {"status": "count_change_too_large"}
    offsets: dict[str, list[float]] = {}
    for index, segment in enumerate(reference["segments"][:-1]):
        offsets.setdefault(segment["skill"], []).append((segment["end"] - r_events[index][0]) / reference["fps"])
    cuts = [
        int(round(t_events[i][0] + float(np.median(offsets.get(sequence[i], [0.0]))) * target["fps"]))
        for i in range(len(sequence) - 1)
    ]
    points = [0, *cuts, target["length"]]
    if any(b <= a for a, b in zip(points, points[1:])):
        return parent, {"status": "collapsing_partition"}
    template = {"pick": None, "place": None}
    for segment in parent["segments"]:
        template[segment["skill"]] = template[segment["skill"]] or segment
    segments = []
    for label, start, end in zip(sequence, points, points[1:]):
        segment = copy.deepcopy(template[label])
        segment.update(start=start, end=end, start_frame=start, end_frame=end)
        segments.append(segment)
    return dict(copy.deepcopy(parent), segments=segments), {
        "status": "event_transcript",
        "before": labels,
        "after": sequence,
    }


def _transfer(mode: str, cap: float, references: list[dict[str, Any]], target: dict[str, Any], labels: list[str]) -> Any:
    from roboatom.physextend.decoders.gripper_cycle import transfer_boundaries

    warps = []
    for seed in references:
        if [s["skill"] for s in seed["segments"]] != labels:
            continue
        fit = (
            base.joint_alignment(seed["data_uri"], target["data_uri"], target["dataset"])
            if mode == "j"
            else combined_alignment(seed["data_uri"], target["data_uri"], target["dataset"], mode)
        )
        if fit is not None and fit.cost < cap:
            warps.append(np.asarray(transfer_boundaries(fit, seed["segments"]), dtype=float))
    return np.median(warps, axis=0) if warps else None


def reference_loo_f1(references: list[dict[str, Any]], mode: str, cap: float) -> tuple[float | None, int]:
    """Skill-F1@0.7 of DTW transfer among the task's own annotated references (leave-one-out)."""
    from roboatom.physextend.refine import score as score

    total = np.zeros(3, dtype=int)
    used = 0
    for index, held in enumerate(references):
        others = references[:index] + references[index + 1 :]
        labels = [s["skill"] for s in held["segments"]]
        warp = _transfer(mode, cap, others, held, labels)
        if warp is None:
            continue
        cuts = np.clip(np.rint(warp), 1, held["length"] - 1)
        total += score.counts(
            score.with_boundaries(labels, list(cuts), held["length"]),
            [(s["skill"], s["start"], s["end"]) for s in held["segments"]],
        )
        used += 1
    return (score.f1(tuple(total)) if used else None), used


def loo_gated_dtw(
    row: dict[str, Any], parent: dict[str, Any], pairs: list[Any], config: dict[str, Any], proposal: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply DTW boundary transfer only for tasks where it reproduces held-out references well."""
    from roboatom.physextend.refine import rounds as rounds

    groups = base.target_groups(row["target"])
    key = next((g for g in sorted(groups, key=len, reverse=True) if g in proposal["params"]), None)
    if key is None:
        return parent, {"status": "no_rule"}
    mode, cap, threshold = proposal["params"][key]
    quality, used = reference_loo_f1(row["references"], mode, cap)
    if quality is None or used < 2 or quality < threshold:
        return parent, {"status": "loo_gate_closed", "loo_f1": quality, "loo_n": used}
    labels = [s["skill"] for s in parent["segments"]]
    if len(labels) < 2:
        return parent, {"status": "no_internal_boundary"}
    warp = _transfer(mode, cap, row["references"], row["target"], labels)
    if warp is None:
        return parent, {"status": "no_compatible_fit"}
    final = np.clip(np.rint(warp), 1, row["target"]["length"] - 1)
    points = [0, *final, row["target"]["length"]]
    if any(b <= a for a, b in zip(points, points[1:])):
        return parent, {"status": "collapsing_partition"}
    return rounds.set_boundaries(parent, final.astype(int)), {"status": "loo_gated_dtw", "loo_f1": quality, "loo_n": used}


def offset_snap(
    row: dict[str, Any], parent: dict[str, Any], pairs: list[Any], config: dict[str, Any], proposal: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Snap each cut to the nearest target gripper event + a per-group offset (fit on dev targets)."""
    from roboatom.physextend.refine import events as events
    from roboatom.physextend.refine import rounds as rounds

    groups = base.target_groups(row["target"])
    key = next((g for g in sorted(groups, key=len, reverse=True) if g in proposal["params"]), None)
    if key is None:
        return parent, {"status": "no_rule"}
    window, offset = proposal["params"][key]
    target = row["target"]
    cuts = np.asarray([s["end_frame"] for s in parent["segments"][:-1]], dtype=float)
    frames = np.asarray([e[0] for e in events.events(target["data_uri"], target["dataset"])], dtype=float)
    if not len(cuts) or not len(frames):
        return parent, {"status": "no_events"}
    final, moved = events.snap(cuts, frames, target["fps"], window, offset, target["length"])
    if not moved:
        return parent, {"status": "no_event_in_window"}
    return rounds.set_boundaries(parent, final.astype(int)), {"status": "offset_snap", "moved": moved}


@run_cache(maxsize=4000)
def joint_speed(uri: str, dataset: str) -> Any:
    from scipy.ndimage import gaussian_filter1d

    units = [u for v in base.joint_channels(uri, dataset).values() if (u := base.normalize_channel(v)) is not None]
    if not units:
        return None
    return np.mean([np.abs(np.gradient(gaussian_filter1d(u, 2))) for u in units], axis=0) * current_run().episodes[uri]["fps"]


def idle_absorb(
    row: dict[str, Any], parent: dict[str, Any], pairs: list[Any], config: dict[str, Any], proposal: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Move a cut past a still period (arms not moving) that starts at the cut: annotators end the
    previous action when the arm resumes motion, not when it stops."""
    from roboatom.physextend.refine import rounds as rounds

    groups = base.target_groups(row["target"])
    key = next((g for g in sorted(groups, key=len, reverse=True) if g in proposal["params"]), None)
    if key is None:
        return parent, {"status": "no_rule"}
    relative, minimum = proposal["params"][key]
    target = row["target"]
    speed = joint_speed(target["data_uri"], target["dataset"])
    cuts = np.asarray([s["end_frame"] for s in parent["segments"][:-1]], dtype=float)
    if speed is None or not len(cuts):
        return parent, {"status": "no_joint_signal"}
    still = speed < relative * np.percentile(speed, 90)
    final = cuts.copy()
    for index, cut in enumerate(cuts.astype(int)):
        end = cut
        while end < target["length"] - 1 and still[end]:
            end += 1
        if (end - cut) / target["fps"] >= minimum:
            high = cuts[index + 1] - 1 if index + 1 < len(cuts) else target["length"] - 1
            final[index] = min(end, high)
    if np.array_equal(final, cuts):
        return parent, {"status": "no_idle"}
    return rounds.set_boundaries(parent, final.astype(int)), {"status": "idle_absorbed"}


def transition_shift(
    row: dict[str, Any], parent: dict[str, Any], pairs: list[Any], config: dict[str, Any], proposal: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Subtract a per-(group, transition) systematic bias (seconds) fitted on dev targets only."""
    from roboatom.physextend.refine import rounds as rounds

    groups = base.target_groups(row["target"])
    key = next((g for g in sorted(groups, key=len, reverse=True) if g in proposal["params"]), None)
    if key is None:
        return parent, {"status": "no_rule"}
    shifts = proposal["params"][key]
    target = row["target"]
    labels = [s["skill"] for s in parent["segments"]]
    cuts = np.asarray([s["end_frame"] for s in parent["segments"][:-1]], dtype=float)
    final = cuts.copy()
    for index in range(len(cuts)):
        shift = shifts.get(labels[index] + ">" + labels[index + 1])
        if shift is not None:
            final[index] = cuts[index] - shift * target["fps"]
    final = np.clip(np.rint(final), 1, target["length"] - 1)
    points = [0, *final, target["length"]]
    if np.array_equal(final, cuts) or any(b <= a for a, b in zip(points, points[1:])):
        return parent, {"status": "no_shift"}
    return rounds.set_boundaries(parent, final.astype(int)), {"status": "transition_shift"}


METHODS: dict[str, Any] = {"transition_shift": transition_shift, "idle_absorb": idle_absorb, "offset_snap": offset_snap, "loo_dtw": loo_gated_dtw, "event_transcript": event_transcript, "joint_dtw": joint_dtw_transfer, "event_anchor": event_anchor, "sequence_vote": sequence_vote, "merge_short": merge_short_segments}

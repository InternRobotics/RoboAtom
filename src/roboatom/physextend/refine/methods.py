"""Sensor-only hypotheses introduced one at a time after observed feedback."""

from __future__ import annotations

from roboatom.physextend.runtime import current_run

from roboatom.physextend.runtime import run_cache
from collections import defaultdict
import re
import json
from typing import Any

import numpy as np
from roboatom.physextend.decoders.gripper_cycle import normalize_channel, align_channels, transfer_boundaries, semantic_signature

from roboatom.physextend.refine import evidence as ev
from roboatom.physextend.refine import iterative as previous
from roboatom.physextend.refine import rounds as rounds


def scalar_effector_column(column: str, width: int) -> bool:
    """Recognize explicit scalar end-effector opening, excluding Cartesian pose."""
    return width == 1 and bool(re.search(r"(?:states?\.)end_effector_(?:left|right)_position$", column))


@run_cache(maxsize=3000)
def opening_channels(uri: str, dataset: str) -> dict[str, np.ndarray]:
    """Restore missing explicit scalar opening columns without merging hands."""
    episode = current_run().episodes[uri]
    result = dict(ev.schema_grippers(uri, dataset, episode["length"], True))
    raw, _ = ev.raw_episode(uri, dataset, episode["length"])
    for column, matrix in raw.items():
        if not scalar_effector_column(column, matrix.shape[1]):
            continue
        value = np.asarray(matrix[:, 0], dtype=float)
        if len(value) != episode["length"]:
            value = np.interp(np.linspace(0, 1, episode["length"]), np.linspace(0, 1, len(value)), value)
        result[f"{column}[0]/{ev.side_of(column)}"] = value
    return result


def apply(
    row: dict[str, Any], parent: dict[str, Any], pairs: list[Any], config: dict[str, Any], proposal: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply only an implemented and explicitly registered experiment."""
    group = target_groups(row["target"])
    if proposal.get("scope") and not group & set(proposal["scope"]):
        return parent, {"status": "out_of_scope"}
    if proposal["method"] == "compose":
        routes = proposal["routes"]
        selected = next((routes[g] for g in sorted(group, key=len, reverse=True) if g in routes), None)
        if selected is None:
            return parent, {"status": "compose_parent"}
        return reference_predictions(selected, len(pairs))[row["record_id"]], {"status": "compose", "version": selected}
    if proposal["method"] in group_methods():
        return group_methods()[proposal["method"]](row, parent, pairs, config, proposal)
    if proposal["method"] == "weak_channel_filter":

        return previous.integrated_prediction(row, pairs, config)
    if proposal["method"] == "sequence_event_prior":
        return sequence_event_prior(row, parent, pairs, config, proposal)
    if proposal["method"] == "compatible_reference_group":
        return compatible_group_refine(row, parent, pairs, config)
    if proposal["method"] == "joint_state_refine":
        base = (
            reference_predictions(proposal["input_baseline"], len(pairs))[row["record_id"]]
            if proposal.get("input_baseline")
            else parent
        )
        return joint_state_refine(row, base, pairs, proposal)
    if proposal["method"] == "local_opening_search":
        return local_opening_search(row, parent, pairs, proposal)
    raise ValueError("Unknown or not-yet-implemented method")


def target_groups(target: dict[str, Any]) -> set[str]:
    """Metadata-only routing keys: dataset and dataset|embodiment."""
    robot = router_profiles()[target["data_uri"]]["robot_type"]
    return {target["dataset"], target["dataset"] + "|" + robot}


def group_methods() -> dict[str, Any]:
    from roboatom.physextend.refine import group_methods as module

    return module.METHODS


def reference_predictions(name: str, count: int) -> dict[str, dict[str, Any]]:
    """Resolve a recipe dependency in the active target's computation graph."""
    provider = current_run().prediction_provider
    if provider is None:
        raise RuntimeError('No active PhysExtend recipe resolver')
    return provider(name, count)


def local_profile_cost(
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    source_frame: float,
    target_frame: float,
    source_fps: float,
    target_fps: float,
    seconds: float = 0.5,
) -> float | None:
    """Compare equal physical-time windows using locally informative channels."""
    offsets = np.linspace(-seconds, seconds, 17)
    costs = []
    for channel in sorted(source.keys() & target.keys()):
        left = normalize_channel(source[channel])
        right = normalize_channel(target[channel])
        if left is None or right is None:
            continue
        a = np.interp(source_frame + offsets * source_fps, np.arange(len(left)), left)
        if np.ptp(a) < 0.05:
            continue
        b = np.interp(target_frame + offsets * target_fps, np.arange(len(right)), right)
        costs.append(float(np.mean((a - b) ** 2)))
    return float(np.mean(costs)) if costs else None


@run_cache(maxsize=3000)
def dexterous_channels(uri: str, dataset: str) -> dict[str, np.ndarray]:
    """Add documented hand joint positions as separate articulation evidence."""
    result = dict(opening_channels(uri, dataset))
    episode = current_run().episodes[uri]
    raw, schema = ev.raw_episode(uri, dataset, episode["length"])
    for column, matrix in raw.items():
        if any(
            word in column.lower() for word in ["action", "command", "velocity", "effort", "force", "current", "torque"]
        ):
            continue
        names = ev.flatten_names(schema.get(column, {}).get("names", []))
        if len(names) != matrix.shape[1]:
            continue
        for index, name in enumerate(names):
            if not re.search(r"hand_joint|finger|thumb", name.lower()):
                continue
            values = np.asarray(matrix[:, index], dtype=float)
            if len(values) != episode["length"]:
                values = np.interp(np.linspace(0, 1, episode["length"]), np.linspace(0, 1, len(values)), values)
            result[f"{column}[{index}]/{ev.side_of(name + ' ' + column)}"] = values
    return result


@run_cache(maxsize=3000)
def synergy_channels(uri: str, dataset: str) -> dict[str, np.ndarray]:
    """Represent each missing scalar effector by one normalized hand synergy."""
    base = opening_channels(uri, dataset)
    if any(normalize_channel(value) is not None for value in base.values()):
        return base
    groups = {}
    for key, value in dexterous_channels(uri, dataset).items():
        side = key.rsplit("/", 1)[-1]
        normalized = normalize_channel(value)
        if normalized is not None:
            groups.setdefault(side, []).append((key, normalized))
    result = {}
    for side, entries in groups.items():
        explicit = [v for key, v in entries if not key.startswith("observation.state[")]
        selected = explicit or [v for key, v in entries]
        result["hand_synergy/" + side] = np.mean(selected, axis=0)
    return result or base


def suppress_weak_sibling(channels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Remove tiny paired-hand jitter before normalization amplifies it."""
    groups = {}
    for key, value in channels.items():
        side = key.rsplit("/", 1)[-1]
        if side not in {"left", "right"}:
            continue
        stem = re.sub("left|right", "SIDE", key)
        stem = re.sub(r"\[\d+\]", "[]", stem)
        amplitude = float(np.percentile(value, 95) - np.percentile(value, 5))
        groups.setdefault(stem, {})[side] = (key, amplitude)
    dropped = set()
    for pair in groups.values():
        if set(pair) != {"left", "right"}:
            continue
        left, right = pair["left"], pair["right"]
        if left[1] < 0.01 * right[1]:
            dropped.add(left[0])
        elif right[1] < 0.01 * left[1]:
            dropped.add(right[0])
    return {key: value for key, value in channels.items() if key not in dropped}


@run_cache(maxsize=3000)
def clean_channels(uri: str, dataset: str) -> dict[str, np.ndarray]:
    return suppress_weak_sibling(synergy_channels(uri, dataset))


def hysteresis_transitions(values: np.ndarray, fps: float) -> int | None:
    """Count persistent low/high transitions independently of opening direction."""
    unit = normalize_channel(values)
    if unit is None:
        return None
    state = None
    pending = None
    run = 0
    transitions = 0
    persistence = max(2, int(round(0.1 * fps)))
    for value in unit:
        level = 0 if value < 0.25 else 1 if value > 0.75 else None
        if level is None:
            pending = None
            run = 0
            continue
        if level == pending:
            run += 1
        else:
            pending = level
            run = 1
        if run >= persistence and state != level:
            transitions += int(state is not None)
            state = level
    return transitions


@run_cache(maxsize=3000)
def episode_transitions(uri: str, dataset: str) -> float | None:
    episode = current_run().episodes[uri]
    groups = defaultdict(list)
    for key, value in clean_channels(uri, dataset).items():
        if key.startswith("hand_synergy/"):
            continue
        count = hysteresis_transitions(value, episode["fps"])
        if count is not None:
            groups[key.rsplit("/", 1)[-1]].append(count)
    return sum(float(np.mean(v)) for v in groups.values()) if groups else None


def duration_discrepancy(source: dict[str, Any], target: dict[str, Any]) -> float:
    """Compare physical durations, not frame counts across different rates."""
    return abs(float(np.log((source["length"] / source["fps"]) / (target["length"] / target["fps"]))))


def event_contrast(penalties: dict[str, float]) -> bool:
    return bool(penalties) and np.ptp(list(penalties.values())) > 1e-9


def main_skill_skeleton(signature: tuple[str, ...]) -> tuple[str, ...]:
    """Identify sequences differing solely in free-motion annotation granularity."""
    return tuple(skill for skill in signature if skill not in {"move", "recover"})


def sequence_event_prior(
    row: dict[str, Any], parent: dict[str, Any], pairs: list[Any], config: dict[str, Any], proposal: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select an observed complete sequence using extra episode-level evidence."""
    from roboatom.physextend.decoders import fusion as decoder

    groups = defaultdict(list)
    for _, candidate in pairs:
        if candidate["status"] == "ok":
            groups[tuple(candidate["seed_signature"])].append(candidate)
    if len(groups) < 2:
        return parent, {"status": "one_observed_sequence"}
    target = row["target"]
    target_events = episode_transitions(target["data_uri"], target["dataset"])
    if target_events is None:
        return parent, {"status": "no_scalar_event_evidence"}
    selection_config = dict(
        config, signature_weight=config.get("sequence_signature_weight", config.get("signature_weight", 0))
    )
    scores = {}
    penalties = {}
    duration_penalties = {}
    motion_scores = {}
    for seed, candidate in pairs:
        count = episode_transitions(seed["data_uri"], seed["dataset"])
        penalty = abs(count - target_events) / max(count, target_events, 1) if count is not None else 0
        penalties[candidate["seed_record_id"]] = penalty
        duration_penalties[candidate["seed_record_id"]] = duration_discrepancy(seed, target)
        motion_scores[candidate["seed_record_id"]] = decoder.candidate_score(candidate, selection_config)
        scores[candidate["seed_record_id"]] = (
            motion_scores[candidate["seed_record_id"]]
            + proposal["event_weight"] * penalty
            + proposal.get("duration_weight", 0) * duration_penalties[candidate["seed_record_id"]]
        )
    if proposal.get("duration_requires_event_contrast") and not event_contrast(penalties):
        scores = {key: motion_scores[key] + proposal["event_weight"] * penalties[key] for key in scores}
    blocked = []
    if proposal.get("preserve_motion_granularity"):
        skeletons = defaultdict(dict)
        for signature, values in groups.items():
            skeletons[main_skill_skeleton(signature)][signature] = values
        for alternatives in skeletons.values():
            identifiers = [c["seed_record_id"] for values in alternatives.values() for c in values]
            if len(alternatives) < 2 or event_contrast({key: penalties[key] for key in identifiers}):
                continue
            original_scores = {
                key: motion_scores[key] + proposal["event_weight"] * penalties[key] for key in identifiers
            }
            allowed, _ = decoder.choose_sequence(alternatives, original_scores, config)
            blocked.extend(signature for signature in alternatives if signature != allowed)
        groups = {signature: values for signature, values in groups.items() if signature not in blocked}
    selected, rankings = decoder.choose_sequence(groups, scores, config)
    old = tuple(s["skill"] for s in parent["segments"])
    diagnostic = {
        "status": "event_sequence_unchanged",
        "target_transitions": target_events,
        "event_penalties": penalties,
        "duration_penalties": duration_penalties,
        "sequence_scores": rankings,
        "selected": list(selected),
        "blocked_motion_granularity_sequences": [list(x) for x in blocked],
    }
    if selected == old:
        return parent, diagnostic
    original = decoder.choose_sequence

    def forced_choice(groups: Any, scores: Any, config: Any) -> Any:
        return selected, rankings

    decoder.choose_sequence = forced_choice

    try:
        output, pipeline = previous.integrated_prediction(row, pairs, config)
    finally:
        decoder.choose_sequence = original
    return output, dict(diagnostic, status="event_sequence_changed", pipeline=pipeline)


def compatible_pairs(pairs: list[Any], signature: tuple[str, ...], config: dict[str, Any]) -> tuple[list[Any], int]:
    """Choose the target-motion-supported reference's full semantic group."""
    from roboatom.physextend.decoders.fusion import candidate_score

    matching = [(seed, candidate) for seed, candidate in pairs if tuple(candidate["seed_signature"]) == signature]
    if not matching:
        raise ValueError("Selected complete sequence absent from permitted references")
    groups = defaultdict(list)
    for seed, candidate in matching:
        groups[semantic_signature(seed["segments"])].append((seed, candidate))
    anchor = min(matching, key=lambda pair: (candidate_score(pair[1], config), pair[1]["seed_record_id"]))
    return groups[semantic_signature(anchor[0]["segments"])], len(groups)


def compatible_group_refine(
    row: dict[str, Any], parent: dict[str, Any], pairs: list[Any], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = tuple(s["skill"] for s in parent["segments"])
    selected, group_count = compatible_pairs(pairs, signature, config)
    if group_count < 2:
        return parent, {"status": "already_compatible_reference_group"}
    references = [seed for seed, _ in selected]
    allowed = dict(row, references=references, seeds=references)

    output, diagnostic = previous.integrated_prediction(allowed, selected, config)
    output, shared = previous.shared_pair_refine(row, output, pairs, config, True)
    return output, {
        "status": "consistent_semantic_reference_group",
        "original_k": len(pairs),
        "compatible_k": len(selected),
        "semantic_groups": group_count,
        "selected_ids": [s["benchmark_id"] for s in references],
        "pipeline": diagnostic,
        "shared_only_extra_references": shared,
    }


def joint_indices(column: str, names: list[str], width: int) -> list[int]:
    """Select documented arm position axes, excluding command and hand axes."""
    if any(
        word in column.lower()
        for word in ["action", "command", "velocity", "effort", "torque", "force", "current", "accel"]
    ):
        return []
    explicit = bool(re.search(r"joint|arm", column.lower()))
    result = []
    for index in range(width):
        name = names[index].lower() if len(names) == width else ""
        if any(word in name for word in ["grip", "finger", "thumb", "hand_joint"]):
            continue
        if explicit or re.search(r"joint|(?:^|[._])arm(?:[._]|$)", name):
            result.append(index)
    return result


@run_cache(maxsize=3000)
def joint_channels(uri: str, dataset: str) -> dict[str, np.ndarray]:
    episode = current_run().episodes[uri]
    raw, schema = ev.raw_episode(uri, dataset, episode["length"])
    explicit = {}
    generic = {}
    for column, matrix in raw.items():
        names = ev.flatten_names(schema.get(column, {}).get("names", []))
        for index in joint_indices(column, names, matrix.shape[1]):
            name = names[index] if len(names) == matrix.shape[1] else column
            side = ev.side_of(name + " " + column)
            value = np.asarray(matrix[:, index], dtype=float)
            if len(value) != episode["length"]:
                value = np.interp(np.linspace(0, 1, episode["length"]), np.linspace(0, 1, len(value)), value)
            (generic if column == "observation.state" else explicit)[f"{column}[{index}]/{side}"] = value
    sides = {key.rsplit("/", 1)[-1] for key in explicit}
    return dict(explicit, **{key: value for key, value in generic.items() if key.rsplit("/", 1)[-1] not in sides})


@run_cache(maxsize=3000)
def unit_joints(uri: str, dataset: str) -> dict[str, np.ndarray]:
    return {
        key: unit
        for key, value in joint_channels(uri, dataset).items()
        if (unit := normalize_channel(value)) is not None
    }


@run_cache(maxsize=6000)
def joint_alignment(source: str, target: str, dataset: str) -> Any:
    return align_channels(joint_channels(source, dataset), joint_channels(target, dataset), cycle=True)


def unit_profile_cost(
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    source_frame: float,
    target_frame: float,
    source_fps: float,
    target_fps: float,
) -> float | None:
    costs = []
    offsets = np.linspace(-0.5, 0.5, 17)
    for key in sorted(source.keys() & target.keys()):
        a = source[key]
        b = target[key]
        left = np.interp(source_frame + offsets * source_fps, np.arange(len(a)), a)
        if np.ptp(left) < 0.05:
            continue
        right = np.interp(target_frame + offsets * target_fps, np.arange(len(b)), b)
        costs.append(float(np.mean((left - right) ** 2)))
    return float(np.mean(costs)) if costs else None


def opening_veto(before: float | None, after: float | None) -> bool:
    return before is not None and after is not None and after > before + 0.01


def joint_state_refine(
    row: dict[str, Any], parent: dict[str, Any], pairs: list[Any], proposal: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = row["target"]
    length = target["length"]
    old = np.asarray([s["end_frame"] for s in parent["segments"][:-1]])
    if not len(old):
        return parent, {"status": "no_internal_boundary"}
    signature = semantic_signature(parent["segments"])
    votes = defaultdict(list)
    details = []
    target_signals = unit_joints(target["data_uri"], target["dataset"])
    target_opening = (
        clean_channels(target["data_uri"], target["dataset"]) if proposal.get("opening_nonworsening") else {}
    )
    for seed, candidate in pairs:
        if semantic_signature(seed["segments"]) != signature:
            continue
        fit = joint_alignment(seed["data_uri"], target["data_uri"], target["dataset"])
        if fit is None or fit.cost >= 0.08:
            continue
        warped = transfer_boundaries(fit, seed["segments"])
        proposed = np.rint(0.5 * old + 0.5 * warped)
        source_signals = unit_joints(seed["data_uri"], seed["dataset"])
        source_opening = (
            clean_channels(seed["data_uri"], seed["dataset"]) if proposal.get("opening_nonworsening") else {}
        )
        for index, frame in enumerate(proposed):
            if abs(frame - old[index]) > 0.08 * length:
                continue
            sf = seed["segments"][index]["end"]
            before = unit_profile_cost(source_signals, target_signals, sf, old[index], seed["fps"], target["fps"])
            after = unit_profile_cost(source_signals, target_signals, sf, frame, seed["fps"], target["fps"])
            if before is not None and after is not None and before - after > 0.01:
                if proposal.get("opening_nonworsening"):
                    opening_before = local_profile_cost(
                        source_opening, target_opening, sf, old[index], seed["fps"], target["fps"]
                    )
                    opening_after = local_profile_cost(
                        source_opening, target_opening, sf, frame, seed["fps"], target["fps"]
                    )
                    if opening_veto(opening_before, opening_after):
                        continue
                votes[index].append(frame)
        details.append({"seed": seed["benchmark_id"], "cost": fit.cost, "channels": list(fit.channels)})
    final = old.copy()
    for index, values in votes.items():
        if np.ptp(values) <= 0.035 * length:
            final[index] = int(round(float(np.median(values))))
    final, _ = ev.guard_local_durations(old, final, length, 0.5)
    changed = np.flatnonzero(old != final).tolist()
    return rounds.set_boundaries(parent, final) if changed else parent, {
        "status": "independent_joint_state_support",
        "changed": changed,
        "reference_fits": details,
    }


@run_cache(maxsize=3000)
def unit_opening(uri: str, dataset: str) -> dict[str, np.ndarray]:
    return {
        key: unit
        for key, value in clean_channels(uri, dataset).items()
        if (unit := normalize_channel(value)) is not None
    }


@run_cache(maxsize=6000)
def clean_alignment(source: str, target: str, dataset: str) -> Any:
    return align_channels(clean_channels(source, dataset), clean_channels(target, dataset), cycle=True)


def local_profile_curve(
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    source_frame: float,
    target_frames: np.ndarray,
    source_fps: float,
    target_fps: float,
) -> np.ndarray | None:
    costs = []
    offsets = np.linspace(-0.5, 0.5, 17)
    for key in sorted(source.keys() & target.keys()):
        a = source[key]
        b = target[key]
        left = np.interp(source_frame + offsets * source_fps, np.arange(len(a)), a)
        if np.ptp(left) < 0.05:
            continue
        right = np.interp(target_frames[:, None] + offsets[None, :] * target_fps, np.arange(len(b)), b)
        costs.append(np.mean((right - left[None, :]) ** 2, axis=1))
    return np.mean(costs, axis=0) if costs else None


def local_opening_search(
    row: dict[str, Any], parent: dict[str, Any], pairs: list[Any], proposal: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = row["target"]
    old = np.asarray([s["end_frame"] for s in parent["segments"][:-1]])
    final = old.copy()
    duration = np.diff(np.r_[0, old, target["length"]])
    signature = semantic_signature(parent["segments"])
    target_signals = unit_opening(target["data_uri"], target["dataset"])
    matching = []
    for seed, candidate in pairs:
        if semantic_signature(seed["segments"]) != signature:
            continue
        fit = clean_alignment(seed["data_uri"], target["data_uri"], target["dataset"])
        if fit is not None and fit.cost < 0.08:
            matching.append(seed)
    details = {}
    for index, frame in enumerate(old):
        radius = min(
            int(round(proposal["search_seconds"] * target["fps"])), int(0.25 * min(duration[index], duration[index + 1]))
        )
        if radius < 1:
            continue
        grid = np.arange(max(1, frame - radius), min(target["length"] - 1, frame + radius) + 1)
        curves = []
        for seed in matching:
            source = unit_opening(seed["data_uri"], seed["dataset"])
            curve = local_profile_curve(
                source, target_signals, seed["segments"][index]["end"], grid, seed["fps"], target["fps"]
            )
            if curve is not None:
                curves.append(curve)
        if not curves:
            continue
        cost = np.mean(curves, axis=0)
        old_index = int(np.flatnonzero(grid == frame)[0])
        best = int(np.argmin(cost + 1e-12 * np.abs(grid - frame)))
        gain = float(cost[old_index] - cost[best])
        if gain > proposal["minimum_gain"]:
            final[index] = int(grid[best])
            details[str(index)] = {"gain": gain, "support": len(curves)}
    final, conflicts = ev.guard_local_durations(old, final, target["length"], 0.5)
    changed = np.flatnonzero(final != old).tolist()
    return rounds.set_boundaries(parent, final) if changed else parent, {
        "status": "bounded_local_opening_search",
        "changed": changed,
        "local_evidence": details,
        "conflicts": conflicts,
    }


@run_cache(maxsize=1)
def router_profiles() -> dict[str, Any]:
    return json.loads((current_run().workdir / "audit/embodiments.json").read_text())

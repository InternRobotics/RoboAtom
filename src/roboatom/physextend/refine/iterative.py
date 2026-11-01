"""Run exactly one preregistered sensor-only change after reading prior results."""

from __future__ import annotations

from roboatom.physextend.runtime import current_run

from collections import defaultdict
import logging
from typing import Any

from roboatom.physextend.refine import gripper
from roboatom.physextend.refine.sequential import input_row
from roboatom.physextend.runtime import run_cache
from roboatom.physextend.decoders.gripper_cycle import align_channels
from roboatom.physextend.refine.evidence import dominant_side


@run_cache(maxsize=6000)
def hand_alignment(source_uri: str, target_uri: str, dataset: str, cycle: bool) -> Any:
    """Ignore a weak opening side only when both trajectories agree on the active hand."""
    source = mapped_grippers(source_uri, dataset)
    target = mapped_grippers(target_uri, dataset)
    source_side = dominant_side(source)
    target_side = dominant_side(target)
    if source_side is not None and source_side == target_side:
        source = {
            key: value
            for key, value in source.items()
            if key.endswith("/" + source_side)
        }
        target = {
            key: value
            for key, value in target.items()
            if key.endswith("/" + source_side)
        }
    return align_channels(source, target, cycle=cycle)


def mapped_grippers(uri: str, dataset: str) -> dict[str, Any]:
    """Use the final recipe's weak-channel filter for alignment."""
    from roboatom.physextend.refine.methods import clean_channels
    return clean_channels(uri, dataset)


LOG = logging.getLogger(__name__)


def single_profile_refine(
    row: dict[str, Any],
    parent: dict[str, Any],
    pairs: list[Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Allow one reference only with local profile gain and bounded movement."""
    import copy
    import numpy as np
    from roboatom.physextend.decoders.gripper_cycle import (
        normalize_channel,
        boundary_profile_cost,
        ordered_boundaries,
        transfer_boundaries,
    )
    from roboatom.physextend.decoders.fusion import refined_path

    if len(pairs) != 1:
        return parent, {"status": "multireference_parent_retained"}
    seed, candidate = pairs[0]
    fit = gripper.alignment(
        gripper.episode_uri(seed),
        gripper.episode_uri(row["target"]),
        row["target"]["dataset"],
        True,
    )
    if fit is None or fit.cost >= 0.08:
        return parent, {
            "status": "single_alignment_unreliable",
            "cost": None if fit is None else fit.cost,
        }
    length = row["target"]["length"]
    old = np.asarray([segment["end_frame"] for segment in parent["segments"][:-1]])
    if not len(old):
        return parent, {"status": "one_segment_no_boundary"}
    warped = transfer_boundaries(fit, seed["segments"])
    original = np.asarray(refined_path(candidate, config), dtype=float)
    if len(original) != len(old):
        return parent, {"status": "sequence_incompatible"}
    proposed = ordered_boundaries(0.2 * original + 0.8 * warped, length)

    def normalized(episode: dict[str, Any]) -> dict[str, np.ndarray]:
        values = mapped_grippers(episode["data_uri"], episode["dataset"])
        return {
            key: unit
            for key, value in values.items()
            if key in fit.channels
            if (unit := normalize_channel(value)) is not None
        }

    source = normalized(seed)
    target = normalized(row["target"])
    source_frames = [segment["end"] for segment in seed["segments"][:-1]]
    gains = np.asarray(
        [
            boundary_profile_cost(source, target, s, o)
            - boundary_profile_cost(source, target, s, n)
            for s, o, n in zip(source_frames, old, proposed)
        ]
    )
    accept = (gains > 0.01) & (np.abs(proposed - old) <= 0.08 * length)
    if not accept.any():
        return parent, {
            "status": "single_profile_no_supported_move",
            "alignment_cost": fit.cost,
            "local_gains": gains.tolist(),
        }
    final = ordered_boundaries(np.where(accept, proposed, old), length)
    output = copy.deepcopy(parent)
    points = np.r_[0, final, length]
    for index, segment in enumerate(output["segments"]):
        segment.update(
            start=int(points[index]),
            start_frame=int(points[index]),
            end=int(points[index + 1]),
            end_frame=int(points[index + 1]),
        )
    return output, {
        "status": "single_profile_refined",
        "alignment_cost": fit.cost,
        "accepted_boundaries": int(accept.sum()),
        "local_gains": gains.tolist(),
    }


def shared_pair_refine(
    row: dict[str, Any],
    parent: dict[str, Any],
    pairs: list[Any],
    config: dict[str, Any],
    guard_durations: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Borrow boundary evidence only through explicit compatible shared subtasks."""
    import copy
    import numpy as np
    from roboatom.physextend.decoders.gripper_cycle import transfer_boundaries, ordered_boundaries
    from roboatom.physextend.refine.evidence import shared_adjacent_pairs

    signature = tuple(segment["skill"] for segment in parent["segments"])
    if len({tuple(candidate["seed_signature"]) for _, candidate in pairs}) < 2:
        return parent, {"status": "no_cross_sequence_reference"}
    old = np.asarray([segment["end_frame"] for segment in parent["segments"][:-1]])
    if not len(old):
        return parent, {"status": "one_segment_no_boundary"}
    anchors = defaultdict(list)
    for seed, candidate in pairs:
        correspondence = shared_adjacent_pairs(parent["segments"], seed["segments"])
        if not correspondence:
            continue
        fit = gripper.alignment(
            seed["data_uri"], row["target"]["data_uri"], row["target"]["dataset"], True
        )
        if fit is None or fit.cost >= 0.08:
            continue
        warped = transfer_boundaries(fit, seed["segments"])
        same = tuple(candidate["seed_signature"]) == signature
        sides = tuple(sorted({channel.rsplit("/", 1)[-1] for channel in fit.channels}))
        for index, reference_index in correspondence.items():
            anchors[index].append(
                {
                    "frame": float(warped[reference_index]),
                    "same": same,
                    "sides": sides,
                    "source": seed["benchmark_id"],
                }
            )
    final = old.copy()
    accepted = []
    length = row["target"]["length"]
    for index, items in anchors.items():
        if (
            len(items) < 2
            or not any(item["same"] for item in items)
            or all(item["same"] for item in items)
        ):
            continue
        if len({item["sides"] for item in items}) != 1:
            continue
        frames = np.asarray([item["frame"] for item in items])
        proposal = float(frames.mean())
        if (
            np.ptp(frames) > 0.035 * length
            or abs(proposal - old[index]) > 0.08 * length
        ):
            continue
        final[index] = round(proposal)
        accepted.append(
            {
                "boundary_index": index,
                "sources": [item["source"] for item in items],
                "old_frame": int(old[index]),
                "new_frame": int(final[index]),
            }
        )
    rolled_back = []
    if guard_durations:
        from roboatom.physextend.refine.evidence import guard_local_durations

        final, rolled_back = guard_local_durations(old, final, length)
    if np.array_equal(old, final):
        return parent, {
            "status": "no_reliable_shared_move",
            "supported_pairs": len(anchors),
        }
    final = ordered_boundaries(final, length)
    output = copy.deepcopy(parent)
    points = np.r_[0, final, length]
    for index, segment in enumerate(output["segments"]):
        segment.update(start_frame=int(points[index]), end_frame=int(points[index + 1]))
        if "start" in segment:
            segment["start"] = int(points[index])
        if "end" in segment:
            segment["end"] = int(points[index + 1])
    return output, {
        "status": "shared_subtask_boundaries",
        "accepted": accepted,
        "duration_guard_rollbacks": rolled_back,
    }


def integrated_prediction(
    row: dict[str, Any], pairs: list[Any], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Predict from references and signals without intermediate experiment predictions."""
    from roboatom.physextend.decoders.fusion import decode_candidates

    validated = input_row({"target": row["target"], "seeds": row["references"]})
    current_run().episodes.update(
        {
            episode["data_uri"]: episode
            for episode in [validated["target"], *validated["references"]]
        }
    )
    base, sequence_diagnostic = decode_candidates(
        validated,
        [candidate for _, candidate in pairs],
        config,
        "single" if len(pairs) == 1 else "multi",
    )
    mapped, mapping_diagnostic = gripper.refine(
        validated, base, pairs, config, gripper.CONSERVATIVE_CONFIG
    )
    local, local_diagnostic = single_profile_refine(validated, mapped, pairs, config)
    output, shared_diagnostic = shared_pair_refine(
        validated, local, pairs, config, True
    )
    return output, {
        "status": "integrated_boundary_candidate",
        "sequence": sequence_diagnostic,
        "mapping": mapping_diagnostic,
        "single_reference": local_diagnostic,
        "shared_subtasks": shared_diagnostic,
        "mixed_sequence_uncertainty": len(
            {tuple(c["seed_signature"]) for _, c in pairs}
        )
        > 1,
    }

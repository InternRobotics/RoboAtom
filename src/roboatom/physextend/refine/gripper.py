"""Generate four frozen offline ablations; target labels are never decoded."""

from __future__ import annotations

from collections import defaultdict
import copy
from roboatom.physextend.runtime import run_cache
import logging
import re
from typing import Any

import numpy as np

from roboatom.physextend.decoders.gripper_cycle import (
    transfer_boundaries,
    semantic_signature,
    ordered_boundaries,
    normalize_channel,
    boundary_profile_cost,
)
from roboatom.physextend.decoders.fusion import (
    candidate_score,
    decode_candidates,
    refined_path,
)
from roboatom.physextend.features.sources import _rewrite_current_uri

LOGGER = logging.getLogger(__name__)
CONSERVATIVE_CONFIG = {
    "cycle": True,
    "blend": 0.8,
    "fusion": "semantic_global",
    "conservative": True,
}


def identifier(row: dict[str, Any]) -> str:
    return str(row.get("episode_index") or row.get("example_id") or row["record_id"])


def canonical_uri(value: str, dataset: str) -> str:
    result = value.replace(":s3://", ":")
    return _rewrite_current_uri(dataset, result)


def episode_uri(episode: dict[str, Any]) -> str:
    value = (
        episode.get("resolved_sensor_uri")
        or episode.get("data_uri")
        or episode.get("signals_path")
        or episode.get("path")
        or (episode.get("video_paths") or [""])[0]
    )
    value = re.sub(
        r"/videos/(chunk-\d+)/[^/]+/(episode_\d+)\.mp4$",
        r"/data/\1/\2.parquet",
        str(value),
    )
    return canonical_uri(value, episode["dataset"])


def load_channels(uri: str, dataset: str) -> dict[str, np.ndarray]:
    """Use the final recipe's documented and filtered sensor channels."""
    from roboatom.physextend.refine.methods import clean_channels
    return clean_channels(uri, dataset)


def alignment(source_uri: str, target_uri: str, dataset: str, cycle: bool) -> Any:
    """Align the active hand using the final channel selection rules."""
    from roboatom.physextend.refine.iterative import hand_alignment
    return hand_alignment(source_uri, target_uri, dataset, cycle)


@run_cache(maxsize=600)
def normalized_channels(uri: str, dataset: str) -> dict[str, np.ndarray]:
    result = {}
    for name, values in load_channels(uri, dataset).items():
        normalized = normalize_channel(values)
        if normalized is not None:
            result[name] = normalized
    return result


def modified_candidate(
    candidate: dict[str, Any], boundaries: np.ndarray
) -> dict[str, Any]:
    result = copy.deepcopy(candidate)
    points = np.r_[0, boundaries, result["prediction"]["frames"]]
    for index, segment in enumerate(result["prediction"]["segments"]):
        segment["start_frame"] = int(points[index])
        segment["end_frame"] = int(points[index + 1])
    for index, item in enumerate(result["diagnostic"]["decisions"]):
        for key in [
            "frame",
            "ours_frame",
            "sensor_frame",
            "primary_frame",
            "coarse_frame",
        ]:
            item[key] = int(boundaries[index])
        item["anchor_applied"] = False
        item["anchor_proposals"] = []
    return result


def refine(
    row: dict[str, Any],
    baseline: dict[str, Any],
    pairs: list[Any],
    config: dict[str, Any],
    version: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refine predictions using sensors and reference-only labels."""
    length = int(row["target"]["length"])
    signature = tuple(segment["skill"] for segment in baseline["segments"])
    eligible = []
    for seed, candidate in pairs:
        if tuple(candidate["seed_signature"]) != signature:
            continue
        fit = alignment(
            episode_uri(seed),
            episode_uri(row["target"]),
            row["target"]["dataset"],
            version["cycle"],
        )
        if fit is None:
            continue
        warped = transfer_boundaries(fit, seed["segments"])
        original = np.asarray(refined_path(candidate, config), dtype=float)
        proposal = ordered_boundaries(
            (1 - version["blend"]) * original + version["blend"] * warped, length
        )
        eligible.append(
            {
                "seed": seed,
                "candidate": candidate,
                "fit": fit,
                "proposal": proposal,
                "original": original,
                "score": candidate_score(candidate, config),
            }
        )
    if not eligible:
        return baseline, {"status": "no_compatible_active_channels"}
    # Do not drop a reference merely because its telemetry was unavailable.
    total = sum(tuple(c["seed_signature"]) == signature for _, c in pairs)
    if len(eligible) != total:
        return baseline, {"status": "partial_signal_coverage"}
    if version["fusion"] == "original":
        adjusted = [
            modified_candidate(item["candidate"], item["proposal"]) for item in eligible
        ]
        adapted = {
            "record_id": identifier(row),
            "canonical_task_id": baseline["task_id"],
            "target": row["target"],
            "support_mode": "same",
            "protocol": baseline["protocol"],
        }
        frozen = dict(config, anchor_blend=0, sensor_scale=0)
        pred, diagnostic = decode_candidates(
            adapted, adjusted, frozen, "single" if len(row["seeds"]) == 1 else "multi"
        )
        proposed = np.asarray(
            [segment["end_frame"] for segment in pred["segments"][:-1]]
        )
    else:
        groups = defaultdict(list)
        for item in eligible:
            groups[semantic_signature(item["seed"]["segments"])].append(item)
        baseline_signature = semantic_signature(baseline["segments"])
        eligible = groups.get(baseline_signature, [])
        if not eligible:
            return baseline, {"status": "no_compatible_subtask_semantics"}
        costs = np.asarray(
            [item["score"] + 0.5 * item["fit"].cost for item in eligible]
        )
        weights = np.exp(-(costs - costs.min()) / 2)
        proposals = np.stack([item["proposal"] for item in eligible])
        # A single weight per reference over the complete path makes durations
        # convex combinations: boundary mixing cannot create a collapsed segment.
        proposed = ordered_boundaries(
            np.average(proposals, axis=0, weights=weights), length
        )
        diagnostic = {
            "semantic_groups": len(groups),
            "global_weights": weights.tolist(),
        }
        if version.get("conservative"):
            old = np.asarray(
                [segment["end_frame"] for segment in baseline["segments"][:-1]]
            )
            spread = (
                np.ptp(proposals, axis=0)
                if len(proposals) > 1
                else np.full(len(old), length)
            )
            reliable = np.mean([item["fit"].cost for item in eligible]) < 0.08
            agrees = spread <= 0.035 * length
            bounded = np.abs(proposed - old) <= 0.08 * length
            # Require two independently supplied references to support the move;
            # keep K1 unchanged in this conservative multi-reference variant.
            accept = agrees & bounded & reliable & (len(eligible) >= 2)
            if version.get("local_evidence"):
                target_signals = normalized_channels(
                    episode_uri(row["target"]), row["target"]["dataset"]
                )
                old_costs = []
                new_costs = []
                for item in eligible:
                    seed = item["seed"]
                    source_signals = normalized_channels(
                        episode_uri(seed), seed["dataset"]
                    )
                    source_frames = [
                        segment["end"] for segment in seed["segments"][:-1]
                    ]
                    old_costs.append(
                        [
                            boundary_profile_cost(
                                source_signals,
                                target_signals,
                                source_frame,
                                target_frame,
                            )
                            for source_frame, target_frame in zip(source_frames, old)
                        ]
                    )
                    new_costs.append(
                        [
                            boundary_profile_cost(
                                source_signals,
                                target_signals,
                                source_frame,
                                target_frame,
                            )
                            for source_frame, target_frame in zip(
                                source_frames, proposed
                            )
                        ]
                    )
                old_costs = np.asarray(old_costs)
                new_costs = np.asarray(new_costs)
                gains = np.mean(old_costs - new_costs, axis=0)
                supports = np.all(new_costs <= old_costs + 0.03, axis=0)
                accept = accept & (gains > 0.01) & supports
                diagnostic["local_profile_gains"] = gains.tolist()
            proposed = ordered_boundaries(np.where(accept, proposed, old), length)
            diagnostic["accepted_boundaries"] = int(accept.sum())
    output = copy.deepcopy(baseline)
    points = np.r_[0, proposed, length]
    for index, segment in enumerate(output["segments"]):
        segment.update(
            start=int(points[index]),
            start_frame=int(points[index]),
            end=int(points[index + 1]),
            end_frame=int(points[index + 1]),
        )
    diagnostic.update(
        status="refined",
        channels=[item["fit"].channels for item in eligible],
        alignment_costs=[item["fit"].cost for item in eligible],
        references=[item["candidate"]["seed_physical_id"] for item in eligible],
    )
    return output, diagnostic

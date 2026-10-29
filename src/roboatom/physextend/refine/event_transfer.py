"""Reference-calibrated event anchoring.

For each predicted boundary, find the reference's own annotated boundary and the
gripper event of each kind closest to it; if the target has an event of the same
side/kind near the predicted cut, place the cut at target_event + reference_offset.
Uses only permitted inputs (reference annotation + both trajectories).
"""

from __future__ import annotations

from roboatom.physextend.runtime import current_run

from roboatom.physextend.runtime import run_cache
from typing import Any

import numpy as np

from roboatom.physextend.refine import evidence as ev
from roboatom.physextend.refine import events as X
from roboatom.physextend.refine import methods as base

SCALAR_OPENING = {
    # Source columns documented as scalar opening; generic extractor otherwise mixes in pose axes.
    "InternData-A1|franka": ["states.gripper.position", "actions.gripper.openness"],
}


@run_cache(maxsize=6000)
def episode_events(uri: str, dataset: str, group: str) -> tuple[tuple[int, str, str], ...]:
    columns = SCALAR_OPENING.get(group)
    if not columns:
        return X.events(uri, dataset)
    episode = current_run().episodes[uri]
    raw, _ = ev.raw_episode(uri, dataset, episode["length"])
    for column in columns:
        if column in raw:
            unit = base.normalize_channel(raw[column][:, 0])
            if unit is not None:
                return tuple((f, k, "single") for f, k in X.channel_events(unit, episode["fps"]))
    return ()


def anchored(
    row: dict[str, Any], labels: list[str], cuts: np.ndarray, group: str, window: float, attach: float, spread: float
) -> tuple[np.ndarray, dict[str, Any]]:
    target = row["target"]
    fps = target["fps"]
    t_events = episode_events(target["data_uri"], target["dataset"], group)
    if not t_events:
        return cuts, {}
    votes: dict[int, list[float]] = {}
    for seed in row["references"]:
        if [s["skill"] for s in seed["segments"]] != labels:
            continue
        r_events = episode_events(seed["data_uri"], seed["dataset"], group)
        if not r_events:
            continue
        for index, segment in enumerate(seed["segments"][:-1]):
            boundary = segment["end"]
            near = [(abs(b - boundary), b, kind, side) for b, kind, side in r_events if abs(b - boundary) <= attach * seed["fps"]]
            if not near:
                continue
            _, frame, kind, side = min(near)
            offset = (boundary - frame) / seed["fps"]
            candidates = [f for f, k, s in t_events if k == kind and s == side and abs(f - cuts[index]) <= window * fps]
            if not candidates:
                continue
            anchor = min(candidates, key=lambda f: abs(f - cuts[index]))
            votes.setdefault(index, []).append(anchor + offset * fps)
    final = cuts.astype(float).copy()
    moved = {}
    for index in sorted(votes):
        values = votes[index]
        if np.ptp(values) > spread * fps:
            continue
        proposal = float(np.median(values))
        low = final[index - 1] + 1 if index else 1
        high = cuts[index + 1] - 1 if index + 1 < len(cuts) else target["length"] - 1
        if low <= proposal <= high:
            final[index] = proposal
            moved[str(index)] = round((proposal - cuts[index]) / fps, 3)
    return np.rint(final), moved

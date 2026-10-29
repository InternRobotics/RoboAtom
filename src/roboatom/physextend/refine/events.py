"""Persistent gripper open/close events of one trajectory (sensor-only, no annotation)."""

from __future__ import annotations

from roboatom.physextend.runtime import current_run

from roboatom.physextend.runtime import run_cache

import numpy as np

from roboatom.physextend.refine import methods as base


def channel_events(unit: np.ndarray, fps: float, persist_seconds: float = 0.15) -> list[tuple[int, str]]:
    """0.5-crossings that persist; returned as (frame, rise|fall)."""
    high = unit > 0.5
    persist = max(2, int(round(persist_seconds * fps)))
    events = []
    current = bool(high[0])
    index = 0
    while index < len(high):
        if bool(high[index]) != current:
            end = index
            while end < len(high) and high[end] == high[index] and end - index < persist:
                end += 1
            if end - index >= persist:
                events.append((index, "rise" if high[index] else "fall"))
                current = bool(high[index])
            index = end
        else:
            index += 1
    return events


@run_cache(maxsize=6000)
def events(uri: str, dataset: str) -> tuple[tuple[int, str, str], ...]:
    """All opening-channel events as (frame, kind, side), sorted by frame."""
    episode = current_run().episodes[uri]
    out = []
    for key, value in base.clean_channels(uri, dataset).items():
        if key.startswith("hand_synergy/"):
            continue
        unit = base.normalize_channel(value)
        if unit is None:
            continue
        side = key.rsplit("/", 1)[-1]
        out.extend((frame, kind, side) for frame, kind in channel_events(unit, episode["fps"]))
    return tuple(sorted(out))


def snap(
    cuts: np.ndarray, frames: np.ndarray, fps: float, window: float, offset: float, length: int
) -> tuple[np.ndarray, list[int]]:
    """Move each cut to (nearest event + offset) when that event lies within the window."""
    final = cuts.astype(float).copy()
    moved = []
    if not len(frames):
        return final, moved
    for index, cut in enumerate(cuts):
        nearest = frames[np.argmin(np.abs(frames - cut))]
        if abs(nearest - cut) <= window * fps:
            proposal = nearest + offset * fps
            low = final[index - 1] + 1 if index else 1
            high = cuts[index + 1] - 1 if index + 1 < len(cuts) else length - 1
            if low <= proposal <= high:
                final[index] = proposal
                moved.append(index)
    return np.rint(final), moved

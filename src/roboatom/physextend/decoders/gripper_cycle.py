"""Offline channel-preserving gripper alignment, without target annotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class Alignment:
    """A reference-to-query monotone alignment and observable fit cost."""

    source_grid: np.ndarray
    target_frames: np.ndarray
    cost: float
    channels: tuple[str, ...]


def normalize_channel(values: np.ndarray) -> np.ndarray | None:
    """Normalize an active channel without assigning opening direction."""
    low, high = np.percentile(values, [5, 95])
    if high - low < 1e-5:
        return None
    return np.clip((values - low) / (high - low), 0, 1)


def monotone_path(cost: np.ndarray, band: float = 0.25) -> tuple[np.ndarray, float]:
    """Find an order-preserving path with bounded normalized time distortion."""
    size = cost.shape[0]
    if cost.shape != (size, size) or size < 2:
        raise ValueError("Expected a square alignment cost with at least two frames")
    distance = np.full((size + 1, size + 1), np.inf)
    distance[0, 0] = 0
    predecessors = np.zeros((size + 1, size + 1), dtype=np.uint8)
    for diagonal in range(2, 2 * size + 1):
        rows = np.arange(max(1, diagonal - size), min(size, diagonal - 1) + 1)
        cols = diagonal - rows
        keep = np.abs(rows - cols) <= max(2, round(band * size))
        rows, cols = rows[keep], cols[keep]
        choices = np.stack(
            [
                distance[rows - 1, cols - 1],
                distance[rows - 1, cols] + 0.02,
                distance[rows, cols - 1] + 0.02,
            ]
        )
        selected = np.argmin(choices, axis=0)
        distance[rows, cols] = (
            cost[rows - 1, cols - 1] + choices[selected, np.arange(len(rows))]
        )
        predecessors[rows, cols] = selected
    row, col = size, size
    pairs = []
    while row and col:
        pairs.append((row - 1, col - 1))
        step = predecessors[row, col]
        row -= step != 2
        col -= step != 1
    pairs.reverse()
    return np.asarray(pairs), float(distance[size, size] / len(pairs))


def align_channels(
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    *,
    cycle: bool,
    samples: int = 200,
) -> Alignment | None:
    """Align complete multichannel levels, optionally emphasizing transitions.

    Channel keys retain source-column and dimension identity. Constant channels
    are ignored, and no maximum across left/right or state/action is taken.
    """
    source_values, target_values, names = [], [], []
    for name in sorted(source.keys() & target.keys()):
        left, right = normalize_channel(source[name]), normalize_channel(target[name])
        if left is None or right is None:
            continue
        source_values.append(
            np.interp(np.linspace(0, 1, samples), np.linspace(0, 1, len(left)), left)
        )
        target_values.append(
            np.interp(np.linspace(0, 1, samples), np.linspace(0, 1, len(right)), right)
        )
        names.append(name)
    if not names:
        return None
    left = np.stack(source_values, axis=1)
    right = np.stack(target_values, axis=1)
    level_cost = cdist(left, right, "sqeuclidean") / len(names)
    cost = level_cost
    if cycle:
        left_rate = np.gradient(gaussian_filter1d(left, 1, axis=0), axis=0) * 6
        right_rate = np.gradient(gaussian_filter1d(right, 1, axis=0), axis=0) * 6
        cost = cost + 0.7 * cdist(left_rate, right_rate, "sqeuclidean") / len(names)
    path, fit = monotone_path(cost)
    source_length = len(next(iter(source.values())))
    target_length = len(next(iter(target.values())))
    mapped = np.array(
        [np.median(path[path[:, 0] == index, 1]) for index in range(samples)]
    )
    return Alignment(
        np.linspace(0, source_length - 1, samples),
        mapped / (samples - 1) * (target_length - 1),
        fit,
        tuple(names),
    )


def transfer_boundaries(
    alignment: Alignment, segments: list[dict[str, Any]]
) -> np.ndarray:
    """Transfer reference-only labels through a signal-derived monotone path."""
    return np.interp(
        [segment["end"] for segment in segments[:-1]],
        alignment.source_grid,
        alignment.target_frames,
    )


def boundary_profile_cost(
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    source_frame: float,
    target_frame: float,
) -> float:
    """Compare normalized channel neighborhoods without query annotations."""
    costs = []
    for name in sorted(source.keys() & target.keys()):
        left, right = source[name], target[name]
        offsets = np.linspace(-0.02, 0.02, 15)
        source_profile = np.interp(
            source_frame + offsets * len(left), np.arange(len(left)), left
        )
        target_profile = np.interp(
            target_frame + offsets * len(right), np.arange(len(right)), right
        )
        costs.append(float(np.mean((source_profile - target_profile) ** 2)))
    return float(np.mean(costs)) if costs else float("inf")


def semantic_signature(
    segments: list[dict[str, Any]],
) -> tuple[tuple[str, str, str], ...]:
    """Conservatively prevent fusion of different explicit objects or hands."""
    result = []
    for segment in segments:
        text = str(segment.get("action_text", "")).lower().strip()
        side = (
            "both"
            if "both" in text
            else (
                "left"
                if "left" in text
                else "right" if "right" in text else "unspecified"
            )
        )
        phrase = re.sub(r"^(picks? up|places?|puts?|grasps?|moves?)\s+", "", text)
        phrase = re.split(
            r"\s+(?:onto|into|on|to|with|from|at|in)\s+", phrase, maxsplit=1
        )[0]
        phrase = re.sub(r"\b(?:the|a|an)\b", "", phrase)
        phrase = re.sub(r"[^\w\s]", "", phrase)
        phrase = " ".join(phrase.split())
        result.append((str(segment["skill"]).lower(), phrase, side))
    return tuple(result)


def ordered_boundaries(values: np.ndarray, length: int) -> np.ndarray:
    """Project rounded proposals to a nonempty partition of the full video."""
    count = len(values)
    if length <= count:
        raise ValueError("Not enough frames for a nonempty partition")
    result = np.clip(np.rint(values).astype(int), 1, length - 1)
    for index in range(count):
        result[index] = max(
            index + 1, result[index], result[index - 1] + 1 if index else 1
        )
    for index in range(count - 1, -1, -1):
        result[index] = min(
            length - (count - index),
            result[index],
            result[index + 1] - 1 if index < count - 1 else length - 1,
        )
    return result

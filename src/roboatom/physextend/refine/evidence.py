"""Sensor-only evidence for sequential extend experiments."""

from __future__ import annotations

from roboatom.physextend.runtime import run_cache
import json
import re
from typing import Any

import numpy as np

from roboatom.physextend.refine.sequential import paths, seeded_decoder


def flatten_names(value: Any) -> list[str]:
    """Flatten documented source feature names without assuming arm order."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [name for item in value for name in flatten_names(item)]
    if isinstance(value, dict):
        return [name for item in value.values() for name in flatten_names(item)]
    return []


def side_of(name: str) -> str:
    """Recognize only explicitly named sides."""
    text = name.lower()
    if "left" in text or re.search(r"(?:^|[._])l_", text):
        return "left"
    if "right" in text or re.search(r"(?:^|[._])r_", text):
        return "right"
    return "unspecified"


def gripper_name(name: str) -> bool:
    """Recognize source-documented opening axes, excluding arm pose dimensions."""
    return bool(re.search(r"grip|(?:^|[._])(?:l_g|r_g)$", name.lower()))


@run_cache(maxsize=3000)
def raw_episode(
    uri: str, dataset: str, length: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read only already projected numeric sensor caches from legal episodes."""
    episode = {"data_uri": uri, "dataset": dataset, "length": length}
    feature, raw_path, metadata_path = paths(episode)
    metadata = json.loads(metadata_path.read_text())
    with np.load(raw_path, allow_pickle=False) as raw:
        values = {
            name: np.asarray(raw[f"column_{index}"], dtype=float)
            for index, name in enumerate(metadata["columns"])
        }
    schema = seeded_decoder.sensor_schema(uri)
    return values, schema


@run_cache(maxsize=3000)
def schema_grippers(
    uri: str, dataset: str, length: int, opening_only: bool = False
) -> dict[str, np.ndarray]:
    """Recover real state opening axes from source schema, retaining each side."""
    from roboatom.physextend.features.feature_mapping import schema_gripper_indices

    raw, schema = raw_episode(uri, dataset, length)
    result = {}
    explicit = []
    generic = []
    for column, matrix in raw.items():
        lower = column.lower()
        if any(word in lower for word in ["action", "command", "mode", "activity"]):
            continue
        if opening_only and any(
            word in lower
            for word in ["velocity", "effort", "torque", "force", "current", "accel"]
        ):
            continue
        description = schema.get(column, {})
        names = flatten_names(description.get("names", []))
        indices = set(schema_gripper_indices(description, matrix.shape[1]))
        if len(names) == matrix.shape[1]:
            indices.update(
                index for index, name in enumerate(names) if gripper_name(name)
            )
        if "gripper" in lower and not indices:
            indices = set(range(matrix.shape[1]))
        for index in sorted(indices):
            name = (
                names[index] if len(names) == matrix.shape[1] else f"{column}[{index}]"
            )
            if opening_only and any(
                word in name.lower()
                for word in [
                    "velocity",
                    "effort",
                    "torque",
                    "force",
                    "current",
                    "accel",
                ]
            ):
                continue
            side = side_of(name + " " + column)
            values = np.asarray(matrix[:, index], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite schema-documented gripper values")
            if len(values) != length:
                values = np.interp(
                    np.linspace(0, 1, length), np.linspace(0, 1, len(values)), values
                )
            entry = (f"{column}[{index}]/{side}", values)
            (generic if column == "observation.state" else explicit).append(entry)
    # Prefer explicit opening measurements per side, retaining a missing arm.
    explicit_sides = {key.rsplit("/", 1)[-1] for key, _ in explicit}
    selected = explicit + [
        (key, values)
        for key, values in generic
        if key.rsplit("/", 1)[-1] not in explicit_sides
    ]
    for key, values in selected:
        result[key] = values
    return result


def dominant_side(channels: dict[str, np.ndarray], ratio: float = 4.0) -> str | None:
    """Identify a clearly dominant opening side only in comparable channel pairs."""
    groups = {}
    for key, value in channels.items():
        side = key.rsplit("/", 1)[-1]
        if side not in {"left", "right"}:
            continue
        stem = re.sub(r"left|right", "SIDE", key)
        stem = re.sub(r"\[\d+\]", "[]", stem)
        groups.setdefault(stem, {})[side] = float(
            np.percentile(value, 95) - np.percentile(value, 5)
        )
    votes = []
    for pair in groups.values():
        if set(pair) != {"left", "right"}:
            continue
        left, right = pair["left"], pair["right"]
        if left > ratio * max(right, 1e-8):
            votes.append("left")
        elif right > ratio * max(left, 1e-8):
            votes.append("right")
        else:
            return None
    return votes[0] if votes and len(set(votes)) == 1 else None


def shared_adjacent_pairs(
    selected_steps: list[dict[str, Any]], reference_steps: list[dict[str, Any]]
) -> dict[int, int]:
    """Match unique adjacent semantic pairs, never segment ordinals across sequences."""
    from collections import Counter
    from roboatom.physextend.decoders.gripper_cycle import semantic_signature

    selected = list(semantic_signature(selected_steps))
    reference = list(semantic_signature(reference_steps))
    selected_pairs = list(zip(selected, selected[1:]))
    reference_pairs = list(zip(reference, reference[1:]))
    selected_counts = Counter(selected_pairs)
    reference_counts = Counter(reference_pairs)
    return {
        index: reference_pairs.index(pair)
        for index, pair in enumerate(selected_pairs)
        if selected_counts[pair] == 1 and reference_counts[pair] == 1
    }


def guard_local_durations(
    old: np.ndarray, proposed: np.ndarray, length: int, minimum_fraction: float = 0.5
) -> tuple[np.ndarray, list[int]]:
    """Rollback conflicting local moves instead of creating one-frame segments."""
    old = np.asarray(old, dtype=int)
    result = np.asarray(proposed, dtype=int).copy()
    old_duration = np.diff(np.r_[0, old, length])
    if np.any(old_duration <= 0):
        raise ValueError("Parent segmentation must be a positive partition")
    rolled = set()
    for _ in range(len(old) + 1):
        duration = np.diff(np.r_[0, result, length])
        invalid = np.flatnonzero(
            (duration <= 0) | (duration < minimum_fraction * old_duration)
        )
        if not len(invalid):
            return result, sorted(rolled)
        changed = False
        for segment in invalid:
            for index in [segment - 1, segment]:
                if 0 <= index < len(old) and result[index] != old[index]:
                    result[index] = old[index]
                    rolled.add(int(index))
                    changed = True
        if not changed:
            raise ValueError("Duration guard could not restore a valid parent span")
    raise ValueError("Duration rollback failed to converge")

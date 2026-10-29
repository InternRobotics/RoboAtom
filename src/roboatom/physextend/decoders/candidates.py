"""Per-segment sensor evidence comparing a decoded candidate with its reference."""

from __future__ import annotations

from typing import Any

import numpy as np

from roboatom.physextend.features.extraction import feature_paths
from roboatom.physextend.decoders.seeded_decoder import raw_sensor_cache


def resample(values: np.ndarray, length: int) -> np.ndarray:
    """Resample numeric telemetry to the declared annotation grid."""
    if len(values) == length:
        return values
    return np.stack(
        [
            np.interp(
                np.linspace(0, 1, length),
                np.linspace(0, 1, len(values)),
                values[:, index],
            )
            for index in range(values.shape[1])
        ],
        axis=1,
    )


def segment_evidence(
    row: dict[str, Any], seed: dict[str, Any], prediction: dict[str, Any]
) -> dict[str, Any]:
    """Measure target/seed segment similarity using numeric sensor fields only."""
    target_paths, seed_paths = feature_paths(row["target"]), feature_paths(seed)
    target_raw = raw_sensor_cache(str(target_paths[1]), str(target_paths[2]))
    seed_raw = raw_sensor_cache(str(seed_paths[1]), str(seed_paths[2]))
    columns = sorted(
        k
        for k in set(target_raw) & set(seed_raw)
        if target_raw[k].shape[1] == seed_raw[k].shape[1]
        and "action" not in k.lower()
        and "command" not in k.lower()
    )
    if not columns:
        columns = sorted(
            k
            for k in set(target_raw) & set(seed_raw)
            if target_raw[k].shape[1] == seed_raw[k].shape[1]
        )
    if not columns:
        raise ValueError("No compatible sensor columns for candidate scoring")
    source = np.concatenate(
        [resample(seed_raw[k], int(seed["length"])) for k in columns], axis=1
    ).astype(float)
    target = np.concatenate(
        [resample(target_raw[k], int(row["target"]["length"])) for k in columns], axis=1
    ).astype(float)
    source = np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    scale = np.maximum(np.std(source, axis=0), 0.05)
    source = source / scale
    target = target / scale
    items = []
    for pred, label in zip(prediction["segments"], seed["segments"], strict=True):
        left, right = int(pred["start_frame"]), int(pred["end_frame"])
        query = target[left:right]
        support = source[int(label["start"]) : int(label["end"])]
        grid = np.linspace(0, 1, 12)
        source_shape = np.stack(
            [
                np.interp(grid, np.linspace(0, 1, len(support)), support[:, j])
                for j in range(support.shape[1])
            ],
            axis=1,
        )
        target_shape = np.stack(
            [
                np.interp(grid, np.linspace(0, 1, len(query)), query[:, j])
                for j in range(query.shape[1])
            ],
            axis=1,
        )
        phase_distance = float(
            np.mean(np.minimum((source_shape - target_shape) ** 2, 25.0))
        )
        centered_distance = float(
            np.mean(
                np.minimum(
                    (
                        (source_shape - source_shape[0])
                        - (target_shape - target_shape[0])
                    )
                    ** 2,
                    25.0,
                )
            )
        )
        mean_distance = float(
            np.mean(
                np.minimum(
                    (np.mean(support, axis=0) - np.mean(query, axis=0)) ** 2, 25.0
                )
            )
        )
        source_motion = np.mean(np.abs(np.diff(source_shape, axis=0)), axis=0)
        target_motion = np.mean(np.abs(np.diff(target_shape, axis=0)), axis=0)
        motion_distance = float(
            np.mean(np.minimum((source_motion - target_motion) ** 2, 25.0))
        )
        source_duration = (label["end"] - label["start"]) / seed["length"]
        duration = (right - left) / row["target"]["length"]
        items.append(
            {
                "phase": phase_distance,
                "centered": centered_distance,
                "state_mean": mean_distance,
                "motion": motion_distance,
                "duration_log": abs(
                    float(np.log(max(duration, 1e-9) / max(source_duration, 1e-9)))
                ),
                "target_ratio": duration,
                "seed_ratio": source_duration,
                "skill": label["skill"],
            }
        )
    return {"sensor_columns": columns, "segment_evidence": items}

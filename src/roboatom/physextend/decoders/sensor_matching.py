"""Sensor-feature Gaussian models and dynamic-programming boundary search."""

from __future__ import annotations

from collections import Counter
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq


EpisodeId = int | str
logger = logging.getLogger(__name__)


@dataclass
class ManualLabel:
    episode_index: EpisodeId
    path: Path | str
    length: int
    tasks: list[str]
    action_config: list[dict[str, Any]]


@dataclass
class EpisodeFeatures:
    episode_index: EpisodeId
    length: int
    sample_frames: np.ndarray
    state_features: np.ndarray
    visual_features: np.ndarray
    combined_motion_full: np.ndarray
    state_motion_full: np.ndarray
    visual_motion_full: np.ndarray


@dataclass
class FittedModel:
    template_key: str
    source_episode_indices: list[EpisodeId]
    num_subtasks: int
    tasks_fallback: list[str]
    action_texts: list[str]
    skills: list[str]
    start_ratio_median: np.ndarray
    boundary_ratio_median: np.ndarray
    segment_ratio_median: np.ndarray
    segment_ratio_min: np.ndarray
    state_mean: np.ndarray
    state_var: np.ndarray
    state_global_mean: np.ndarray
    state_global_std: np.ndarray
    visual_mean: np.ndarray
    visual_var: np.ndarray
    visual_global_mean: np.ndarray
    visual_global_std: np.ndarray


def read_column_matrix(table: pq.Table, column_name: str) -> np.ndarray:
    values = table[column_name].to_pylist()
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def temporal_context(base: np.ndarray, offsets: Sequence[int]) -> np.ndarray:
    t_len = base.shape[0]
    pieces = []
    base_indices = np.arange(t_len)
    for offset in offsets:
        idx = np.clip(base_indices + int(offset), 0, t_len - 1)
        pieces.append(base[idx])
    return np.concatenate(pieces, axis=1).astype(np.float32)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if window <= 1 or values.size == 0:
        return values
    window = int(window)
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def robust_01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    q05, q95 = np.percentile(values, [5, 95])
    scale = max(float(q95 - q05), 1e-6)
    return np.clip((values - q05) / scale, 0.0, 1.0).astype(np.float32)


def normalize_train_test(train_parts: list[np.ndarray], test: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    train_all = np.concatenate(train_parts, axis=0)
    mean = train_all.mean(axis=0)
    std = train_all.std(axis=0)
    std = np.maximum(std, 1e-4)
    return [(x - mean) / std for x in train_parts], (test - mean) / std, mean, std


def fit_modality_gaussian(parts_by_subtask: list[list[np.ndarray]], dim: int) -> tuple[np.ndarray, np.ndarray]:
    means = []
    vars_ = []
    for subtask_index, chunks in enumerate(parts_by_subtask):
        if not chunks:
            means.append(np.zeros((dim,), dtype=np.float32))
            vars_.append(np.ones((dim,), dtype=np.float32))
            continue
        cat = np.concatenate(chunks, axis=0).astype(np.float32)
        means.append(cat.mean(axis=0))
        vars_.append(np.maximum(cat.var(axis=0), 1e-4))
    return np.stack(means, axis=0).astype(np.float32), np.stack(vars_, axis=0).astype(np.float32)


def fit_model(
    manual_labels: list[ManualLabel],
    feature_by_episode: dict[EpisodeId, EpisodeFeatures],
) -> FittedModel:
    num_subtasks = len(manual_labels[0].action_config)
    if any(len(label.action_config) != num_subtasks for label in manual_labels):
        raise ValueError("Manual labels have different numbers of subtasks")

    state_train = [feature_by_episode[label.episode_index].state_features for label in manual_labels]
    visual_train = [feature_by_episode[label.episode_index].visual_features for label in manual_labels]
    state_norm_parts, _, state_global_mean, state_global_std = normalize_train_test(state_train, state_train[0])
    visual_norm_parts, _, visual_global_mean, visual_global_std = normalize_train_test(visual_train, visual_train[0])

    state_by_subtask: list[list[np.ndarray]] = [[] for _ in range(num_subtasks)]
    visual_by_subtask: list[list[np.ndarray]] = [[] for _ in range(num_subtasks)]
    start_ratios: list[list[float]] = [[] for _ in range(num_subtasks)]
    segment_ratios: list[list[float]] = [[] for _ in range(num_subtasks)]

    for label_index, label in enumerate(manual_labels):
        feat = feature_by_episode[label.episode_index]
        sample_frames = feat.sample_frames
        for subtask_index, segment in enumerate(label.action_config):
            start = int(segment["start_frame"])
            end = int(segment["end_frame"])
            mask = (sample_frames >= start) & (sample_frames < end)
            if int(mask.sum()) < 2:
                nearest = np.argmin(np.abs(sample_frames - (start + end) // 2))
                mask[nearest] = True
            state_by_subtask[subtask_index].append(state_norm_parts[label_index][mask])
            visual_by_subtask[subtask_index].append(visual_norm_parts[label_index][mask])
            start_ratios[subtask_index].append(start / float(label.length))
            segment_ratios[subtask_index].append((end - start) / float(label.length))

    state_mean, state_var = fit_modality_gaussian(state_by_subtask, dim=state_norm_parts[0].shape[1])
    visual_mean, visual_var = fit_modality_gaussian(visual_by_subtask, dim=visual_norm_parts[0].shape[1])

    def most_common_nonempty(values: Sequence[str]) -> str:
        cleaned = [str(value).strip() for value in values if str(value).strip()]
        return Counter(cleaned).most_common(1)[0][0] if cleaned else ""

    tasks_fallback = next((label.tasks for label in manual_labels if label.tasks), [])
    action_texts = [
        most_common_nonempty(
            str(label.action_config[index].get("action_text", "")) for label in manual_labels
        )
        for index in range(num_subtasks)
    ]
    skills = [
        most_common_nonempty(
            str(label.action_config[index].get("skill", "")) for label in manual_labels
        )
        for index in range(num_subtasks)
    ]
    start_ratio_median = np.asarray([np.median(values) for values in start_ratios], dtype=np.float32)
    segment_ratio_median = np.asarray([np.median(values) for values in segment_ratios], dtype=np.float32)
    segment_ratio_min = np.asarray([np.min(values) for values in segment_ratios], dtype=np.float32)
    boundaries = []
    for subtask_index in range(1, num_subtasks):
        starts = [label.action_config[subtask_index]["start_frame"] / float(label.length) for label in manual_labels]
        boundaries.append(float(np.median(starts)))

    return FittedModel(
        template_key=f"{num_subtasks}-subtasks",
        source_episode_indices=[label.episode_index for label in manual_labels],
        num_subtasks=num_subtasks,
        tasks_fallback=tasks_fallback,
        action_texts=action_texts,
        skills=skills,
        start_ratio_median=start_ratio_median,
        boundary_ratio_median=np.asarray(boundaries, dtype=np.float32),
        segment_ratio_median=segment_ratio_median,
        segment_ratio_min=segment_ratio_min,
        state_mean=state_mean,
        state_var=state_var,
        state_global_mean=state_global_mean.astype(np.float32),
        state_global_std=state_global_std.astype(np.float32),
        visual_mean=visual_mean,
        visual_var=visual_var,
        visual_global_mean=visual_global_mean.astype(np.float32),
        visual_global_std=visual_global_std.astype(np.float32),
    )


def gaussian_cost(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - mean[None, :, :]
    cost = 0.5 * ((diff * diff) / var[None, :, :] + np.log(var[None, :, :]))
    return cost.mean(axis=2).astype(np.float32)


def normalize_cost(cost: np.ndarray) -> np.ndarray:
    centered = cost - cost.min(axis=1, keepdims=True)
    scale = np.percentile(centered, 90, axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-6)
    return (centered / scale).astype(np.float32)


def episode_costs(features: EpisodeFeatures, model: FittedModel, state_weight: float, visual_weight: float) -> np.ndarray:
    state = (features.state_features - model.state_global_mean) / model.state_global_std
    visual = (features.visual_features - model.visual_global_mean) / model.visual_global_std
    state_cost = normalize_cost(gaussian_cost(state, model.state_mean, model.state_var))
    visual_cost = normalize_cost(gaussian_cost(visual, model.visual_mean, model.visual_var))
    return (float(state_weight) * state_cost + float(visual_weight) * visual_cost).astype(np.float32)


def local_minima_indices(values: np.ndarray, lo: int, hi: int) -> list[int]:
    lo = max(1, lo)
    hi = min(len(values) - 2, hi)
    if lo > hi:
        return []
    segment = values[lo : hi + 1]
    indices: list[int] = []
    for offset in range(1, len(segment) - 1):
        if segment[offset] <= segment[offset - 1] and segment[offset] <= segment[offset + 1]:
            indices.append(lo + offset)
    return indices


def candidate_boundaries(
    features: EpisodeFeatures,
    model: FittedModel,
    candidate_keep: int,
    boundary_window_ratio: float,
) -> list[list[int]]:
    candidates_all: list[list[int]] = []
    motion = features.combined_motion_full
    length = features.length
    sample_frames = features.sample_frames
    for boundary_index, ratio in enumerate(model.boundary_ratio_median):
        expected = int(round(float(ratio) * length))
        left_seg = float(model.segment_ratio_median[boundary_index]) * length
        right_seg = float(model.segment_ratio_median[boundary_index + 1]) * length
        window = int(max(65, min(460, boundary_window_ratio * max(left_seg, right_seg))))
        lo = max(1, expected - window)
        hi = min(length - 1, expected + window)
        in_window = sample_frames[(sample_frames >= lo) & (sample_frames <= hi)]
        if in_window.size == 0:
            in_window = np.asarray([expected], dtype=np.int32)
        minima = local_minima_indices(motion, lo=lo, hi=hi)
        pooled = set(int(x) for x in in_window.tolist())
        pooled.add(expected)
        pooled.update(int(x) for x in minima)
        ranked = sorted(
            pooled,
            key=lambda frame: (
                float(motion[min(max(frame, 0), len(motion) - 1)]),
                abs(frame - expected),
            ),
        )
        keep = sorted(ranked[:candidate_keep])
        if expected not in keep:
            keep.append(expected)
        candidates_all.append(sorted(set(max(1, min(length - 1, int(x))) for x in keep)))
    return candidates_all


def segment_sample_slice(sample_frames: np.ndarray, start_frame: int, end_frame: int) -> tuple[int, int]:
    a = int(np.searchsorted(sample_frames, start_frame, side="left"))
    b = int(np.searchsorted(sample_frames, end_frame, side="left"))
    if b <= a:
        nearest = int(np.searchsorted(sample_frames, (start_frame + end_frame) // 2, side="left"))
        nearest = min(max(nearest, 0), len(sample_frames) - 1)
        return nearest, nearest + 1
    return a, b


def solve_boundaries(
    features: EpisodeFeatures,
    cost: np.ndarray,
    model: FittedModel,
    duration_weight: float,
    boundary_motion_weight: float,
    min_segment_frames: int,
    candidate_keep: int,
    boundary_window_ratio: float,
    refine_window: int,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[list[int], list[float], list[int]]:
    if diagnostics is not None:
        diagnostics["used_fallback"] = False
    if model.num_subtasks == 1:
        return [], [], []

    expected = [int(round(float(ratio) * features.length)) for ratio in model.boundary_ratio_median]
    candidates = candidate_boundaries(features, model, candidate_keep, boundary_window_ratio)
    num_boundaries = len(candidates)
    num_subtasks = model.num_subtasks
    length = features.length
    sample_frames = features.sample_frames
    prefix = np.vstack([np.zeros((1, num_subtasks), dtype=np.float64), np.cumsum(cost.astype(np.float64), axis=0)])
    expected_lengths = np.maximum(model.segment_ratio_median * length, float(min_segment_frames))
    min_lengths = np.maximum(model.segment_ratio_min * length * 0.45, float(min_segment_frames))

    def interval_cost(subtask_index: int, start: int, end: int) -> float:
        if end <= start:
            return math.inf
        a, b = segment_sample_slice(sample_frames, start, end)
        count = max(1, b - a)
        emission = float((prefix[b, subtask_index] - prefix[a, subtask_index]) / count)
        duration = end - start
        expected = float(expected_lengths[subtask_index])
        duration_cost = ((duration - expected) / max(expected, 1.0)) ** 2
        return emission + float(duration_weight) * duration_cost

    dp: list[np.ndarray] = []
    back: list[np.ndarray] = []

    first = np.full((len(candidates[0]),), np.inf, dtype=np.float64)
    for i, boundary in enumerate(candidates[0]):
        if boundary < min_lengths[0]:
            continue
        motion_cost = float(features.combined_motion_full[boundary]) * float(boundary_motion_weight)
        first[i] = interval_cost(0, 0, boundary) + motion_cost
    dp.append(first)
    back.append(np.full((len(candidates[0]),), -1, dtype=np.int32))

    for boundary_index in range(1, num_boundaries):
        current = np.full((len(candidates[boundary_index]),), np.inf, dtype=np.float64)
        current_back = np.full((len(candidates[boundary_index]),), -1, dtype=np.int32)
        for cur_i, cur_boundary in enumerate(candidates[boundary_index]):
            best_cost = math.inf
            best_prev = -1
            for prev_i, prev_boundary in enumerate(candidates[boundary_index - 1]):
                if cur_boundary - prev_boundary < min_lengths[boundary_index]:
                    continue
                value = (
                    float(dp[boundary_index - 1][prev_i])
                    + interval_cost(boundary_index, prev_boundary, cur_boundary)
                    + float(features.combined_motion_full[cur_boundary]) * float(boundary_motion_weight)
                )
                if value < best_cost:
                    best_cost = value
                    best_prev = prev_i
            current[cur_i] = best_cost
            current_back[cur_i] = best_prev
        dp.append(current)
        back.append(current_back)

    best_final = math.inf
    best_idx = -1
    last_boundary_list = candidates[-1]
    for prev_i, prev_boundary in enumerate(last_boundary_list):
        if length - prev_boundary < min_lengths[-1]:
            continue
        value = float(dp[-1][prev_i]) + interval_cost(num_subtasks - 1, prev_boundary, length)
        if value < best_final:
            best_final = value
            best_idx = prev_i
    if best_idx < 0 or not math.isfinite(best_final):
        if diagnostics is not None:
            diagnostics["used_fallback"] = True
        boundaries = fallback_boundaries_from_model(features.length, model)
        confidences = boundary_confidences(features, boundaries, model)
        return boundaries, confidences, expected

    boundary_choices = [0] * num_boundaries
    boundary_choices[-1] = best_idx
    for boundary_index in range(num_boundaries - 1, 0, -1):
        boundary_choices[boundary_index - 1] = int(back[boundary_index][boundary_choices[boundary_index]])
    boundaries = [int(candidates[i][boundary_choices[i]]) for i in range(num_boundaries)]
    boundaries = refine_boundaries(
        features=features,
        boundaries=boundaries,
        model=model,
        min_segment_frames=min_segment_frames,
        refine_window=refine_window,
    )
    confidences = boundary_confidences(features, boundaries, model)
    return boundaries, confidences, expected


def fallback_boundaries_from_model(length: int, model: FittedModel) -> list[int]:
    """Build valid monotonic boundaries from a model's median boundary ratios.

    Args:
        length: Episode length in frames.
        model: Fitted subtask boundary model.

    Returns:
        Boundary frame indices with at least one frame per segment.
    """
    if model.num_subtasks <= 1:
        return []
    boundaries: list[int] = []
    num_boundaries = model.num_subtasks - 1
    previous = 0
    for boundary_index, ratio in enumerate(model.boundary_ratio_median):
        remaining_boundaries = num_boundaries - boundary_index - 1
        lower = previous + 1
        upper = max(lower, int(length) - remaining_boundaries - 1)
        boundary = int(round(float(ratio) * int(length)))
        boundary = min(max(boundary, lower), upper)
        boundaries.append(boundary)
        previous = boundary
    return boundaries


def refine_boundaries(
    features: EpisodeFeatures,
    boundaries: Sequence[int],
    model: FittedModel,
    min_segment_frames: int,
    refine_window: int,
) -> list[int]:
    motion = features.combined_motion_full
    length = features.length
    refined: list[int] = []
    original = list(int(x) for x in boundaries)
    for i, boundary in enumerate(original):
        left_limit = min_segment_frames if i == 0 else refined[-1] + min_segment_frames
        right_limit = length - min_segment_frames if i == len(original) - 1 else original[i + 1] - min_segment_frames
        lo = max(left_limit, boundary - refine_window)
        hi = min(right_limit, boundary + refine_window)
        if lo > hi:
            refined.append(boundary)
            continue
        expected = int(round(float(model.boundary_ratio_median[i]) * length))
        best = min(
            range(lo, hi + 1),
            key=lambda frame: float(motion[frame]) + 0.08 * abs(frame - expected) / max(1, hi - lo),
        )
        refined.append(int(best))
    return refined


def boundary_confidences(features: EpisodeFeatures, boundaries: Sequence[int], model: FittedModel) -> list[float]:
    motion = features.combined_motion_full
    length = features.length
    confidences = []
    for i, boundary in enumerate(boundaries):
        lo = max(0, boundary - 90)
        hi = min(len(motion), boundary + 91)
        local = motion[lo:hi]
        if local.size == 0:
            pause_conf = 0.0
        else:
            local_min = float(local.min())
            local_p80 = float(np.percentile(local, 80))
            pause_conf = (local_p80 - float(motion[boundary])) / max(local_p80 - local_min, 1e-6)
        expected = int(round(float(model.boundary_ratio_median[i]) * length))
        left_seg = float(model.segment_ratio_median[i]) * length
        right_seg = float(model.segment_ratio_median[i + 1]) * length
        window = max(80.0, 0.55 * max(left_seg, right_seg))
        timing_conf = math.exp(-abs(boundary - expected) / window)
        confidences.append(round(float(np.clip(0.55 * pause_conf + 0.45 * timing_conf, 0.0, 1.0)), 4))
    return confidences

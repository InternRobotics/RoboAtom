"""Evaluate saved action-segmentation predictions after inference."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import itertools
from . import segments as base_evaluator

DEFAULT_F1_OVERLAPS = (0.3, 0.5, 0.7)
DEFAULT_SKILL_MATCH_TIOU = 0.50
DEFAULT_SKILL_F1_OVERLAP = 0.70
MISSING_SKILL_LABEL = "<missing-skill>"


@dataclass(frozen=True)
class PreparedEpisode:
    """Parsed inputs and window audit for one evaluation episode."""

    episode_id: str
    gt_row: Mapping[str, Any]
    raw_gold: tuple[Mapping[str, Any], ...]
    raw_predicted: tuple[Mapping[str, Any], ...]
    gold: tuple[Mapping[str, Any], ...]
    predicted: tuple[Mapping[str, Any], ...]
    prediction_present: bool
    evaluation_window: Mapping[str, Any]
    raw_prediction_segment_count: int
    missing_subtask_segment_count: int
    missing_skill_segment_count: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional command-line arguments. Uses ``sys.argv`` when omitted.

    Returns:
        Parsed and validated arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate completed action-segmentation inference with temporal "
            "segmentation and skill metrics."
        )
    )
    parser.add_argument(
        "--gt",
        required=True,
        type=Path,
        help="Ground-truth Bench JSON/JSONL or LlamaFactory dataset file.",
    )
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help=(
            "Saved inference run name and prediction path. Repeat to compare "
            "runs. A directory resolves to episodes.jsonl."
        ),
    )
    parser.add_argument(
        "--f1-overlaps",
        default=",".join(str(value) for value in DEFAULT_F1_OVERLAPS),
        help="Comma-separated segment IoU thresholds (default: 0.3,0.5,0.7).",
    )
    parser.add_argument(
        "--skill-match-tiou",
        type=float,
        default=DEFAULT_SKILL_MATCH_TIOU,
        help="Temporal-only matching threshold for Skill Acc (default: 0.50).",
    )
    parser.add_argument(
        "--skill-f1-overlap",
        type=float,
        default=DEFAULT_SKILL_F1_OVERLAP,
        help="Same-skill segment IoU threshold for Skill F1 (default: 0.70).",
    )
    parser.add_argument(
        "--background-label",
        action="append",
        default=[],
        help=(
            "Skill label excluded from Edit/F1/Skill Acc. Repeat for multiple "
            "labels. No labels are excluded by default."
        ),
    )
    parser.add_argument(
        "--skip-skill-metrics",
        action="store_true",
        help=(
            "Report Edit, Skill F1, and Skill Acc as N/A when GT has no skill "
            "labels; temporal F1 remains enabled."
        ),
    )
    parser.add_argument(
        "--evaluation-window",
        choices=base_evaluator.EVALUATION_WINDOWS,
        default="full",
        help="Evaluation window policy (default: full).",
    )
    parser.add_argument(
        "--allow-extra-episodes",
        action="store_true",
        help="Ignore prediction episode IDs absent from GT instead of failing.",
    )
    parser.add_argument("--output-json", type=Path, help="Aggregate JSON report path.")
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        help="Per-run, per-episode JSONL report path.",
    )
    parser.add_argument("--output-tsv", type=Path, help="Run comparison TSV path.")
    args = parser.parse_args(argv)

    try:
        args.f1_overlaps = base_evaluator.parse_thresholds(args.f1_overlaps)
    except ValueError as exc:
        parser.error(str(exc))
    if not 0.0 <= args.skill_match_tiou <= 1.0:
        parser.error("--skill-match-tiou must be between 0 and 1")
    if not 0.0 <= args.skill_f1_overlap <= 1.0:
        parser.error("--skill-f1-overlap must be between 0 and 1")
    return args


def normalized_background_labels(labels: Iterable[str]) -> frozenset[str]:
    """Return normalized configured background skill labels."""
    return frozenset(
        base_evaluator.normalize_label(label) for label in labels if str(label).strip()
    )


def is_background_segment(
    segment: Mapping[str, Any],
    background_labels: frozenset[str],
) -> bool:
    """Return whether a segment carries a configured background skill."""
    label = segment.get("skill_label")
    return (
        label is not None
        and base_evaluator.normalize_label(str(label)) in background_labels
    )


def validate_non_overlapping(
    segments: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> None:
    """Reject positive-duration overlaps in one ordered segment sequence.

    Args:
        segments: Time-sorted parsed segments.
        context: Human-readable role and episode description.

    Raises:
        ValueError: If two consecutive segments overlap.
    """
    for previous, current in itertools.pairwise(segments):
        if float(current["start"]) < float(previous["end"]) - base_evaluator.EPSILON:
            raise ValueError(
                f"{context} has overlapping segments "
                f"{previous['index']} and {current['index']}"
            )


def prediction_error_exists(
    parse_errors: Sequence[Mapping[str, Any]],
    episode_id: str,
) -> bool:
    """Return whether an error is already recorded for an episode."""
    return any(str(error.get("episode_id")) == episode_id for error in parse_errors)


def raw_prediction_count(
    row: Mapping[str, Any] | None,
    episode_id: str,
) -> int:
    """Best-effort raw prediction segment count for diagnostics."""
    if row is None:
        return 0
    try:
        return len(base_evaluator.raw_segment_values(row, episode_id, "Prediction"))
    except ValueError:
        return 0


def prepare_run(
    gt_index: Mapping[str, dict[str, Any]],
    prediction_index: Mapping[str, dict[str, Any]],
    *,
    run_name: str,
    initial_parse_errors: Sequence[Mapping[str, Any]],
    allow_extra_episodes: bool,
    evaluation_window: str,
) -> tuple[list[PreparedEpisode], list[dict[str, Any]], list[str]]:
    """Parse and validate all episodes for one saved inference run.

    Prediction parsing failures are recorded and scored as empty predictions;
    invalid ground truth fails the evaluation.

    Args:
        gt_index: Ground-truth records indexed by episode ID.
        prediction_index: Prediction records indexed by episode ID.
        run_name: Human-readable run name.
        initial_parse_errors: Errors emitted while reading inference output.
        allow_extra_episodes: Whether prediction-only IDs may be ignored.
        evaluation_window: Requested window policy.

    Returns:
        Prepared episodes, combined parse errors, and ignored extra IDs.

    Raises:
        ValueError: If GT is invalid or disallowed extra episodes exist.
    """
    extra_ids = sorted(set(prediction_index) - set(gt_index))
    if extra_ids and not allow_extra_episodes:
        preview = ", ".join(extra_ids[:10])
        raise ValueError(
            f"Prediction {run_name!r} contains {len(extra_ids)} episode(s) "
            f"absent from GT: {preview}. Pass --allow-extra-episodes to ignore them."
        )

    parse_errors = [dict(error) for error in initial_parse_errors]
    prepared: list[PreparedEpisode] = []
    for episode_id, gt_row in gt_index.items():
        raw_gold = base_evaluator.parse_segments(gt_row, episode_id, "GT")
        validate_non_overlapping(raw_gold, context=f"GT {episode_id}")

        prediction_row = prediction_index.get(episode_id)
        prediction_present = prediction_row is not None
        raw_count = raw_prediction_count(prediction_row, episode_id)
        raw_predicted: list[dict[str, Any]] = []
        if prediction_row is not None:
            try:
                raw_predicted = base_evaluator.parse_segments(
                    prediction_row,
                    episode_id,
                    "Prediction",
                )
                validate_non_overlapping(
                    raw_predicted,
                    context=f"Prediction {episode_id}",
                )
            except ValueError as exc:
                if not prediction_error_exists(parse_errors, episode_id):
                    parse_errors.append(
                        {
                            "episode_id": episode_id,
                            "error": str(exc),
                            "source": "evaluation_validation",
                        }
                    )
                raw_predicted = []

        missing_subtask_count = sum(
            segment.get("subtask_label") is None for segment in raw_predicted
        )
        missing_skill_count = sum(
            segment.get("skill_label") is None for segment in raw_predicted
        )
        predicted, gold, window_audit = base_evaluator.apply_evaluation_window(
            raw_predicted,
            raw_gold,
            evaluation_window=evaluation_window,
            episode=episode_id,
        )
        prepared.append(
            PreparedEpisode(
                episode_id=episode_id,
                gt_row=gt_row,
                raw_gold=tuple(raw_gold),
                raw_predicted=tuple(raw_predicted),
                gold=tuple(gold),
                predicted=tuple(predicted),
                prediction_present=prediction_present,
                evaluation_window=window_audit,
                raw_prediction_segment_count=raw_count,
                missing_subtask_segment_count=missing_subtask_count,
                missing_skill_segment_count=missing_skill_count,
            )
        )
    return prepared, parse_errors, extra_ids


def skill_sequence(
    segments: Sequence[Mapping[str, Any]],
    *,
    background_labels: frozenset[str],
    is_ground_truth: bool,
) -> list[str]:
    """Build a compressed, normalized segment-level skill sequence."""
    sequence: list[str] = []
    for segment in segments:
        if is_background_segment(segment, background_labels):
            continue
        value = segment.get("skill_label")
        if value is None:
            if is_ground_truth:
                continue
            label = MISSING_SKILL_LABEL
        else:
            label = base_evaluator.normalize_label(str(value))
        if not sequence or sequence[-1] != label:
            sequence.append(label)
    return sequence


def levenshtein_distance(left: Sequence[str], right: Sequence[str]) -> int:
    """Return unit-cost Levenshtein distance between two label sequences."""
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            substitution_cost = 0 if left_value == right_value else 1
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def edit_score(
    predicted: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
    *,
    background_labels: frozenset[str],
) -> dict[str, Any]:
    """Compute normalized segment-level skill Edit Score."""
    predicted_sequence = skill_sequence(
        predicted,
        background_labels=background_labels,
        is_ground_truth=False,
    )
    gt_sequence = skill_sequence(
        gold,
        background_labels=background_labels,
        is_ground_truth=True,
    )
    if not gt_sequence:
        return {
            "score": None,
            "distance": None,
            "prediction": predicted_sequence,
            "gt": gt_sequence,
        }
    distance = levenshtein_distance(predicted_sequence, gt_sequence)
    normalizer = max(len(predicted_sequence), len(gt_sequence))
    score = max(0.0, (1.0 - distance / normalizer) * 100.0)
    return {
        "score": score,
        "distance": distance,
        "prediction": predicted_sequence,
        "gt": gt_sequence,
    }


def metric_values(tp: int, fp: int, fn: int) -> dict[str, Any]:
    """Return precision, recall, and F1 with their raw counts."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def temporal_segment_f1(
    predicted: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
    *,
    overlap_threshold: float,
) -> dict[str, Any]:
    """Compute label-free temporal segment F1 using the original matcher.

    The ordered one-to-one alignment maximizes the number of pairs meeting the
    temporal-IoU threshold first and their total IoU second. Neither subtask
    nor skill labels participate in matching.

    Args:
        predicted: Predicted segments sorted by time.
        gold: Ground-truth segments sorted by time.
        overlap_threshold: Inclusive temporal-IoU threshold.

    Returns:
        Micro-count inputs and pair-level temporal matching diagnostics.
    """
    pairs = base_evaluator.match_temporal_segments(
        predicted,
        gold,
        tiou_threshold=overlap_threshold,
    )
    return base_evaluator.temporal_threshold_report(predicted, gold, pairs)


def skill_segment_f1(
    predicted: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
    *,
    overlap_threshold: float,
    background_labels: frozenset[str],
) -> dict[str, Any]:
    """Compute classic same-skill one-to-one segmental F1 counts.

    Predictions are processed in temporal order and matched to the unmatched GT
    segment with the greatest IoU, as in the MS-TCN evaluation protocol.
    """
    predicted_segments = [
        segment
        for segment in predicted
        if not is_background_segment(segment, background_labels)
    ]
    gt_segments = [
        segment
        for segment in gold
        if segment.get("skill_label") is not None
        and not is_background_segment(segment, background_labels)
    ]
    hit_gt_indexes: set[int] = set()
    matches: list[dict[str, Any]] = []
    tp = 0
    fp = 0
    for predicted_segment in predicted_segments:
        predicted_label = predicted_segment.get("skill_label")
        best_gt_index: int | None = None
        best_iou = 0.0
        if predicted_label is not None:
            normalized_prediction = base_evaluator.normalize_label(str(predicted_label))
            for gt_index, gt_segment in enumerate(gt_segments):
                gt_label = base_evaluator.normalize_label(
                    str(gt_segment["skill_label"])
                )
                if normalized_prediction != gt_label:
                    continue
                tiou = base_evaluator.temporal_iou(predicted_segment, gt_segment)
                if tiou > best_iou + base_evaluator.EPSILON:
                    best_iou = tiou
                    best_gt_index = gt_index
        is_match = (
            best_gt_index is not None
            and best_gt_index not in hit_gt_indexes
            and best_iou + base_evaluator.EPSILON >= overlap_threshold
        )
        if is_match:
            tp += 1
            hit_gt_indexes.add(best_gt_index)
        else:
            fp += 1
        matches.append(
            {
                "prediction_index": int(predicted_segment["index"]),
                "gt_index": (
                    int(gt_segments[best_gt_index]["index"])
                    if is_match and best_gt_index is not None
                    else None
                ),
                "temporal_iou": best_iou,
                "is_match": is_match,
            }
        )
    fn = len(gt_segments) - len(hit_gt_indexes)
    return {
        "metrics": metric_values(tp, fp, fn),
        "matches": matches,
        "evaluable_gt_segment_count": len(gt_segments),
        "prediction_segment_count": len(predicted_segments),
    }


def filtered_skill_segments(
    segments: Sequence[Mapping[str, Any]],
    background_labels: frozenset[str],
) -> list[Mapping[str, Any]]:
    """Remove only explicitly configured background skill segments."""
    return [
        segment
        for segment in segments
        if not is_background_segment(segment, background_labels)
    ]


def skill_accuracy(
    predicted: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
    *,
    tiou_threshold: float,
    background_labels: frozenset[str],
) -> dict[str, Any]:
    """Compute original temporal-first, segment-level Skill Accuracy."""
    filtered_predicted = filtered_skill_segments(predicted, background_labels)
    filtered_gold = filtered_skill_segments(gold, background_labels)
    pairs = base_evaluator.match_temporal_segments(
        filtered_predicted,
        filtered_gold,
        tiou_threshold=tiou_threshold,
    )
    return base_evaluator.evaluate_skill_accuracy(
        filtered_predicted,
        filtered_gold,
        pairs,
    )


def temporal_miou(
    predicted: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute temporal mIoU over order-preserving aligned GT segments."""
    report = base_evaluator.timelens_temporal_metrics(
        predicted,
        gold,
        tiou_thresholds=(),
    )
    return {
        "score": report["miou"],
        "aligned_iou_sum": report["aligned_iou_sum"],
        "target_count": report["target_count"],
        "matches": report["matches"],
    }


def evaluate_episode(
    episode: PreparedEpisode,
    *,
    f1_overlaps: Sequence[float],
    skill_f1_overlap: float,
    skill_match_tiou: float,
    background_labels: frozenset[str],
    skip_skill_metrics: bool = False,
) -> dict[str, Any]:
    """Evaluate all requested metrics for one prepared episode."""
    f1_reports = {
        base_evaluator.threshold_key(overlap): temporal_segment_f1(
            episode.predicted,
            episode.gold,
            overlap_threshold=overlap,
        )
        for overlap in f1_overlaps
    }
    if skip_skill_metrics:
        edit_report: dict[str, Any] = {
            "status": "not_applicable",
            "score": None,
            "distance": None,
            "prediction": None,
            "gt": None,
        }
        skill_f1_report: dict[str, Any] = {
            "status": "not_applicable",
            "metrics": None,
            "matches": [],
            "temporal_iou_threshold": skill_f1_overlap,
        }
        skill_accuracy_report: dict[str, Any] = {
            "status": "not_applicable",
            "score": None,
            "correct": None,
            "evaluated_pairs": None,
            "missing_prediction": None,
            "unevaluable_gt": None,
            "temporal_iou_threshold": skill_match_tiou,
        }
    else:
        edit_report = edit_score(
            episode.predicted,
            episode.gold,
            background_labels=background_labels,
        )
        skill_f1_report = skill_segment_f1(
            episode.predicted,
            episode.gold,
            overlap_threshold=skill_f1_overlap,
            background_labels=background_labels,
        )
        skill_f1_report["temporal_iou_threshold"] = skill_f1_overlap
        skill_accuracy_report = skill_accuracy(
            episode.predicted,
            episode.gold,
            tiou_threshold=skill_match_tiou,
            background_labels=background_labels,
        )
    return {
        "episode_id": episode.episode_id,
        "prediction_present": episode.prediction_present,
        "raw_gt_segment_count": len(episode.raw_gold),
        "raw_prediction_segment_count": episode.raw_prediction_segment_count,
        "gt_segment_count": len(episode.gold),
        "prediction_segment_count": len(episode.predicted),
        "missing_subtask_segment_count": episode.missing_subtask_segment_count,
        "missing_skill_segment_count": episode.missing_skill_segment_count,
        "evaluation_window": dict(episode.evaluation_window),
        "edit": edit_report,
        "f1": f1_reports,
        "skill_f1": skill_f1_report,
        "skill_accuracy": skill_accuracy_report,
        "miou": temporal_miou(episode.predicted, episode.gold),
    }


def aggregate_skill_accuracy(
    episode_reports: Sequence[Mapping[str, Any]],
    *,
    tiou_threshold: float,
) -> dict[str, Any]:
    """Aggregate conditional Skill Accuracy by matched-pair counts."""
    correct = sum(
        int(episode["skill_accuracy"]["correct"]) for episode in episode_reports
    )
    evaluated_pairs = sum(
        int(episode["skill_accuracy"]["evaluated_pairs"]) for episode in episode_reports
    )
    return {
        "score": correct / evaluated_pairs if evaluated_pairs else None,
        "correct": correct,
        "evaluated_pairs": evaluated_pairs,
        "missing_prediction": sum(
            int(episode["skill_accuracy"]["missing_prediction"])
            for episode in episode_reports
        ),
        "unevaluable_gt": sum(
            int(episode["skill_accuracy"]["unevaluable_gt"])
            for episode in episode_reports
        ),
        "temporal_iou_threshold": tiou_threshold,
    }


def aggregate_temporal_miou(
    episode_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate temporal mIoU with unmatched GT segments contributing zero."""
    aligned_iou_sum = sum(
        float(episode["miou"]["aligned_iou_sum"]) for episode in episode_reports
    )
    target_count = sum(
        int(episode["miou"]["target_count"]) for episode in episode_reports
    )
    episode_scores = [
        float(episode["miou"]["score"])
        for episode in episode_reports
        if int(episode["miou"]["target_count"]) > 0
    ]
    return {
        "score": aligned_iou_sum / target_count if target_count else None,
        "aligned_iou_sum": aligned_iou_sum,
        "target_count": target_count,
        "macro_episode_miou": (
            sum(episode_scores) / len(episode_scores) if episode_scores else None
        ),
        "matching": "order_preserving_one_to_one_max_total_iou",
        "unmatched_gt_iou": 0.0,
    }


def evaluate_run(
    episodes: Sequence[PreparedEpisode],
    *,
    run_name: str,
    prediction_path: Path,
    prediction_format: str,
    parse_errors: Sequence[Mapping[str, Any]],
    ignored_extra_ids: Sequence[str],
    f1_overlaps: Sequence[float],
    skill_f1_overlap: float,
    skill_match_tiou: float,
    background_labels: frozenset[str],
    evaluation_window: str,
    skip_skill_metrics: bool = False,
) -> dict[str, Any]:
    """Evaluate and aggregate one completed inference run."""
    episode_reports = [
        evaluate_episode(
            episode,
            f1_overlaps=f1_overlaps,
            skill_f1_overlap=skill_f1_overlap,
            skill_match_tiou=skill_match_tiou,
            background_labels=background_labels,
            skip_skill_metrics=skip_skill_metrics,
        )
        for episode in episodes
    ]
    f1_report = {}
    for overlap in f1_overlaps:
        key = base_evaluator.threshold_key(overlap)
        metrics = metric_values(
            sum(
                int(episode["f1"][key]["metrics"]["tp"])
                for episode in episode_reports
            ),
            sum(
                int(episode["f1"][key]["metrics"]["fp"])
                for episode in episode_reports
            ),
            sum(
                int(episode["f1"][key]["metrics"]["fn"])
                for episode in episode_reports
            ),
        )
        f1_report[key] = {
            "micro": metrics,
            "macro_episode_f1": (
                sum(
                    float(episode["f1"][key]["metrics"]["f1"])
                    for episode in episode_reports
                )
                / len(episode_reports)
                if episode_reports
                else 0.0
            ),
        }
    if skip_skill_metrics:
        edit_report: dict[str, Any] = {
            "status": "not_applicable",
            "macro_episode_score": None,
            "evaluable_episode_count": 0,
        }
        skill_f1_report: dict[str, Any] = {
            "status": "not_applicable",
            "micro": None,
            "macro_episode_f1": None,
            "temporal_iou_threshold": skill_f1_overlap,
        }
        skill_accuracy_report: dict[str, Any] = {
            "status": "not_applicable",
            "score": None,
            "correct": None,
            "evaluated_pairs": None,
            "missing_prediction": None,
            "unevaluable_gt": None,
            "temporal_iou_threshold": skill_match_tiou,
        }
    else:
        skill_f1_metrics = metric_values(
            sum(
                int(episode["skill_f1"]["metrics"]["tp"])
                for episode in episode_reports
            ),
            sum(
                int(episode["skill_f1"]["metrics"]["fp"])
                for episode in episode_reports
            ),
            sum(
                int(episode["skill_f1"]["metrics"]["fn"])
                for episode in episode_reports
            ),
        )
        skill_f1_report = {
            "micro": skill_f1_metrics,
            "macro_episode_f1": (
                sum(
                    float(episode["skill_f1"]["metrics"]["f1"])
                    for episode in episode_reports
                )
                / len(episode_reports)
                if episode_reports
                else 0.0
            ),
            "temporal_iou_threshold": skill_f1_overlap,
        }
        edit_scores = [
            float(episode["edit"]["score"])
            for episode in episode_reports
            if episode["edit"]["score"] is not None
        ]
        edit_report = {
            "macro_episode_score": (
                sum(edit_scores) / len(edit_scores) if edit_scores else None
            ),
            "evaluable_episode_count": len(edit_scores),
        }
        skill_accuracy_report = aggregate_skill_accuracy(
            episode_reports,
            tiou_threshold=skill_match_tiou,
        )
    missing_ids = [
        episode.episode_id for episode in episodes if not episode.prediction_present
    ]
    available_count = len(episodes) - len(missing_ids)
    return {
        "name": run_name,
        "prediction_path": str(prediction_path),
        "prediction_format": prediction_format,
        "gt_episode_count": len(episodes),
        "available_prediction_episode_count": available_count,
        "coverage": available_count / len(episodes) if episodes else 0.0,
        "missing_prediction_episode_count": len(missing_ids),
        "missing_prediction_episode_ids": missing_ids,
        "ignored_extra_episode_count": len(ignored_extra_ids),
        "ignored_extra_episode_ids": list(ignored_extra_ids),
        "parse_error_count": len(parse_errors),
        "parse_errors": [
            dict(error)
            for error in parse_errors[: base_evaluator.MAX_PARSE_ERROR_EXAMPLES]
        ],
        "evaluation_window": base_evaluator.aggregate_evaluation_window(
            episode_reports,
            policy=evaluation_window,
        ),
        "format": base_evaluator.format_diagnostics(episode_reports, parse_errors),
        "edit": edit_report,
        "f1": f1_report,
        "skill_f1": skill_f1_report,
        "skill_accuracy": skill_accuracy_report,
        "miou": aggregate_temporal_miou(episode_reports),
        "episodes": episode_reports,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    """Build an evaluation report from saved inference artifacts."""
    gt_index, llamafactory_samples, gt_format = base_evaluator.read_gt_input(args.gt)
    prediction_specs = [
        base_evaluator.parse_prediction_spec(spec) for spec in args.prediction
    ]
    run_names = [name for name, _ in prediction_specs]
    if len(run_names) != len(set(run_names)):
        raise ValueError("--prediction run names must be unique")

    prepared_runs: list[tuple[Any, ...]] = []
    for run_name, prediction_path in prediction_specs:
        prediction_index, initial_errors, prediction_format = (
            base_evaluator.read_prediction_input(
                path=prediction_path,
                llamafactory_samples=llamafactory_samples,
            )
        )
        episodes, parse_errors, extra_ids = prepare_run(
            gt_index,
            prediction_index,
            run_name=run_name,
            initial_parse_errors=initial_errors,
            allow_extra_episodes=args.allow_extra_episodes,
            evaluation_window=args.evaluation_window,
        )
        prepared_runs.append(
            (
                run_name,
                prediction_path,
                prediction_format,
                episodes,
                parse_errors,
                extra_ids,
            )
        )

    background_labels = normalized_background_labels(args.background_label)
    runs = [
        evaluate_run(
            episodes,
            run_name=run_name,
            prediction_path=prediction_path,
            prediction_format=prediction_format,
            parse_errors=parse_errors,
            ignored_extra_ids=extra_ids,
            f1_overlaps=args.f1_overlaps,
            skill_f1_overlap=args.skill_f1_overlap,
            skill_match_tiou=args.skill_match_tiou,
            background_labels=background_labels,
            evaluation_window=args.evaluation_window,
            skip_skill_metrics=args.skip_skill_metrics,
        )
        for (
            run_name,
            prediction_path,
            prediction_format,
            episodes,
            parse_errors,
            extra_ids,
        ) in prepared_runs
    ]
    return {
        "protocol": {
            "task": "post_inference_action_segmentation",
            "inference_included": False,
            "skill_metrics_skipped": args.skip_skill_metrics,
            "edit": {
                "enabled": not args.skip_skill_metrics,
                "not_applicable_reason": "ground_truth_skill_labels_unavailable"
                if args.skip_skill_metrics
                else None,
                "evaluation_unit": "ordered_skill_segment_sequence",
                "adjacent_duplicate_skills": "collapsed",
                "normalization": "100*(1-distance/max_sequence_length)",
            },
            "f1": {
                "enabled": True,
                "not_applicable_reason": None,
                "evaluation_unit": "segment",
                "matching": "ordered_temporal_only_max_cardinality_then_total_iou",
                "temporal_iou_thresholds": list(args.f1_overlaps),
                "skill_matching": None,
                "aggregation": "micro_tp_fp_fn",
            },
            "skill_f1": {
                "enabled": not args.skip_skill_metrics,
                "not_applicable_reason": "ground_truth_skill_labels_unavailable"
                if args.skip_skill_metrics
                else None,
                "evaluation_unit": "skill_segment",
                "matching": "prediction_order_greedy_best_same_skill_unmatched_gt",
                "temporal_iou_threshold": args.skill_f1_overlap,
                "skill_matching": "normalized_exact",
                "aggregation": "micro_tp_fp_fn",
            },
            "skill_accuracy": {
                "enabled": not args.skip_skill_metrics,
                "not_applicable_reason": "ground_truth_skill_labels_unavailable"
                if args.skip_skill_metrics
                else None,
                "evaluation_unit": "temporally_matched_segment_pair",
                "matching": "ordered_temporal_only_max_cardinality_then_total_iou",
                "temporal_iou_threshold": args.skill_match_tiou,
                "skill_matching": "normalized_exact",
            },
            "miou": {
                "evaluation_unit": "ground_truth_segment",
                "matching": "order_preserving_one_to_one_max_total_iou",
                "unmatched_gt_iou": 0.0,
                "skill_matching": None,
            },
            "background_labels": sorted(background_labels),
            "evaluation_window": args.evaluation_window,
            "time_unit": "seconds",
        },
        "gt_path": str(args.gt.resolve()),
        "gt_format": gt_format,
        "runs": runs,
    }


def format_optional(value: Any, *, percentage: bool = False, missing: str = "-") -> str:
    """Format an optional report value for tabular output."""
    if value is None:
        return missing
    numeric = float(value)
    if percentage:
        numeric *= 100.0
    return f"{numeric:.3f}"


def summary_headers(report: Mapping[str, Any]) -> list[str]:
    """Return stable terminal and TSV summary headers."""
    overlaps = report["protocol"]["f1"]["temporal_iou_thresholds"]
    skill_f1_overlap = float(
        report["protocol"]["skill_f1"]["temporal_iou_threshold"]
    )
    return [
        "Run",
        "Coverage",
        "FormatValid",
        "Edit",
        "mIoU",
        *(f"F1@{float(overlap):.2f}" for overlap in overlaps),
        f"SkillF1@{skill_f1_overlap:.2f}",
        "SkillAcc",
    ]


def summary_rows(report: Mapping[str, Any]) -> list[list[str]]:
    """Return formatted summary rows for all inference runs."""
    overlaps = report["protocol"]["f1"]["temporal_iou_thresholds"]
    rows: list[list[str]] = []
    for run in report["runs"]:
        row = [
            str(run["name"]),
            format_optional(run["coverage"], percentage=True),
            format_optional(
                run["format"]["format_valid_episode_rate"], percentage=True
            ),
            format_optional(run["edit"]["macro_episode_score"], missing="N/A"),
            format_optional(run["miou"]["score"], percentage=True),
        ]
        for overlap in overlaps:
            f1_report = run["f1"][base_evaluator.threshold_key(float(overlap))]
            micro = f1_report["micro"]
            row.append(
                format_optional(
                    None if micro is None else micro["f1"],
                    percentage=True,
                    missing="N/A",
                )
            )
        skill_f1_micro = run["skill_f1"]["micro"]
        row.append(
            format_optional(
                None if skill_f1_micro is None else skill_f1_micro["f1"],
                percentage=True,
                missing="N/A",
            )
        )
        row.append(
            format_optional(
                run["skill_accuracy"]["score"],
                percentage=True,
                missing="N/A",
            )
        )
        rows.append(row)
    return rows


def print_summary(report: Mapping[str, Any]) -> None:
    """Write the compact comparison table to standard output."""
    lines = ["\t".join(summary_headers(report))]
    lines.extend("\t".join(row) for row in summary_rows(report))
    sys.stdout.write("\n".join(lines) + "\n")


def write_summary_tsv(path: Path, report: Mapping[str, Any]) -> None:
    """Atomically write the run comparison TSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(summary_headers(report))
            writer.writerows(summary_rows(report))
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Run the post-inference action-segmentation evaluator."""
    try:
        args = parse_args(argv)
        report = build_report(args)
        print_summary(report)
        if args.output_json:
            base_evaluator.atomic_write_json(args.output_json, report)
        if args.output_jsonl:
            base_evaluator.write_episode_jsonl(args.output_jsonl, report)
        if args.output_tsv:
            write_summary_tsv(args.output_tsv, report)
    except (OSError, RuntimeError, ValueError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

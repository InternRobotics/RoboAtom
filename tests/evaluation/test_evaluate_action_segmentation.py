"""Tests for the post-inference action-segmentation evaluator."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from roboatom.evaluation import action as evaluator


def parsed_segment(
    index: int,
    start: float,
    end: float,
    *,
    subtask: str | None = None,
    skill: str | None = None,
) -> dict[str, Any]:
    """Build one already-parsed evaluator segment."""
    return {
        "index": index,
        "start": float(start),
        "end": float(end),
        "label": subtask or skill or "",
        "subtask_label": subtask,
        "skill_label": skill,
    }


def bench_episode(
    episode_id: str,
    segments: list[dict[str, Any]],
    *,
    frames: int = 10,
    fps: float = 10.0,
    frame_interval: str = "half_open",
) -> dict[str, Any]:
    """Build one Bench-format episode."""
    return {
        "episode_index": episode_id,
        "frames": frames,
        "fps": fps,
        "frame_interval": frame_interval,
        "segments": segments,
    }


def frame_segment(
    start_frame: int,
    end_frame: int,
    *,
    subtask: str | None = "pick cup",
    skill: str | None = "Pick",
) -> dict[str, Any]:
    """Build one raw frame-boundary segment."""
    segment: dict[str, Any] = {
        "start_frame": start_frame,
        "end_frame": end_frame,
    }
    if subtask is not None:
        segment["subtask"] = subtask
    if skill is not None:
        segment["skill"] = skill
    return segment


def write_json(path: Path, value: Any) -> None:
    """Write a JSON fixture."""
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def report_args(gt_path: Path, prediction_path: Path, *extra: str) -> Any:
    """Return parsed evaluator arguments."""
    return evaluator.parse_args(
        [
            "--gt",
            str(gt_path),
            "--prediction",
            f"demo={prediction_path}",
            *extra,
        ]
    )


def test_edit_collapses_adjacent_duplicate_skills() -> None:
    """Adjacent duplicate skill segments must not reduce Edit."""
    predicted = [
        parsed_segment(0, 0, 1, skill="Pick"),
        parsed_segment(1, 1, 2, skill=" pick "),
        parsed_segment(2, 2, 3, skill="Place"),
    ]
    gold = [
        parsed_segment(0, 0, 2, skill="Pick"),
        parsed_segment(1, 2, 3, skill="Place"),
    ]

    report = evaluator.edit_score(
        predicted,
        gold,
        background_labels=frozenset(),
    )

    assert report["score"] == pytest.approx(100.0)
    assert report["prediction"] == ["pick", "place"]


def test_edit_penalizes_skill_oversegmentation() -> None:
    """Alternating extra skill segments must reduce sequence Edit."""
    predicted = [
        parsed_segment(0, 0, 1, skill="Pick"),
        parsed_segment(1, 1, 2, skill="Place"),
        parsed_segment(2, 2, 3, skill="Pick"),
        parsed_segment(3, 3, 4, skill="Place"),
    ]
    gold = [
        parsed_segment(0, 0, 2, skill="Pick"),
        parsed_segment(1, 2, 4, skill="Place"),
    ]

    report = evaluator.edit_score(
        predicted,
        gold,
        background_labels=frozenset(),
    )

    assert report["distance"] == 2
    assert report["score"] == pytest.approx(50.0)


def test_skill_f1_threshold_is_inclusive() -> None:
    """A same-skill segment with IoU 0.3 passes only inclusive 0.3."""
    predicted = [parsed_segment(0, 0, 3, skill="Pick")]
    gold = [parsed_segment(0, 0, 10, skill="Pick")]

    at_03 = evaluator.skill_segment_f1(
        predicted,
        gold,
        overlap_threshold=0.3,
        background_labels=frozenset(),
    )
    at_05 = evaluator.skill_segment_f1(
        predicted,
        gold,
        overlap_threshold=0.5,
        background_labels=frozenset(),
    )
    at_07 = evaluator.skill_segment_f1(
        predicted,
        gold,
        overlap_threshold=0.7,
        background_labels=frozenset(),
    )

    assert at_03["metrics"] == {
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert at_05["metrics"]["tp"] == 0
    assert at_05["metrics"]["fp"] == 1
    assert at_05["metrics"]["fn"] == 1
    assert at_07["metrics"] == at_05["metrics"]


def test_skill_f1_requires_exact_normalized_skill() -> None:
    """Temporal overlap alone must not produce a segmental F1 TP."""
    predicted = [parsed_segment(0, 0, 1, skill="Place")]
    gold = [parsed_segment(0, 0, 1, skill="Pick")]

    report = evaluator.skill_segment_f1(
        predicted,
        gold,
        overlap_threshold=0.3,
        background_labels=frozenset(),
    )

    assert report["metrics"]["tp"] == 0
    assert report["metrics"]["fp"] == 1
    assert report["metrics"]["fn"] == 1


def test_temporal_f1_ignores_skill_labels() -> None:
    """The original temporal F1 matches boundaries without using labels."""
    predicted = [parsed_segment(0, 0, 1, skill="Place")]
    gold = [parsed_segment(0, 0, 1, skill="Pick")]

    report = evaluator.temporal_segment_f1(
        predicted,
        gold,
        overlap_threshold=0.7,
    )

    assert report["metrics"]["tp"] == 1
    assert report["metrics"]["fp"] == 0
    assert report["metrics"]["fn"] == 0
    assert report["metrics"]["f1"] == 1.0


def test_temporal_f1_uses_one_to_one_matching() -> None:
    """Two predictions cannot both match the same GT segment."""
    predicted = [
        parsed_segment(0, 0, 1),
        parsed_segment(1, 0, 1),
    ]
    gold = [parsed_segment(0, 0, 1)]

    report = evaluator.temporal_segment_f1(
        predicted,
        gold,
        overlap_threshold=0.7,
    )

    assert report["metrics"]["tp"] == 1
    assert report["metrics"]["fp"] == 1
    assert report["metrics"]["fn"] == 0


def test_skill_accuracy_is_temporal_first_and_not_frame_wise() -> None:
    """Skill Acc compares skills only after fixed temporal correspondence."""
    predicted = [
        parsed_segment(0, 0, 1, subtask="same", skill="Place"),
        parsed_segment(1, 1, 2, subtask="same", skill="Pick"),
    ]
    gold = [
        parsed_segment(0, 0, 1, subtask="same", skill="Pick"),
        parsed_segment(1, 1, 2, subtask="same", skill="Place"),
    ]

    report = evaluator.skill_accuracy(
        predicted,
        gold,
        tiou_threshold=0.5,
        background_labels=frozenset(),
    )

    assert report["evaluated_pairs"] == 2
    assert report["correct"] == 0
    assert report["score"] == 0.0
    assert [comparison["temporal_iou"] for comparison in report["comparisons"]] == [
        1.0,
        1.0,
    ]


def test_skill_accuracy_counts_missing_prediction_skill_as_wrong() -> None:
    """A matched segment without predicted skill remains in the denominator."""
    predicted = [parsed_segment(0, 0, 1, subtask="pick cup", skill=None)]
    gold = [parsed_segment(0, 0, 1, subtask="pick cup", skill="Pick")]

    report = evaluator.skill_accuracy(
        predicted,
        gold,
        tiou_threshold=0.5,
        background_labels=frozenset(),
    )

    assert report["score"] == 0.0
    assert report["missing_prediction"] == 1


def test_temporal_miou_aligns_segments_and_zeros_unmatched_gt() -> None:
    """mIoU averages aligned temporal IoU across every GT segment."""
    predicted = [
        parsed_segment(0, 0, 0.5, skill="Wrong"),
        parsed_segment(1, 1, 2, skill="AlsoWrong"),
    ]
    gold = [
        parsed_segment(0, 0, 1, skill="Pick"),
        parsed_segment(1, 1, 2, skill="Place"),
        parsed_segment(2, 2, 3, skill="Push"),
    ]

    report = evaluator.temporal_miou(predicted, gold)

    assert report["aligned_iou_sum"] == pytest.approx(1.5)
    assert report["target_count"] == 3
    assert report["score"] == pytest.approx(0.5)


def test_full_and_gt_active_windows_change_only_evaluable_frames(
    tmp_path: Path,
) -> None:
    """GT-active removes only recognized edge-idle spans before scoring."""
    gt_path = tmp_path / "gt.json"
    prediction_path = tmp_path / "prediction.json"
    gt_segments = [
        frame_segment(
            0, 10, subtask="wait before the task starts", skill="PreTaskIdle"
        ),
        frame_segment(10, 20, subtask="pick cup", skill="Pick"),
        frame_segment(
            20, 30, subtask="wait after the task finishes", skill="PostTaskIdle"
        ),
    ]
    write_json(
        gt_path,
        [bench_episode("demo", gt_segments, frames=30, fps=10.0)],
    )
    write_json(
        prediction_path,
        [bench_episode("demo", [frame_segment(10, 20)], frames=30, fps=10.0)],
    )

    full_report = evaluator.build_report(report_args(gt_path, prediction_path))
    active_report = evaluator.build_report(
        report_args(gt_path, prediction_path, "--evaluation-window", "gt-active")
    )

    assert full_report["runs"][0]["miou"]["score"] == pytest.approx(1 / 3)
    assert active_report["runs"][0]["miou"]["score"] == 1.0


def test_closed_gt_frames_and_second_predictions_align(tmp_path: Path) -> None:
    """Closed-inclusive GT and second-based predictions share one time grid."""
    gt_path = tmp_path / "gt.json"
    prediction_path = tmp_path / "prediction.json"
    write_json(
        gt_path,
        [
            bench_episode(
                "demo",
                [frame_segment(0, 9)],
                frames=10,
                fps=10.0,
                frame_interval="closed",
            )
        ],
    )
    write_json(
        prediction_path,
        [
            {
                "episode_index": "demo",
                "fps": 10.0,
                "segments": [
                    {
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "subtask": "pick cup",
                        "skill": "Pick",
                    }
                ],
            }
        ],
    )

    report = evaluator.build_report(report_args(gt_path, prediction_path))["runs"][0]

    assert report["f1"]["0.70"]["micro"]["f1"] == 1.0
    assert report["skill_f1"]["micro"]["f1"] == 1.0
    assert report["skill_accuracy"]["score"] == 1.0


def test_prediction_overlap_is_a_format_error_scored_as_empty(tmp_path: Path) -> None:
    """Ambiguous frame ownership must not be silently resolved."""
    gt_path = tmp_path / "gt.json"
    prediction_path = tmp_path / "prediction.json"
    write_json(
        gt_path,
        [bench_episode("demo", [frame_segment(0, 10)])],
    )
    write_json(
        prediction_path,
        [
            bench_episode(
                "demo",
                [frame_segment(0, 6), frame_segment(5, 10)],
            )
        ],
    )

    report = evaluator.build_report(report_args(gt_path, prediction_path))["runs"][0]

    assert report["parse_error_count"] == 1
    assert report["format"]["format_valid_episode_rate"] == 0.0
    assert report["f1"]["0.30"]["micro"]["fn"] == 1


def test_missing_prediction_episode_reduces_coverage(tmp_path: Path) -> None:
    """GT episodes without inference output remain in every denominator."""
    gt_path = tmp_path / "gt.json"
    prediction_path = tmp_path / "prediction.json"
    write_json(
        gt_path,
        [
            bench_episode("one", [frame_segment(0, 10)]),
            bench_episode("two", [frame_segment(0, 10)]),
        ],
    )
    write_json(
        prediction_path,
        [bench_episode("one", [frame_segment(0, 10)])],
    )

    report = evaluator.build_report(report_args(gt_path, prediction_path))["runs"][0]

    assert report["coverage"] == pytest.approx(0.5)
    assert report["miou"]["score"] == pytest.approx(0.5)
    assert report["f1"]["0.50"]["micro"]["tp"] == 1
    assert report["f1"]["0.50"]["micro"]["fn"] == 1


def test_llamafactory_saved_predictions_are_supported(tmp_path: Path) -> None:
    """Evaluation accepts a dataset plus completed generated_predictions JSONL."""
    gt_path = tmp_path / "dataset.json"
    prediction_path = tmp_path / "generated_predictions.jsonl"
    target = (
        '{"segments":[{"skill":"Pick","subtask":"pick cup",'
        '"start_frame":0,"end_frame":10}]}'
    )
    sample = {
        "messages": [
            {"role": "system", "content": "Segment the video."},
            {
                "role": "user",
                "content": (
                    "<video>\n\ntotal_frames: 10\nsource_fps: 10.0\n"
                    "task_instruction: pick cup"
                ),
            },
            {"role": "assistant", "content": target},
        ],
        "videos": ["cluster:s3://bucket/demo.mp4"],
    }
    write_json(gt_path, [sample])
    prediction_path.write_text(
        json.dumps({"predict": target}) + "\n",
        encoding="utf-8",
    )

    report = evaluator.build_report(report_args(gt_path, prediction_path))["runs"][0]

    assert report["prediction_format"] == "llamafactory_generated"
    assert report["coverage"] == 1.0
    assert report["edit"]["macro_episode_score"] == 100.0


def test_cli_writes_json_jsonl_and_tsv(tmp_path: Path) -> None:
    """All post-inference output formats carry the same perfect-run metrics."""
    gt_path = tmp_path / "gt.json"
    prediction_path = tmp_path / "prediction.json"
    output_json = tmp_path / "report.json"
    output_jsonl = tmp_path / "episodes.jsonl"
    output_tsv = tmp_path / "summary.tsv"
    episode = bench_episode("demo", [frame_segment(0, 10)])
    write_json(gt_path, [episode])
    write_json(prediction_path, [episode])

    return_code = evaluator.main(
        [
            "--gt",
            str(gt_path),
            "--prediction",
            f"demo={prediction_path}",
            "--output-json",
            str(output_json),
            "--output-jsonl",
            str(output_jsonl),
            "--output-tsv",
            str(output_tsv),
        ]
    )

    assert return_code == 0
    report = json.loads(output_json.read_text(encoding="utf-8"))
    episode_row = json.loads(output_jsonl.read_text(encoding="utf-8"))
    tsv_lines = output_tsv.read_text(encoding="utf-8").splitlines()
    assert report["protocol"]["inference_included"] is False
    assert report["runs"][0]["f1"]["0.70"]["micro"]["f1"] == 1.0
    assert report["runs"][0]["skill_f1"]["micro"]["f1"] == 1.0
    assert report["runs"][0]["miou"]["score"] == 1.0
    assert episode_row["run_name"] == "demo"
    assert "mIoU" in tsv_lines[0]
    assert "SkillF1@0.70" in tsv_lines[0]
    assert tsv_lines[1].startswith("demo\t")


def test_skip_skill_metrics_reports_na_but_keeps_open_metrics(
    tmp_path: Path,
) -> None:
    """Skill-free GT keeps temporal scores and marks skill-only metrics N/A."""
    gt_path = tmp_path / "gt.json"
    prediction_path = tmp_path / "prediction.json"
    output_tsv = tmp_path / "summary.tsv"
    episode = bench_episode(
        "demo",
        [frame_segment(0, 10, subtask="place cup", skill=None)],
    )
    write_json(gt_path, [episode])
    write_json(prediction_path, [episode])

    args = report_args(
        gt_path,
        prediction_path,
        "--skip-skill-metrics",
        "--output-tsv",
        str(output_tsv),
    )
    report = evaluator.build_report(args)
    run = report["runs"][0]
    evaluator.write_summary_tsv(output_tsv, report)

    assert report["protocol"]["skill_metrics_skipped"] is True
    assert report["protocol"]["edit"]["enabled"] is False
    assert report["protocol"]["f1"]["enabled"] is True
    assert report["protocol"]["skill_f1"]["enabled"] is False
    assert run["miou"]["score"] == pytest.approx(1.0)
    assert run["edit"]["macro_episode_score"] is None
    assert run["skill_accuracy"]["score"] is None
    assert all(value["micro"]["f1"] == 1.0 for value in run["f1"].values())
    assert run["skill_f1"]["micro"] is None
    assert output_tsv.read_text(encoding="utf-8").splitlines()[1].count("N/A") == 3

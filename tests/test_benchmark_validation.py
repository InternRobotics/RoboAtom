"""Benchmark validation without a separate manifest or bundled dataset."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from roboatom.validation.benchmark import validate_benchmark


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a temporary benchmark JSONL fixture."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def benchmark_root(tmp_path: Path) -> Path:
    """Create 500 episodes sharing two local video paths."""
    inputs = []
    gold = []
    for index in range(500):
        identifier = f"episode-{index}"
        inputs.append({
            "benchmark_id": identifier, "length": 10, "fps": 5.0,
            "task": "Pick the object.", "views": {"front": "videos/front.mp4"},
            "evaluation_views": ["front"], "model_view_key": "front",
        })
        gold.append({
            "benchmark_id": identifier,
            "action_config": [{
                "start_frame": 0, "end_frame": 9,
                "skill": "Pick", "action_text": "Pick the object.",
            }],
        })
    inputs[0]["views"]["wrist"] = "videos/wrist.mp4"
    inputs[0]["evaluation_views"].append("wrist")
    write_rows(tmp_path / "benchmark.jsonl", inputs)
    write_rows(tmp_path / "gt.jsonl", gold)
    (tmp_path / "videos").mkdir()
    for name in ("front.mp4", "wrist.mp4"):
        (tmp_path / "videos" / name).touch()
    return tmp_path


def test_validation_without_manifest(benchmark_root: Path) -> None:
    result = validate_benchmark(benchmark_root)

    assert result == {
        "coverage": 500, "expected": 500, "missing": [], "extra": [],
        "media_checked": 0, "media_expected": 2,
    }


def test_media_checks_unique_camera_paths(benchmark_root: Path) -> None:
    result = validate_benchmark(benchmark_root, benchmark_root)

    assert result["media_checked"] == 2
    assert result["media_expected"] == 2


def test_missing_auxiliary_video(benchmark_root: Path) -> None:
    (benchmark_root / "videos/wrist.mp4").unlink()

    with pytest.raises(ValueError, match="Missing video: videos/wrist.mp4"):
        validate_benchmark(benchmark_root, benchmark_root)


@pytest.mark.parametrize("filename", ["benchmark.jsonl", "gt.jsonl"])
def test_duplicate_ids_are_rejected(benchmark_root: Path, filename: str) -> None:
    path = benchmark_root / filename
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows.append(rows[0])
    write_rows(path, rows)

    with pytest.raises(ValueError, match="duplicate"):
        validate_benchmark(benchmark_root)


def test_missing_gt_id_is_rejected(benchmark_root: Path) -> None:
    path = benchmark_root / "gt.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    write_rows(path, rows[:-1])

    with pytest.raises(ValueError, match="IDs differ"):
        validate_benchmark(benchmark_root)


def test_full_benchmark_requires_500_episodes(benchmark_root: Path) -> None:
    for filename in ("benchmark.jsonl", "gt.jsonl"):
        path = benchmark_root / filename
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        write_rows(path, rows[:-1])

    with pytest.raises(ValueError, match="Expected 500"):
        validate_benchmark(benchmark_root)


@pytest.mark.parametrize("video_path", ["../outside.mp4", "/outside.mp4"])
def test_video_paths_are_relative(benchmark_root: Path, video_path: str) -> None:
    path = benchmark_root / "benchmark.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["views"]["front"] = video_path
    write_rows(path, rows)

    with pytest.raises(ValueError, match="Video path must be relative"):
        validate_benchmark(benchmark_root)


def test_camera_metadata_matches_views(benchmark_root: Path) -> None:
    path = benchmark_root / "benchmark.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["evaluation_views"] = ["front"]
    write_rows(path, rows)

    with pytest.raises(ValueError, match="Invalid camera metadata"):
        validate_benchmark(benchmark_root)


def test_invalid_gt_segment_is_rejected(benchmark_root: Path) -> None:
    path = benchmark_root / "gt.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["action_config"][0]["end_frame"] = 20
    write_rows(path, rows)

    with pytest.raises(ValueError, match="invalid"):
        validate_benchmark(benchmark_root)

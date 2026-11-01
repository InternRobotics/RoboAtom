"""Validate benchmark input, ground truth and referenced local videos."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .data import check_coverage, read_rows, validate_rows


def validate_benchmark(root: Path, media_root: Path | None = None) -> dict[str, Any]:
    """Check the 500-episode benchmark and optionally its video files.

    Args:
        root: Directory containing benchmark.jsonl and gt.jsonl.
        media_root: Root for the relative video paths, or None to skip media checks.

    Returns:
        Episode coverage and the number of expected and checked video files.

    Raises:
        ValueError: If IDs, annotations, camera paths or media files are invalid.
    """
    rows = read_rows(root / "benchmark.jsonl")
    input_ids = validate_rows(rows, require_segments=False)
    gold_rows = read_rows(root / "gt.jsonl")
    gold_ids = [row["benchmark_id"] for row in gold_rows]
    if len(gold_ids) != len(set(gold_ids)) or input_ids != set(gold_ids):
        raise ValueError("Benchmark input/GT IDs differ or contain duplicates")
    inputs = {row["benchmark_id"]: row for row in rows}
    gold = [{**inputs[row["benchmark_id"]], **row} for row in gold_rows]
    result = check_coverage(gold, gold, expected=500)
    videos: set[Path] = set()
    for row in rows:
        views = row["views"]
        if (
            not views
            or row["model_view_key"] not in views
            or set(row["evaluation_views"]) != set(views)
        ):
            raise ValueError(f"Invalid camera metadata: {row['benchmark_id']}")
        for value in views.values():
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Video path must be relative: {value}")
            videos.add(path)
    if media_root is not None:
        for path in sorted(videos):
            if not (media_root / path).is_file():
                raise ValueError(f"Missing video: {path}")
    return {
        **result,
        "media_checked": len(videos) if media_root is not None else 0,
        "media_expected": len(videos),
    }

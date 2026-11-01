"""Build the frame-index prompt and camera layout used by RoboAtom-Anno."""
from __future__ import annotations

from typing import Any

SYSTEM = "You are a robotic video subtask segmenter."
LAYOUTS = {
    "single": "FULL=PRIMARY",
    "vertical_stack": "TOP=PRIMARY\nBOTTOM=AUXILIARY",
    "primary_large_right_stack": (
        "LEFT_LARGE=PRIMARY\nRIGHT_TOP=LEFT_WRIST\nRIGHT_BOTTOM=RIGHT_WRIST"
    ),
    "grid_2x2": (
        "TOP_LEFT=PRIMARY\nTOP_RIGHT=AUXILIARY\n"
        "BOTTOM_LEFT=LEFT_WRIST\nBOTTOM_RIGHT=RIGHT_WRIST"
    ),
}


def camera_layout(row: dict[str, Any]) -> dict[str, Any]:
    """Select synchronized cameras from a benchmark input row.

    Args:
        row: Benchmark metadata with views, evaluation_views and model_view_key.

    Returns:
        Layout name and an ordered list of camera roles and relative paths.

    Raises:
        ValueError: If the input does not contain one to four evaluation views.
    """
    primary = row["model_view_key"]
    secondary = [key for key in row["evaluation_views"] if key != primary]
    ordered = [primary, *secondary]
    left = [key for key in secondary if "left" in key.casefold()]
    right = [key for key in secondary if "right" in key.casefold()]
    if len(ordered) == 1:
        roles, layout = ["primary"], "single"
    elif len(ordered) == 2:
        roles, layout = ["primary", "aux"], "vertical_stack"
    elif len(ordered) == 3:
        if left and right:
            ordered = [primary, left[0], right[0]]
        roles = ["primary", "left_wrist", "right_wrist"]
        layout = "primary_large_right_stack"
    elif len(ordered) == 4:
        if left and right:
            auxiliary = next(key for key in secondary if key not in {left[0], right[0]})
            ordered = [primary, auxiliary, left[0], right[0]]
        roles = ["primary", "aux", "left_wrist", "right_wrist"]
        layout = "grid_2x2"
    else:
        raise ValueError("Expected one to four evaluation views")
    return {
        "layout": layout,
        "views": [
            {"role": role, "path": row["views"][key]}
            for key, role in zip(ordered, roles, strict=True)
        ],
    }


def user_prompt(row: dict[str, Any], layout: str) -> str:
    """Format video metadata and task instruction without ground truth.

    Args:
        row: Benchmark input metadata.
        layout: Camera layout returned by camera_layout.

    Returns:
        User instruction for zero-based, closed frame intervals.
    """
    return (
        f"<video>\n\nview_layout:\n{LAYOUTS[layout]}\n\n"
        "Given the full video, output all atomic subtasks and their start/end frame indices.\n\n"
        "Frame-index convention:\n"
        "- Intervals are closed and inclusive: [start_frame, end_frame].\n"
        "- Every frame must be covered exactly once with no gaps or overlaps.\n"
        "- The first start_frame must be 0.\n"
        "- Each next start_frame must equal the previous end_frame plus 1.\n"
        "- The final end_frame must equal total_frames minus 1.\n"
        "- Therefore 0 <= start_frame <= end_frame < total_frames.\n\n"
        f"total_frames: {row['length']}\n"
        f"source_fps: {row['fps']:g}\n"
        f"task_instruction: {row['task']}\n\n"
        "Return valid JSON only."
    )

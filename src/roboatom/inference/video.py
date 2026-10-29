# Adapted from LLaMA-Factory. Copyright 2025 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.
"""Local video sampling and synchronized camera mosaics for RoboAtom-Anno."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import av
import numpy as np
from PIL import Image


def sample_indices(total_frames: int, duration: float) -> list[int]:
    """Select up to 448 frames at a target rate of two frames per second.

    Args:
        total_frames: Original video frame count.
        duration: Video duration in seconds.

    Returns:
        Evenly spaced original frame indices including both endpoints.
    """
    count = min(total_frames, 448, max(1, math.floor(duration * 2.0)))
    return np.linspace(0, total_frames - 1, count).astype(np.int32).tolist()


def decode_video(
    path: Path, target_times: list[float] | None = None,
) -> tuple[list[Image.Image], list[int], float, float]:
    """Decode sampled source frames or synchronized auxiliary frames.

    Args:
        path: Local video file.
        target_times: Primary-camera sample times, or None for primary sampling.

    Returns:
        RGB frames, original indices, source FPS and duration.

    Raises:
        ValueError: If requested frames are absent from the decoded video.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "FRAME"
        stream.thread_count = 2
        fps = float(stream.average_rate)
        duration = float(stream.duration * stream.time_base)
        if target_times is None:
            indices = sample_indices(stream.frames, duration)
        else:
            indices = [
                min(stream.frames - 1, max(0, round(timestamp * fps)))
                for timestamp in target_times
            ]
        required = set(indices)
        decoded = {}
        for index, frame in enumerate(container.decode(stream)):
            if index in required:
                decoded[index] = frame.to_image()
            if index >= indices[-1]:
                break
        if required != decoded.keys():
            raise ValueError(f"Video is missing sampled frames: {path}")
        return [decoded[index] for index in indices], indices, fps, duration


def _fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    scale = min(size[0] / image.width, size[1] / image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    resized = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size)
    panel.paste(resized, ((size[0] - width) // 2, (size[1] - height) // 2))
    return panel


def compose_mosaic(frames: dict[str, Image.Image], layout: str) -> Image.Image:
    """Arrange synchronized RGB frames using the model's camera layout.

    Args:
        frames: Frames indexed by primary, aux, left_wrist and right_wrist roles.
        layout: single, vertical_stack, primary_large_right_stack or grid_2x2.

    Returns:
        Composed RGB image.

    Raises:
        ValueError: If the layout is unsupported.
    """
    primary = frames["primary"]
    width, height = max(2, primary.width), max(2, primary.height)
    if layout == "single":
        return primary
    if layout == "vertical_stack":
        canvas = Image.new("RGB", (width, height * 2))
        placements = [
            ("primary", (0, 0), (width, height)),
            ("aux", (0, height), (width, height)),
        ]
    elif layout == "primary_large_right_stack":
        small_width, upper_height = max(1, width // 2), max(1, height // 2)
        canvas = Image.new("RGB", (width + small_width, height))
        placements = [
            ("primary", (0, 0), (width, height)),
            ("left_wrist", (width, 0), (small_width, upper_height)),
            ("right_wrist", (width, upper_height), (small_width, height - upper_height)),
        ]
    elif layout == "grid_2x2":
        canvas = Image.new("RGB", (width * 2, height * 2))
        placements = [
            ("primary", (0, 0), (width, height)),
            ("aux", (width, 0), (width, height)),
            ("left_wrist", (0, height), (width, height)),
            ("right_wrist", (width, height), (width, height)),
        ]
    else:
        raise ValueError(f"Unsupported camera layout: {layout}")
    for role, position, size in placements:
        canvas.paste(_fit_panel(frames[role], size), position)
    return canvas


def _resize_frame(image: Image.Image) -> Image.Image:
    pixels = image.width * image.height
    if pixels > 16384:
        scale = math.sqrt(16384 / pixels)
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    pixels = image.width * image.height
    if pixels < 256:
        scale = math.sqrt(256 / pixels)
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    if min(image.size) < 28:
        image = image.resize((max(image.width, 28), max(image.height, 28)))
    if image.width / image.height > 200:
        image = image.resize((image.height * 180, image.height))
    if image.height / image.width > 200:
        image = image.resize((image.width, image.width * 180))
    return image.convert("RGB")


def preprocess_video(
    video: dict[str, Any], media_root: Path, processor: Any,
) -> tuple[dict[str, Any], list[int]]:
    """Sample and preprocess a local single-view or multiview video.

    Args:
        video: Structured camera layout and relative video paths.
        media_root: Root containing the video files.
        processor: Qwen3-VL processor from the released model.

    Returns:
        Model video tensors and original source indices for visual blocks.

    Raises:
        ValueError: If camera paths escape media_root or durations differ.
    """
    root = media_root.resolve()
    paths = {view["role"]: (root / view["path"]).resolve() for view in video["views"]}
    if any(not path.is_relative_to(root) for path in paths.values()):
        raise ValueError("Video paths must be inside media_root")
    primary, indices, fps, duration = decode_video(paths["primary"])
    cameras = {"primary": primary}
    times = [index / fps for index in indices]
    for role, path in paths.items():
        if role == "primary":
            continue
        frames, _, view_fps, view_duration = decode_video(path, times)
        if abs(view_duration - duration) > max(0.1, 2.0 / min(fps, view_fps)):
            raise ValueError(f"Camera duration differs from the primary video: {role}")
        cameras[role] = frames
    frames = [
        compose_mosaic({role: images[index] for role, images in cameras.items()}, video["layout"])
        for index in range(len(primary))
    ]
    temporal_size = processor.video_processor.temporal_patch_size
    frames = [_resize_frame(frame) for frame in frames for _ in range(temporal_size)]
    sampled_axis = [index / fps * 2.0 for index in indices for _ in range(temporal_size)]
    metadata = [{
        "fps": 2.0,
        "duration": duration,
        "total_num_frames": len(frames),
        "frames_indices": sampled_axis,
    }]
    processed = processor.video_processor(
        videos=[frames], video_metadata=metadata, fps=2.0,
        return_metadata=True, do_sample_frames=False,
    )
    processed.pop("video_metadata")
    return dict(processed), indices

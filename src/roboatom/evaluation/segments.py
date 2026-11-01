"""Parsing, windowing, and temporal matching helpers for the benchmark evaluator."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EPSILON = 1e-12
SUBTASK_FIELDS = ("action_text", "subtask", "substask", "label", "description")
SKILL_FIELDS = ("primitive_skill", "skill")
LABEL_FIELDS = SUBTASK_FIELDS + SKILL_FIELDS
MAX_PARSE_ERROR_EXAMPLES = 20
EVALUATION_WINDOWS = ("full", "gt-active")
BOUNDARY_IDLE_LABEL_FIELDS = ("label", "subtask_label", "skill_label")
BOUNDARY_IDLE_LABELS = frozenset(
    {
        "idle",
        "pre task idle",
        "pretaskidle",
        "post task idle",
        "posttaskidle",
        "wait before the task starts",
        "wait after the task finishes",
    }
)


def parse_thresholds(raw_value: str) -> tuple[float, ...]:
    """Parse a comma-separated threshold list."""
    values: list[float] = []
    for item in raw_value.split(","):
        text = item.strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"Invalid tIoU threshold {text!r}") from exc
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"tIoU threshold must be between 0 and 1: {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one tIoU threshold is required")
    return tuple(dict.fromkeys(values))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON array, JSON object, or JSONL records from a file.

    Args:
        path: Input file path.

    Returns:
        A list of JSON object records.

    Raises:
        ValueError: If the path is missing or contains non-object records.
    """
    if not path.is_file():
        raise ValueError(f"JSON/JSONL file does not exist: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"Expected JSON array in {path}")
        records = value
    elif "\n" not in text:
        value = json.loads(text)
        if isinstance(value, list):
            records = value
        else:
            records = [value]
    else:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc

    rows: list[dict[str, Any]] = []
    for row_index, value in enumerate(records):
        row = unwrap_row(value)
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object at {path} record {row_index}")
        rows.append(row)
    return rows


def unwrap_row(value: Any) -> Any:
    """Accept plain episode rows and common result wrappers."""
    if isinstance(value, dict):
        result = value.get("result")
        if isinstance(result, dict):
            return result
        if isinstance(result, list):
            objects = [item for item in result if isinstance(item, dict)]
            if len(objects) == 1:
                return objects[0]
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and (
                isinstance(item.get("action_config"), list)
                or isinstance(item.get("segments"), list)
            ):
                return item
    return value


def parse_prediction_spec(spec: str) -> tuple[str, Path]:
    """Parse a NAME=PATH prediction specification."""
    if "=" not in spec:
        raise ValueError(f"Invalid --prediction {spec!r}; expected NAME=PATH")
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name or not raw_path:
        raise ValueError(f"Invalid --prediction {spec!r}; NAME and PATH are required")
    path = Path(raw_path).expanduser()
    if path.is_dir():
        path = path / "episodes.jsonl"
    return name, path.resolve()


def strip_code_fence(text: str) -> str:
    """Remove one surrounding markdown code fence if present.

    Args:
        text: Raw model output text.

    Returns:
        Text without one leading and trailing markdown fence.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_json_value(text: str) -> Any:
    """Parse a JSON value from raw model output.

    Args:
        text: Model output, optionally surrounded by markdown or extra text.

    Returns:
        Parsed JSON value.

    Raises:
        ValueError: If no JSON object or array can be parsed.
    """
    stripped = strip_code_fence(text)
    decoder = json.JSONDecoder()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for index, character in enumerate(stripped):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        return value

    raise ValueError("could not parse JSON from model output")


def is_llamafactory_sample(row: Mapping[str, Any]) -> bool:
    """Return whether a row looks like a LlamaFactory multimodal sample."""
    return isinstance(row.get("messages"), list) and isinstance(row.get("videos"), list)


def is_llamafactory_prediction_row(row: Mapping[str, Any]) -> bool:
    """Return whether a row looks like one generated_predictions.jsonl record."""
    return "predict" in row


def message_text(message: Mapping[str, Any]) -> str:
    """Return text content from a chat message.

    Args:
        message: Chat message dictionary.

    Returns:
        Concatenated text content.
    """
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    return str(content)


def parse_prompt_metadata(prompt: str) -> tuple[int, float]:
    """Extract total frame count and source FPS from a LlamaFactory prompt.

    Args:
        prompt: User prompt text.

    Returns:
        A ``(total_frames, fps)`` tuple.

    Raises:
        ValueError: If either metadata field is missing or invalid.
    """
    total_frames: int | None = None
    source_fps: float | None = None
    for line in prompt.splitlines():
        stripped = line.strip()
        normalized = stripped[2:].strip() if stripped.startswith("- ") else stripped
        if normalized.startswith("total_frames:"):
            total_frames = int(normalized.split(":", maxsplit=1)[1].strip())
        elif normalized.startswith("source_fps:") or normalized.startswith("fps:"):
            source_fps = float(normalized.split(":", maxsplit=1)[1].strip())

    if total_frames is None or total_frames <= 0:
        raise ValueError(f"missing or invalid total_frames in prompt: {prompt[:200]}")
    if source_fps is None or not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"missing or invalid source_fps/fps in prompt: {prompt[:200]}")
    return total_frames, source_fps


def llamafactory_sample_metadata(sample: Mapping[str, Any], sample_index: int) -> tuple[str, int, float]:
    """Return episode id/path, frame count, and FPS for one LlamaFactory sample.

    Args:
        sample: LlamaFactory dataset row.
        sample_index: Row index used for error messages.

    Returns:
        A tuple of ``(video_path, total_frames, fps)``.

    Raises:
        ValueError: If the sample is malformed.
    """
    videos = sample.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError(f"LlamaFactory sample {sample_index} has no videos list")
    video_path = str(videos[0])

    messages = sample.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"LlamaFactory sample {sample_index} has no messages list")
    user_messages = [
        message
        for message in messages
        if isinstance(message, dict) and str(message.get("role", "")).lower() == "user"
    ]
    if not user_messages:
        raise ValueError(f"LlamaFactory sample {sample_index} has no user message")
    total_frames, fps = parse_prompt_metadata(message_text(user_messages[0]))
    return video_path, total_frames, fps


def llamafactory_assistant_target(sample: Mapping[str, Any], sample_index: int) -> str:
    """Return the assistant target text from a LlamaFactory dataset row.

    Args:
        sample: LlamaFactory dataset row.
        sample_index: Row index used for error messages.

    Returns:
        Assistant message text.

    Raises:
        ValueError: If no assistant message is present.
    """
    messages = sample.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"LlamaFactory sample {sample_index} has no messages list")
    for message in reversed(messages):
        if isinstance(message, dict) and str(message.get("role", "")).lower() == "assistant":
            return message_text(message)
    raise ValueError(f"LlamaFactory sample {sample_index} has no assistant target")


def raw_segments_from_model_output(value: Any) -> list[dict[str, Any]]:
    """Extract segment dictionaries from a model JSON value.

    Args:
        value: Parsed model JSON output.

    Returns:
        Segment dictionaries.

    Raises:
        ValueError: If the output has no segment list or contains malformed items.
    """
    if isinstance(value, dict):
        raw_segments = value.get("segments")
    elif isinstance(value, list):
        raw_segments = value
    else:
        raise ValueError("model output must be a JSON object or segment list")

    if not isinstance(raw_segments, list):
        raise ValueError("model output has no segments list")

    segments: list[dict[str, Any]] = []
    for segment_index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, dict):
            raise ValueError(f"segment {segment_index} is not an object")
        segments.append(dict(raw_segment))
    return segments


def episode_row(
    *,
    video_path: str,
    total_frames: int,
    fps: float,
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build one canonical episode row for evaluation."""
    return {
        "episode_index": video_path,
        "path": video_path,
        "fps": fps,
        "frames": total_frames,
        "segments": segments,
    }


def llamafactory_gt_rows(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert LlamaFactory dataset rows to in-memory evaluator episode rows.

    Args:
        samples: LlamaFactory JSON dataset records.

    Returns:
        Canonical evaluator rows.
    """
    rows: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        video_path, total_frames, fps = llamafactory_sample_metadata(sample, sample_index)
        target = parse_json_value(llamafactory_assistant_target(sample, sample_index))
        rows.append(
            episode_row(
                video_path=video_path,
                total_frames=total_frames,
                fps=fps,
                segments=raw_segments_from_model_output(target),
            )
        )
    return rows


def preview_text(text: str, limit: int = 500) -> str:
    """Return a compact single-line text preview."""
    return " ".join(text.split())[:limit]


def empty_prediction_error(
    *,
    sample_index: int,
    episode: str,
    error: Exception,
    prediction_text: str,
) -> dict[str, Any]:
    """Build one parse-error report row."""
    return {
        "sample_index": sample_index,
        "episode_id": episode,
        "error": f"{type(error).__name__}: {error}",
        "prediction_preview": preview_text(prediction_text),
    }


def llamafactory_prediction_rows(
    *,
    samples: Sequence[dict[str, Any]],
    prediction_records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert generated_predictions records to evaluator rows aligned by sample order.

    Args:
        samples: LlamaFactory eval dataset records.
        prediction_records: LlamaFactory generated prediction records.

    Returns:
        A tuple of canonical prediction rows and parse error rows.

    Raises:
        ValueError: If row counts differ.
    """
    if len(samples) != len(prediction_records):
        raise ValueError(
            "LlamaFactory prediction row count mismatch: "
            f"gt={len(samples)} prediction={len(prediction_records)}"
        )

    rows: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    for sample_index, (sample, prediction_record) in enumerate(
        zip(samples, prediction_records, strict=True)
    ):
        video_path, total_frames, fps = llamafactory_sample_metadata(sample, sample_index)
        prediction_text = str(prediction_record.get("predict", ""))
        try:
            prediction_value = parse_json_value(prediction_text)
            segments = raw_segments_from_model_output(prediction_value)
            row = episode_row(
                video_path=video_path,
                total_frames=total_frames,
                fps=fps,
                segments=segments,
            )
            parse_segments(row, video_path, "Prediction")
        except ValueError as exc:
            segments = []
            parse_errors.append(
                empty_prediction_error(
                    sample_index=sample_index,
                    episode=video_path,
                    error=exc,
                    prediction_text=prediction_text,
                )
            )

        rows.append(
            episode_row(
                video_path=video_path,
                total_frames=total_frames,
                fps=fps,
                segments=segments,
            )
        )
    return rows, parse_errors


def read_gt_input(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]] | None, str]:
    """Read GT input as indexed evaluator rows.

    Args:
        path: GT path.

    Returns:
        ``(gt_index, llamafactory_samples, format_name)``.
    """
    rows = read_jsonl(path)
    if rows and all(is_llamafactory_sample(row) for row in rows):
        gt_rows = llamafactory_gt_rows(rows)
        return index_rows(gt_rows, prediction=False), rows, "llamafactory"
    if any(is_llamafactory_sample(row) for row in rows):
        raise ValueError(f"Mixed LlamaFactory and bench GT records are not supported: {path}")
    return index_rows(rows, prediction=False), None, "bench_jsonl"


def read_prediction_input(
    *,
    path: Path,
    llamafactory_samples: Sequence[dict[str, Any]] | None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], str]:
    """Read one prediction input as indexed evaluator rows.

    Args:
        path: Prediction path.
        llamafactory_samples: GT samples for row-aligned LlamaFactory predictions.

    Returns:
        ``(prediction_index, parse_errors, format_name)``.

    Raises:
        ValueError: If a LlamaFactory prediction lacks LlamaFactory GT rows.
    """
    rows = read_jsonl(path)
    if rows and all(is_llamafactory_prediction_row(row) for row in rows):
        if llamafactory_samples is None:
            raise ValueError("LlamaFactory generated_predictions.jsonl requires a LlamaFactory --gt dataset")
        prediction_rows, parse_errors = llamafactory_prediction_rows(
            samples=llamafactory_samples,
            prediction_records=rows,
        )
        return index_rows(prediction_rows, prediction=True), parse_errors, "llamafactory_generated"
    if any(is_llamafactory_prediction_row(row) for row in rows):
        raise ValueError(f"Mixed LlamaFactory and bench prediction records are not supported: {path}")
    return index_rows(rows, prediction=True), [], "bench_jsonl"


def episode_id(row: Mapping[str, Any], *, prediction: bool) -> str:
    """Return a stable episode identifier from a GT or prediction row."""
    keys = (
        ("episode_id", "benchmark_id", "episode_index", "sample_id", "id", "path")
        if prediction
        else ("episode_id", "benchmark_id", "episode_index", "sample_id", "id", "path")
    )
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    role = "Prediction" if prediction else "GT"
    raise ValueError(f"{role} row has no episode ID")


def index_rows(rows: Iterable[dict[str, Any]], *, prediction: bool) -> dict[str, dict[str, Any]]:
    """Index rows by episode ID and reject duplicates."""
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = episode_id(row, prediction=prediction)
        if identifier in indexed:
            role = "Prediction" if prediction else "GT"
            raise ValueError(f"{role} contains duplicate episode ID {identifier!r}")
        indexed[identifier] = row
    return indexed


def finite_number(value: Any, context: str) -> float:
    """Convert a value to a finite float."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite, got {value!r}")
    return number


def positive_fps(row: Mapping[str, Any], episode: str, role: str) -> float:
    """Read a positive FPS value from an episode row."""
    fps = finite_number(row.get("fps"), f"{role} {episode} fps")
    if fps <= 0:
        raise ValueError(f"{role} {episode} fps must be positive")
    return fps


def raw_segment_values(row: Mapping[str, Any], episode: str, role: str) -> list[dict[str, Any]]:
    """Return the raw segment list from action_config or segments."""
    values = row.get("action_config")
    if values is None:
        values = row.get("segments")
    if values is None:
        raise ValueError(f"{role} {episode} has no action_config or segments list")
    if not isinstance(values, list):
        raise ValueError(f"{role} {episode} segments must be a list")
    result: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"{role} {episode} segment {index} is not an object")
        result.append(value)
    return result


def optional_segment_label(
    segment: Mapping[str, Any],
    fields: Sequence[str],
) -> str | None:
    """Extract the first non-empty label from a sequence of fields."""
    for field in fields:
        value = segment.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def segment_label(segment: Mapping[str, Any], context: str) -> str:
    """Extract the legacy label text used for joint semantic matching."""
    label = optional_segment_label(segment, LABEL_FIELDS)
    if label is not None:
        return label
    raise ValueError(f"{context} has no usable label field")


def detect_frame_interval(
    row: Mapping[str, Any],
    raw_segments: Sequence[Mapping[str, Any]],
) -> str:
    """Detect whether frame end indices are half-open or closed-inclusive."""
    explicit = str(row.get("frame_interval", "")).strip().lower()
    if explicit in {"closed", "closed_inclusive", "inclusive"}:
        return "closed"
    if explicit in {"open", "half_open", "exclusive"}:
        return "open"

    open_votes = 0
    closed_votes = 0
    frames_value = row.get("num_frames", row.get("num_frames", row.get("frames", row.get("length"))))
    if frames_value is not None and raw_segments:
        try:
            frames = int(frames_value)
            last_end = int(raw_segments[-1].get("end_frame"))
        except (TypeError, ValueError):
            frames = 0
            last_end = -1
        if frames > 0 and last_end == frames:
            open_votes += 1
        if frames > 0 and last_end == frames - 1:
            closed_votes += 1

    for previous, current in zip(raw_segments, raw_segments[1:]):
        try:
            previous_end = int(previous.get("end_frame"))
            current_start = int(current.get("start_frame"))
        except (TypeError, ValueError):
            continue
        if current_start == previous_end:
            open_votes += 1
        if current_start == previous_end + 1:
            closed_votes += 1

    return "closed" if closed_votes > open_votes else "open"


def segment_times(
    segment: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    frame_interval: str,
    context: str,
) -> tuple[float, float]:
    """Return segment start/end seconds."""
    if segment.get("start_sec") is not None and segment.get("end_sec") is not None:
        return (
            finite_number(segment.get("start_sec"), f"{context} start_sec"),
            finite_number(segment.get("end_sec"), f"{context} end_sec"),
        )

    fps = positive_fps(row, str(row.get("episode_index", "")), context)
    start_frame = finite_number(segment.get("start_frame"), f"{context} start_frame")
    end_frame = finite_number(segment.get("end_frame"), f"{context} end_frame")
    start = start_frame / fps
    end = (end_frame + 1.0) / fps if frame_interval == "closed" else end_frame / fps
    return start, end


def parse_segments(row: Mapping[str, Any], episode: str, role: str) -> list[dict[str, Any]]:
    """Parse and time-sort an episode's segments."""
    raw_segments = raw_segment_values(row, episode, role)
    frame_interval = detect_frame_interval(row, raw_segments)
    parsed: list[dict[str, Any]] = []

    for index, raw_segment in enumerate(raw_segments):
        context = f"{role} {episode} segment {index}"
        start, end = segment_times(
            raw_segment,
            row,
            frame_interval=frame_interval,
            context=context,
        )
        label = segment_label(raw_segment, context)
        if start < 0:
            raise ValueError(f"{context} start is negative: {start}")
        if end <= start:
            continue
        parsed.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "label": label,
                "subtask_label": optional_segment_label(raw_segment, SUBTASK_FIELDS),
                "skill_label": optional_segment_label(raw_segment, SKILL_FIELDS),
            }
        )

    parsed.sort(key=lambda item: (item["start"], item["end"], item["index"]))
    return parsed


def normalize_boundary_idle_label(value: Any) -> str:
    """Normalize a label for exact boundary-idle recognition.

    Args:
        value: Candidate segment label.

    Returns:
        A case-folded, whitespace-separated alphanumeric label.
    """
    normalized_characters = [
        character if character.isalnum() else " "
        for character in str(value).casefold()
    ]
    return " ".join("".join(normalized_characters).split())


def is_boundary_idle_segment(segment: Mapping[str, Any]) -> bool:
    """Return whether a parsed segment has a recognized exact idle label."""
    return any(
        normalize_boundary_idle_label(segment[field]) in BOUNDARY_IDLE_LABELS
        for field in BOUNDARY_IDLE_LABEL_FIELDS
        if segment.get(field) is not None and str(segment[field]).strip()
    )


def ground_truth_active_window(
    gold: Sequence[Mapping[str, Any]],
    *,
    episode: str,
) -> tuple[float, float]:
    """Derive the active time window from meaningful GT segments.

    Args:
        gold: Parsed and time-sorted ground-truth segments.
        episode: Episode identifier used in validation errors.

    Returns:
        The earliest meaningful GT start and latest meaningful GT end.

    Raises:
        ValueError: If the episode has no non-idle GT segment.
    """
    meaningful_segments = [
        segment for segment in gold if not is_boundary_idle_segment(segment)
    ]
    if not meaningful_segments:
        raise ValueError(
            f"GT episode {episode!r} has no non-idle segment for gt-active evaluation"
        )

    window_start = min(float(segment["start"]) for segment in meaningful_segments)
    window_end = max(float(segment["end"]) for segment in meaningful_segments)
    if window_end <= window_start:
        raise ValueError(
            f"GT episode {episode!r} has an invalid active window: "
            f"[{window_start}, {window_end}]"
        )
    return window_start, window_end


def clip_segments_to_window(
    segments: Sequence[Mapping[str, Any]],
    *,
    window_start: float,
    window_end: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Intersect segments with a time window without rebasing coordinates.

    Args:
        segments: Parsed segments in original time coordinates.
        window_start: Inclusive active-window start in seconds.
        window_end: Exclusive active-window end in seconds.

    Returns:
        The retained segment copies and drop/clip audit counts.
    """
    retained_segments: list[dict[str, Any]] = []
    dropped_segment_count = 0
    clipped_segment_count = 0
    for segment in segments:
        original_start = float(segment["start"])
        original_end = float(segment["end"])
        clipped_start = max(original_start, window_start)
        clipped_end = min(original_end, window_end)
        if clipped_end - clipped_start <= EPSILON:
            dropped_segment_count += 1
            continue

        retained_segment = dict(segment)
        retained_segment["start"] = clipped_start
        retained_segment["end"] = clipped_end
        if (
            abs(clipped_start - original_start) > EPSILON
            or abs(clipped_end - original_end) > EPSILON
        ):
            clipped_segment_count += 1
        retained_segments.append(retained_segment)

    return retained_segments, {
        "original_segment_count": len(segments),
        "retained_segment_count": len(retained_segments),
        "dropped_segment_count": dropped_segment_count,
        "clipped_segment_count": clipped_segment_count,
    }


def apply_evaluation_window(
    predicted: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
    *,
    evaluation_window: str,
    episode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Apply one evaluation-window policy to GT and predictions.

    Args:
        predicted: Parsed prediction segments.
        gold: Parsed ground-truth segments.
        evaluation_window: ``full`` or ``gt-active``.
        episode: Episode identifier used in validation errors.

    Returns:
        Cropped predictions, cropped GT, and a per-episode audit payload.

    Raises:
        ValueError: If the policy is unsupported or GT cannot define a window.
    """
    if evaluation_window not in EVALUATION_WINDOWS:
        raise ValueError(
            f"Unsupported evaluation window {evaluation_window!r}; "
            f"expected one of {EVALUATION_WINDOWS}"
        )

    if evaluation_window == "full":
        return [dict(segment) for segment in predicted], [
            dict(segment) for segment in gold
        ], {
            "policy": "full",
            "start_sec": None,
            "end_sec": None,
            "timestamps_rebased": False,
            "gt": {
                "original_segment_count": len(gold),
                "retained_segment_count": len(gold),
                "dropped_segment_count": 0,
                "clipped_segment_count": 0,
            },
            "prediction": {
                "original_segment_count": len(predicted),
                "retained_segment_count": len(predicted),
                "dropped_segment_count": 0,
                "clipped_segment_count": 0,
            },
        }

    window_start, window_end = ground_truth_active_window(gold, episode=episode)
    cropped_gold, gt_audit = clip_segments_to_window(
        gold,
        window_start=window_start,
        window_end=window_end,
    )
    cropped_predicted, prediction_audit = clip_segments_to_window(
        predicted,
        window_start=window_start,
        window_end=window_end,
    )
    return cropped_predicted, cropped_gold, {
        "policy": "gt-active",
        "start_sec": window_start,
        "end_sec": window_end,
        "timestamps_rebased": False,
        "gt": gt_audit,
        "prediction": prediction_audit,
    }


def temporal_iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """Compute temporal IoU for two half-open time intervals."""
    intersection = max(
        0.0,
        min(float(left["end"]), float(right["end"]))
        - max(float(left["start"]), float(right["start"])),
    )
    union = max(float(left["end"]), float(right["end"])) - min(
        float(left["start"]), float(right["start"])
    )
    return intersection / union if union > 0 else 0.0


def normalize_label(label: str) -> str:
    """Normalize label text for exact equality."""
    return " ".join(str(label).casefold().split())


def better_threshold_alignment(
    candidate: tuple[int, float, tuple[tuple[int, int, float], ...]],
    current: tuple[int, float, tuple[tuple[int, int, float], ...]],
) -> bool:
    """Return whether a thresholded temporal alignment is better.

    Thresholded temporal metrics maximize the number of valid one-to-one
    matches first, then total IoU, with a deterministic lexicographic final
    tie-breaker.

    Args:
        candidate: Candidate ``(match_count, total_iou, pairs)`` state.
        current: Current best state in the same format.

    Returns:
        Whether ``candidate`` should replace ``current``.
    """

    if candidate[0] != current[0]:
        return candidate[0] > current[0]
    if abs(candidate[1] - current[1]) > EPSILON:
        return candidate[1] > current[1]
    return candidate[2] < current[2]


def match_temporal_segments(
    predicted: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
    *,
    tiou_threshold: float,
) -> tuple[tuple[int, int, float], ...]:
    """Match segments by thresholded, ordered, one-to-one temporal overlap.

    Labels do not participate in correspondence. Dynamic programming
    maximizes valid match count first and total IoU second, so semantic scoring
    can be applied after a stable temporal alignment.

    Args:
        predicted: Predicted segments sorted by time.
        gold: Ground-truth segments sorted by time.
        tiou_threshold: Inclusive temporal-IoU threshold.

    Returns:
        Matched ``(prediction_index, gt_index, tiou)`` tuples.
    """
    pred_count = len(predicted)
    gt_count = len(gold)
    empty_state: tuple[
        int,
        float,
        tuple[tuple[int, int, float], ...],
    ] = (0, 0.0, ())
    table = [
        [empty_state for _ in range(gt_count + 1)]
        for _ in range(pred_count + 1)
    ]

    for pred_index in range(pred_count - 1, -1, -1):
        for gt_index in range(gt_count - 1, -1, -1):
            best = table[pred_index + 1][gt_index]
            skip_gt = table[pred_index][gt_index + 1]
            if better_threshold_alignment(skip_gt, best):
                best = skip_gt

            tiou = temporal_iou(predicted[pred_index], gold[gt_index])
            if tiou + EPSILON >= tiou_threshold:
                tail = table[pred_index + 1][gt_index + 1]
                pair = (pred_index, gt_index, tiou)
                matched = (
                    tail[0] + 1,
                    tail[1] + tiou,
                    (pair,) + tail[2],
                )
                if better_threshold_alignment(matched, best):
                    best = matched
            table[pred_index][gt_index] = best
    return table[0][0][2]


def better_temporal_alignment(
    candidate: tuple[float, int, tuple[tuple[int, int, float], ...]],
    current: tuple[float, int, tuple[tuple[int, int, float], ...]],
) -> bool:
    """Return whether a monotonic temporal alignment candidate is better.

    Alignments maximize total temporal IoU, then matched-pair count, and use
    lexicographic pair order as a deterministic final tie-breaker.
    """

    if abs(candidate[0] - current[0]) > EPSILON:
        return candidate[0] > current[0]
    if candidate[1] != current[1]:
        return candidate[1] > current[1]
    return candidate[2] < current[2]


def align_temporal_segments(
    predicted: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
) -> tuple[tuple[int, int, float], ...]:
    """Find an order-preserving one-to-one maximum-total-IoU alignment.

    This alignment adapts TimeLens-style single-window metrics to videos that
    contain multiple ordered atomic segments. A prediction and a GT segment
    can each participate in at most one pair, and zero-IoU pairs are omitted.

    Args:
        predicted: Predicted segments sorted by time.
        gold: Ground-truth segments sorted by time.

    Returns:
        Matched ``(prediction_index, gt_index, temporal_iou)`` tuples.
    """

    pred_count = len(predicted)
    gt_count = len(gold)
    empty_state: tuple[float, int, tuple[tuple[int, int, float], ...]] = (
        0.0,
        0,
        (),
    )
    table = [
        [empty_state for _ in range(gt_count + 1)]
        for _ in range(pred_count + 1)
    ]
    for pred_index in range(pred_count - 1, -1, -1):
        for gt_index in range(gt_count - 1, -1, -1):
            best = table[pred_index + 1][gt_index]
            skip_gt = table[pred_index][gt_index + 1]
            if better_temporal_alignment(skip_gt, best):
                best = skip_gt

            tiou = temporal_iou(predicted[pred_index], gold[gt_index])
            if tiou > EPSILON:
                tail = table[pred_index + 1][gt_index + 1]
                pair = (pred_index, gt_index, tiou)
                matched = (
                    tail[0] + tiou,
                    tail[1] + 1,
                    (pair,) + tail[2],
                )
                if better_temporal_alignment(matched, best):
                    best = matched
            table[pred_index][gt_index] = best
    return table[0][0][2]


def timelens_temporal_metrics(
    predicted: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
    *,
    tiou_thresholds: Sequence[float],
) -> dict[str, Any]:
    """Compute TimeLens-style R1 and mIoU for multi-segment videos.

    Every GT atomic segment is treated as one evaluation item. Segments are
    aligned monotonically one-to-one by maximum total IoU; unmatched GT items
    receive IoU zero. R1 is the fraction of GT items at or above a threshold,
    and mIoU is the mean aligned IoU over all GT items.

    Args:
        predicted: Predicted segments sorted by time.
        gold: Ground-truth segments sorted by time.
        tiou_thresholds: R1 IoU thresholds.

    Returns:
        Per-episode R1 counts, mIoU, and alignment details.
    """

    pairs = align_temporal_segments(predicted, gold)
    aligned_ious = [0.0] * len(gold)
    for _pred_index, gt_index, tiou in pairs:
        aligned_ious[gt_index] = tiou
    r1 = {}
    for threshold in tiou_thresholds:
        hit_count = sum(
            1 for tiou in aligned_ious if tiou + EPSILON >= threshold
        )
        r1[threshold_key(threshold)] = {
            "hits": hit_count,
            "total": len(gold),
            "score": hit_count / len(gold) if gold else 0.0,
        }
    return {
        "evaluation_unit": "ground_truth_segment",
        "matching": "order_preserving_one_to_one_max_total_iou",
        "r1": r1,
        "miou": sum(aligned_ious) / len(gold) if gold else 0.0,
        "aligned_iou_sum": sum(aligned_ious),
        "target_count": len(gold),
        "matches": temporal_match_payload(predicted, gold, pairs),
    }


def metric_values(tp: int, fp: int, fn: int) -> dict[str, Any]:
    """Compute precision, recall, and F1 from counts."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def segment_payload(segment: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable segment summary."""
    return {
        "index": int(segment["index"]),
        "start_sec": float(segment["start"]),
        "end_sec": float(segment["end"]),
        "label": str(segment["label"]),
        "subtask": segment.get("subtask_label"),
        "skill": segment.get("skill_label"),
    }


def temporal_match_payload(
    predicted: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
    pairs: Sequence[tuple[int, int, float]],
) -> list[dict[str, Any]]:
    """Return JSON-serializable temporal-only match details."""
    return [
        {
            "prediction": segment_payload(predicted[pred_index]),
            "gt": segment_payload(gold[gt_index]),
            "temporal_iou": tiou,
        }
        for pred_index, gt_index, tiou in pairs
    ]


def temporal_threshold_report(
    predicted: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
    pairs: Sequence[tuple[int, int, float]],
) -> dict[str, Any]:
    """Build one temporal-only threshold report.

    Args:
        predicted: Predicted segments sorted by time.
        gold: Ground-truth segments sorted by time.
        pairs: Accepted temporal-only matches.

    Returns:
        Metrics, matches, and unmatched segment payloads.
    """
    matched_pred = {pred_index for pred_index, _, _ in pairs}
    matched_gt = {gt_index for _, gt_index, _ in pairs}
    return {
        "metrics": metric_values(
            len(pairs),
            len(predicted) - len(pairs),
            len(gold) - len(pairs),
        ),
        "matches": temporal_match_payload(predicted, gold, pairs),
        "unmatched_predictions": [
            segment_payload(segment)
            for index, segment in enumerate(predicted)
            if index not in matched_pred
        ],
        "unmatched_gt": [
            segment_payload(segment)
            for index, segment in enumerate(gold)
            if index not in matched_gt
        ],
    }


def evaluate_skill_accuracy(
    predicted: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
    pairs: Sequence[tuple[int, int, float]],
) -> dict[str, Any]:
    """Evaluate exact normalized skill labels on temporally aligned pairs.

    Args:
        predicted: Predicted segments sorted by time.
        gold: Ground-truth segments sorted by time.
        pairs: Accepted temporal-only matches.

    Returns:
        Conditional accuracy counts and per-pair comparisons.
    """
    correct = 0
    evaluated_pairs = 0
    missing_prediction = 0
    unevaluable_gt = 0
    comparisons: list[dict[str, Any]] = []

    for pred_index, gt_index, tiou in pairs:
        predicted_label = predicted[pred_index].get("skill_label")
        gt_label = gold[gt_index].get("skill_label")
        is_correct: bool | None = None
        if gt_label is None:
            unevaluable_gt += 1
            status = "gt_unavailable"
        else:
            evaluated_pairs += 1
            if predicted_label is None:
                missing_prediction += 1
                is_correct = False
                status = "prediction_missing"
            else:
                is_correct = normalize_label(str(predicted_label)) == normalize_label(
                    str(gt_label)
                )
                if is_correct:
                    correct += 1
                status = "correct" if is_correct else "incorrect"
        comparisons.append(
            {
                "prediction_index": int(predicted[pred_index]["index"]),
                "gt_index": int(gold[gt_index]["index"]),
                "temporal_iou": tiou,
                "prediction": predicted_label,
                "gt": gt_label,
                "is_correct": is_correct,
                "status": status,
            }
        )

    return {
        "score": correct / evaluated_pairs if evaluated_pairs else None,
        "correct": correct,
        "evaluated_pairs": evaluated_pairs,
        "missing_prediction": missing_prediction,
        "unevaluable_gt": unevaluable_gt,
        "comparisons": comparisons,
    }


def format_diagnostics(
    episode_reports: Sequence[Mapping[str, Any]],
    parse_errors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize parse validity, empty outputs, and missing fields.

    Args:
        episode_reports: Evaluated episode reports.
        parse_errors: Prediction parse errors for this run or group.

    Returns:
        Episode- and segment-level format diagnostics.
    """

    episode_count = len(episode_reports)
    parse_error_episodes = {
        str(error["episode_id"])
        for error in parse_errors
        if error.get("episode_id") is not None
    }
    present_count = sum(bool(episode["prediction_present"]) for episode in episode_reports)
    valid_count = sum(
        bool(episode["prediction_present"])
        and str(episode["episode_id"]) not in parse_error_episodes
        for episode in episode_reports
    )
    empty_count = sum(
        int(
            episode.get(
                "raw_prediction_segment_count",
                episode["prediction_segment_count"],
            )
        )
        == 0
        for episode in episode_reports
    )
    return {
        "episode_count": episode_count,
        "prediction_present_episode_count": present_count,
        "format_valid_episode_count": valid_count,
        "format_valid_episode_rate": valid_count / episode_count if episode_count else 0.0,
        "parse_error_count": len(parse_errors),
        "parse_error_episode_count": len(parse_error_episodes),
        "empty_prediction_episode_count": empty_count,
        "empty_prediction_episode_rate": empty_count / episode_count if episode_count else 0.0,
        "missing_subtask_segment_count": sum(
            int(episode["missing_subtask_segment_count"])
            for episode in episode_reports
        ),
        "missing_skill_segment_count": sum(
            int(episode["missing_skill_segment_count"])
            for episode in episode_reports
        ),
    }


def aggregate_evaluation_window(
    episode_reports: Sequence[Mapping[str, Any]],
    *,
    policy: str,
) -> dict[str, Any]:
    """Aggregate per-episode evaluation-window audit counts.

    Args:
        episode_reports: Evaluated episode reports.
        policy: Applied evaluation-window policy.

    Returns:
        Run-level counts describing window effects on GT and predictions.
    """
    gt_dropped = 0
    gt_clipped = 0
    prediction_dropped = 0
    prediction_clipped = 0
    changed_episode_count = 0
    for episode_report in episode_reports:
        audit = episode_report["evaluation_window"]
        gt_audit = audit["gt"]
        prediction_audit = audit["prediction"]
        gt_dropped += int(gt_audit["dropped_segment_count"])
        gt_clipped += int(gt_audit["clipped_segment_count"])
        prediction_dropped += int(prediction_audit["dropped_segment_count"])
        prediction_clipped += int(prediction_audit["clipped_segment_count"])
        if any(
            int(value) > 0
            for value in (
                gt_audit["dropped_segment_count"],
                gt_audit["clipped_segment_count"],
                prediction_audit["dropped_segment_count"],
                prediction_audit["clipped_segment_count"],
            )
        ):
            changed_episode_count += 1

    return {
        "policy": policy,
        "episode_count": len(episode_reports),
        "changed_episode_count": changed_episode_count,
        "gt_dropped_segment_count": gt_dropped,
        "gt_clipped_segment_count": gt_clipped,
        "prediction_dropped_segment_count": prediction_dropped,
        "prediction_clipped_segment_count": prediction_clipped,
    }


def threshold_key(value: float) -> str:
    """Return a stable JSON key for a threshold."""
    return f"{value:.2f}"


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def write_episode_jsonl(path: Path, report: Mapping[str, Any]) -> None:
    """Write per-run, per-episode evaluation rows as JSONL.

    Args:
        path: Output JSONL path.
        report: Full evaluation report.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for run in report.get("runs", []):
            if not isinstance(run, Mapping):
                continue
            run_name = str(run.get("name", ""))
            for episode in run.get("episodes", []):
                if not isinstance(episode, Mapping):
                    continue
                row = {"run_name": run_name, **dict(episode)}
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

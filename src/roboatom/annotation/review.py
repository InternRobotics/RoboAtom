#!/usr/bin/env python3
"""Offline model reviewer for SubtaskAnnotator JSONL annotations.

The script reads annotation JSONL rows, samples evidence frames around each
subtask, calls an OpenAI-compatible vision model, and writes a compact review
JSONL that can be loaded by app.py --mode model-review.
"""

from __future__ import annotations

import argparse
import base64
import configparser
import io
import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import av
except ImportError:  # pragma: no cover - reported by require_dependencies
    av = None

try:
    import cv2
except ImportError:  # pragma: no cover - reported by require_dependencies
    cv2 = None

try:
    import httpx
except ImportError:  # pragma: no cover - reported by require_dependencies
    httpx = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - reported by require_dependencies
    np = None

try:
    from openai import APIConnectionError, APITimeoutError, BadRequestError, OpenAI
except ImportError:  # pragma: no cover - reported by require_dependencies
    APIConnectionError = APITimeoutError = BadRequestError = OpenAI = None

try:
    from mmengine import fileio
except ImportError:  # pragma: no cover - optional remote reader
    fileio = None

from roboatom.annotation.prompt_logging import append_text_prompt, resolve_prompt_log_path


DEFAULT_BASE_URL = ""
DEFAULT_MODEL = ""
DEFAULT_API_KEY = ""
DEFAULT_VIEW_KEY = "head"
DEFAULT_FALLBACK_VIEW_KEYS = ["hand_right", "hand_left"]
DEFAULT_BOUNDARY_OFFSETS = [-45, -15, 0, 15, 45]
DEFAULT_MAX_GLOBAL_FRAMES = 24
DEFAULT_REQUEST_TIMEOUT = 3000.0
DEFAULT_CONNECT_TIMEOUT = 60.0
DEFAULT_STREAM_LOG_CHARS = 200
DEFAULT_SKILL_OPTIONS = [
    "Pick",
    "Place",
    "Press",
    "Push",
    "Pull",
    "Move",
    "Fold",
    "Wipe",
    "Close",
    "Open",
    "Pour",
    "Cut",
    "Screw",
    "Stack",
    "Handover",
    "Sweep",
]
ISSUE_TYPES = {
    "boundary_error",
    "wrong_object",
    "wrong_action",
    "wrong_bin",
    "missing_action",
    "extra_action",
    "naming_inconsistent",
    "format_error",
    "unclear_visual",
}
VERDICTS = {"correct", "incorrect", "uncertain"}
IMAGE_EXT = ".jpg"

logger = logging.getLogger("model_review_score")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def require_dependencies() -> None:
    missing = []
    if OpenAI is None:
        missing.append("openai")
    if httpx is None:
        missing.append("httpx")
    if av is None:
        missing.append("av")
    if cv2 is None:
        missing.append("opencv-python-headless")
    if np is None:
        missing.append("numpy")
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing dependencies: {joined}. Install them from requirements.txt before running this scorer."
        )


def env_first(names: Sequence[str], default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return default


def default_config_path() -> Path:
    raise ValueError("Annotation configuration must be supplied explicitly")


def parse_csv_ints(raw: str, default: Sequence[int]) -> List[int]:
    values: List[int] = []
    for part in str(raw or "").split(","):
        text = part.strip()
        if not text:
            continue
        try:
            values.append(int(text))
        except ValueError as exc:
            raise ValueError(f"Invalid integer in comma-separated list: {text}") from exc
    return values or list(default)


def parse_csv_strings(raw: str) -> List[str]:
    result: List[str] = []
    seen = set()
    for part in str(raw or "").split(","):
        text = part.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def parse_multiline_options(raw: str) -> List[str]:
    result: List[str] = []
    seen = set()
    for line in str(raw or "").splitlines():
        text = line.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def load_skill_options(config_path: Path) -> List[str]:
    if not config_path.is_file():
        logger.warning("Config file not found for skill options, using defaults: %s", config_path)
        return list(DEFAULT_SKILL_OPTIONS)
    parser = configparser.ConfigParser()
    parser.read(config_path, encoding="utf-8")
    if not parser.has_section("annotation"):
        return list(DEFAULT_SKILL_OPTIONS)
    options = parse_multiline_options(parser.get("annotation", "skill_options", fallback=""))
    return options or list(DEFAULT_SKILL_OPTIONS)


def resolve_base_url(arg_value: str) -> str:
    if arg_value.strip():
        return arg_value.strip()
    return env_first(["NEWAPI_BASE_URL", "DASHSCOPE_BASE_URL", "VLLM_BASE_URL"], DEFAULT_BASE_URL)


def resolve_model(arg_value: str) -> str:
    if arg_value.strip():
        return arg_value.strip()
    return env_first(["NEWAPI_MODEL", "DASHSCOPE_MODEL", "OPENAI_MODEL", "VLLM_MODEL"], DEFAULT_MODEL)


def resolve_api_key(arg_value: str) -> str:
    if arg_value.strip():
        raise ValueError("API keys must be provided through environment variables")
    return env_first(["NEWAPI_API_KEY", "DASHSCOPE_API_KEY", "VLLM_API_KEY", "OPENAI_API_KEY"], DEFAULT_API_KEY)


def safe_api_key_tail(api_key: str) -> str:
    key = str(api_key or "").strip()
    if not key:
        return "EMPTY"
    return f"***{key[-6:]}" if len(key) > 6 else "***"


def make_client(
    base_url: str,
    api_key: str,
    *,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
) -> OpenAI:
    if not api_key:
        raise RuntimeError("Missing API key. Set NEWAPI_API_KEY/DASHSCOPE_API_KEY/OPENAI_API_KEY or pass --api-key.")
    http_client = httpx.Client(
        trust_env=True,
        timeout=httpx.Timeout(float(request_timeout), connect=float(connect_timeout)),
    )
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=http_client,
        max_retries=0,
    )


def iter_exception_chain(exc: BaseException) -> List[BaseException]:
    seen = set()
    chain: List[BaseException] = []
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def api_create_with_retry(
    client: OpenAI,
    *,
    max_retries: int,
    debug_meta: Dict[str, Any],
    **kwargs: Any,
) -> Any:
    last_error: Optional[BaseException] = None
    sample_id = debug_meta.get("sample_id", "unknown")
    frame_count = debug_meta.get("frame_count", "unknown")
    base_url = debug_meta.get("base_url", "unknown")
    api_key_tail = debug_meta.get("api_key_tail", "unknown")
    logger.info(
        "LLM request | sample_id=%s | model=%s | base_url=%s | api_key=%s | images=%s",
        sample_id,
        kwargs.get("model"),
        base_url,
        api_key_tail,
        frame_count,
    )

    for attempt in range(max_retries):
        try:
            logger.info("LLM attempt %s/%s | sample_id=%s", attempt + 1, max_retries, sample_id)
            response = client.chat.completions.create(**kwargs)
            usage = getattr(response, "usage", None)
            request_id = getattr(response, "_request_id", None)
            logger.info("LLM response | sample_id=%s | request_id=%s | usage=%s", sample_id, request_id, usage)
            return response
        except BadRequestError as exc:
            logger.error(
                "LLM BadRequest | sample_id=%s | status=%s | request_id=%s | body=%s",
                sample_id,
                getattr(exc, "status_code", None),
                getattr(exc, "request_id", None),
                getattr(exc, "body", None),
            )
            raise
        except (APIConnectionError, APITimeoutError, ConnectionError, TimeoutError) as exc:
            last_error = exc
            if attempt >= max_retries - 1:
                logger.error("LLM request failed finally | sample_id=%s | error=%s", sample_id, exc)
                raise
            wait_sec = attempt + 1
            logger.warning(
                "LLM request failed | sample_id=%s | attempt=%s/%s | retry_in=%ss | error=%s",
                sample_id,
                attempt + 1,
                max_retries,
                wait_sec,
                exc,
            )
            time.sleep(wait_sec)
    raise RuntimeError(f"LLM request failed: {last_error}")


def is_retryable_llm_exception(exc: BaseException) -> bool:
    for err in iter_exception_chain(exc):
        if APIConnectionError is not None and isinstance(err, APIConnectionError):
            return True
        if APITimeoutError is not None and isinstance(err, APITimeoutError):
            return True
        if isinstance(err, (ConnectionError, TimeoutError)):
            return True
        if httpx is not None and isinstance(err, (httpx.TimeoutException, httpx.TransportError)):
            return True
    return False


def preview_text(text: str, limit: int = 500) -> Tuple[str, str]:
    normalized = str(text or "").replace("\n", "\\n")
    if len(normalized) <= limit:
        return normalized, normalized
    return normalized[:limit], normalized[-limit:]


def stream_delta_content(event: Any) -> str:
    choices = getattr(event, "choices", None)
    if choices is None and isinstance(event, dict):
        choices = event.get("choices")
    if not choices:
        return ""

    choice = choices[0]
    delta = getattr(choice, "delta", None)
    if delta is None and isinstance(choice, dict):
        delta = choice.get("delta")
    if delta is None:
        return ""

    content = getattr(delta, "content", None)
    if content is None and isinstance(delta, dict):
        content = delta.get("content")
    return str(content or "")


def stream_completion_text(client: OpenAI, *, debug_meta: Dict[str, Any], **kwargs: Any) -> str:
    sample_id = debug_meta.get("sample_id", "unknown")
    log_every = max(1, int(debug_meta.get("stream_log_chars", DEFAULT_STREAM_LOG_CHARS)))
    started = time.monotonic()
    first_chunk_at: Optional[float] = None
    chunks: List[str] = []
    total_chars = 0
    chars_since_log = 0
    event_count = 0
    content_chunk_count = 0

    logger.info("LLM stream start | sample_id=%s", sample_id)
    try:
        stream = client.chat.completions.create(stream=True, **kwargs)
        for event in stream:
            event_count += 1
            text = stream_delta_content(event)
            if not text:
                continue

            content_chunk_count += 1
            if first_chunk_at is None:
                first_chunk_at = time.monotonic()
                logger.info(
                    "LLM stream first chunk | sample_id=%s | latency=%.1fs",
                    sample_id,
                    first_chunk_at - started,
                )

            chunks.append(text)
            total_chars += len(text)
            chars_since_log += len(text)
            if chars_since_log >= log_every:
                logger.info(
                    "LLM stream progress | sample_id=%s | chunks=%s | events=%s | chars=%s | elapsed=%.1fs",
                    sample_id,
                    content_chunk_count,
                    event_count,
                    total_chars,
                    time.monotonic() - started,
                )
                chars_since_log = 0
    except Exception as exc:
        partial_text = "".join(chunks).strip()
        head, tail = preview_text(partial_text)
        logger.error(
            "LLM stream failed | sample_id=%s | chunks=%s | events=%s | chars=%s | elapsed=%.1fs | "
            "partial_head=%s | partial_tail=%s | error=%s",
            sample_id,
            content_chunk_count,
            event_count,
            len(partial_text),
            time.monotonic() - started,
            head,
            tail,
            exc,
        )
        raise

    response_text = "".join(chunks).strip()
    logger.info(
        "LLM stream done | sample_id=%s | chunks=%s | events=%s | chars=%s | elapsed=%.1fs",
        sample_id,
        content_chunk_count,
        event_count,
        len(response_text),
        time.monotonic() - started,
    )
    return response_text


def api_completion_text_with_retry(
    client: OpenAI,
    *,
    max_retries: int,
    stream: bool,
    debug_meta: Dict[str, Any],
    **kwargs: Any,
) -> str:
    last_error: Optional[BaseException] = None
    sample_id = debug_meta.get("sample_id", "unknown")
    frame_count = debug_meta.get("frame_count", "unknown")
    base_url = debug_meta.get("base_url", "unknown")
    api_key_tail = debug_meta.get("api_key_tail", "unknown")
    logger.info(
        "LLM request | sample_id=%s | model=%s | base_url=%s | api_key=%s | images=%s | "
        "stream=%s | max_tokens=%s | temperature=%s | top_p=%s | max_retries=%s",
        sample_id,
        kwargs.get("model"),
        base_url,
        api_key_tail,
        frame_count,
        stream,
        kwargs.get("max_tokens"),
        kwargs.get("temperature"),
        kwargs.get("top_p"),
        max_retries,
    )

    for attempt in range(max_retries):
        started = time.monotonic()
        try:
            logger.info("LLM attempt %s/%s | sample_id=%s", attempt + 1, max_retries, sample_id)
            if stream:
                return stream_completion_text(client, debug_meta=debug_meta, **kwargs)

            response = client.chat.completions.create(**kwargs)
            usage = getattr(response, "usage", None)
            request_id = getattr(response, "_request_id", None)
            response_text = str(response.choices[0].message.content or "").strip()
            logger.info(
                "LLM response | sample_id=%s | request_id=%s | usage=%s | chars=%s | elapsed=%.1fs",
                sample_id,
                request_id,
                usage,
                len(response_text),
                time.monotonic() - started,
            )
            return response_text
        except BadRequestError as exc:
            logger.error(
                "LLM BadRequest | sample_id=%s | status=%s | request_id=%s | body=%s",
                sample_id,
                getattr(exc, "status_code", None),
                getattr(exc, "request_id", None),
                getattr(exc, "body", None),
            )
            raise
        except Exception as exc:
            if not is_retryable_llm_exception(exc):
                raise
            last_error = exc
            if attempt >= max_retries - 1:
                logger.error("LLM request failed finally | sample_id=%s | error=%s", sample_id, exc)
                raise
            wait_sec = attempt + 1
            logger.warning(
                "LLM request failed | sample_id=%s | attempt=%s/%s | retry_in=%ss | elapsed=%.1fs | error=%s",
                sample_id,
                attempt + 1,
                max_retries,
                wait_sec,
                time.monotonic() - started,
                exc,
            )
            time.sleep(wait_sec)
    raise RuntimeError(f"LLM request failed: {last_error}")


def read_video_bytes(video_path: str) -> bytes:
    path = str(video_path)
    local_path = Path(path).expanduser()
    if not local_path.is_absolute():
        local_path = Path(os.environ.get("ROBOATOM_MEDIA_ROOT", ".")) / local_path
    if local_path.is_file():
        return local_path.read_bytes()
    if fileio is None:
        raise RuntimeError(f"Video is not local and mmengine is unavailable: {video_path}")
    data = fileio.get(path)
    if not data:
        raise RuntimeError(f"Empty video bytes: {video_path}")
    return data


def probe_video(video_path: str, video_bytes: bytes) -> Tuple[float, int]:
    with av.open(io.BytesIO(video_bytes)) as container:
        if not container.streams.video:
            raise RuntimeError(f"No video stream found: {video_path}")
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        total_frames = int(stream.frames or 0)
        if total_frames <= 0 and stream.duration and stream.time_base and stream.average_rate:
            total_frames = int(float(stream.duration * stream.time_base * stream.average_rate))
        if total_frames <= 0:
            total_frames = sum(1 for _ in container.decode(video=0))
    if total_frames <= 0:
        raise RuntimeError(f"No decodable frames found: {video_path}")
    return fps or 30.0, total_frames


def decode_target_frames(video_path: str, video_bytes: bytes, target_frames: Iterable[int]) -> Dict[int, Any]:
    targets = sorted({int(frame) for frame in target_frames if int(frame) >= 0})
    decoded: Dict[int, Any] = {}
    if not targets:
        return decoded
    target_set = set(targets)
    max_target = targets[-1]

    with av.open(io.BytesIO(video_bytes)) as container:
        if not container.streams.video:
            raise RuntimeError(f"No video stream found: {video_path}")
        for frame_id, frame in enumerate(container.decode(video=0)):
            if frame_id in target_set:
                decoded[frame_id] = frame.to_ndarray(format="bgr24")
                if len(decoded) >= len(target_set):
                    break
            if frame_id > max_target:
                break
    return decoded


def is_annotation_candidate(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("action_config"), list)
        and isinstance(value.get("tasks"), list)
        and ("length" in value or "frames" in value)
    )


def iter_annotation_records(input_jsonl: Path) -> Iterable[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
    with input_jsonl.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if isinstance(row, list):
                if row and is_annotation_candidate(row[0]):
                    companion = row[1] if len(row) > 1 and isinstance(row[1], dict) else {}
                    yield line_no, row[0], companion
                continue
            if is_annotation_candidate(row):
                yield line_no, row, {}


def iter_annotation_rows(input_jsonl: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    for line_no, annotation, _companion in iter_annotation_records(input_jsonl):
        yield line_no, annotation


def detect_interval_style(segments: List[Dict[str, Any]], frames: int) -> str:
    if not segments or frames <= 0:
        return "open"
    looks_open = bool(segments and int(segments[-1].get("end_frame", -1)) == frames)
    looks_closed = bool(segments and int(segments[-1].get("end_frame", -1)) == frames - 1)
    for prev, nxt in zip(segments, segments[1:]):
        prev_end = int(prev.get("end_frame", 0))
        next_start = int(nxt.get("start_frame", 0))
        if next_start != prev_end:
            looks_open = False
        if next_start != prev_end + 1:
            looks_closed = False
    if looks_closed and not looks_open:
        return "closed"
    return "open"


def get_video_path(annotation: Dict[str, Any], view_key: str) -> str:
    views = annotation.get("views")
    if isinstance(views, dict):
        if view_key in views and str(views[view_key]).strip():
            return str(views[view_key]).strip()
        for key in (DEFAULT_VIEW_KEY, "main"):
            if key in views and str(views[key]).strip():
                return str(views[key]).strip()
        for value in views.values():
            if str(value).strip():
                return str(value).strip()
    return str(annotation.get("path") or annotation.get("episode_index") or "").strip()


def get_view_paths(annotation: Dict[str, Any]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    views = annotation.get("views")
    if isinstance(views, dict):
        for key, value in views.items():
            text = str(value or "").strip()
            if str(key).strip() and text:
                result[str(key).strip()] = text
    fallback = str(annotation.get("path") or annotation.get("episode_index") or "").strip()
    if fallback and DEFAULT_VIEW_KEY not in result:
        result[DEFAULT_VIEW_KEY] = fallback
    return result


def resolve_fallback_view_keys(annotation: Dict[str, Any], primary_view_key: str, raw_keys: str) -> List[str]:
    explicit_keys = parse_csv_strings(raw_keys)
    if explicit_keys:
        candidates = explicit_keys
    else:
        view_order = annotation.get("view_order")
        if isinstance(view_order, list):
            candidates = [str(key).strip() for key in view_order if str(key).strip()]
        else:
            candidates = []
        if not candidates:
            candidates = list(DEFAULT_FALLBACK_VIEW_KEYS)
    result: List[str] = []
    seen = {primary_view_key}
    for key in candidates:
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def normalize_segments_for_review(annotation: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    frames = int(annotation.get("length", annotation.get("frames", 0)))
    raw_segments = annotation.get("action_config") or []
    interval_style = detect_interval_style(raw_segments, frames)
    result: List[Dict[str, Any]] = []
    for idx, segment in enumerate(raw_segments):
        start = int(segment.get("start_frame", 0))
        raw_end = int(segment.get("end_frame", start))
        end_inclusive = raw_end if interval_style == "closed" else raw_end - 1
        end_inclusive = clamp(end_inclusive, start, max(frames - 1, 0))
        result.append(
            {
                "seg_id": int(segment.get("seg_id", idx)),
                "start_frame": clamp(start, 0, max(frames - 1, 0)),
                "end_frame": end_inclusive,
                "action_text": str(segment.get("action_text", "")).strip(),
                "skill": str(segment.get("skill", "")).strip(),
            }
        )
    return result, interval_style


def uniform_frame_indices(total_frames: int, max_count: int) -> List[int]:
    total = max(0, int(total_frames))
    count = max(0, int(max_count))
    if total <= 0 or count <= 0:
        return []
    if total <= count:
        return list(range(total))
    if count == 1:
        return [0]
    last = total - 1
    frames = [int(round(index * last / (count - 1))) for index in range(count)]
    result: List[int] = []
    seen = set()
    for frame in frames:
        safe_frame = clamp(frame, 0, last)
        if safe_frame in seen:
            continue
        seen.add(safe_frame)
        result.append(safe_frame)
    return result


def add_frame_manifest_item(
    manifest: List[Dict[str, Any]],
    seen: set[Tuple[str, int]],
    *,
    view_key: str,
    frame: int,
    role: str,
    total_frames: int,
    segment: Optional[Dict[str, Any]] = None,
) -> None:
    safe_frame = clamp(int(frame), 0, max(total_frames - 1, 0))
    key = (str(view_key), safe_frame)
    if key in seen:
        return
    seen.add(key)
    item: Dict[str, Any] = {
        "view_key": str(view_key),
        "frame": safe_frame,
        "role": str(role),
    }
    if segment is not None:
        item.update(
            {
                "seg_id": int(segment["seg_id"]),
                "start_frame": int(segment.get("start_frame", 0)),
                "end_frame": int(segment.get("end_frame", 0)),
                "action_text": str(segment.get("action_text", "")),
                "skill": str(segment.get("skill", "")),
            }
        )
    manifest.append(item)


def append_frame_manifest_for_view(
    manifest: List[Dict[str, Any]],
    seen: set[Tuple[str, int]],
    *,
    segments: List[Dict[str, Any]],
    total_frames: int,
    view_key: str,
    max_global_frames: int,
    boundary_offsets: Sequence[int],
    include_global: bool,
) -> None:
    for segment in segments:
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        if end < start:
            end = start
        span = max(1, end - start + 1)
        for pct in (0.25, 0.5, 0.75):
            frame = start + int(round((span - 1) * pct))
            add_frame_manifest_item(
                manifest,
                seen,
                view_key=view_key,
                frame=frame,
                role=f"inside_{int(pct * 100)}",
                total_frames=total_frames,
                segment=segment,
            )

    by_seg = {int(segment["seg_id"]): segment for segment in segments}
    for prev, nxt in zip(segments, segments[1:]):
        prev_id = int(prev["seg_id"])
        next_id = int(nxt["seg_id"])
        boundary_frame = int(nxt["start_frame"])
        for offset in boundary_offsets:
            segment = by_seg.get(next_id if offset >= 0 else prev_id)
            add_frame_manifest_item(
                manifest,
                seen,
                view_key=view_key,
                frame=boundary_frame + int(offset),
                role=f"boundary_{prev_id}_{next_id}{int(offset):+d}",
                total_frames=total_frames,
                segment=segment,
            )

    if include_global:
        for index, frame in enumerate(uniform_frame_indices(total_frames, max_global_frames)):
            add_frame_manifest_item(
                manifest,
                seen,
                view_key=view_key,
                frame=frame,
                role=f"global_{index:03d}",
                total_frames=total_frames,
                segment=None,
            )


def build_frame_manifest(
    segments: List[Dict[str, Any]],
    total_frames: int,
    boundary_window: Optional[int] = None,
    *,
    view_key: str = DEFAULT_VIEW_KEY,
    max_global_frames: int = DEFAULT_MAX_GLOBAL_FRAMES,
    boundary_offsets: Optional[Sequence[int]] = None,
    include_global: bool = True,
) -> List[Dict[str, Any]]:
    del boundary_window  # Legacy argument retained for callers from older scorer versions.
    manifest: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int]] = set()
    append_frame_manifest_for_view(
        manifest,
        seen,
        segments=segments,
        total_frames=total_frames,
        view_key=view_key,
        max_global_frames=max_global_frames,
        boundary_offsets=boundary_offsets or DEFAULT_BOUNDARY_OFFSETS,
        include_global=include_global,
    )
    manifest.sort(key=lambda item: (str(item.get("view_key", "")), int(item["frame"]), str(item["role"])))
    return manifest


def build_multi_view_frame_manifest(
    *,
    segments: List[Dict[str, Any]],
    total_frames: int,
    view_keys: Sequence[str],
    max_global_frames: int,
    boundary_offsets: Sequence[int],
    include_global: bool,
) -> List[Dict[str, Any]]:
    manifest: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int]] = set()
    for view_key in view_keys:
        append_frame_manifest_for_view(
            manifest,
            seen,
            segments=segments,
            total_frames=total_frames,
            view_key=str(view_key),
            max_global_frames=max_global_frames,
            boundary_offsets=boundary_offsets,
            include_global=include_global,
        )
    manifest.sort(key=lambda item: (str(item.get("view_key", "")), int(item["frame"]), str(item["role"])))
    return manifest


def safe_sample_dir_name(line_no: int, sample_id: str) -> str:
    name = Path(str(sample_id or f"line_{line_no}")).stem
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or f"line_{line_no}"
    return f"line_{line_no:06d}_{safe[:80]}"


def resize_to_height(frame: Any, target_height: int) -> Any:
    height, width = frame.shape[:2]
    if height <= 0:
        return frame
    scale = target_height / height
    target_width = max(1, int(round(width * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def render_tile(frame: Any, meta: Dict[str, Any], target_height: int) -> Any:
    frame = resize_to_height(frame, target_height)
    panel_height = 64
    canvas = cv2.copyMakeBorder(
        frame,
        panel_height,
        0,
        0,
        0,
        borderType=cv2.BORDER_CONSTANT,
        value=(18, 18, 18),
    )
    view_key = str(meta.get("view_key", DEFAULT_VIEW_KEY))
    seg_id = meta.get("seg_id")
    seg_label = f"seg {seg_id}" if seg_id is not None else "global"
    lines = [
        f"view={view_key} | {seg_label} | frame {meta['frame']}",
        f"role={meta['role']}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 24
    for line in lines:
        cv2.putText(canvas, line, (10, y), font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28
    return canvas


def make_contact_sheets(
    *,
    video_path: Optional[str] = None,
    video_bytes: Optional[bytes] = None,
    video_sources: Optional[Dict[str, Tuple[str, bytes]]] = None,
    frame_manifest: List[Dict[str, Any]],
    output_dir: Path,
    target_height: int,
    frames_per_sheet: int,
    columns: int,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if video_sources is None:
        if video_path is None or video_bytes is None:
            raise RuntimeError("make_contact_sheets requires video_sources or video_path/video_bytes")
        video_sources = {DEFAULT_VIEW_KEY: (video_path, video_bytes)}

    decoded: Dict[Tuple[str, int], Any] = {}
    target_frames_by_view: Dict[str, List[int]] = {}
    for item in frame_manifest:
        view_key = str(item.get("view_key", DEFAULT_VIEW_KEY))
        target_frames_by_view.setdefault(view_key, []).append(int(item["frame"]))

    for view_key, target_frames in target_frames_by_view.items():
        source = video_sources.get(view_key)
        if source is None:
            logger.warning("Missing video source for view %s", view_key)
            continue
        source_path, source_bytes = source
        view_decoded = decode_target_frames(source_path, source_bytes, target_frames)
        for frame_id, frame in view_decoded.items():
            decoded[(view_key, int(frame_id))] = frame

    if not decoded:
        raise RuntimeError("No evidence frames decoded for contact sheets")

    tiles: List[Any] = []
    for meta in frame_manifest:
        view_key = str(meta.get("view_key", DEFAULT_VIEW_KEY))
        frame = decoded.get((view_key, int(meta["frame"])))
        if frame is None:
            logger.warning(
                "Missing decoded frame %s for view %s role %s",
                meta["frame"],
                view_key,
                meta.get("role"),
            )
            continue
        tiles.append(render_tile(frame, meta, target_height))

    if not tiles:
        raise RuntimeError(f"No contact-sheet tiles rendered for {video_path}")

    sheet_paths: List[Path] = []
    per_sheet = max(1, int(frames_per_sheet))
    cols = max(1, int(columns))
    for sheet_index, start in enumerate(range(0, len(tiles), per_sheet)):
        page_tiles = tiles[start:start + per_sheet]
        tile_h = max(tile.shape[0] for tile in page_tiles)
        tile_w = max(tile.shape[1] for tile in page_tiles)
        rows = math.ceil(len(page_tiles) / cols)
        sheet = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
        sheet[:, :] = (18, 18, 18)
        for idx, tile in enumerate(page_tiles):
            row = idx // cols
            col = idx % cols
            y = row * tile_h
            x = col * tile_w
            sheet[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
        sheet_path = output_dir / f"sheet_{sheet_index:03d}{IMAGE_EXT}"
        if not cv2.imwrite(str(sheet_path), sheet):
            raise RuntimeError(f"Failed to write contact sheet: {sheet_path}")
        sheet_paths.append(sheet_path)
    return sheet_paths


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def extract_companion_context(companion_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(companion_metadata, dict):
        return {}

    context: Dict[str, Any] = {}
    official_task = str(companion_metadata.get("official_task_instruction", "")).strip()
    if official_task:
        context["official_task_instruction"] = official_task

    inventory = companion_metadata.get("ground_truth_inventory")
    if isinstance(inventory, dict) and inventory:
        context["ground_truth_inventory"] = inventory

    return context


def build_review_prompt(
    *,
    sample_id: str,
    task_text: str,
    total_frames: int,
    fps: float,
    segments: List[Dict[str, Any]],
    frame_manifest: List[Dict[str, Any]],
    skill_options: Sequence[str],
    available_view_keys: Sequence[str],
    review_pass: str,
    companion_metadata: Optional[Dict[str, Any]] = None,
    allow_needs_more_views: bool = True,
) -> str:
    segment_lines = [
        {
            "seg_id": segment["seg_id"],
            "start_frame": segment["start_frame"],
            "end_frame": segment["end_frame"],
            "skill": segment["skill"],
            "action_text": segment["action_text"],
        }
        for segment in segments
    ]
    frame_lines = [
        {
            "view_key": item.get("view_key", DEFAULT_VIEW_KEY),
            "seg_id": item["seg_id"],
            "frame": item["frame"],
            "role": item["role"],
        }
        for item in frame_manifest
        if "seg_id" in item
    ]
    global_frame_lines = [
        {
            "view_key": item.get("view_key", DEFAULT_VIEW_KEY),
            "frame": item["frame"],
            "role": item["role"],
        }
        for item in frame_manifest
        if "seg_id" not in item
    ]
    needs_more_views_rule = (
        'Set "needs_more_views": true only when the current view cannot decide a concrete '
        "grasp/release/boundary/object issue and another available camera view may resolve it."
        if allow_needs_more_views
        else 'Set "needs_more_views": false; no additional camera views will be provided in this pass.'
    )
    companion_context = extract_companion_context(companion_metadata)
    companion_context_block = ""
    if companion_context:
        companion_context_block = f"""
Additional task context (ignore any old evaluation scores or prior grading fields):
{json.dumps(companion_context, ensure_ascii=False, indent=2)}
"""
    return f"""
You are a strict episode-level reviewer for human subtask annotations of a robot manipulation video.

Review only annotation quality. Do not grade delivery time, service plan, or aesthetics.
First understand the whole episode and task flow, then review every segment in context.

Input:
- sample_id: {sample_id}
- task: {task_text}
- total_frames: {total_frames}
- fps: {fps}
- review_pass: {review_pass}
- available_view_keys: {json.dumps(list(available_view_keys), ensure_ascii=False)}
- allowed_skills: {json.dumps(list(skill_options), ensure_ascii=False)}

Human annotation segments use inclusive frame ranges:
{json.dumps(segment_lines, ensure_ascii=False, indent=2)}

The attached contact-sheet images contain sampled evidence frames. Each tile has:
- view key
- seg id
- exact frame id
- sampling role, such as global_000, inside_50, boundary_2_3+10

Segment evidence frame manifest:
{json.dumps(frame_lines, ensure_ascii=False, indent=2)}

Global episode evidence frame manifest:
{json.dumps(global_frame_lines, ensure_ascii=False, indent=2)}

{companion_context_block}

Judge whether the human annotation is correct along two axes:
1. Description accuracy: skill, action_text, object identity, target/bin/container, and wording.
2. Segmentation accuracy: whether each segment contains exactly one atomic subtask and whether adjacent boundaries split at the correct action-event transition.

Mark incorrect if you find any issue that needs human repair:
- boundary_error: the boundary cuts an atomic action event incorrectly, merges Pick and Place, puts grasp/lift/release/stabilize events in the wrong segment, or lets one segment contain the core event of another subtask.
- wrong_object: action_text names the wrong object, material, color, or manipulated item.
- wrong_action: action_text or skill names the wrong action, or skill is outside/contradicts the allowed skill set.
- wrong_bin: the destination bin/container/color/location is wrong.
- missing_action: an obvious pick/place/action is absent.
- extra_action: a segment describes an action not shown.
- naming_inconsistent: the same physical object is named inconsistently across related segments, especially Pick/Place pairs.
- format_error: text has spelling/format problems that need cleanup.
- unclear_visual: visual evidence is insufficient or occluded after careful inspection.

Segmentation rules:
- Boundaries are important. Do not judge them by a fixed frame tolerance. Judge whether the atomic action event is assigned to the correct segment.
- For Pick, the segment should contain the grasp/lift event where the gripper changes from not holding to stably holding the object.
- For Place, the segment should contain the placement/release event where the held object is released and becomes stable at the target.
- Pick and Place should not be collapsed into one segment. A Pick segment should not include the core release event; a Place segment should not include the core grasp/lift event.
- Light approach, hovering, tiny pauses, or short cleanup frames are acceptable only when they do not change the segment's atomic subtask meaning.
- The first segment should begin at the episode start, the last segment should end at the episode end, and adjacent segments should be continuous.

Description consistency rules:
- Review object names across the whole episode. The same physical object should keep the same name between its Pick and Place segments.
- Bin/container colors and target locations must match the task and visible evidence.
- Use the official task instruction and inventory only as helpful context when provided; visual evidence still controls the final judgment.
- "Move" means robot base/chassis movement only. Arm approach/reach/hover should not be labeled as Move.

Rules:
- Use "correct" only when no repair-worthy issue is visible.
- Use "incorrect" when there is at least one actionable issue.
- Use "uncertain" when evidence is insufficient; do not invent errors.
- Every issue must cite evidence_frames using exact frame ids visible in the contact sheets.
- Suggested fixes should be concise and actionable.
- {needs_more_views_rule}

Return valid JSON only in this exact schema:
{{
  "verdict": "correct | incorrect | uncertain",
  "needs_more_views": false,
  "issues": [
    {{
      "seg_id": 0,
      "issue_type": "boundary_error",
      "evidence_frames": [0],
      "reason": "Brief reason.",
      "suggested_fix": "Brief fix.",
      "confidence": 0.0
    }}
  ],
  "summary": "Brief Chinese summary."
}}
""".strip()


def parse_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("No JSON object found in model response")
    return json.loads(text[start:end])


def normalize_review_response(raw: Dict[str, Any], *, segments: List[Dict[str, Any]], total_frames: int) -> Dict[str, Any]:
    verdict = str(raw.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        verdict = "uncertain"
    needs_more_views = bool(raw.get("needs_more_views", False))
    valid_seg_ids = {int(segment["seg_id"]) for segment in segments}
    max_frame = max(0, total_frames - 1)
    issues: List[Dict[str, Any]] = []
    raw_issues = raw.get("issues", [])
    if not isinstance(raw_issues, list):
        raw_issues = []
    for issue in raw_issues:
        if not isinstance(issue, dict):
            continue
        try:
            seg_id = int(issue.get("seg_id"))
        except (TypeError, ValueError):
            continue
        if seg_id not in valid_seg_ids:
            continue
        issue_type = str(issue.get("issue_type", "")).strip()
        if issue_type not in ISSUE_TYPES:
            issue_type = "unclear_visual"
        raw_frames = issue.get("evidence_frames", [])
        if not isinstance(raw_frames, list):
            raw_frames = [raw_frames]
        frames: List[int] = []
        for raw_frame in raw_frames:
            if isinstance(raw_frame, dict):
                raw_frame = raw_frame.get("frame", raw_frame.get("frame_id"))
            try:
                frame = clamp(int(raw_frame), 0, max_frame)
            except (TypeError, ValueError):
                continue
            if frame not in frames:
                frames.append(frame)
        try:
            confidence = float(issue.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        issues.append(
            {
                "seg_id": seg_id,
                "issue_type": issue_type,
                "evidence_frames": frames,
                "reason": str(issue.get("reason", "")).strip(),
                "suggested_fix": str(issue.get("suggested_fix", "")).strip(),
                "confidence": max(0.0, min(1.0, confidence)),
            }
        )
    if issues and verdict == "correct":
        verdict = "incorrect"
    if verdict == "incorrect" and not issues:
        verdict = "uncertain"
    return {
        "verdict": verdict,
        "needs_more_views": needs_more_views,
        "issues": issues,
        "summary": str(raw.get("summary", "")).strip(),
    }


def should_run_view_fallback(raw_review: Dict[str, Any], review: Dict[str, Any]) -> bool:
    if bool(raw_review.get("needs_more_views")) or bool(review.get("needs_more_views")):
        return True
    if str(review.get("verdict", "")).strip().lower() == "uncertain":
        return True
    return any(str(issue.get("issue_type", "")) == "unclear_visual" for issue in review.get("issues", []))


def call_model_review(
    *,
    client: OpenAI,
    model: str,
    base_url: str,
    api_key: str,
    sample_id: str,
    prompt: str,
    sheet_paths: List[Path],
    max_retries: int,
    max_tokens: int,
    stream: bool,
    stream_log_chars: int,
) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for sheet_path in sheet_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_to_base64(sheet_path)}",
                },
            }
        )
    response_text = api_completion_text_with_retry(
        client,
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.1,
        top_p=1.0,
        max_tokens=max_tokens,
        stream=stream,
        max_retries=max_retries,
        debug_meta={
            "sample_id": sample_id,
            "frame_count": len(sheet_paths),
            "base_url": base_url,
            "api_key_tail": safe_api_key_tail(api_key),
            "stream_log_chars": int(stream_log_chars),
        },
    )
    return parse_json_object(response_text)


def existing_review_line_numbers(output_jsonl: Path) -> set[int]:
    if not output_jsonl.is_file():
        return set()
    result: set[int] = set()
    with output_jsonl.open("r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            try:
                result.add(int(row.get("line_no")))
            except (TypeError, ValueError, AttributeError):
                continue
    return result


def append_jsonl(path: Path, item: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    verdict_counts: Dict[str, int] = {}
    issue_counts: Dict[str, int] = {}
    for row in rows:
        verdict = str(row.get("verdict", "unknown"))
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        for issue in row.get("issues", []):
            issue_type = str(issue.get("issue_type", "unknown"))
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
    payload = {
        "items": len(rows),
        "verdict_counts": verdict_counts,
        "issue_counts": issue_counts,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_review_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def call_review_pass(
    *,
    client: OpenAI,
    model: str,
    base_url: str,
    api_key: str,
    sample_id: str,
    task_text: str,
    total_frames: int,
    fps: float,
    segments: List[Dict[str, Any]],
    frame_manifest: List[Dict[str, Any]],
    sheet_paths: List[Path],
    args: argparse.Namespace,
    review_pass: str,
    available_view_keys: Sequence[str],
    companion_metadata: Dict[str, Any],
    allow_needs_more_views: bool,
    line_no: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    prompt = build_review_prompt(
        sample_id=sample_id,
        task_text=task_text,
        total_frames=total_frames,
        fps=fps,
        segments=segments,
        frame_manifest=frame_manifest,
        skill_options=getattr(args, "skill_options", DEFAULT_SKILL_OPTIONS),
        available_view_keys=available_view_keys,
        review_pass=review_pass,
        companion_metadata=companion_metadata,
        allow_needs_more_views=allow_needs_more_views,
    )
    if getattr(args, "prompt_file", None):
        prompt = Path(args.prompt_file).read_text(encoding="utf-8") + "\n\nRuntime annotation context:\n" + prompt
    append_text_prompt(
        getattr(args, "prompt_log_path", None),
        script="model_review_score",
        line_no=line_no,
        sample_id=sample_id,
        review_pass=review_pass,
        prompt=prompt,
    )
    raw_review = call_model_review(
        client=client,
        model=model,
        base_url=base_url,
        api_key=api_key,
        sample_id=sample_id,
        prompt=prompt,
        sheet_paths=sheet_paths,
        max_retries=int(args.max_retries),
        max_tokens=int(args.max_tokens),
        stream=bool(args.stream),
        stream_log_chars=int(args.stream_log_chars),
    )
    review = normalize_review_response(raw_review, segments=segments, total_frames=total_frames)
    return raw_review, review


def read_available_video_sources(view_paths: Dict[str, str], view_keys: Sequence[str]) -> Dict[str, Tuple[str, bytes]]:
    sources: Dict[str, Tuple[str, bytes]] = {}
    for view_key in view_keys:
        video_path = str(view_paths.get(view_key, "")).strip()
        if not video_path:
            logger.warning("Fallback view %s is unavailable in annotation views", view_key)
            continue
        try:
            sources[view_key] = (video_path, read_video_bytes(video_path))
        except Exception as exc:
            logger.warning("Failed to read fallback view %s from %s: %s", view_key, video_path, exc)
    return sources


def process_one(
    *,
    line_no: int,
    annotation: Dict[str, Any],
    companion_metadata: Dict[str, Any],
    args: argparse.Namespace,
    client: OpenAI,
    model: str,
    base_url: str,
    api_key: str,
) -> Dict[str, Any]:
    total_frames = int(annotation.get("length", annotation.get("frames", 0)))
    fps = float(annotation.get("fps", 30.0) or 30.0)
    sample_id = str(annotation.get("episode_index") or annotation.get("path") or f"line_{line_no}")
    task_text = " ".join(str(task).strip() for task in annotation.get("tasks", []) if str(task).strip())
    view_paths = get_view_paths(annotation)
    primary_view_key = str(args.view_key)
    video_path = view_paths.get(primary_view_key) or get_video_path(annotation, primary_view_key)
    if not video_path:
        raise RuntimeError(f"No video path found for line {line_no}")
    view_paths[primary_view_key] = video_path

    segments, interval_style = normalize_segments_for_review(annotation)
    if not segments:
        raise RuntimeError(f"No segments found for line {line_no}")

    video_bytes = read_video_bytes(video_path)
    probed_fps, probed_frames = probe_video(video_path, video_bytes)
    if total_frames <= 0:
        total_frames = probed_frames
    if abs(probed_fps - fps) > 0.01:
        logger.info("FPS differs | line=%s | annotation=%s | video=%s", line_no, fps, probed_fps)

    boundary_offsets = parse_csv_ints(str(args.boundary_offsets), DEFAULT_BOUNDARY_OFFSETS)
    sample_dir = Path(args.assets_dir) / safe_sample_dir_name(line_no, sample_id)
    primary_sources = {primary_view_key: (video_path, video_bytes)}
    primary_manifest = build_multi_view_frame_manifest(
        segments=segments,
        total_frames=total_frames,
        view_keys=[primary_view_key],
        max_global_frames=int(args.max_global_frames),
        boundary_offsets=boundary_offsets,
        include_global=True,
    )
    primary_sheet_paths = make_contact_sheets(
        video_sources=primary_sources,
        frame_manifest=primary_manifest,
        output_dir=sample_dir / "primary",
        target_height=int(args.target_height),
        frames_per_sheet=int(args.frames_per_sheet),
        columns=int(args.sheet_columns),
    )

    raw_review, review = call_review_pass(
        client=client,
        model=model,
        base_url=base_url,
        api_key=api_key,
        sample_id=sample_id,
        task_text=task_text,
        total_frames=total_frames,
        fps=fps,
        segments=segments,
        frame_manifest=primary_manifest,
        sheet_paths=primary_sheet_paths,
        args=args,
        review_pass="primary",
        available_view_keys=[primary_view_key],
        companion_metadata=companion_metadata,
        allow_needs_more_views=not bool(args.no_view_fallback),
        line_no=line_no,
    )
    final_manifest = list(primary_manifest)
    final_sheet_paths = list(primary_sheet_paths)
    views_used = [primary_view_key]
    review_pass = "primary"
    passes: List[Dict[str, Any]] = [
        {
            "name": "primary",
            "views": [primary_view_key],
            "frames": primary_manifest,
            "contact_sheets": [str(path) for path in primary_sheet_paths],
        }
    ]

    if not args.no_view_fallback and should_run_view_fallback(raw_review, review):
        fallback_keys = resolve_fallback_view_keys(annotation, primary_view_key, str(args.fallback_view_keys))
        fallback_sources = read_available_video_sources(view_paths, fallback_keys)
        loaded_fallback_keys = list(fallback_sources.keys())
        if loaded_fallback_keys:
            logger.info(
                "Running fallback multi-view review | line=%s | views=%s",
                line_no,
                ",".join(loaded_fallback_keys),
            )
            fallback_manifest = build_multi_view_frame_manifest(
                segments=segments,
                total_frames=total_frames,
                view_keys=loaded_fallback_keys,
                max_global_frames=0,
                boundary_offsets=boundary_offsets,
                include_global=False,
            )
            try:
                fallback_sheet_paths = make_contact_sheets(
                    video_sources=fallback_sources,
                    frame_manifest=fallback_manifest,
                    output_dir=sample_dir / "fallback",
                    target_height=int(args.target_height),
                    frames_per_sheet=int(args.frames_per_sheet),
                    columns=int(args.sheet_columns),
                )
                combined_manifest = primary_manifest + fallback_manifest
                combined_sheet_paths = primary_sheet_paths + fallback_sheet_paths
                fallback_raw, fallback_review = call_review_pass(
                    client=client,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    sample_id=sample_id,
                    task_text=task_text,
                    total_frames=total_frames,
                    fps=fps,
                    segments=segments,
                    frame_manifest=combined_manifest,
                    sheet_paths=combined_sheet_paths,
                    args=args,
                    review_pass="fallback",
                    available_view_keys=[primary_view_key] + loaded_fallback_keys,
                    companion_metadata=companion_metadata,
                    allow_needs_more_views=False,
                    line_no=line_no,
                )
                del fallback_raw
                review = fallback_review
                review_pass = "fallback"
                views_used = [primary_view_key] + loaded_fallback_keys
                final_manifest = combined_manifest
                final_sheet_paths = combined_sheet_paths
                passes.append(
                    {
                        "name": "fallback",
                        "views": loaded_fallback_keys,
                        "frames": fallback_manifest,
                        "contact_sheets": [str(path) for path in fallback_sheet_paths],
                    }
                )
            except Exception:
                logger.exception("Fallback multi-view review failed for line_no=%s; keeping primary review", line_no)
        else:
            logger.info("No fallback views available for line_no=%s; keeping primary review", line_no)

    manifest_path = sample_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "line_no": line_no,
                "sample_id": sample_id,
                "video_path": video_path,
                "view_key": primary_view_key,
                "views_used": views_used,
                "interval_style": interval_style,
                "boundary_offsets": boundary_offsets,
                "frames": final_manifest,
                "contact_sheets": [str(path) for path in final_sheet_paths],
                "passes": passes,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    review.update(
        {
            "line_no": line_no,
            "sample_id": sample_id,
            "model": model,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "view_key": primary_view_key,
            "views_used": views_used,
            "review_pass": review_pass,
            "assets_dir": str(sample_dir),
            "contact_sheets": [str(path) for path in final_sheet_paths],
        }
    )
    return review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline model review scorer for SubtaskAnnotator JSONL")
    parser.add_argument("--prompt-file", type=Path, help="External prompt; runtime context is appended")
    parser.add_argument("--input-jsonl", required=True, help="Input annotation JSONL")
    parser.add_argument("--output-jsonl", required=True, help="Output model review JSONL")
    parser.add_argument("--assets-dir", required=True, help="Directory for evidence contact sheets")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default="", help="API key")
    parser.add_argument("--model", default="", help="Model name")
    parser.add_argument("--view-key", default=DEFAULT_VIEW_KEY, help="Primary view key to score")
    parser.add_argument("--config", required=True, help="Path to annotation_ui config.ini")
    parser.add_argument("--target-height", type=int, default=260, help="Tile image height before header")
    parser.add_argument("--frames-per-sheet", type=int, default=24, help="Evidence tiles per contact sheet")
    parser.add_argument("--sheet-columns", type=int, default=4, help="Contact sheet columns")
    parser.add_argument(
        "--boundary-window",
        type=int,
        default=15,
        help="Legacy option retained for compatibility; use --boundary-offsets for sampling",
    )
    parser.add_argument(
        "--max-global-frames",
        type=int,
        default=DEFAULT_MAX_GLOBAL_FRAMES,
        help="Maximum uniformly sampled episode-level frames for the primary view",
    )
    parser.add_argument(
        "--boundary-offsets",
        default=",".join(str(value) for value in DEFAULT_BOUNDARY_OFFSETS),
        help="Comma-separated frame offsets around each adjacent segment boundary",
    )
    parser.add_argument(
        "--fallback-view-keys",
        default="",
        help="Comma-separated fallback view keys; default uses view_order except the primary view, then hand_right,hand_left",
    )
    parser.add_argument("--no-view-fallback", action="store_true", help="Disable fallback review with additional views")
    parser.add_argument("--stream", action="store_true", help="Use streaming chat completions for better timeout debugging")
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="HTTP request timeout in seconds",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=DEFAULT_CONNECT_TIMEOUT,
        help="HTTP connect timeout in seconds",
    )
    parser.add_argument(
        "--stream-log-chars",
        type=int,
        default=DEFAULT_STREAM_LOG_CHARS,
        help="Log streaming progress after this many generated characters",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="LLM retry count")
    parser.add_argument("--max-tokens", type=int, default=2048, help="LLM max output tokens")
    parser.add_argument("--start-line", type=int, default=1, help="Skip JSONL rows before this original line number")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit processed valid samples; 0 means all")
    parser.add_argument("--resume", action="store_true", help="Skip line_no values already present in output JSONL")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON path")
    parser.add_argument(
        "--prompt-log",
        default="",
        help="Append each LLM text user-prompt here, records separated by ==== (override path).",
    )
    parser.add_argument(
        "--no-prompt-log",
        action="store_true",
        help="Disable appending prompts to the prompt log file.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    args.prompt_log_path = None if args.no_prompt_log or not args.prompt_log else resolve_prompt_log_path(args.prompt_log)
    require_dependencies()

    input_jsonl = Path(args.input_jsonl).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    if not input_jsonl.is_file():
        raise SystemExit(f"Input JSONL not found: {input_jsonl}")

    base_url = resolve_base_url(args.base_url)
    api_key = resolve_api_key(args.api_key)
    model = resolve_model(args.model)
    if not base_url or not model or not api_key:
        raise SystemExit("Provide API base URL, model, and an API key environment variable")
    args.skill_options = load_skill_options(Path(args.config).expanduser().resolve())
    client = make_client(
        base_url,
        api_key,
        request_timeout=float(args.request_timeout),
        connect_timeout=float(args.connect_timeout),
    )
    skip_lines = existing_review_line_numbers(output_jsonl) if args.resume else set()
    processed_rows: List[Dict[str, Any]] = []
    processed_count = 0

    logger.info("Input: %s", input_jsonl)
    logger.info("Output: %s", output_jsonl)
    logger.info("Assets: %s", Path(args.assets_dir).resolve())
    logger.info("Skill options: %s", ", ".join(args.skill_options))
    logger.info(
        "LLM transport: stream=%s request_timeout=%ss connect_timeout=%ss",
        bool(args.stream),
        args.request_timeout,
        args.connect_timeout,
    )
    if args.prompt_log_path:
        logger.info("Prompt log: %s", args.prompt_log_path)

    for line_no, annotation, companion_metadata in iter_annotation_records(input_jsonl):
        if line_no < int(args.start_line):
            continue
        if line_no in skip_lines:
            logger.info("Skip existing line_no=%s", line_no)
            continue
        if args.max_samples and processed_count >= int(args.max_samples):
            break

        logger.info("Reviewing line_no=%s", line_no)
        try:
            review = process_one(
                line_no=line_no,
                annotation=annotation,
                companion_metadata=companion_metadata,
                args=args,
                client=client,
                model=model,
                base_url=base_url,
                api_key=api_key,
            )
        except Exception as exc:
            logger.exception("Review failed for line_no=%s", line_no)
            review = {
                "line_no": line_no,
                "sample_id": str(annotation.get("episode_index") or annotation.get("path") or f"line_{line_no}"),
                "verdict": "uncertain",
                "issues": [
                    {
                        "seg_id": 0,
                        "issue_type": "unclear_visual",
                        "evidence_frames": [],
                        "reason": f"Scorer failed: {exc}",
                        "suggested_fix": "Run manual review or rerun scorer after fixing the failure.",
                        "confidence": 0.0,
                    }
                ],
                "summary": "模型复审脚本执行失败，需要人工检查或重跑。",
                "model": model,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
            }
        append_jsonl(output_jsonl, review)
        processed_rows.append(review)
        processed_count += 1

    if args.summary_json:
        write_summary(Path(args.summary_json).resolve(), read_review_rows(output_jsonl))
    logger.info("Done. Processed %s samples.", processed_count)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import configparser
import copy
import hashlib
import json
import logging
import mimetypes
import os
import re
import tempfile
import threading
import time
import uuid
from email.utils import format_datetime
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, TypedDict
from urllib.parse import quote, urlparse

from flask import Flask, Response, g, jsonify, render_template, request

try:
    from botocore.exceptions import BotoCoreError, ClientError

    RECOVERABLE_REMOTE_ERRORS = (BotoCoreError, ClientError, OSError, ValueError, TypeError)
except ImportError:
    RECOVERABLE_REMOTE_ERRORS = (OSError, ValueError, TypeError)

REQUIRED_SEGMENT_FIELDS = ("seg_id", "start_frame", "end_frame", "action_text", "skill")
INTERVAL_STYLE_OPEN = "open"
INTERVAL_STYLE_CLOSED = "closed"
INPUT_FORMAT_JSON = "json"
INPUT_FORMAT_JSONL = "jsonl"
MULTI_USER_MANIFEST_VERSION = 1
ANNOTATOR_COOKIE_NAME = "annotator_id"
AUDIT_SESSION_COOKIE_NAME = "annotation_session_id"
AUDIT_SCHEMA_VERSION = 2
AUDIT_STATUS_SCHEMA_VERSION = 1
ANNOTATOR_ROLE_ANNOTATOR = "annotator"
ANNOTATOR_ROLE_REVIEWER = "reviewer"
ANNOTATOR_ROLES = {ANNOTATOR_ROLE_ANNOTATOR, ANNOTATOR_ROLE_REVIEWER}
REVIEW_BACKUP_FIELD = "pre_review_annotation"
APP_MODE_ANNOTATION = "annotation"
APP_MODE_MODEL_REVIEW = "model-review"
APP_MODES = {APP_MODE_ANNOTATION, APP_MODE_MODEL_REVIEW}
MODEL_REVIEW_VERDICTS = {"correct", "incorrect", "uncertain", "unreviewed"}
MODEL_REVIEW_ISSUE_TYPES = {
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
REMOTE_URI_RE = re.compile(r"^s3://(?P<bucket>[^/]+)(?:/(?P<key>.*))?$")
CLUSTERED_REMOTE_RE = re.compile(r"^(?P<cluster>[^:]+):(?P<uri>s3://.+)$")
VIDEO_S3_MODE_VALUES = {"hybrid", "direct", "proxy"}
PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy")
MAX_VIEW_COUNT = 4
AUDIT_STATUS_UNOPENED = "unopened"
AUDIT_STATUS_OPENED = "opened"
AUDIT_STATUS_MODIFIED = "modified"
AUDIT_STATUS_SAVED = "saved"
AUDIT_STATUS_VALIDATION_FAILED = "validation_failed"
AUDIT_STATUS_REVIEWED = "reviewed"
AUDIT_STATUS_LEGACY = "legacy_output_without_audit"
AUDIT_BEHAVIOR_EVENT_TYPES = {"item_open", "item_close"}
AUDIT_MUTATION_EVENT_TYPES = {
    "annotation_update", "review_update", "annotation_save", "review_save", "discard_update"
}
DISCARD_REASON_LABELS = {
    "video_unplayable": "视频损坏或无法播放",
    "content_mismatch": "内容与任务不符",
    "missing_view": "关键视角缺失",
    "unclear": "画面无法判断",
    "duplicate": "重复数据",
    "other": "其他",
}
DISCARD_FIELDS = ("discarded", "discard_reason", "discard_note", "discarded_by", "discarded_at")


def format_http_datetime(value: datetime) -> str:
    """Format S3 timestamps as RFC-compliant GMT, including OSS tzinfo values."""
    if value.tzinfo is None:
        normalized = value.replace(tzinfo=timezone.utc)
    else:
        normalized = value.astimezone(timezone.utc)
    return format_datetime(normalized, usegmt=True)


CANONICAL_SKILLS = [
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

_AUDIT_LOCKS: Dict[str, threading.Lock] = {}
_AUDIT_LOCKS_GUARD = threading.Lock()

logger = logging.getLogger("annotation_ui")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )


@dataclass(frozen=True)
class RemotePathParts:
    raw_path: str
    cluster: str
    uri: str
    bucket: str
    key: str


@dataclass(frozen=True)
class S3ClusterConfig:
    section_name: str
    access_key: str
    secret_key: str
    endpoint_url: str
    region_name: str = "us-east-1"


@dataclass(frozen=True)
class RuntimeConfig:
    config_path: Path
    s3_mode: str
    presigned_expire_sec: int
    fallback_retry_max: int
    default_cluster: str
    disable_proxy: bool
    no_proxy_extra: str
    configured_skill_options: List[str]
    configured_action_text_options: List[str]
    s3_clusters: Dict[str, S3ClusterConfig]

    @classmethod
    def load(cls, config_path: Path) -> "RuntimeConfig":
        path = config_path.expanduser().resolve()
        if not path.is_file():
            raise ValidationError(f"Config file not found: {path}")

        parser = configparser.ConfigParser(interpolation=None)
        parser.read(path, encoding="utf-8")

        if not parser.sections():
            raise ValidationError(f"Config file has no sections: {path}")

        video_section = parser["video"] if parser.has_section("video") else None
        s3_mode = str(video_section.get("s3_mode", "hybrid")).strip().lower() if video_section else "hybrid"
        if s3_mode not in VIDEO_S3_MODE_VALUES:
            allowed = ", ".join(sorted(VIDEO_S3_MODE_VALUES))
            raise ValidationError(f"[video].s3_mode must be one of: {allowed}")

        default_cluster = str(video_section.get("default_cluster", "")).strip() if video_section else ""

        try:
            presigned_expire_sec = int(video_section.get("presigned_expire_sec", 3600)) if video_section else 3600
        except (TypeError, ValueError) as exc:
            raise ValidationError("[video].presigned_expire_sec must be an integer") from exc
        if presigned_expire_sec <= 0:
            raise ValidationError("[video].presigned_expire_sec must be > 0")

        try:
            fallback_retry_max = int(video_section.get("fallback_retry_max", 1)) if video_section else 1
        except (TypeError, ValueError) as exc:
            raise ValidationError("[video].fallback_retry_max must be an integer") from exc
        if fallback_retry_max < 0:
            raise ValidationError("[video].fallback_retry_max must be >= 0")

        disable_proxy_raw = str(video_section.get("disable_proxy", "true")).strip().lower() if video_section else "true"
        disable_proxy = disable_proxy_raw in {"1", "true", "yes", "on"}
        no_proxy_extra = str(video_section.get("no_proxy_extra", "")).strip() if video_section else ""

        annotation_section = parser["annotation"] if parser.has_section("annotation") else None
        configured_skill_options = parse_option_list(
            annotation_section.get("skill_options", "") if annotation_section else ""
        )
        if not configured_skill_options:
            configured_skill_options = list(CANONICAL_SKILLS)

        configured_action_text_options = parse_option_list(
            annotation_section.get("action_text_options", "") if annotation_section else ""
        )

        s3_clusters: Dict[str, S3ClusterConfig] = {}
        for section_name in parser.sections():
            if section_name in {"video", "annotation"}:
                continue

            section = parser[section_name]
            access_key = os.environ.get(str(section.get("access_key_env", "")), str(section.get("access_key", ""))).strip()
            secret_key = os.environ.get(str(section.get("secret_key_env", "")), str(section.get("secret_key", ""))).strip()
            endpoint_raw = (
                str(section.get("endpoint", "")).strip()
                or str(section.get("host_base", "")).strip()
                or str(section.get("endpoint_url", "")).strip()
            )
            if not access_key or not secret_key or not endpoint_raw:
                raise ValidationError(
                    f"Config section [{section_name}] must contain access_key, secret_key and endpoint/host_base"
                )

            endpoint_url = endpoint_raw.split(",", 1)[0].strip()
            if endpoint_url and not endpoint_url.startswith(("http://", "https://")):
                endpoint_url = f"http://{endpoint_url}"
            region_name = str(section.get("region_name", "us-east-1")).strip() or "us-east-1"

            s3_clusters[section_name] = S3ClusterConfig(
                section_name=section_name,
                access_key=access_key,
                secret_key=secret_key,
                endpoint_url=endpoint_url,
                region_name=region_name,
            )

        if default_cluster and default_cluster not in s3_clusters:
            raise ValidationError(f"[video].default_cluster '{default_cluster}' not found in config sections")

        return cls(
            config_path=path,
            s3_mode=s3_mode,
            presigned_expire_sec=presigned_expire_sec,
            fallback_retry_max=fallback_retry_max,
            default_cluster=default_cluster,
            disable_proxy=disable_proxy,
            no_proxy_extra=no_proxy_extra,
            configured_skill_options=configured_skill_options,
            configured_action_text_options=configured_action_text_options,
            s3_clusters=s3_clusters,
        )


class ValidationError(ValueError):
    def __init__(
        self,
        message: str,
        field_errors: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(message)
        self.field_errors = field_errors or []


class SegmentPayload(TypedDict):
    seg_id: int
    start_frame: int
    end_frame: int
    action_text: str
    skill: str


class PlaybackPayload(TypedDict, total=False):
    primary_url: str
    fallback_url: str
    mode: str
    fallback_retry_max: int
    error: str


class VideoPayload(TypedDict, total=False):
    index: int
    episode_index: str
    path: str
    frames: int
    fps: float
    tasks: List[str]
    action_config: List[SegmentPayload]
    unavailable: bool
    error: str
    playback: PlaybackPayload
    primary_view_key: str
    view_order: List[str]
    views: List[Dict[str, Any]]
    view_count: int
    frame_map: Dict[str, Any]
    discarded: bool
    discard_reason: str
    discard_note: str
    discarded_by: str
    discarded_at: str


def parse_option_list(raw_value: Any) -> List[str]:
    text = str(raw_value or "")
    values: List[str] = []
    seen = set()
    for line in text.splitlines():
        for token in line.split(","):
            value = token.strip()
            if not value or value in seen:
                continue
            seen.add(value)
            values.append(value)
    return values


def resolve_media_path(path: str) -> Path:
    """Resolve local media under an explicit media root."""
    value = Path(path).expanduser()
    return value if value.is_absolute() else Path(os.environ.get("ROBOATOM_MEDIA_ROOT", ".")) / value


def is_remote_path(path: str) -> bool:
    value = str(path or "").strip()
    if not value:
        return False
    return value.startswith("s3://") or bool(CLUSTERED_REMOTE_RE.match(value))


def parse_remote_path(path: str) -> RemotePathParts:
    value = str(path or "").strip()
    if not value:
        raise ValidationError("Remote path is empty")

    cluster = ""
    uri = value
    matched = CLUSTERED_REMOTE_RE.match(value)
    if matched:
        cluster = str(matched.group("cluster") or "").strip()
        uri = str(matched.group("uri") or "").strip()

    uri_match = REMOTE_URI_RE.match(uri)
    if not uri_match:
        raise ValidationError(f"Invalid remote path format: {value}")

    bucket = str(uri_match.group("bucket") or "").strip()
    key = str(uri_match.group("key") or "").strip()
    if not bucket:
        raise ValidationError(f"Invalid remote path bucket: {value}")

    return RemotePathParts(
        raw_path=value,
        cluster=cluster,
        uri=uri,
        bucket=bucket,
        key=key,
    )


def _ensure_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field_name} must be an integer") from exc


def _ensure_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field_name} must be a number") from exc


def normalize_views(item: Dict[str, Any], item_index: int) -> Tuple[Dict[str, str], List[str], bool]:
    has_views = "views" in item
    raw_episode_index = str(item.get("episode_index", item.get("episode_id", ""))).strip()
    raw_path = str(item.get("path", "")).strip()

    if has_views:
        raw_views = item.get("views")
        if not isinstance(raw_views, dict):
            raise ValidationError(f"item[{item_index}].views must be an object")
        if not raw_views:
            raise ValidationError(f"item[{item_index}].views cannot be empty")
        if len(raw_views) > MAX_VIEW_COUNT:
            raise ValidationError(f"item[{item_index}].views supports at most {MAX_VIEW_COUNT} views")

        views: Dict[str, str] = {}
        for raw_key, raw_value in raw_views.items():
            view_key = str(raw_key).strip()
            if not view_key:
                raise ValidationError(f"item[{item_index}].views contains an empty view key")
            if view_key in views:
                raise ValidationError(f"item[{item_index}].views contains duplicate key '{view_key}'")
            view_path = str(raw_value or "").strip()
            if not view_path:
                raise ValidationError(f"item[{item_index}].views['{view_key}'] cannot be empty")
            views[view_key] = view_path

        raw_view_order = item.get("view_order")
        if raw_view_order is None:
            view_order = list(views.keys())
            return views, view_order, True

        if not isinstance(raw_view_order, list):
            raise ValidationError(f"item[{item_index}].view_order must be a list of view keys")

        order_seen = set()
        view_order: List[str] = []
        for order_idx, value in enumerate(raw_view_order):
            view_key = str(value).strip()
            if not view_key:
                raise ValidationError(f"item[{item_index}].view_order[{order_idx}] cannot be empty")
            if view_key in order_seen:
                raise ValidationError(f"item[{item_index}].view_order has duplicate key '{view_key}'")
            if view_key not in views:
                raise ValidationError(
                    f"item[{item_index}].view_order contains unknown key '{view_key}'"
                )
            order_seen.add(view_key)
            view_order.append(view_key)

        for view_key in views.keys():
            if view_key not in order_seen:
                view_order.append(view_key)

        return views, view_order, True

    if "view_order" in item:
        raise ValidationError(f"item[{item_index}].view_order requires item[{item_index}].views")

    effective_path = raw_path or raw_episode_index
    return {"main": effective_path}, ["main"], False


def normalize_segments(action_config: Any) -> List[Dict[str, Any]]:
    if not isinstance(action_config, list):
        raise ValidationError("action_config must be a list")

    normalized: List[Dict[str, Any]] = []
    for idx, segment in enumerate(action_config):
        if not isinstance(segment, dict):
            raise ValidationError(f"action_config[{idx}] must be an object")

        missing = [field for field in REQUIRED_SEGMENT_FIELDS if field not in segment]
        if missing:
            raise ValidationError(
                f"action_config[{idx}] missing required fields: {', '.join(missing)}"
            )

        start_frame = _ensure_int(segment.get("start_frame"), f"action_config[{idx}].start_frame")
        end_frame = _ensure_int(segment.get("end_frame"), f"action_config[{idx}].end_frame")
        action_text = str(segment.get("action_text", "")).strip()
        skill = str(segment.get("skill", "")).strip()

        if not action_text:
            raise ValidationError(f"action_config[{idx}].action_text cannot be empty")
        if not skill:
            raise ValidationError(f"action_config[{idx}].skill cannot be empty")

        normalized.append(
            {
                "seg_id": idx,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "action_text": action_text,
                "skill": skill,
            }
        )

    return normalized


def validate_segments(segments: List[Dict[str, Any]], frames: int) -> None:
    if frames <= 0:
        raise ValidationError("frames must be greater than 0")
    if not segments:
        raise ValidationError("action_config cannot be empty")

    for idx, segment in enumerate(segments):
        start_frame = segment["start_frame"]
        end_frame = segment["end_frame"]

        if start_frame < 0 or end_frame < 0:
            raise ValidationError(f"segment[{idx}] has negative frame index")
        if start_frame >= end_frame:
            raise ValidationError(
                f"segment[{idx}] must satisfy start_frame < end_frame, got [{start_frame}, {end_frame})"
            )
        if end_frame > frames:
            raise ValidationError(
                f"segment[{idx}] end_frame={end_frame} exceeds frames={frames}"
            )

    if segments[0]["start_frame"] != 0:
        raise ValidationError(
            f"first segment must start at frame 0, got {segments[0]['start_frame']}"
        )
    if segments[-1]["end_frame"] != frames:
        raise ValidationError(
            f"last segment must end at frame {frames}, got {segments[-1]['end_frame']}"
        )

    for idx in range(len(segments) - 1):
        cur_end = segments[idx]["end_frame"]
        next_start = segments[idx + 1]["start_frame"]
        if cur_end != next_start:
            raise ValidationError(
                f"segments must be contiguous: segment[{idx}].end_frame={cur_end} != segment[{idx + 1}].start_frame={next_start}"
            )


def build_frame_map(frames: int, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    frame_map: List[Dict[str, Any]] = [
        {
            "frame": frame,
            "seg_id": None,
            "action_text": "",
            "skill": "",
            "is_boundary": False,
        }
        for frame in range(frames)
    ]

    boundaries = {segment["start_frame"] for segment in segments}

    for segment in segments:
        start_frame = segment["start_frame"]
        end_frame = segment["end_frame"]
        for frame in range(start_frame, end_frame):
            frame_map[frame] = {
                "frame": frame,
                "seg_id": segment["seg_id"],
                "action_text": segment["action_text"],
                "skill": segment["skill"],
                "is_boundary": frame in boundaries,
            }

    return frame_map


def sanitize_tasks(tasks: Any) -> List[str]:
    if not isinstance(tasks, list):
        raise ValidationError("tasks must be a list of strings")
    cleaned = [str(item).strip() for item in tasks if str(item).strip()]
    if not cleaned:
        raise ValidationError("tasks cannot be empty")
    return cleaned


def detect_interval_style(segments: List[Dict[str, Any]], frames: int) -> str:
    if not segments or frames <= 0:
        return INTERVAL_STYLE_OPEN

    looks_open = True
    looks_closed = True

    for idx in range(len(segments) - 1):
        current = segments[idx]
        nxt = segments[idx + 1]
        if nxt["start_frame"] != current["end_frame"]:
            looks_open = False
        if nxt["start_frame"] != current["end_frame"] + 1:
            looks_closed = False

    if segments[0]["start_frame"] != 0:
        looks_open = False
        looks_closed = False

    last_end = segments[-1]["end_frame"]
    if last_end != frames:
        looks_open = False
    if last_end != frames - 1:
        looks_closed = False

    if looks_closed and not looks_open:
        return INTERVAL_STYLE_CLOSED
    return INTERVAL_STYLE_OPEN


def convert_segments_to_open(segments: List[Dict[str, Any]], interval_style: str) -> List[Dict[str, Any]]:
    if interval_style != INTERVAL_STYLE_CLOSED:
        return copy.deepcopy(segments)
    converted = copy.deepcopy(segments)
    for segment in converted:
        segment["end_frame"] = int(segment["end_frame"]) + 1
    return converted


def convert_segments_from_open(segments: List[Dict[str, Any]], interval_style: str) -> List[Dict[str, Any]]:
    converted = copy.deepcopy(segments)
    if interval_style != INTERVAL_STYLE_CLOSED:
        return converted
    for idx, segment in enumerate(converted):
        end_frame = int(segment["end_frame"]) - 1
        if end_frame < int(segment["start_frame"]):
            raise ValidationError(
                f"cannot convert segment[{idx}] back to closed interval because end_frame < start_frame"
            )
        segment["end_frame"] = end_frame
    return converted


def is_annotation_candidate(candidate: Any) -> bool:
    if not isinstance(candidate, dict):
        return False
    if isinstance(candidate.get("segments"), list) and "num_frames" in candidate:
        return True
    if not isinstance(candidate.get("action_config"), list):
        return False
    if not isinstance(candidate.get("tasks"), list):
        return False
    return ("frames" in candidate) or ("length" in candidate)


@dataclass
class LoadedInput:
    input_format: str
    items: List[Dict[str, Any]]
    source_mappings: List[Dict[str, Any]]
    load_warnings: List[str] = field(default_factory=list)
    jsonl_rows: Optional[List[Dict[str, Any]]] = None


def resolve_user_path(raw_path: Any, base_dir: Path, field_name: str) -> Path:
    value = str(raw_path or "").strip()
    if not value:
        raise ValidationError(f"{field_name} is required")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path.resolve()


def default_seed_input_path(master_input_path: Path, annotator_id: str) -> Path:
    suffix = master_input_path.suffix
    stem = master_input_path.stem
    return (master_input_path.parent / f"{stem}_{annotator_id}{suffix}").resolve()


def serialize_jsonl_rows(rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for row in rows:
        if row.get("blank"):
            lines.append("")
            continue
        lines.append(json.dumps(row["value"], ensure_ascii=False))
    return "\n".join(lines) + "\n"


def build_serialized_output(
    input_format: str,
    source_mappings: List[Dict[str, Any]],
    serialized_items: List[Dict[str, Any]],
    jsonl_rows: Optional[List[Dict[str, Any]]] = None,
) -> str:
    if input_format == INPUT_FORMAT_JSONL:
        if jsonl_rows is None:
            raise ValidationError("JSONL save failed: missing source rows")

        rows = copy.deepcopy(jsonl_rows)
        for idx, serialized_item in enumerate(serialized_items):
            mapping = source_mappings[idx]
            row_index = mapping.get("line_index")
            if row_index is None or row_index < 0 or row_index >= len(rows):
                raise ValidationError(f"JSONL save failed: invalid source line index for item[{idx}]")
            if rows[row_index].get("blank"):
                raise ValidationError(f"JSONL save failed: target line for item[{idx}] is blank")

            row_value = rows[row_index]["value"]
            target_type = mapping.get("target_type")
            if target_type == "dict_result":
                if not isinstance(row_value, dict):
                    raise ValidationError(
                        f"JSONL save failed at line {mapping.get('line_no')}: expected dict"
                    )
                row_value["result"] = serialized_item
            elif target_type == "root_dict":
                rows[row_index]["value"] = serialized_item
            elif target_type == "list_item":
                list_index = int(mapping.get("list_index", -1))
                if not isinstance(row_value, list) or list_index < 0 or list_index >= len(row_value):
                    raise ValidationError(
                        f"JSONL save failed at line {mapping.get('line_no')}: invalid list index"
                    )
                row_value[list_index] = serialized_item
            else:
                raise ValidationError(f"JSONL save failed: unsupported target_type={target_type}")

        return serialize_jsonl_rows(rows)

    return json.dumps(serialized_items, ensure_ascii=False, indent=2)


def write_output_atomically(output_path: Path, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as file:
            file.write(content)
        os.replace(temp_path, output_path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def canonical_json_dumps(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return str(value)


def audit_payload_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_dumps(value).encode("utf-8")).hexdigest()


def audit_sidecar_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.audit.jsonl")


def status_sidecar_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.status.json")


def audit_lock_for_path(audit_path: Path) -> threading.Lock:
    key = str(audit_path.expanduser().resolve())
    with _AUDIT_LOCKS_GUARD:
        lock = _AUDIT_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _AUDIT_LOCKS[key] = lock
        return lock


def item_audit_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "tasks": copy.deepcopy(item.get("tasks", [])),
        "action_config": copy.deepcopy(item.get("action_config", [])),
    }
    for field_name in DISCARD_FIELDS:
        if field_name in item:
            payload[field_name] = copy.deepcopy(item.get(field_name))
    return payload


def sample_id_for_item(item: Dict[str, Any]) -> str:
    for key in ("sample_id", "episode_index", "path"):
        value = str(item.get(key, "")).strip()
        if value:
            return value
    return ""


def item_key_for_mapping(
    output_path: Path,
    source_mappings: List[Dict[str, Any]],
    local_index: Optional[int],
    global_index: Optional[int],
    sample_id: str,
) -> str:
    if local_index is None:
        return ""
    mapping: Dict[str, Any] = {}
    if 0 <= local_index < len(source_mappings):
        mapping = source_mappings[local_index]
    if "line_no" in mapping:
        origin = f"jsonl:{mapping.get('line_no')}"
    elif "item_index" in mapping:
        origin = f"json:{mapping.get('item_index')}"
    else:
        origin = f"index:{local_index}"
    global_part = "" if global_index is None else str(global_index)
    sample_part = sample_id or str(output_path.name)
    return f"{origin}|global:{global_part}|sample:{sample_part}"


def _segment_identity_values(segments: List[Dict[str, Any]]) -> List[str]:
    return [str(segment.get("seg_id", "")).strip() for segment in segments]


def _can_match_segments_by_seg_id(
    before_segments: List[Dict[str, Any]],
    after_segments: List[Dict[str, Any]],
) -> bool:
    before_ids = _segment_identity_values(before_segments)
    after_ids = _segment_identity_values(after_segments)
    return (
        all(before_ids)
        and all(after_ids)
        and len(before_ids) == len(set(before_ids))
        and len(after_ids) == len(set(after_ids))
    )


def annotation_field_diff(
    before_payload: Dict[str, Any],
    after_payload: Dict[str, Any],
) -> Dict[str, Any]:
    diff: Dict[str, Any] = {}
    before_tasks = list(before_payload.get("tasks") or [])
    after_tasks = list(after_payload.get("tasks") or [])
    if before_tasks != after_tasks:
        diff["tasks"] = {
            "before": before_tasks,
            "after": after_tasks,
            "before_count": len(before_tasks),
            "after_count": len(after_tasks),
        }

    before_segments = [
        copy.deepcopy(segment)
        for segment in before_payload.get("action_config") or []
        if isinstance(segment, dict)
    ]
    after_segments = [
        copy.deepcopy(segment)
        for segment in after_payload.get("action_config") or []
        if isinstance(segment, dict)
    ]
    segment_changes: List[Dict[str, Any]] = []
    segment_order_changed = False
    fields = ("seg_id", "start_frame", "end_frame", "skill", "action_text")

    if _can_match_segments_by_seg_id(before_segments, after_segments):
        before_ids = _segment_identity_values(before_segments)
        after_ids = _segment_identity_values(after_segments)
        segment_order_changed = before_ids != after_ids
        before_by_id = {str(segment.get("seg_id")): (idx, segment) for idx, segment in enumerate(before_segments)}
        after_by_id = {str(segment.get("seg_id")): (idx, segment) for idx, segment in enumerate(after_segments)}
        for seg_id in sorted(set(before_by_id) | set(after_by_id), key=str):
            before_entry = before_by_id.get(seg_id)
            after_entry = after_by_id.get(seg_id)
            if before_entry is None and after_entry is not None:
                after_idx, after_segment = after_entry
                segment_changes.append(
                    {
                        "operation": "added",
                        "seg_id": after_segment.get("seg_id"),
                        "index_after": after_idx,
                        "after": {field: after_segment.get(field) for field in fields},
                    }
                )
                continue
            if after_entry is None and before_entry is not None:
                before_idx, before_segment = before_entry
                segment_changes.append(
                    {
                        "operation": "removed",
                        "seg_id": before_segment.get("seg_id"),
                        "index_before": before_idx,
                        "before": {field: before_segment.get(field) for field in fields},
                    }
                )
                continue
            if before_entry is None or after_entry is None:
                continue
            before_idx, before_segment = before_entry
            after_idx, after_segment = after_entry
            field_changes = {
                field: {
                    "before": before_segment.get(field),
                    "after": after_segment.get(field),
                }
                for field in fields
                if before_segment.get(field) != after_segment.get(field)
            }
            if field_changes or before_idx != after_idx:
                segment_changes.append(
                    {
                        "operation": "modified",
                        "seg_id": after_segment.get("seg_id"),
                        "index_before": before_idx,
                        "index_after": after_idx,
                        "fields": field_changes,
                    }
                )
    else:
        max_len = max(len(before_segments), len(after_segments))
        for idx in range(max_len):
            before_segment = before_segments[idx] if idx < len(before_segments) else None
            after_segment = after_segments[idx] if idx < len(after_segments) else None
            if before_segment is None and after_segment is not None:
                segment_changes.append(
                    {
                        "operation": "added",
                        "seg_id": after_segment.get("seg_id"),
                        "index_after": idx,
                        "after": {field: after_segment.get(field) for field in fields},
                    }
                )
                continue
            if after_segment is None and before_segment is not None:
                segment_changes.append(
                    {
                        "operation": "removed",
                        "seg_id": before_segment.get("seg_id"),
                        "index_before": idx,
                        "before": {field: before_segment.get(field) for field in fields},
                    }
                )
                continue
            if before_segment is None or after_segment is None:
                continue
            field_changes = {
                field: {
                    "before": before_segment.get(field),
                    "after": after_segment.get(field),
                }
                for field in fields
                if before_segment.get(field) != after_segment.get(field)
            }
            if field_changes:
                segment_changes.append(
                    {
                        "operation": "modified",
                        "seg_id": after_segment.get("seg_id", before_segment.get("seg_id")),
                        "index_before": idx,
                        "index_after": idx,
                        "fields": field_changes,
                    }
                )

    if segment_order_changed:
        diff["segment_order"] = {
            "before": _segment_identity_values(before_segments),
            "after": _segment_identity_values(after_segments),
        }
    if segment_changes:
        diff["segments"] = segment_changes
    return diff


def summarize_annotation_diff(diff: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "tasks_changed": "tasks" in diff,
        "segments_added": 0,
        "segments_removed": 0,
        "segments_modified": 0,
        "segments_reordered": "segment_order" in diff,
        "fields_changed": {},
        "boundary_changes": 0,
        "text_changes": 0,
        "skill_changes": 0,
        "total_changes": 0,
    }
    if "tasks" in diff:
        summary["total_changes"] += 1

    field_counts: Dict[str, int] = {}
    for change in diff.get("segments", []):
        operation = str(change.get("operation", ""))
        if operation == "added":
            summary["segments_added"] += 1
            summary["total_changes"] += 1
        elif operation == "removed":
            summary["segments_removed"] += 1
            summary["total_changes"] += 1
        elif operation == "modified":
            fields = change.get("fields") if isinstance(change.get("fields"), dict) else {}
            if fields:
                summary["segments_modified"] += 1
            for field_name in fields.keys():
                field_counts[field_name] = field_counts.get(field_name, 0) + 1
                summary["total_changes"] += 1
    if summary["segments_reordered"]:
        summary["total_changes"] += 1
    summary["fields_changed"] = field_counts
    summary["boundary_changes"] = field_counts.get("start_frame", 0) + field_counts.get("end_frame", 0)
    summary["text_changes"] = field_counts.get("action_text", 0)
    summary["skill_changes"] = field_counts.get("skill", 0)
    return summary


def default_audit_status_item(index: int, status: str = AUDIT_STATUS_UNOPENED) -> Dict[str, Any]:
    return {
        "index": index,
        "status": status,
        "modified": False,
        "first_opened_at": None,
        "last_modified_at": None,
        "last_saved_at": None,
        "last_modified_by": "",
        "revision": 0,
        "change_count": 0,
        "validation_error_count": 0,
        "last_event_id": "",
        "active_ms": 0,
    }


def _coerce_status_item(raw: Any, index: int, default_status: str) -> Dict[str, Any]:
    item = default_audit_status_item(index, default_status)
    if isinstance(raw, dict):
        for key in item.keys():
            if key in raw:
                item[key] = raw[key]
    item["index"] = index
    item["status"] = str(item.get("status") or default_status)
    item["modified"] = bool(item.get("modified", False))
    for key in ("revision", "change_count", "validation_error_count", "active_ms"):
        try:
            item[key] = max(0, int(item.get(key) or 0))
        except (TypeError, ValueError):
            item[key] = 0
    for key in ("last_modified_by", "last_event_id"):
        item[key] = str(item.get(key) or "")
    return item


def _audit_event_target_indices(event: Dict[str, Any], item_count: int) -> List[int]:
    targets: List[int] = []
    local_index = event.get("local_index", event.get("index"))
    if isinstance(local_index, int):
        targets.append(local_index)
    summary = event.get("change_summary") if isinstance(event.get("change_summary"), dict) else {}
    dirty_indices = summary.get("dirty_indices")
    if isinstance(dirty_indices, list):
        for value in dirty_indices:
            try:
                targets.append(int(value))
            except (TypeError, ValueError):
                continue
    seen: Set[int] = set()
    result: List[int] = []
    for idx in targets:
        if 0 <= idx < item_count and idx not in seen:
            seen.add(idx)
            result.append(idx)
    return result


def apply_audit_event_to_status_items(items: List[Dict[str, Any]], event: Dict[str, Any]) -> None:
    item_count = len(items)
    targets = _audit_event_target_indices(event, item_count)
    if not targets:
        return

    event_type = str(event.get("event_type", ""))
    event_status = str(event.get("status", ""))
    created_at = str(event.get("created_at", ""))
    annotator_id = str(event.get("annotator_id", ""))
    role = str(event.get("role", ""))
    event_id = str(event.get("event_id", ""))
    summary = event.get("change_summary") if isinstance(event.get("change_summary"), dict) else {}
    total_changes = int(summary.get("total_changes") or 0)
    try:
        event_revision_after = max(0, int(event.get("revision_after") or 0))
    except (TypeError, ValueError):
        event_revision_after = 0
    try:
        active_ms = max(0, int(event.get("active_ms") or summary.get("active_ms") or 0))
    except (TypeError, ValueError):
        active_ms = 0

    for idx in targets:
        item = items[idx]
        item["last_event_id"] = event_id
        if event_type == "item_open" and event_status == "success":
            if not item.get("first_opened_at"):
                item["first_opened_at"] = created_at
            if item.get("status") == AUDIT_STATUS_UNOPENED:
                item["status"] = AUDIT_STATUS_OPENED
            continue

        if event_type == "item_close" and event_status == "success":
            item["active_ms"] = max(0, int(item.get("active_ms") or 0)) + active_ms
            continue

        if event_type in {"annotation_update", "review_update"}:
            if event_status == "success":
                if total_changes > 0:
                    item["status"] = AUDIT_STATUS_MODIFIED
                    item["modified"] = True
                    item["last_modified_at"] = created_at
                    item["last_modified_by"] = annotator_id
                    item["change_count"] = max(0, int(item.get("change_count") or 0)) + total_changes
                    item["revision"] = max(
                        max(0, int(item.get("revision") or 0)) + 1,
                        event_revision_after,
                    )
                elif item.get("status") == AUDIT_STATUS_UNOPENED:
                    item["status"] = AUDIT_STATUS_OPENED
            else:
                item["status"] = AUDIT_STATUS_VALIDATION_FAILED
                item["validation_error_count"] = max(0, int(item.get("validation_error_count") or 0)) + 1
                item["revision"] = max(
                    max(0, int(item.get("revision") or 0)) + 1,
                    event_revision_after,
                )
            continue

        if event_type == "discard_update":
            if event_status == "success":
                item["status"] = AUDIT_STATUS_REVIEWED if role == ANNOTATOR_ROLE_REVIEWER else AUDIT_STATUS_SAVED
                item["modified"] = True
                item["last_modified_at"] = created_at
                item["last_saved_at"] = created_at
                item["last_modified_by"] = annotator_id
                item["change_count"] = max(0, int(item.get("change_count") or 0)) + 1
            else:
                item["status"] = AUDIT_STATUS_VALIDATION_FAILED
                item["validation_error_count"] = max(0, int(item.get("validation_error_count") or 0)) + 1
            item["revision"] = max(
                max(0, int(item.get("revision") or 0)) + 1,
                event_revision_after,
            )
            continue

        if event_type in {"annotation_save", "review_save"}:
            if event_status == "success":
                item["status"] = AUDIT_STATUS_REVIEWED if role == ANNOTATOR_ROLE_REVIEWER or event_type == "review_save" else AUDIT_STATUS_SAVED
                item["last_saved_at"] = created_at
                item["last_modified_by"] = annotator_id
                item["revision"] = max(
                    max(0, int(item.get("revision") or 0)) + 1,
                    event_revision_after,
                )
            else:
                item["status"] = AUDIT_STATUS_VALIDATION_FAILED
                item["validation_error_count"] = max(0, int(item.get("validation_error_count") or 0)) + 1
                item["revision"] = max(
                    max(0, int(item.get("revision") or 0)) + 1,
                    event_revision_after,
                )


def _new_status_snapshot(output_path: Path, item_count: int, default_status: str) -> Dict[str, Any]:
    return {
        "schema_version": AUDIT_STATUS_SCHEMA_VERSION,
        "output_path": str(output_path),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "items": [
            default_audit_status_item(index, default_status)
            for index in range(item_count)
        ],
    }


def rebuild_status_snapshot_from_audit(output_path: Path, item_count: int) -> Dict[str, Any]:
    snapshot = _new_status_snapshot(output_path, item_count, AUDIT_STATUS_UNOPENED)
    audit_path = audit_sidecar_path(output_path)
    if not audit_path.is_file():
        return snapshot
    try:
        with audit_path.open("r", encoding="utf-8") as file:
            for line in file:
                text = line.strip()
                if not text:
                    continue
                try:
                    event = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    apply_audit_event_to_status_items(snapshot["items"], event)
    except OSError:
        return snapshot
    snapshot["updated_at"] = datetime.now(timezone.utc).isoformat()
    return snapshot


def load_status_snapshot(output_path: Path, item_count: int) -> Dict[str, Any]:
    status_path = status_sidecar_path(output_path)
    audit_path = audit_sidecar_path(output_path)
    default_status = AUDIT_STATUS_UNOPENED
    try:
        legacy_output = output_path.is_file() and output_path.stat().st_size > 0 and not audit_path.is_file()
    except OSError:
        legacy_output = False
    if legacy_output:
        default_status = AUDIT_STATUS_LEGACY

    if status_path.is_file():
        try:
            with status_path.open("r", encoding="utf-8") as file:
                raw = json.load(file)
        except (OSError, json.JSONDecodeError):
            raw = {}
        if isinstance(raw, dict):
            raw_items = raw.get("items") if isinstance(raw.get("items"), list) else []
            items = [
                _coerce_status_item(raw_items[index] if index < len(raw_items) else None, index, default_status)
                for index in range(item_count)
            ]
            return {
                "schema_version": AUDIT_STATUS_SCHEMA_VERSION,
                "output_path": str(output_path),
                "updated_at": str(raw.get("updated_at") or datetime.now(timezone.utc).isoformat()),
                "items": items,
            }

    if audit_path.is_file():
        return rebuild_status_snapshot_from_audit(output_path, item_count)

    return _new_status_snapshot(output_path, item_count, default_status)


def write_status_snapshot(output_path: Path, snapshot: Dict[str, Any]) -> None:
    snapshot["schema_version"] = AUDIT_STATUS_SCHEMA_VERSION
    snapshot["output_path"] = str(output_path)
    snapshot["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_output_atomically(
        status_sidecar_path(output_path),
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def compact_status_items(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = snapshot.get("items") if isinstance(snapshot.get("items"), list) else []
    result: List[Dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            item = default_audit_status_item(index)
        result.append(
            {
                "status": str(item.get("status") or AUDIT_STATUS_UNOPENED),
                "modified": bool(item.get("modified", False)),
                "last_modified_by": str(item.get("last_modified_by") or ""),
                "last_saved_at": item.get("last_saved_at"),
                "change_count": max(0, int(item.get("change_count") or 0)),
            }
        )
    return result


def read_last_audit_event_hash(audit_path: Path) -> str:
    if not audit_path.is_file():
        return ""
    try:
        lines = audit_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in reversed(lines):
        text = line.strip()
        if not text:
            continue
        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            event_hash = str(event.get("event_hash", "")).strip()
            if event_hash:
                return event_hash
    return ""


def finalize_audit_event_hash(event: Dict[str, Any]) -> str:
    payload = copy.deepcopy(event)
    payload.pop("event_hash", None)
    return audit_payload_hash(payload)


@dataclass(frozen=True)
class AnnotatorConfig:
    annotator_id: str
    name: str
    role: str
    start_index: int
    count: int
    seed_input_path: Path
    output_path: Path

    @property
    def stop_index(self) -> int:
        return self.start_index + self.count

    @property
    def is_reviewer(self) -> bool:
        return self.role == ANNOTATOR_ROLE_REVIEWER


@dataclass(frozen=True)
class MultiUserManifest:
    manifest_path: Path
    version: int
    master_input_path: Path
    merged_output_path: Path
    annotators: List[AnnotatorConfig]
    input_format: str
    total_editable_items: int
    load_warnings: List[str] = field(default_factory=list)

    @classmethod
    def load(cls, manifest_path: Path) -> "MultiUserManifest":
        path = manifest_path.expanduser().resolve()
        if not path.is_file():
            raise ValidationError(f"Multi-user manifest not found: {path}")

        try:
            with path.open("r", encoding="utf-8") as file:
                raw = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Invalid multi-user manifest JSON: {exc.msg}") from exc

        if not isinstance(raw, dict):
            raise ValidationError("Multi-user manifest must be a JSON object")

        version = _ensure_int(raw.get("version"), "manifest.version")
        if version != MULTI_USER_MANIFEST_VERSION:
            raise ValidationError(
                f"manifest.version must be {MULTI_USER_MANIFEST_VERSION}, got {version}"
            )

        base_dir = path.parent
        master_input_path = resolve_user_path(raw.get("master_input_json"), base_dir, "master_input_json")
        if not master_input_path.is_file():
            raise ValidationError(f"master_input_json not found: {master_input_path}")

        loaded_master = AnnotationStore._load_items_from_path(master_input_path)
        total_editable_items = len(loaded_master.items)
        if total_editable_items <= 0:
            raise ValidationError(f"No editable annotation items found in {master_input_path}")

        expected_suffix = ".jsonl" if loaded_master.input_format == INPUT_FORMAT_JSONL else ".json"
        merged_output_path = resolve_user_path(
            raw.get("merged_output_json"),
            base_dir,
            "merged_output_json",
        )
        if merged_output_path.suffix.lower() != expected_suffix:
            raise ValidationError(
                f"merged_output_json must use {expected_suffix} for master input format {loaded_master.input_format}"
            )

        raw_annotators = raw.get("annotators")
        if not isinstance(raw_annotators, list) or not raw_annotators:
            raise ValidationError("annotators must be a non-empty list")

        annotators: List[AnnotatorConfig] = []
        annotator_ids: Set[str] = set()
        seed_paths: Set[Path] = set()
        output_paths: Set[Path] = set()
        for idx, item in enumerate(raw_annotators):
            if not isinstance(item, dict):
                raise ValidationError(f"annotators[{idx}] must be an object")

            annotator_id = str(item.get("id", "")).strip()
            if not annotator_id:
                raise ValidationError(f"annotators[{idx}].id is required")
            if annotator_id in annotator_ids:
                raise ValidationError(f"annotators[{idx}].id is duplicated: {annotator_id}")

            name = str(item.get("name", "")).strip() or annotator_id
            role = str(item.get("role", ANNOTATOR_ROLE_ANNOTATOR)).strip().lower()
            if role not in ANNOTATOR_ROLES:
                allowed = ", ".join(sorted(ANNOTATOR_ROLES))
                raise ValidationError(f"annotators[{idx}].role must be one of: {allowed}")

            if role == ANNOTATOR_ROLE_REVIEWER:
                start_index = _ensure_int(
                    item.get("start_index", 0),
                    f"annotators[{idx}].start_index",
                )
                count = _ensure_int(
                    item.get("count", total_editable_items),
                    f"annotators[{idx}].count",
                )
            else:
                start_index = _ensure_int(item.get("start_index"), f"annotators[{idx}].start_index")
                count = _ensure_int(item.get("count"), f"annotators[{idx}].count")
            if start_index < 0:
                raise ValidationError(f"annotators[{idx}].start_index must be >= 0")
            if count <= 0:
                raise ValidationError(f"annotators[{idx}].count must be > 0")
            if role == ANNOTATOR_ROLE_REVIEWER:
                if start_index != 0:
                    raise ValidationError(f"annotators[{idx}].start_index must be 0 for reviewer")
                if count != total_editable_items:
                    raise ValidationError(
                        f"annotators[{idx}].count must be {total_editable_items} for reviewer"
                    )

            seed_input_raw = str(item.get("seed_input_json", "")).strip()
            if seed_input_raw:
                seed_input_path = resolve_user_path(
                    seed_input_raw,
                    base_dir,
                    f"annotators[{idx}].seed_input_json",
                )
            else:
                seed_input_path = default_seed_input_path(master_input_path, annotator_id)
            output_path = resolve_user_path(
                item.get("output_json"),
                base_dir,
                f"annotators[{idx}].output_json",
            )

            if seed_input_path.suffix.lower() != expected_suffix:
                raise ValidationError(
                    f"annotators[{idx}].seed_input_json must use {expected_suffix}"
                )
            if output_path.suffix.lower() != expected_suffix:
                raise ValidationError(
                    f"annotators[{idx}].output_json must use {expected_suffix}"
                )
            if output_path in output_paths:
                raise ValidationError(
                    f"annotators[{idx}].output_json is duplicated: {output_path}"
                )
            if seed_input_path in seed_paths:
                raise ValidationError(
                    f"annotators[{idx}].seed_input_json resolves to a duplicated path: {seed_input_path}"
                )

            annotator_ids.add(annotator_id)
            seed_paths.add(seed_input_path)
            output_paths.add(output_path)
            annotators.append(
                AnnotatorConfig(
                    annotator_id=annotator_id,
                    name=name,
                    role=role,
                    start_index=start_index,
                    count=count,
                    seed_input_path=seed_input_path,
                    output_path=output_path,
                )
            )

        load_warnings: List[str] = []
        sorted_annotators = sorted(
            [annotator for annotator in annotators if not annotator.is_reviewer],
            key=lambda item: item.start_index,
        )
        next_expected_start = 0
        for idx, annotator in enumerate(sorted_annotators):
            if annotator.stop_index > total_editable_items:
                raise ValidationError(
                    f"annotator '{annotator.annotator_id}' range [{annotator.start_index}, {annotator.stop_index}) "
                    f"exceeds total editable items {total_editable_items}"
                )
            if idx > 0 and annotator.start_index < next_expected_start:
                raise ValidationError(
                    f"annotator '{annotator.annotator_id}' range overlaps a previous assignment"
                )
            if annotator.start_index > next_expected_start:
                load_warnings.append(
                    f"assignment gap: items [{next_expected_start}, {annotator.start_index}) "
                    "are not assigned to any annotator"
                )
            next_expected_start = annotator.stop_index
        if next_expected_start < total_editable_items:
            load_warnings.append(
                f"assignment gap: items [{next_expected_start}, {total_editable_items}) "
                "are not assigned to any annotator"
            )

        return cls(
            manifest_path=path,
            version=version,
            master_input_path=master_input_path,
            merged_output_path=merged_output_path,
            annotators=annotators,
            input_format=loaded_master.input_format,
            total_editable_items=total_editable_items,
            load_warnings=load_warnings,
        )

    def get_annotator(self, annotator_id: str) -> Optional[AnnotatorConfig]:
        needle = str(annotator_id or "").strip()
        if not needle:
            return None
        for annotator in self.annotators:
            if annotator.annotator_id == needle:
                return annotator
        return None


@dataclass
class MultiUserStoreManager:
    manifest: MultiUserManifest
    runtime_config: RuntimeConfig
    stores: Dict[str, "AnnotationStore"] = field(default_factory=dict)
    store_source_signatures: Dict[str, Tuple[Tuple[str, bool, int, int], ...]] = field(
        default_factory=dict
    )
    _store_locks: Dict[str, threading.RLock] = field(default_factory=dict, repr=False)
    _manager_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _get_lock(self, annotator_id: str) -> threading.RLock:
        with self._manager_lock:
            lock = self._store_locks.get(annotator_id)
            if lock is None:
                lock = threading.RLock()
                self._store_locks[annotator_id] = lock
            return lock

    def get_annotator(self, annotator_id: str) -> Optional[AnnotatorConfig]:
        return self.manifest.get_annotator(annotator_id)

    def require_annotator(self, annotator_id: str) -> AnnotatorConfig:
        annotator = self.get_annotator(annotator_id)
        if annotator is None:
            raise ValidationError(f"Unknown annotator_id: {annotator_id}")
        return annotator

    @staticmethod
    def _path_signature(path: Path) -> Tuple[str, bool, int, int]:
        resolved = str(path.expanduser().resolve())
        try:
            stat_result = path.stat()
        except OSError:
            return resolved, False, -1, -1
        if not path.is_file():
            return resolved, False, -1, -1
        return resolved, True, int(stat_result.st_mtime_ns), int(stat_result.st_size)

    def _source_paths_for_annotator(self, annotator: AnnotatorConfig) -> List[Path]:
        if annotator.is_reviewer:
            paths = [self.manifest.master_input_path, annotator.output_path]
            for candidate in self.manifest.annotators:
                if candidate.is_reviewer:
                    continue
                paths.append(candidate.output_path)
                paths.append(candidate.seed_input_path)
            return paths
        return [annotator.output_path, annotator.seed_input_path]

    def _source_signature_for_annotator(
        self,
        annotator: AnnotatorConfig,
    ) -> Tuple[Tuple[str, bool, int, int], ...]:
        return tuple(
            self._path_signature(path)
            for path in self._source_paths_for_annotator(annotator)
        )

    @staticmethod
    def _annotation_compare_payload(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            payload = {
                "tasks": sanitize_tasks(item.get("tasks", [])),
                "action_config": normalize_segments(item.get("action_config", [])),
            }
            for field_name in DISCARD_FIELDS:
                if field_name in item:
                    payload[field_name] = copy.deepcopy(item.get(field_name))
            return payload
        except ValidationError:
            return None

    @classmethod
    def _annotations_match(cls, current_item: Dict[str, Any], backup_item: Any) -> bool:
        if not isinstance(backup_item, dict):
            return False
        current_payload = cls._annotation_compare_payload(current_item)
        backup_payload = cls._annotation_compare_payload(backup_item)
        return current_payload is not None and current_payload == backup_payload

    @staticmethod
    def _apply_reviewer_overlay(base_item: Dict[str, Any], reviewer_item: Dict[str, Any]) -> None:
        base_item["tasks"] = copy.deepcopy(reviewer_item.get("tasks", []))
        base_item["action_config"] = copy.deepcopy(reviewer_item.get("action_config", []))
        for field_name in DISCARD_FIELDS:
            if field_name in reviewer_item:
                base_item[field_name] = copy.deepcopy(reviewer_item.get(field_name))
            else:
                base_item.pop(field_name, None)
        backup = reviewer_item.get(REVIEW_BACKUP_FIELD)
        if isinstance(backup, dict):
            base_item[REVIEW_BACKUP_FIELD] = copy.deepcopy(backup)

    def _load_store_from_output_or_seed(self, annotator: AnnotatorConfig) -> "AnnotationStore":
        output_path = annotator.output_path
        if output_path.exists() and output_path.is_file():
            try:
                output_store = AnnotationStore.load(
                    input_path=output_path,
                    output_path=output_path,
                    runtime_config=self.runtime_config,
                )
                if len(output_store.session_data) != annotator.count:
                    raise ValidationError(
                        f"annotator '{annotator.annotator_id}' output_json item count mismatch: "
                        f"expected {annotator.count}, got {len(output_store.session_data)}"
                    )
                return output_store
            except (ValidationError, OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "annotator '%s' output restore failed; refusing seed fallback: %s",
                    annotator.annotator_id,
                    exc,
                )
                raise ValidationError(
                    f"annotator '{annotator.annotator_id}' output_json exists but cannot be restored: {exc}"
                ) from exc

        seed_store = AnnotationStore.load(
            input_path=annotator.seed_input_path,
            output_path=annotator.output_path,
            runtime_config=self.runtime_config,
        )
        if len(seed_store.session_data) != annotator.count:
            raise ValidationError(
                f"annotator '{annotator.annotator_id}' seed_input_json item count mismatch: "
                f"expected {annotator.count}, got {len(seed_store.session_data)}"
            )
        return seed_store

    def _load_raw_items_from_output_or_seed(self, annotator: AnnotatorConfig) -> Tuple[List[Dict[str, Any]], List[str]]:
        warnings: List[str] = []
        output_path = annotator.output_path
        if output_path.exists() and output_path.is_file():
            try:
                loaded_output = AnnotationStore._load_items_from_path(output_path)
                if len(loaded_output.items) != annotator.count:
                    raise ValidationError(
                        f"expected {annotator.count}, got {len(loaded_output.items)}"
                    )
                return [copy.deepcopy(item) for item in loaded_output.items], list(loaded_output.load_warnings)
            except (ValidationError, OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "annotator '%s' output restore failed for reviewer merge; refusing seed fallback: %s",
                    annotator.annotator_id,
                    exc,
                )
                raise ValidationError(
                    f"annotator '{annotator.annotator_id}' output_json exists but cannot be restored: {exc}"
                ) from exc

        loaded_seed = AnnotationStore._load_items_from_path(annotator.seed_input_path)
        if len(loaded_seed.items) != annotator.count:
            raise ValidationError(
                f"annotator '{annotator.annotator_id}' seed_input_json item count mismatch: "
                f"expected {annotator.count}, got {len(loaded_seed.items)}"
            )
        warnings.extend(loaded_seed.load_warnings)
        return [copy.deepcopy(item) for item in loaded_seed.items], warnings

    def _build_reviewer_base_store(self, reviewer: AnnotatorConfig) -> "AnnotationStore":
        loaded_master = AnnotationStore._load_items_from_path(self.manifest.master_input_path)
        merged_items = [copy.deepcopy(item) for item in loaded_master.items]
        load_warnings = list(loaded_master.load_warnings)

        for annotator in self.manifest.annotators:
            if annotator.is_reviewer:
                continue
            raw_items, warnings = self._load_raw_items_from_output_or_seed(annotator)
            load_warnings.extend(warnings)
            for local_index, item in enumerate(raw_items):
                global_index = annotator.start_index + local_index
                if global_index < 0 or global_index >= len(merged_items):
                    raise ValidationError(
                        f"annotator '{annotator.annotator_id}' maps local item[{local_index}] "
                        f"to invalid global item[{global_index}]"
                    )
                merged_items[global_index] = copy.deepcopy(item)

        loaded_reviewer = LoadedInput(
            input_format=loaded_master.input_format,
            items=merged_items,
            source_mappings=loaded_master.source_mappings,
            load_warnings=load_warnings,
            jsonl_rows=loaded_master.jsonl_rows,
        )
        return AnnotationStore.from_loaded_input(
            loaded=loaded_reviewer,
            input_path=self.manifest.master_input_path,
            output_path=reviewer.output_path,
            runtime_config=self.runtime_config,
            review_mode=True,
        )

    def _load_reviewer_overlay_store(self, reviewer: AnnotatorConfig) -> Optional["AnnotationStore"]:
        if not reviewer.output_path.exists() or not reviewer.output_path.is_file():
            return None
        try:
            reviewer_store = AnnotationStore.load(
                input_path=reviewer.output_path,
                output_path=reviewer.output_path,
                runtime_config=self.runtime_config,
            )
            if len(reviewer_store.session_data) != self.manifest.total_editable_items:
                raise ValidationError(
                    f"expected {self.manifest.total_editable_items}, "
                    f"got {len(reviewer_store.session_data)}"
            )
            return reviewer_store
        except (ValidationError, OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "reviewer '%s' output overlay restore failed; refusing to skip: %s",
                reviewer.annotator_id,
                exc,
            )
            raise ValidationError(
                f"reviewer '{reviewer.annotator_id}' output_json exists but cannot be restored: {exc}"
            ) from exc

    def _load_reviewer_store(self, reviewer: AnnotatorConfig) -> "AnnotationStore":
        reviewer_store = self._build_reviewer_base_store(reviewer)
        overlay_store = self._load_reviewer_overlay_store(reviewer)
        if overlay_store is None:
            return reviewer_store

        applied_count = 0
        skipped_count = 0
        for idx, overlay_item in enumerate(overlay_store.session_data):
            if idx >= len(reviewer_store.session_data):
                break
            backup = overlay_item.get(REVIEW_BACKUP_FIELD)
            if self._annotations_match(reviewer_store.session_data[idx], backup):
                self._apply_reviewer_overlay(reviewer_store.session_data[idx], overlay_item)
                applied_count += 1
            elif isinstance(backup, dict):
                skipped_count += 1

        if skipped_count:
            logger.info(
                "reviewer '%s' overlay applied=%s skipped_after_annotator_update=%s",
                reviewer.annotator_id,
                applied_count,
                skipped_count,
            )
        if applied_count:
            reviewer_store._rebuild_legacy_options_from_session()
            reviewer_store._refresh_options()
        reviewer_store.load_warnings.extend(overlay_store.load_warnings)
        return reviewer_store

    def get_store(self, annotator_id: str) -> "AnnotationStore":
        annotator = self.require_annotator(annotator_id)
        lock = self._get_lock(annotator.annotator_id)
        with lock:
            cached = self.stores.get(annotator.annotator_id)
            current_signature = self._source_signature_for_annotator(annotator)
            cached_signature = self.store_source_signatures.get(annotator.annotator_id)
            if (
                cached is not None
                and (cached_signature == current_signature or cached.dirty_indices)
            ):
                return cached
            if annotator.is_reviewer:
                loaded = self._load_reviewer_store(annotator)
            else:
                loaded = self._load_store_from_output_or_seed(annotator)
            self.stores[annotator.annotator_id] = loaded
            self.store_source_signatures[annotator.annotator_id] = (
                self._source_signature_for_annotator(annotator)
            )
            return loaded

    def update_annotation(self, annotator_id: str, index: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        lock = self._get_lock(annotator_id)
        with lock:
            store = self.get_store(annotator_id)
            return store.update_annotation(index, payload)

    def update_discard_and_save(
        self,
        annotator_id: str,
        index: int,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        annotator = self.require_annotator(annotator_id)
        lock = self._get_lock(annotator_id)
        with lock:
            store = self.get_store(annotator_id)
            result = store.update_discard_and_save(index, payload, annotator.annotator_id)
            self.store_source_signatures[annotator.annotator_id] = (
                self._source_signature_for_annotator(annotator)
            )
            return result

    def save(self, annotator_id: str) -> Dict[str, Any]:
        annotator = self.require_annotator(annotator_id)
        lock = self._get_lock(annotator_id)
        with lock:
            store = self.get_store(annotator_id)
            result = store.save()
            self.store_source_signatures[annotator.annotator_id] = (
                self._source_signature_for_annotator(annotator)
            )
            return result


@dataclass
class ModelReviewStore:
    review_path: Path
    reviews_by_line_no: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    reviews_by_sample_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    load_warnings: List[str] = field(default_factory=list)

    @classmethod
    def load(cls, review_path: Path) -> "ModelReviewStore":
        path = review_path.expanduser().resolve()
        if not path.is_file():
            raise ValidationError(f"model_review_json not found: {path}")

        store = cls(review_path=path)
        with path.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    raw = json.loads(text)
                except json.JSONDecodeError as exc:
                    store.load_warnings.append(
                        f"model review line {line_no} skipped: invalid JSON ({exc.msg})"
                    )
                    continue
                if not isinstance(raw, dict):
                    store.load_warnings.append(
                        f"model review line {line_no} skipped: expected object"
                    )
                    continue
                try:
                    review = store._normalize_review(raw, line_no)
                except ValidationError as exc:
                    store.load_warnings.append(
                        f"model review line {line_no} skipped: {exc}"
                    )
                    continue

                review_line_no = review.get("line_no")
                if isinstance(review_line_no, int):
                    store.reviews_by_line_no[review_line_no] = review

                sample_id = str(review.get("sample_id", "")).strip()
                if sample_id:
                    store.reviews_by_sample_id[sample_id] = review

        return store

    @staticmethod
    def _coerce_int(value: Any, field_name: str, default: Optional[int] = None) -> Optional[int]:
        if value is None or value == "":
            return default
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            if default is not None:
                return default
            raise ValidationError(f"{field_name} must be an integer") from exc

    @classmethod
    def _normalize_evidence_frames(cls, raw_frames: Any) -> List[int]:
        if raw_frames is None:
            return []
        if not isinstance(raw_frames, list):
            raw_frames = [raw_frames]
        frames: List[int] = []
        seen: Set[int] = set()
        for raw in raw_frames:
            if isinstance(raw, dict):
                raw = raw.get("frame", raw.get("frame_id"))
            frame = cls._coerce_int(raw, "evidence_frames[]", default=None)
            if frame is None or frame in seen:
                continue
            seen.add(frame)
            frames.append(frame)
        return frames

    @classmethod
    def _normalize_issue(cls, raw_issue: Any) -> Dict[str, Any]:
        if not isinstance(raw_issue, dict):
            raise ValidationError("issue must be an object")
        seg_id = cls._coerce_int(raw_issue.get("seg_id"), "issue.seg_id", default=None)
        raw_type = str(raw_issue.get("issue_type", "")).strip() or "unclear_visual"
        issue_type = raw_type if raw_type in MODEL_REVIEW_ISSUE_TYPES else "unclear_visual"
        confidence = raw_issue.get("confidence", 0)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0
        confidence_value = max(0.0, min(1.0, confidence_value))
        return {
            "seg_id": seg_id,
            "issue_type": issue_type,
            "evidence_frames": cls._normalize_evidence_frames(raw_issue.get("evidence_frames", [])),
            "reason": str(raw_issue.get("reason", "")).strip(),
            "suggested_fix": str(raw_issue.get("suggested_fix", "")).strip(),
            "confidence": confidence_value,
        }

    @classmethod
    def _normalize_review(cls, raw: Dict[str, Any], fallback_line_no: int) -> Dict[str, Any]:
        line_no = cls._coerce_int(raw.get("line_no"), "line_no", default=None)
        verdict_raw = str(raw.get("verdict", "")).strip().lower()
        verdict = verdict_raw if verdict_raw in MODEL_REVIEW_VERDICTS else "uncertain"
        issues: List[Dict[str, Any]] = []
        raw_issues = raw.get("issues", [])
        if raw_issues is None:
            raw_issues = []
        if not isinstance(raw_issues, list):
            raise ValidationError("issues must be a list")
        for issue in raw_issues:
            issues.append(cls._normalize_issue(issue))

        if verdict == "incorrect" and not issues:
            verdict = "uncertain"
        if verdict == "correct" and issues:
            verdict = "incorrect"

        return {
            "line_no": line_no,
            "review_line_no": fallback_line_no,
            "sample_id": str(raw.get("sample_id", "")).strip(),
            "verdict": verdict,
            "issues": issues,
            "summary": str(raw.get("summary", "")).strip(),
            "model": str(raw.get("model", "")).strip(),
            "reviewed_at": str(raw.get("reviewed_at", "")).strip(),
        }

    @staticmethod
    def _sample_ids_for_item(item: Dict[str, Any]) -> List[str]:
        candidates = [
            item.get("episode_index"),
            item.get("path"),
        ]
        views = item.get("views")
        if isinstance(views, dict):
            candidates.extend(views.values())
        result: List[str] = []
        seen: Set[str] = set()
        for value in candidates:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    def get_review(self, store: "AnnotationStore", index: int) -> Optional[Dict[str, Any]]:
        if index < 0 or index >= len(store.session_data):
            return None
        mapping = store.source_mappings[index] if index < len(store.source_mappings) else {}
        line_no = mapping.get("line_no")
        if isinstance(line_no, int):
            review = self.reviews_by_line_no.get(line_no)
            if review is not None:
                return copy.deepcopy(review)

        item = store.session_data[index]
        for sample_id in self._sample_ids_for_item(item):
            review = self.reviews_by_sample_id.get(sample_id)
            if review is not None:
                return copy.deepcopy(review)
        return None

    def build_meta(self, store: "AnnotationStore", index: int) -> Dict[str, Any]:
        review = self.get_review(store, index)
        if review is None:
            return {
                "review_verdict": "unreviewed",
                "review_issue_count": 0,
                "review_uncertain_count": 0,
                "review_summary": "",
            }
        issues = review.get("issues", [])
        uncertain_count = sum(
            1
            for issue in issues
            if str(issue.get("issue_type", "")) == "unclear_visual"
        )
        return {
            "review_verdict": str(review.get("verdict", "unreviewed")),
            "review_issue_count": len(issues),
            "review_uncertain_count": uncertain_count,
            "review_summary": str(review.get("summary", "")),
        }


@dataclass
class AnnotationStore:
    input_path: Path
    output_path: Path
    session_data: List[Dict[str, Any]]
    configured_skill_options: List[str]
    legacy_skill_options: List[str]
    skill_options: List[str]
    configured_action_text_options: List[str]
    legacy_action_text_options: List[str]
    action_text_options: List[str]
    input_format: str = INPUT_FORMAT_JSON
    source_mappings: List[Dict[str, Any]] = field(default_factory=list)
    jsonl_rows: Optional[List[Dict[str, Any]]] = None
    load_warnings: List[str] = field(default_factory=list)
    dirty_indices: Set[int] = field(default_factory=set)
    runtime_config: Optional[RuntimeConfig] = None
    review_mode: bool = False
    pending_review_backups: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    _boto3_clients: Dict[str, Any] = field(default_factory=dict, repr=False)
    _proxy_disabled_once: bool = field(default=False, repr=False)
    _playback_cache: Dict[Tuple[int, str, str], Dict[str, Any]] = field(default_factory=dict, repr=False)

    @staticmethod
    def _resolve_action_text_options(
        configured_action_text_options: List[str],
        legacy_action_text_options: List[str],
    ) -> List[str]:
        # Configured action-text options are authoritative when present.
        if configured_action_text_options:
            return list(configured_action_text_options)
        return list(legacy_action_text_options)

    @staticmethod
    def _load_items_from_path(input_path: Path) -> LoadedInput:
        suffix = input_path.suffix.lower()
        if suffix == ".jsonl":
            items: List[Dict[str, Any]] = []
            source_mappings: List[Dict[str, Any]] = []
            load_warnings: List[str] = []
            jsonl_rows: List[Dict[str, Any]] = []
            with input_path.open("r", encoding="utf-8") as file:
                for line_no, line in enumerate(file, start=1):
                    text = line.rstrip("\n")
                    if not text.strip():
                        jsonl_rows.append({"line_no": line_no, "blank": True, "value": None})
                        continue

                    try:
                        row = json.loads(text)
                    except json.JSONDecodeError as exc:
                        raise ValidationError(
                            f"Invalid JSONL at line {line_no}: {exc.msg}"
                        ) from exc

                    jsonl_rows.append({"line_no": line_no, "blank": False, "value": row})

                    selected_item: Optional[Dict[str, Any]] = None
                    mapping: Optional[Dict[str, Any]] = None

                    if isinstance(row, dict):
                        result_obj = row.get("result")
                        if is_annotation_candidate(result_obj):
                            selected_item = result_obj
                            mapping = {
                                "line_index": len(jsonl_rows) - 1,
                                "line_no": line_no,
                                "target_type": "dict_result",
                            }
                        elif is_annotation_candidate(row):
                            selected_item = row
                            mapping = {
                                "line_index": len(jsonl_rows) - 1,
                                "line_no": line_no,
                                "target_type": "root_dict",
                            }
                    elif isinstance(row, list):
                        for idx, element in enumerate(row):
                            if is_annotation_candidate(element):
                                selected_item = element
                                mapping = {
                                    "line_index": len(jsonl_rows) - 1,
                                    "line_no": line_no,
                                    "target_type": "list_item",
                                    "list_index": idx,
                                }
                                break

                    if selected_item is None or mapping is None:
                        load_warnings.append(
                            f"JSONL line {line_no} skipped: no editable annotation object found"
                        )
                        continue

                    items.append(selected_item)
                    source_mappings.append(mapping)

            return LoadedInput(
                input_format=INPUT_FORMAT_JSONL,
                items=items,
                source_mappings=source_mappings,
                load_warnings=load_warnings,
                jsonl_rows=jsonl_rows,
            )

        with input_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        if not isinstance(raw, list):
            raise ValidationError("Input JSON must be an array of VideoItem objects")

        items: List[Dict[str, Any]] = []
        source_mappings: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValidationError(f"Item[{idx}] must be an object")
            items.append(item)
            source_mappings.append(
                {
                    "target_type": "json_array",
                    "item_index": idx,
                }
            )

        return LoadedInput(
            input_format=INPUT_FORMAT_JSON,
            items=items,
            source_mappings=source_mappings,
        )

    @classmethod
    def load(
        cls,
        input_path: Path,
        output_path: Path,
        runtime_config: RuntimeConfig,
    ) -> "AnnotationStore":
        loaded = cls._load_items_from_path(input_path)
        return cls.from_loaded_input(
            loaded=loaded,
            input_path=input_path,
            output_path=output_path,
            runtime_config=runtime_config,
        )

    @classmethod
    def from_loaded_input(
        cls,
        loaded: LoadedInput,
        input_path: Path,
        output_path: Path,
        runtime_config: RuntimeConfig,
        review_mode: bool = False,
    ) -> "AnnotationStore":
        raw = loaded.items

        session_data: List[Dict[str, Any]] = []
        source_mappings: List[Dict[str, Any]] = []
        load_warnings = list(loaded.load_warnings)
        configured_skill_options = list(runtime_config.configured_skill_options)
        configured_skill_set = set(configured_skill_options)
        configured_action_text_options = list(runtime_config.configured_action_text_options)
        configured_action_text_set = set(configured_action_text_options)
        legacy_skill_options: List[str] = []
        legacy_skill_seen = set()
        legacy_action_text_options: List[str] = []
        legacy_action_text_seen = set()

        if len(loaded.source_mappings) != len(raw):
            raise ValidationError("Internal load error: source mapping count mismatch")

        for idx, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValidationError(f"Item[{idx}] must be an object")

            input_mapping = copy.deepcopy(loaded.source_mappings[idx])
            public_input = "segments" in item and "action_config" not in item
            if public_input:
                from roboatom.validation.data import canonical_row
                canonical = canonical_row(item)
                item = copy.deepcopy(item)
                item["action_config"] = [
                    {"seg_id": i, "start_frame": segment["start_frame"],
                     "end_frame": segment["end_frame"] + 1,
                     "skill": segment["skill"], "action_text": segment["subtask"]}
                    for i, segment in enumerate(canonical["segments"])
                ]
                item["tasks"] = item.get("tasks", [str(item.get("task", item.get("task_id", "")))])
                input_mapping["public_input"] = True

            frames_field = "num_frames" if "num_frames" in item else ("frames" if "frames" in item else ("length" if "length" in item else None))
            if not frames_field:
                raise ValidationError(
                    f"Item[{idx}] missing required frame field: need one of frames or length"
                )

            frames = _ensure_int(item.get(frames_field), f"item[{idx}].{frames_field}")
            fps = _ensure_float(item.get("fps"), f"item[{idx}].fps")
            if fps <= 0:
                raise ValidationError(f"item[{idx}].fps must be greater than 0")

            tasks = sanitize_tasks(item.get("tasks"))
            raw_segments = normalize_segments(item.get("action_config"))
            interval_style = detect_interval_style(raw_segments, frames)
            segments = convert_segments_to_open(raw_segments, interval_style)
            validate_segments(segments, frames)

            normalized_item = copy.deepcopy(item)
            raw_episode_index = str(item.get("episode_index", item.get("episode_id", ""))).strip()
            views, view_order, has_views_input = normalize_views(item, idx)
            ordered_views = {view_key: str(views.get(view_key, "")).strip() for view_key in view_order}
            primary_view_key = view_order[0]
            primary_view_path = str(ordered_views.get(primary_view_key, "")).strip()

            view_states: Dict[str, Dict[str, Any]] = {}
            unavailable_view_keys: List[str] = []
            for view_key in view_order:
                view_path = str(ordered_views.get(view_key, "")).strip()
                view_errors: List[str] = []
                unavailable = False
                if not view_path:
                    unavailable = True
                    view_errors.append("Video path is missing")
                elif not is_remote_path(view_path):
                    video_path = resolve_media_path(view_path)
                    unavailable = not video_path.is_file()
                    if unavailable:
                        view_errors.append(f"Video file not found: {video_path}")

                if unavailable:
                    unavailable_view_keys.append(view_key)
                view_states[view_key] = {
                    "key": view_key,
                    "path": view_path,
                    "unavailable": unavailable,
                    "error": " | ".join(view_errors),
                }

            prior_error = str(item.get("error", "")).strip()
            errors: List[str] = []
            if prior_error:
                errors.append(prior_error)

            unavailable_count = len(unavailable_view_keys)
            total_views = len(view_order)
            all_unavailable = unavailable_count == total_views
            available_primary = next(
                (view_key for view_key in view_order if not bool(view_states.get(view_key, {}).get("unavailable", False))),
                primary_view_key,
            )
            primary_view_key = available_primary
            primary_view_path = str(ordered_views.get(primary_view_key, "")).strip()
            effective_episode_index = raw_episode_index or primary_view_path
            if all_unavailable:
                detail = "; ".join(
                    [
                        f"{view_key}: {view_states[view_key]['error'] or 'unavailable'}"
                        for view_key in unavailable_view_keys
                    ]
                )
                errors.append(f"All views are unavailable ({detail})")
                line_no = input_mapping.get("line_no")
                if line_no is not None:
                    load_warnings.append(
                        f"item[{idx}] (line {line_no}) is not playable: all views are unavailable"
                    )
                else:
                    load_warnings.append(
                        f"item[{idx}] is not playable: all views are unavailable"
                    )
            elif unavailable_count > 0:
                errors.append(
                    f"Some views are unavailable: {', '.join(unavailable_view_keys)}"
                )
                line_no = input_mapping.get("line_no")
                if line_no is not None:
                    load_warnings.append(
                        f"item[{idx}] (line {line_no}) has unavailable views: {', '.join(unavailable_view_keys)}"
                    )
                else:
                    load_warnings.append(
                        f"item[{idx}] has unavailable views: {', '.join(unavailable_view_keys)}"
                    )

            normalized_item["frames"] = frames
            normalized_item["fps"] = fps
            normalized_item["tasks"] = tasks
            normalized_item["action_config"] = segments
            review_backup = normalized_item.get(REVIEW_BACKUP_FIELD)
            if isinstance(review_backup, dict) and isinstance(review_backup.get("action_config"), list):
                backup_segments = normalize_segments(review_backup.get("action_config"))
                backup_interval_style = detect_interval_style(backup_segments, frames)
                review_backup["action_config"] = convert_segments_to_open(
                    backup_segments,
                    backup_interval_style,
                )
            normalized_item["views"] = ordered_views
            normalized_item["view_order"] = view_order
            normalized_item["view_states"] = view_states
            normalized_item["primary_view_key"] = primary_view_key
            normalized_item["path"] = primary_view_path
            normalized_item["episode_index"] = effective_episode_index
            normalized_item["unavailable"] = all_unavailable
            normalized_item["error"] = " | ".join([msg for msg in errors if msg])

            for segment in segments:
                skill_value = str(segment["skill"]).strip()
                if (
                    skill_value
                    and skill_value not in configured_skill_set
                    and skill_value not in legacy_skill_seen
                ):
                    legacy_skill_seen.add(skill_value)
                    legacy_skill_options.append(skill_value)

                action_text_value = str(segment["action_text"]).strip()
                if (
                    action_text_value
                    and action_text_value not in configured_action_text_set
                    and action_text_value not in legacy_action_text_seen
                ):
                    legacy_action_text_seen.add(action_text_value)
                    legacy_action_text_options.append(action_text_value)

            input_mapping["had_path"] = "path" in item
            input_mapping["had_episode_index"] = "episode_index" in item
            input_mapping["had_views"] = has_views_input
            input_mapping["had_view_order"] = "view_order" in item
            input_mapping["had_frames"] = "frames" in item
            input_mapping["had_length"] = "length" in item
            input_mapping["had_error"] = "error" in item
            input_mapping["had_unavailable"] = "unavailable" in item
            input_mapping["original_error"] = item.get("error")
            input_mapping["interval_style"] = interval_style

            source_mappings.append(input_mapping)
            session_data.append(normalized_item)

        skill_options = configured_skill_options + legacy_skill_options
        action_text_options = cls._resolve_action_text_options(
            configured_action_text_options,
            legacy_action_text_options,
        )
        return cls(
            input_path=input_path,
            output_path=output_path,
            session_data=session_data,
            configured_skill_options=configured_skill_options,
            legacy_skill_options=legacy_skill_options,
            skill_options=skill_options,
            configured_action_text_options=configured_action_text_options,
            legacy_action_text_options=legacy_action_text_options,
            action_text_options=action_text_options,
            input_format=loaded.input_format,
            source_mappings=source_mappings,
            jsonl_rows=loaded.jsonl_rows,
            load_warnings=load_warnings,
            runtime_config=runtime_config,
            review_mode=review_mode,
        )

    def get_meta(self) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for idx, item in enumerate(self.session_data):
            view_order = list(item.get("view_order") or [])
            view_states = item.get("view_states") if isinstance(item.get("view_states"), dict) else {}
            unavailable_views = 0
            for view_key in view_order:
                state = view_states.get(view_key) if isinstance(view_states, dict) else None
                if isinstance(state, dict) and bool(state.get("unavailable", False)):
                    unavailable_views += 1
            result.append(
                {
                    "index": idx,
                    "episode_index": item["episode_index"],
                    "path": item["path"],
                    "frames": item["frames"],
                    "fps": item["fps"],
                    "tasks_count": len(item["tasks"]),
                    "segments_count": len(item["action_config"]),
                    "unavailable": bool(item.get("unavailable", False)),
                    "error": str(item.get("error", "")),
                    "view_count": len(view_order),
                    "view_keys": view_order,
                    "views_unavailable_count": unavailable_views,
                    "views_available_count": max(0, len(view_order) - unavailable_views),
                    "discarded": bool(item.get("discarded", False)),
                    "discard_reason": str(item.get("discard_reason", "")),
                }
            )
        return result

    def get_video(self, index: int, include_frame_map: bool = False) -> VideoPayload:
        item = self._get_item(index)
        view_order = list(item.get("view_order") or [])
        view_states = item.get("view_states") if isinstance(item.get("view_states"), dict) else {}
        if not view_order:
            fallback_path = str(item.get("path", "")).strip()
            view_order = ["main"]
            view_states = {
                "main": {
                    "key": "main",
                    "path": fallback_path,
                    "unavailable": bool(item.get("unavailable", False)),
                    "error": str(item.get("error", "")),
                }
            }

        views_payload: List[Dict[str, Any]] = []
        for view_key in view_order:
            state = view_states.get(view_key) if isinstance(view_states, dict) else None
            if not isinstance(state, dict):
                state = {
                    "key": view_key,
                    "path": "",
                    "unavailable": True,
                    "error": "View state missing",
                }
            view_error = str(state.get("error", "")).strip()
            view_unavailable = bool(state.get("unavailable", False))
            playback: Dict[str, Any] = {
                "primary_url": "",
                "fallback_url": "",
                "mode": "unavailable",
                "fallback_retry_max": 0,
            }
            playback_error = ""
            if not view_unavailable:
                try:
                    playback = self.build_playback(index, view_key=view_key)
                except (
                    ValidationError,
                    FileNotFoundError,
                    RuntimeError,
                    KeyError,
                    OSError,
                    ValueError,
                    TypeError,
                ) as exc:
                    playback_error = str(exc)
                    view_unavailable = True

            merged_errors = [msg for msg in (view_error, playback_error) if msg]
            views_payload.append(
                {
                    "key": view_key,
                    "path": str(state.get("path", "")),
                    "unavailable": view_unavailable,
                    "error": " | ".join(merged_errors),
                    "playback": playback,
                }
            )

        primary_view_key = str(item.get("primary_view_key", "")).strip()
        if not primary_view_key or primary_view_key not in {view["key"] for view in views_payload}:
            primary_view_key = views_payload[0]["key"] if views_payload else "main"
        primary_view = next((view for view in views_payload if view["key"] == primary_view_key), None)
        top_level_playback = (
            copy.deepcopy(primary_view["playback"])
            if isinstance(primary_view, dict)
            else {
                "primary_url": "",
                "fallback_url": "",
                "mode": "unavailable",
                "fallback_retry_max": 0,
            }
        )
        top_level_path = str(item.get("path", "")).strip()
        if not top_level_path and isinstance(primary_view, dict):
            top_level_path = str(primary_view.get("path", ""))
        payload = {
            "index": index,
            "episode_index": item["episode_index"],
            "path": top_level_path,
            "frames": item["frames"],
            "fps": item["fps"],
            "tasks": item["tasks"],
            "action_config": item["action_config"],
            "unavailable": bool(item.get("unavailable", False)),
            "error": str(item.get("error", "")),
            "playback": top_level_playback,
            "primary_view_key": primary_view_key,
            "view_order": [view["key"] for view in views_payload],
            "views": views_payload,
            "view_count": len(views_payload),
            "discarded": bool(item.get("discarded", False)),
            "discard_reason": str(item.get("discard_reason", "")),
            "discard_note": str(item.get("discard_note", "")),
            "discarded_by": str(item.get("discarded_by", "")),
            "discarded_at": str(item.get("discarded_at", "")),
        }
        if include_frame_map:
            payload["frame_map"] = build_frame_map(item["frames"], item["action_config"])
        return payload

    def update_annotation(self, index: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        item = self._get_item(index)

        if bool(item.get("discarded", False)):
            raise ValidationError("Discarded items must be restored before editing annotations")

        tasks = sanitize_tasks(payload.get("tasks", []))
        segments = normalize_segments(payload.get("action_config", []))
        self._validate_segment_skills(segments)
        validate_segments(segments, int(item["frames"]))

        if self.review_mode and index not in self.pending_review_backups:
            existing_backup = item.get(REVIEW_BACKUP_FIELD)
            if (
                isinstance(existing_backup, dict)
                and isinstance(existing_backup.get("action_config"), list)
            ):
                backup_source = existing_backup
            else:
                backup_source = item
            self.pending_review_backups[index] = {
                "tasks": copy.deepcopy(backup_source.get("tasks", [])),
                "action_config": copy.deepcopy(backup_source.get("action_config", [])),
            }

        item["tasks"] = tasks
        item["action_config"] = segments
        self.dirty_indices.add(index)
        self._rebuild_legacy_options_from_session()
        self._refresh_options()

        return {
            "index": index,
            "tasks": tasks,
            "action_config": segments,
            "skill_options": self.skill_options,
            "action_text_options": self.action_text_options,
        }

    def _ensure_reviewer_backup(self, index: int, item: Dict[str, Any]) -> None:
        if not self.review_mode or index in self.pending_review_backups:
            return
        existing_backup = item.get(REVIEW_BACKUP_FIELD)
        backup_source = existing_backup if isinstance(existing_backup, dict) else item
        backup = {
            "tasks": copy.deepcopy(backup_source.get("tasks", [])),
            "action_config": copy.deepcopy(backup_source.get("action_config", [])),
        }
        for field_name in DISCARD_FIELDS:
            if field_name in backup_source:
                backup[field_name] = copy.deepcopy(backup_source.get(field_name))
        self.pending_review_backups[index] = backup

    def update_discard_and_save(
        self,
        index: int,
        payload: Dict[str, Any],
        actor_id: str,
    ) -> Dict[str, Any]:
        item = self._get_item(index)
        before_item = copy.deepcopy(item)
        before_dirty = set(self.dirty_indices)
        before_backups = copy.deepcopy(self.pending_review_backups)
        discarded = bool(payload.get("discarded", False))
        try:
            self._ensure_reviewer_backup(index, item)
            if discarded:
                reason = str(payload.get("discard_reason", "")).strip()
                note = str(payload.get("discard_note", "")).strip()
                if reason and reason not in DISCARD_REASON_LABELS:
                    raise ValidationError(
                        "discard_reason must be one of: " + ", ".join(DISCARD_REASON_LABELS)
                    )
                if len(note) > 300:
                    raise ValidationError("discard_note must be at most 300 characters")
                item.update(
                    {
                        "discarded": True,
                        "discarded_by": str(actor_id or "single"),
                        "discarded_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                if reason:
                    item["discard_reason"] = reason
                else:
                    item.pop("discard_reason", None)
                if note:
                    item["discard_note"] = note
                else:
                    item.pop("discard_note", None)
            else:
                for field_name in DISCARD_FIELDS:
                    item.pop(field_name, None)
            self.dirty_indices.add(index)
            save_result = self.save(skip_validation_indices={index})
        except Exception:
            self.session_data[index] = before_item
            self.dirty_indices = before_dirty
            self.pending_review_backups = before_backups
            raise
        return {
            "index": index,
            "discarded": bool(item.get("discarded", False)),
            "discard_reason": str(item.get("discard_reason", "")),
            "discard_note": str(item.get("discard_note", "")),
            "discarded_by": str(item.get("discarded_by", "")),
            "discarded_at": str(item.get("discarded_at", "")),
            "saved_path": save_result.get("saved_path", ""),
        }

    def validate_items(self, indices: Optional[Set[int]] = None) -> List[Tuple[int, str]]:
        errors: List[Tuple[int, str]] = []
        if indices is None:
            candidate_indices = range(len(self.session_data))
        else:
            candidate_indices = sorted(idx for idx in indices if 0 <= idx < len(self.session_data))

        for idx in candidate_indices:
            item = self.session_data[idx]
            try:
                sanitize_tasks(item.get("tasks"))
                segments = normalize_segments(item.get("action_config"))
                self._validate_segment_skills(segments)
                validate_segments(segments, int(item["frames"]))
            except ValidationError as exc:
                errors.append((idx, str(exc)))
        return errors

    def validate_all(self) -> List[Tuple[int, str]]:
        return self.validate_items()

    def _serialize_item_for_save(self, item: Dict[str, Any], mapping: Dict[str, Any]) -> Dict[str, Any]:
        payload = copy.deepcopy(item)
        payload.pop("view_states", None)
        payload.pop("primary_view_key", None)
        payload["action_config"] = convert_segments_from_open(
            payload.get("action_config", []),
            str(mapping.get("interval_style", INTERVAL_STYLE_OPEN)),
        )
        review_backup = payload.get(REVIEW_BACKUP_FIELD)
        if isinstance(review_backup, dict) and isinstance(review_backup.get("action_config"), list):
            review_backup["action_config"] = convert_segments_from_open(
                review_backup.get("action_config", []),
                str(mapping.get("interval_style", INTERVAL_STYLE_OPEN)),
            )

        had_frames = bool(mapping.get("had_frames"))
        had_length = bool(mapping.get("had_length"))
        if had_length and not had_frames:
            payload["length"] = int(payload.get("frames", 0))
            payload.pop("frames", None)
        elif had_frames and not had_length:
            payload.pop("length", None)
        elif had_frames and had_length:
            value = int(payload.get("frames", payload.get("length", 0)))
            payload["frames"] = value
            payload["length"] = value

        if not bool(mapping.get("had_views")):
            payload.pop("views", None)
            payload.pop("view_order", None)
        else:
            raw_views = payload.get("views")
            if isinstance(raw_views, dict):
                ordered_views: Dict[str, Any] = {}
                view_order = payload.get("view_order")
                if isinstance(view_order, list):
                    for raw_key in view_order:
                        view_key = str(raw_key)
                        if view_key in raw_views:
                            ordered_views[view_key] = raw_views[view_key]
                for view_key, value in raw_views.items():
                    if view_key not in ordered_views:
                        ordered_views[view_key] = value
                payload["views"] = ordered_views
            if not bool(mapping.get("had_view_order")):
                payload.pop("view_order", None)

        if not bool(mapping.get("had_path")):
            payload.pop("path", None)
        if not bool(mapping.get("had_episode_index")):
            payload.pop("episode_index", None)

        if not bool(mapping.get("had_unavailable")):
            payload.pop("unavailable", None)
        if bool(mapping.get("had_error")):
            payload["error"] = mapping.get("original_error")
        else:
            payload.pop("error", None)

        if mapping.get("public_input"):
            payload["segments"] = [
                {"start_frame": segment["start_frame"], "end_frame": segment["end_frame"] - 1,
                 "skill": segment["skill"], "subtask": segment["action_text"]}
                for segment in payload.pop("action_config")
            ]
            payload["num_frames"] = int(item["frames"])
            payload["frame_interval"] = "closed"
            payload.pop("frames", None)
            payload.pop("length", None)
        return payload

    def _build_serialized_rows(self, serialized_items: List[Dict[str, Any]]) -> str:
        return build_serialized_output(
            input_format=self.input_format,
            source_mappings=self.source_mappings,
            serialized_items=serialized_items,
            jsonl_rows=self.jsonl_rows,
        )

    def _write_output_atomically(self, content: str) -> None:
        write_output_atomically(self.output_path, content)

    def save(self, skip_validation_indices: Optional[Set[int]] = None) -> Dict[str, Any]:
        dirty_indices = set(self.dirty_indices)
        validation_indices = dirty_indices - set(skip_validation_indices or set())
        errors = self.validate_items(validation_indices) if validation_indices else []
        if errors:
            joined = "; ".join([f"item[{idx}]: {msg}" for idx, msg in errors])
            raise ValidationError(
                f"Save failed because validation errors exist: {joined}",
                field_errors=[
                    {"field": f"item[{idx}]", "message": msg}
                    for idx, msg in errors
                ],
            )

        if len(self.source_mappings) != len(self.session_data):
            raise ValidationError("Save failed: source mapping mismatch")

        if self.review_mode:
            for idx in dirty_indices:
                backup = self.pending_review_backups.get(idx)
                if backup is not None and 0 <= idx < len(self.session_data):
                    self.session_data[idx][REVIEW_BACKUP_FIELD] = copy.deepcopy(backup)

        serialized_items = [
            self._serialize_item_for_save(item, self.source_mappings[idx])
            for idx, item in enumerate(self.session_data)
        ]

        serialized_output = self._build_serialized_rows(serialized_items)
        self._write_output_atomically(serialized_output)
        self.dirty_indices.clear()
        for idx in dirty_indices:
            self.pending_review_backups.pop(idx, None)

        return {
            "saved_path": str(self.output_path),
            "video_count": len(self.session_data),
            "dirty_items_saved": len(dirty_indices),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "warnings_count": len(self.load_warnings),
            "skipped_rows": len(self.load_warnings),
            "validation_summary": {
                "videos_validated": len(dirty_indices),
                "errors": len(errors),
                "skill_options_count": len(self.skill_options),
                "action_text_options_count": len(self.action_text_options),
            },
        }

    def _resolve_s3_cluster_config(self, remote_parts: RemotePathParts) -> S3ClusterConfig:
        if self.runtime_config is None:
            raise RuntimeError("Runtime config is not initialized")
        clusters = self.runtime_config.s3_clusters

        cluster_name = str(remote_parts.cluster or "").strip()
        if cluster_name:
            if cluster_name in clusters:
                return clusters[cluster_name]
            raise RuntimeError(
                f"S3 cluster '{cluster_name}' not found in config: {self.runtime_config.config_path}"
            )

        default_cluster = str(self.runtime_config.default_cluster or "").strip()
        if default_cluster:
            if default_cluster in clusters:
                return clusters[default_cluster]
            raise RuntimeError(
                f"default_cluster '{default_cluster}' not found in config: {self.runtime_config.config_path}"
            )

        if len(clusters) == 1:
            return next(iter(clusters.values()))
        raise RuntimeError(
            "Remote path missing '<cluster>:' prefix and no [video].default_cluster configured"
        )

    def _get_boto3_client(self, cluster_config: S3ClusterConfig) -> Any:
        cache_key = cluster_config.section_name
        cached = self._boto3_clients.get(cache_key)
        if cached is not None:
            return cached

        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise RuntimeError("boto3 is required for fast S3 streaming but is not installed") from exc

        endpoint_lower = cluster_config.endpoint_url.lower()
        addressing_style = (
            "virtual"
            if ("aliyuncs.com" in endpoint_lower or "aliyun" in endpoint_lower)
            else "path"
        )

        self._prepare_proxy_env()

        client = boto3.client(
            "s3",
            aws_access_key_id=cluster_config.access_key,
            aws_secret_access_key=cluster_config.secret_key,
            endpoint_url=cluster_config.endpoint_url,
            region_name=cluster_config.region_name,
            config=Config(
                proxies={},
                max_pool_connections=128,
                connect_timeout=4,
                read_timeout=30,
                retries={"max_attempts": 2, "mode": "standard"},
                s3={"addressing_style": addressing_style},
            ),
        )
        self._boto3_clients[cache_key] = client
        return client

    def _prepare_proxy_env(self) -> None:
        if self._proxy_disabled_once:
            return
        if self.runtime_config is None:
            return
        if not self.runtime_config.disable_proxy:
            self._proxy_disabled_once = True
            return

        for key in PROXY_ENV_KEYS:
            os.environ.pop(key, None)

        no_proxy_values: List[str] = [
            "localhost",
            "127.0.0.1",
        ]
        for cluster in self.runtime_config.s3_clusters.values():
            try:
                host = urlparse(cluster.endpoint_url).hostname or ""
            except Exception:
                host = ""
            if host:
                no_proxy_values.append(host)

        extra = str(self.runtime_config.no_proxy_extra or "").strip()
        if extra:
            no_proxy_values.extend([token.strip() for token in extra.split(",") if token.strip()])

        merged: List[str] = []
        for source in (os.environ.get("NO_PROXY", ""), os.environ.get("no_proxy", ""), ",".join(no_proxy_values)):
            if not source:
                continue
            merged.extend([token.strip() for token in source.split(",") if token.strip()])

        unique: List[str] = []
        seen = set()
        for token in merged:
            if token in seen:
                continue
            seen.add(token)
            unique.append(token)

        no_proxy_final = ",".join(unique)
        os.environ["NO_PROXY"] = no_proxy_final
        os.environ["no_proxy"] = no_proxy_final
        self._proxy_disabled_once = True

    def _generate_presigned_url(self, remote_parts: RemotePathParts) -> str:
        if self.runtime_config is None:
            raise RuntimeError("Runtime config is not initialized")
        cluster_config = self._resolve_s3_cluster_config(remote_parts)
        s3_client = self._get_boto3_client(cluster_config)
        return str(
            s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": remote_parts.bucket, "Key": remote_parts.key},
                ExpiresIn=int(self.runtime_config.presigned_expire_sec),
            )
        )

    def _build_stream_url(
        self,
        index: int,
        view_key: Optional[str] = None,
        local_path: Optional[Path] = None,
    ) -> str:
        if view_key is None:
            stream_url = f"/api/stream/{index}"
        else:
            stream_url = f"/api/stream/{index}/{quote(view_key, safe='')}"
        if local_path is not None:
            try:
                stat_result = local_path.stat()
                stream_url += f"?v={stat_result.st_size}-{stat_result.st_mtime_ns}"
            except OSError:
                pass
        return stream_url

    def _playback_cache_key(self, index: int, view_key: Optional[str]) -> Tuple[int, str, str]:
        s3_mode = self.runtime_config.s3_mode if self.runtime_config is not None else ""
        view_key_token = str(view_key) if view_key is not None else "__primary__"
        return (int(index), view_key_token, s3_mode)

    def _playback_cache_ttl(self, playback: Dict[str, Any]) -> Optional[int]:
        if self.runtime_config is None:
            return None
        mode = str(playback.get("mode", "")).strip()
        if mode in {"direct_only", "direct_with_proxy_fallback"}:
            return max(1, int(self.runtime_config.presigned_expire_sec) - 60)
        if str(playback.get("error", "")).strip():
            return 5
        return None

    def _get_cached_playback(self, index: int, view_key: Optional[str]) -> Optional[Dict[str, Any]]:
        cache_key = self._playback_cache_key(index, view_key)
        cached = self._playback_cache.get(cache_key)
        if not isinstance(cached, dict):
            return None
        expires_at_raw = cached.get("expires_at")
        if expires_at_raw is not None and time.monotonic() >= float(expires_at_raw):
            self._playback_cache.pop(cache_key, None)
            return None
        payload = cached.get("payload")
        if not isinstance(payload, dict):
            self._playback_cache.pop(cache_key, None)
            return None
        return copy.deepcopy(payload)

    def _set_cached_playback(self, index: int, view_key: Optional[str], playback: Dict[str, Any]) -> None:
        ttl = self._playback_cache_ttl(playback)
        expires_at = (time.monotonic() + ttl) if ttl is not None else None
        self._playback_cache[self._playback_cache_key(index, view_key)] = {
            "expires_at": expires_at,
            "payload": copy.deepcopy(playback),
        }

    def build_playback(self, index: int, view_key: Optional[str] = None) -> Dict[str, Any]:
        if self.runtime_config is None:
            raise RuntimeError("Runtime config is not initialized")
        cached = self._get_cached_playback(index, view_key=view_key)
        if cached is not None:
            return cached

        source_kind, _raw_path, local_path, remote_parts = self.resolve_video_source(
            index,
            view_key=view_key,
        )
        fallback_url = self._build_stream_url(
            index,
            view_key=view_key,
            local_path=local_path if source_kind == "local" else None,
        )

        if source_kind == "local":
            playback = {
                "primary_url": fallback_url,
                "fallback_url": "",
                "mode": "local_stream",
                "fallback_retry_max": 0,
            }
            self._set_cached_playback(index, view_key=view_key, playback=playback)
            return playback

        if remote_parts is None:
            raise RuntimeError("Invalid remote source")

        s3_mode = self.runtime_config.s3_mode
        if s3_mode == "proxy":
            playback = {
                "primary_url": fallback_url,
                "fallback_url": "",
                "mode": "proxy_only",
                "fallback_retry_max": 0,
            }
            self._set_cached_playback(index, view_key=view_key, playback=playback)
            return playback

        try:
            direct_url = self._generate_presigned_url(remote_parts)
        except RuntimeError:
            raise
        except RECOVERABLE_REMOTE_ERRORS as exc:
            if s3_mode == "hybrid":
                logger.warning(
                    "presigned unavailable, fallback to proxy: index=%s view=%s path=%s error=%s",
                    index,
                    view_key,
                    remote_parts.raw_path,
                    exc,
                )
                playback = {
                    "primary_url": fallback_url,
                    "fallback_url": "",
                    "mode": "proxy_only",
                    "fallback_retry_max": 0,
                    "error": f"presigned unavailable, fallback to proxy: {exc}",
                }
                self._set_cached_playback(index, view_key=view_key, playback=playback)
                return playback
            raise RuntimeError(f"Failed to generate presigned URL: {exc}") from exc

        if s3_mode == "direct":
            playback = {
                "primary_url": direct_url,
                "fallback_url": "",
                "mode": "direct_only",
                "fallback_retry_max": 0,
            }
            self._set_cached_playback(index, view_key=view_key, playback=playback)
            return playback

        playback = {
            "primary_url": direct_url,
            "fallback_url": fallback_url,
            "mode": "direct_with_proxy_fallback",
            "fallback_retry_max": max(0, int(self.runtime_config.fallback_retry_max)),
        }
        self._set_cached_playback(index, view_key=view_key, playback=playback)
        return playback

    def _resolve_view_key(self, item: Dict[str, Any], view_key: Optional[str]) -> str:
        views = item.get("views")
        if not isinstance(views, dict):
            raise RuntimeError("View mapping is missing")
        key = str(view_key or item.get("primary_view_key", "")).strip()
        if not key:
            raise KeyError("View key is missing")
        if key not in views:
            raise KeyError(f"View key not found: {key}")
        return key

    def resolve_video_source(
        self,
        index: int,
        view_key: Optional[str] = None,
    ) -> Tuple[str, str, Optional[Path], Optional[RemotePathParts]]:
        item = self._get_item(index)
        resolved_view_key = self._resolve_view_key(item, view_key=view_key)
        views = item.get("views") if isinstance(item.get("views"), dict) else {}
        raw_path = str(views.get(resolved_view_key, "")).strip()
        if not raw_path:
            raise FileNotFoundError(f"Video path is empty for view '{resolved_view_key}'")

        if is_remote_path(raw_path):
            remote_parts = parse_remote_path(raw_path)
            return ("remote", raw_path, None, remote_parts)

        video_path = resolve_media_path(raw_path)
        return ("local", raw_path, video_path, None)

    def open_remote_stream(
        self,
        remote_parts: RemotePathParts,
        range_header: Optional[str],
    ) -> Tuple[Iterator[bytes], int, Dict[str, str], str]:
        cluster_config = self._resolve_s3_cluster_config(remote_parts)
        s3_client = self._get_boto3_client(cluster_config)
        get_kwargs: Dict[str, Any] = {"Bucket": remote_parts.bucket, "Key": remote_parts.key}
        if range_header:
            get_kwargs["Range"] = range_header

        try:
            obj = s3_client.get_object(**get_kwargs)
            body = obj["Body"]
            status_code = int(
                obj.get("ResponseMetadata", {}).get("HTTPStatusCode", 206 if range_header else 200)
            )

            def stream_chunks() -> Iterator[bytes]:
                try:
                    while True:
                        chunk = body.read(1024 * 1024)
                        if not chunk:
                            break
                        yield chunk
                finally:
                    try:
                        body.close()
                    except Exception:
                        pass

            headers: Dict[str, str] = {"Accept-Ranges": str(obj.get("AcceptRanges") or "bytes")}
            if obj.get("ContentRange"):
                headers["Content-Range"] = str(obj["ContentRange"])
            if obj.get("ContentLength") is not None:
                headers["Content-Length"] = str(obj["ContentLength"])
            if obj.get("ETag"):
                headers["ETag"] = str(obj["ETag"])
            if obj.get("LastModified"):
                headers["Last-Modified"] = format_http_datetime(obj["LastModified"])

            mime_type, _ = mimetypes.guess_type(remote_parts.uri)
            content_type = str(obj.get("ContentType") or mime_type or "video/mp4")
            return stream_chunks(), status_code, headers, content_type
        except RECOVERABLE_REMOTE_ERRORS as exc:
            logger.error(
                "remote stream failed: path=%s cluster=%s bucket=%s key=%s range=%s error=%s",
                remote_parts.raw_path,
                remote_parts.cluster,
                remote_parts.bucket,
                remote_parts.key,
                range_header,
                exc,
            )
            raise FileNotFoundError(
                f"Failed to read remote video: {remote_parts.raw_path} ({exc})"
            ) from exc

    def _get_item(self, index: int) -> Dict[str, Any]:
        if index < 0 or index >= len(self.session_data):
            raise IndexError(f"Video index out of range: {index}")
        return self.session_data[index]

    def warmup_playback(self) -> None:
        if self.runtime_config is None:
            return
        if not self.session_data:
            logger.info("playback warmup skipped: no videos loaded")
            return

        for index, item in enumerate(self.session_data):
            view_order = item.get("view_order") if isinstance(item.get("view_order"), list) else []
            if not view_order:
                continue
            primary_view_key = str(item.get("primary_view_key", "")).strip() or str(view_order[0]).strip()
            view_states = item.get("view_states") if isinstance(item.get("view_states"), dict) else {}
            state = view_states.get(primary_view_key) if isinstance(view_states, dict) else None
            if isinstance(state, dict) and bool(state.get("unavailable", False)):
                continue

            started_at = time.perf_counter()
            try:
                self.build_playback(index=index, view_key=primary_view_key)
            except (
                ValidationError,
                FileNotFoundError,
                RuntimeError,
                KeyError,
                OSError,
                ValueError,
                TypeError,
            ) as exc:
                logger.warning(
                    "playback warmup failed: index=%s view=%s error=%s",
                    index,
                    primary_view_key,
                    exc,
                )
                continue

            elapsed = time.perf_counter() - started_at
            logger.info(
                "playback warmup completed: index=%s view=%s elapsed=%.3fs",
                index,
                primary_view_key,
                elapsed,
            )
            return

        logger.info("playback warmup skipped: no playable view found")

    def _refresh_options(self) -> None:
        self.skill_options = self.configured_skill_options + self.legacy_skill_options
        self.action_text_options = self._resolve_action_text_options(
            self.configured_action_text_options,
            self.legacy_action_text_options,
        )

    def _rebuild_legacy_options_from_session(self) -> None:
        configured_skill_set = set(self.configured_skill_options)
        configured_action_text_set = set(self.configured_action_text_options)

        legacy_skill_options: List[str] = []
        legacy_skill_seen = set()
        legacy_action_text_options: List[str] = []
        legacy_action_text_seen = set()

        for item in self.session_data:
            raw_segments = item.get("action_config")
            if not isinstance(raw_segments, list):
                continue
            for raw_segment in raw_segments:
                if not isinstance(raw_segment, dict):
                    continue

                skill_value = str(raw_segment.get("skill", "")).strip()
                if (
                    skill_value
                    and skill_value not in configured_skill_set
                    and skill_value not in legacy_skill_seen
                ):
                    legacy_skill_seen.add(skill_value)
                    legacy_skill_options.append(skill_value)

                action_text_value = str(raw_segment.get("action_text", "")).strip()
                if (
                    action_text_value
                    and action_text_value not in configured_action_text_set
                    and action_text_value not in legacy_action_text_seen
                ):
                    legacy_action_text_seen.add(action_text_value)
                    legacy_action_text_options.append(action_text_value)

        self.legacy_skill_options = legacy_skill_options
        self.legacy_action_text_options = legacy_action_text_options

    def _validate_segment_skills(self, segments: List[Dict[str, Any]]) -> None:
        allowed = set(self.configured_skill_options) | set(self.legacy_skill_options)
        if not allowed:
            return
        invalid: List[Tuple[int, str]] = []
        for idx, segment in enumerate(segments):
            skill = segment.get("skill", "")
            if skill not in allowed:
                invalid.append((idx, str(skill)))
        if invalid:
            details = ", ".join([f"segment[{idx}]='{skill}'" for idx, skill in invalid])
            raise ValidationError(
                f"skill must be one of configured or legacy options: {details}",
                field_errors=[
                    {
                        "field": f"action_config[{idx}].skill",
                        "message": "skill must be configured or existing legacy value",
                        "value": skill,
                    }
                    for idx, skill in invalid
                ],
            )


def prepare_multi_user_seed_files(manifest: MultiUserManifest, force: bool = False) -> List[Path]:
    loaded_master = AnnotationStore._load_items_from_path(manifest.master_input_path)
    written_paths: List[Path] = []
    for annotator in manifest.annotators:
        if annotator.is_reviewer:
            continue
        if annotator.seed_input_path.exists():
            if not annotator.seed_input_path.is_file():
                raise ValidationError(
                    f"annotator '{annotator.annotator_id}' seed_input_json exists but is not a file: "
                    f"{annotator.seed_input_path}"
                )
            if not force:
                logger.info(
                    "seed input exists; skipping annotator '%s': %s",
                    annotator.annotator_id,
                    annotator.seed_input_path,
                )
                continue
        start = annotator.start_index
        stop = annotator.stop_index

        if loaded_master.input_format == INPUT_FORMAT_JSONL:
            if loaded_master.jsonl_rows is None:
                raise ValidationError("JSONL prepare failed: missing source rows")
            subset_rows: List[Dict[str, Any]] = []
            for global_index in range(start, stop):
                mapping = loaded_master.source_mappings[global_index]
                row_index = mapping.get("line_index")
                if row_index is None:
                    raise ValidationError(
                        f"annotator '{annotator.annotator_id}' has invalid line index at item[{global_index}]"
                    )
                subset_rows.append(copy.deepcopy(loaded_master.jsonl_rows[int(row_index)]))
            content = serialize_jsonl_rows(subset_rows)
        else:
            subset_items = [copy.deepcopy(loaded_master.items[idx]) for idx in range(start, stop)]
            content = json.dumps(subset_items, ensure_ascii=False, indent=2)

        write_output_atomically(annotator.seed_input_path, content)
        written_paths.append(annotator.seed_input_path)

    return written_paths


def _runtime_config_for_merge(runtime_config: Optional[RuntimeConfig]) -> RuntimeConfig:
    if runtime_config is not None:
        return runtime_config
    return RuntimeConfig.load(Path(os.environ["ROBOATOM_ANNOTATION_CONFIG"]))


def _validated_serialized_annotator_items(
    annotator: AnnotatorConfig,
    runtime_config: RuntimeConfig,
) -> List[Dict[str, Any]]:
    output_path = annotator.output_path
    if not output_path.is_file():
        raise ValidationError(
            f"annotator '{annotator.annotator_id}' output_json not found: {output_path}"
        )

    try:
        output_store = AnnotationStore.load(
            input_path=output_path,
            output_path=output_path,
            runtime_config=runtime_config,
        )
    except (ValidationError, OSError, json.JSONDecodeError) as exc:
        raise ValidationError(
            f"annotator '{annotator.annotator_id}' output_json is invalid: {exc}"
        ) from exc

    if len(output_store.session_data) != annotator.count:
        raise ValidationError(
            f"annotator '{annotator.annotator_id}' output_json item count mismatch: "
            f"expected {annotator.count}, got {len(output_store.session_data)}"
        )

    errors = output_store.validate_all()
    if errors:
        joined = "; ".join([f"item[{idx}]: {msg}" for idx, msg in errors])
        raise ValidationError(
            f"annotator '{annotator.annotator_id}' output_json validation failed: {joined}",
            field_errors=[
                {
                    "field": f"annotators.{annotator.annotator_id}.item[{idx}]",
                    "message": msg,
                }
                for idx, msg in errors
            ],
        )

    return [
        output_store._serialize_item_for_save(item, output_store.source_mappings[idx])
        for idx, item in enumerate(output_store.session_data)
    ]


def _load_reviewer_output_store(
    reviewer: AnnotatorConfig,
    runtime_config: RuntimeConfig,
    expected_count: int,
) -> Optional["AnnotationStore"]:
    output_path = reviewer.output_path
    if not output_path.is_file():
        return None

    try:
        output_store = AnnotationStore.load(
            input_path=output_path,
            output_path=output_path,
            runtime_config=runtime_config,
        )
    except (ValidationError, OSError, json.JSONDecodeError) as exc:
        raise ValidationError(
            f"reviewer '{reviewer.annotator_id}' output_json is invalid: {exc}"
        ) from exc

    if len(output_store.session_data) != expected_count:
        raise ValidationError(
            f"reviewer '{reviewer.annotator_id}' output_json item count mismatch: "
            f"expected {expected_count}, got {len(output_store.session_data)}"
        )

    return output_store


def _build_merge_comparison_store(
    manifest: MultiUserManifest,
    loaded_master: LoadedInput,
    merged_items: List[Dict[str, Any]],
    runtime_config: RuntimeConfig,
) -> "AnnotationStore":
    loaded_merged = LoadedInput(
        input_format=loaded_master.input_format,
        items=[copy.deepcopy(item) for item in merged_items],
        source_mappings=copy.deepcopy(loaded_master.source_mappings),
        load_warnings=list(loaded_master.load_warnings),
        jsonl_rows=copy.deepcopy(loaded_master.jsonl_rows),
    )
    return AnnotationStore.from_loaded_input(
        loaded=loaded_merged,
        input_path=manifest.master_input_path,
        output_path=manifest.merged_output_path,
        runtime_config=runtime_config,
    )


def _apply_reviewer_outputs_to_merge(
    manifest: MultiUserManifest,
    loaded_master: LoadedInput,
    merged_items: List[Dict[str, Any]],
    runtime_config: RuntimeConfig,
) -> None:
    reviewers = [annotator for annotator in manifest.annotators if annotator.is_reviewer]
    if not reviewers:
        return

    comparison_store = _build_merge_comparison_store(
        manifest=manifest,
        loaded_master=loaded_master,
        merged_items=merged_items,
        runtime_config=runtime_config,
    )
    applied_indices: Set[int] = set()

    for reviewer in reviewers:
        reviewer_store = _load_reviewer_output_store(
            reviewer=reviewer,
            runtime_config=runtime_config,
            expected_count=manifest.total_editable_items,
        )
        if reviewer_store is None:
            logger.info(
                "reviewer '%s' output_json not found for merge; keeping annotator output",
                reviewer.annotator_id,
            )
            continue

        applied_count = 0
        skipped_count = 0
        for idx, reviewer_item in enumerate(reviewer_store.session_data):
            backup = reviewer_item.get(REVIEW_BACKUP_FIELD)
            if not isinstance(backup, dict):
                continue
            if not MultiUserStoreManager._annotations_match(
                comparison_store.session_data[idx],
                backup,
            ):
                skipped_count += 1
                continue

            MultiUserStoreManager._apply_reviewer_overlay(
                comparison_store.session_data[idx],
                reviewer_item,
            )
            comparison_store.session_data[idx].pop(REVIEW_BACKUP_FIELD, None)
            serialized_item = comparison_store._serialize_item_for_save(
                comparison_store.session_data[idx],
                comparison_store.source_mappings[idx],
            )
            serialized_item.pop(REVIEW_BACKUP_FIELD, None)
            merged_items[idx] = serialized_item
            applied_indices.add(idx)
            applied_count += 1

        if applied_count:
            comparison_store._rebuild_legacy_options_from_session()
            comparison_store._refresh_options()
        if applied_count or skipped_count:
            logger.info(
                "reviewer '%s' merge overlay applied=%s skipped_after_annotator_update=%s",
                reviewer.annotator_id,
                applied_count,
                skipped_count,
            )

    errors = comparison_store.validate_items(applied_indices) if applied_indices else []
    if errors:
        joined = "; ".join([f"item[{idx}]: {msg}" for idx, msg in errors])
        raise ValidationError(
            f"reviewer merge overlay validation failed: {joined}",
            field_errors=[
                {
                    "field": f"reviewer_overlay.item[{idx}]",
                    "message": msg,
                }
                for idx, msg in errors
            ],
        )


def merge_multi_user_outputs(
    manifest: MultiUserManifest,
    runtime_config: Optional[RuntimeConfig] = None,
) -> Path:
    effective_runtime_config = _runtime_config_for_merge(runtime_config)
    loaded_master = AnnotationStore._load_items_from_path(manifest.master_input_path)
    merged_items = [copy.deepcopy(item) for item in loaded_master.items]

    for annotator in manifest.annotators:
        if annotator.is_reviewer:
            continue
        serialized_output_items = _validated_serialized_annotator_items(
            annotator,
            effective_runtime_config,
        )
        for local_index, item in enumerate(serialized_output_items):
            global_index = annotator.start_index + local_index
            merged_items[global_index] = copy.deepcopy(item)

    _apply_reviewer_outputs_to_merge(
        manifest=manifest,
        loaded_master=loaded_master,
        merged_items=merged_items,
        runtime_config=effective_runtime_config,
    )

    content = build_serialized_output(
        input_format=loaded_master.input_format,
        source_mappings=loaded_master.source_mappings,
        serialized_items=merged_items,
        jsonl_rows=loaded_master.jsonl_rows,
    )
    write_output_atomically(manifest.merged_output_path, content)
    return manifest.merged_output_path


def parse_range_header(range_header: str, file_size: int) -> Tuple[int, int]:
    if not range_header.startswith("bytes="):
        raise ValidationError("Invalid Range header")

    byte_range = range_header.split("=", 1)[1]
    start_str, end_str = byte_range.split("-", 1)

    if start_str == "":
        suffix_len = int(end_str)
        if suffix_len <= 0:
            raise ValidationError("Invalid Range suffix length")
        start = max(file_size - suffix_len, 0)
        end = file_size - 1
    else:
        start = int(start_str)
        end = int(end_str) if end_str else file_size - 1

    if start < 0 or end < start or end >= file_size:
        raise ValidationError("Range out of file bounds")

    return start, end


def resolve_output_path(input_path: Path, output_json: str) -> Tuple[Path, bool]:
    output_raw = str(output_json or "").strip()
    if output_raw:
        return Path(output_raw).resolve(), False
    default_name = "annotations.json" if input_path.suffix.lower() == ".json" else "annotations.jsonl"
    return (input_path.parent / default_name).resolve(), True


def create_app(
    store: Optional[AnnotationStore],
    multi_user_manager: Optional[MultiUserStoreManager] = None,
    model_review_store: Optional[ModelReviewStore] = None,
    app_mode: str = APP_MODE_ANNOTATION,
) -> Flask:
    if app_mode not in APP_MODES:
        raise ValidationError(f"mode must be one of: {', '.join(sorted(APP_MODES))}")

    base_dir = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        static_folder=str(base_dir / "static"),
        template_folder=str(base_dir / "templates"),
    )
    app.config["JSON_AS_ASCII"] = False
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    @app.before_request
    def _ensure_audit_session_id() -> None:
        session_id = str(request.cookies.get(AUDIT_SESSION_COOKIE_NAME, "")).strip()
        if session_id:
            g.audit_session_id = session_id
            g.audit_session_cookie_new = False
        else:
            g.audit_session_id = str(uuid.uuid4())
            g.audit_session_cookie_new = True

    @app.after_request
    def _persist_audit_session_id(response: Response) -> Response:
        if bool(getattr(g, "audit_session_cookie_new", False)):
            response.set_cookie(
                AUDIT_SESSION_COOKIE_NAME,
                str(getattr(g, "audit_session_id", "")),
                samesite="Lax",
                httponly=True,
            )
        return response

    def _request_annotator_id() -> str:
        return str(request.cookies.get(ANNOTATOR_COOKIE_NAME, "")).strip()

    def _annotator_required_response() -> Response:
        response = jsonify(
            {
                "error": "Annotator selection is required in multi-user mode",
                "requires_annotator_selection": True,
            }
        )
        response.status_code = 409
        response.delete_cookie(ANNOTATOR_COOKIE_NAME)
        return response

    def _get_active_store() -> Tuple[Optional[AnnotationStore], Optional[AnnotatorConfig], Optional[Response]]:
        if multi_user_manager is None:
            if store is None:
                response = jsonify({"error": "Annotation store is not initialized"})
                response.status_code = 500
                return None, None, response
            return store, None, None

        annotator = multi_user_manager.get_annotator(_request_annotator_id())
        if annotator is None:
            return None, None, _annotator_required_response()

        try:
            active_store = multi_user_manager.get_store(annotator.annotator_id)
        except ValidationError as exc:
            response = jsonify({"error": f"Failed to initialize annotator session: {exc}"})
            response.status_code = 400
            return None, annotator, response
        except (OSError, json.JSONDecodeError) as exc:
            response = jsonify({"error": f"Failed to read annotator session: {exc}"})
            response.status_code = 400
            return None, annotator, response

        return active_store, annotator, None

    def _request_remote_addr() -> str:
        forwarded_for = str(request.headers.get("X-Forwarded-For", "")).split(",", 1)[0].strip()
        return forwarded_for or str(request.remote_addr or "")

    def _annotator_log_id(annotator: Optional[AnnotatorConfig]) -> str:
        return annotator.annotator_id if annotator is not None else "single"

    def _annotator_log_role(annotator: Optional[AnnotatorConfig]) -> str:
        return annotator.role if annotator is not None else "single"

    def _dirty_count(active_store: Optional[AnnotationStore]) -> int:
        dirty_indices = getattr(active_store, "dirty_indices", None)
        return len(dirty_indices) if dirty_indices is not None else 0

    def _annotation_payload_hash(value: Any) -> str:
        return audit_payload_hash(value)

    def _item_audit_payload(item: Dict[str, Any]) -> Dict[str, Any]:
        return item_audit_payload(item)

    def _item_audit_hash(active_store: AnnotationStore, index: int) -> str:
        try:
            return _annotation_payload_hash(_item_audit_payload(active_store._get_item(index)))
        except (IndexError, TypeError, ValueError):
            return ""

    def _store_audit_hash(active_store: AnnotationStore) -> str:
        return _annotation_payload_hash(
            [_item_audit_payload(item) for item in active_store.session_data]
        )

    def _request_id() -> str:
        raw = str(request.headers.get("X-Request-ID", "")).strip()
        return raw or str(uuid.uuid4())

    def _request_session_id() -> str:
        return str(getattr(g, "audit_session_id", "") or "")

    def _request_user_agent() -> str:
        return str(request.headers.get("User-Agent", ""))[:500]

    def _global_index_for_annotator(annotator: Optional[AnnotatorConfig], index: Optional[int]) -> Optional[int]:
        if index is None:
            return None
        if annotator is None:
            return index
        return annotator.start_index + index

    def _audit_event_type(base_event_type: str, annotator: Optional[AnnotatorConfig]) -> str:
        if annotator is not None and annotator.role == ANNOTATOR_ROLE_REVIEWER:
            if base_event_type == "annotation_update":
                return "review_update"
            if base_event_type == "annotation_save":
                return "review_save"
        return base_event_type

    def _status_snapshot(active_store: AnnotationStore) -> Dict[str, Any]:
        return load_status_snapshot(active_store.output_path, len(active_store.session_data))

    def _status_for_index(active_store: AnnotationStore, index: int) -> Dict[str, Any]:
        snapshot = _status_snapshot(active_store)
        items = snapshot.get("items") if isinstance(snapshot.get("items"), list) else []
        if 0 <= index < len(items) and isinstance(items[index], dict):
            return compact_status_items({"items": [items[index]]})[0]
        return compact_status_items({"items": [default_audit_status_item(index)]})[0]

    def _annotation_statuses(active_store: AnnotationStore) -> List[Dict[str, Any]]:
        return compact_status_items(_status_snapshot(active_store))

    def _audit_context_from_payload(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        audit_context = payload.get("_audit")
        if not isinstance(audit_context, dict):
            return {}
        result: Dict[str, Any] = {}
        for key in ("active_ms", "lead_time_ms"):
            try:
                result[key] = max(0, int(audit_context.get(key) or 0))
            except (TypeError, ValueError):
                result[key] = 0
        return result

    def _append_audit_event(
        active_store: AnnotationStore,
        event_type: str,
        annotator: Optional[AnnotatorConfig],
        index: Optional[int],
        before_hash: str,
        after_hash: str,
        status: str,
        error: str = "",
        diff: Optional[Dict[str, Any]] = None,
        change_summary: Optional[Dict[str, Any]] = None,
        active_ms: int = 0,
        lead_time_ms: int = 0,
    ) -> Dict[str, Any]:
        audit_path = audit_sidecar_path(active_store.output_path)
        local_index = index
        global_index = _global_index_for_annotator(annotator, index)
        item: Dict[str, Any] = {}
        if index is not None and 0 <= index < len(active_store.session_data):
            item = active_store.session_data[index]
        sample_id = sample_id_for_item(item) if item else ""
        item_key = item_key_for_mapping(
            active_store.output_path,
            active_store.source_mappings,
            local_index,
            global_index,
            sample_id,
        )
        summary = copy.deepcopy(change_summary or {})
        event_diff = copy.deepcopy(diff or {})
        safe_active_ms = max(0, int(active_ms or summary.get("active_ms") or 0))
        safe_lead_time_ms = max(0, int(lead_time_ms or summary.get("lead_time_ms") or 0))
        if safe_active_ms and "active_ms" not in summary:
            summary["active_ms"] = safe_active_ms
        if safe_lead_time_ms and "lead_time_ms" not in summary:
            summary["lead_time_ms"] = safe_lead_time_ms

        lock = audit_lock_for_path(audit_path)
        with lock:
            snapshot = _status_snapshot(active_store)
            targets = [idx for idx in _audit_event_target_indices(
                {"local_index": local_index, "change_summary": summary},
                len(active_store.session_data),
            )]
            revisions = []
            snapshot_items = snapshot.get("items") if isinstance(snapshot.get("items"), list) else []
            for target in targets:
                if 0 <= target < len(snapshot_items) and isinstance(snapshot_items[target], dict):
                    try:
                        revisions.append(int(snapshot_items[target].get("revision") or 0))
                    except (TypeError, ValueError):
                        revisions.append(0)
            revision_before = max(revisions) if revisions else 0
            revision_after = revision_before
            if event_type in AUDIT_MUTATION_EVENT_TYPES and targets:
                revision_after = revision_before + 1

            event = {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "event_id": str(uuid.uuid4()),
                "event_type": event_type,
                "request_id": _request_id(),
                "session_id": _request_session_id(),
                "annotator_id": _annotator_log_id(annotator),
                "role": _annotator_log_role(annotator),
                "index": index,
                "local_index": local_index,
                "global_index": global_index,
                "item_key": item_key,
                "sample_id": sample_id,
                "before_hash": before_hash,
                "after_hash": after_hash,
                "revision_before": revision_before,
                "revision_after": revision_after,
                "diff": event_diff,
                "change_summary": summary,
                "remote_addr": _request_remote_addr(),
                "user_agent": _request_user_agent(),
                "status": status,
                "error": error,
                "active_ms": safe_active_ms,
                "lead_time_ms": safe_lead_time_ms,
                "prev_event_hash": read_last_audit_event_hash(audit_path),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            event["event_hash"] = finalize_audit_event_hash(event)

            try:
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                with audit_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
                apply_audit_event_to_status_items(snapshot["items"], event)
                write_status_snapshot(active_store.output_path, snapshot)
            except OSError as exc:
                logger.error("audit append failed: path=%s error=%s", audit_path, exc)
            return event

    def _get_initial_template_paths() -> Tuple[str, str]:
        active_store, _annotator, _response = _get_active_store()
        if active_store is None:
            if multi_user_manager is not None:
                return str(multi_user_manager.manifest.master_input_path), ""
            return "", ""
        return str(active_store.input_path), str(active_store.output_path)

    def _build_collab_config_payload() -> Dict[str, Any]:
        if multi_user_manager is None:
            return {
                "enabled": False,
                "current_annotator_id": "",
                "current_annotator_name": "",
                "current_annotator_role": "",
                "annotators": [],
            }

        current = multi_user_manager.get_annotator(_request_annotator_id())
        return {
            "enabled": True,
            "current_annotator_id": current.annotator_id if current else "",
            "current_annotator_name": current.name if current else "",
            "current_annotator_role": current.role if current else "",
            "annotators": [
                {
                    "id": annotator.annotator_id,
                    "name": annotator.name,
                    "role": annotator.role,
                    "count": annotator.count,
                    "start_index": annotator.start_index,
                }
                for annotator in multi_user_manager.manifest.annotators
            ],
        }

    @app.get("/")
    def index() -> str:
        input_json, output_json = _get_initial_template_paths()
        return render_template(
            "index.html",
            input_json=input_json,
            output_json=output_json,
        )

    @app.get("/api/collab/config")
    def api_collab_config() -> Response:
        payload = _build_collab_config_payload()
        response = jsonify(payload)
        if (
            multi_user_manager is not None
            and _request_annotator_id()
            and not payload["current_annotator_id"]
        ):
            response.delete_cookie(ANNOTATOR_COOKIE_NAME)
        return response

    @app.post("/api/collab/select")
    def api_collab_select() -> Response:
        if multi_user_manager is None:
            return jsonify({"error": "Multi-user mode is not enabled"}), 404

        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be JSON"}), 400

        annotator_id = str(payload.get("annotator_id", "")).strip()
        if not annotator_id:
            return jsonify({"error": "annotator_id is required"}), 400

        annotator = multi_user_manager.get_annotator(annotator_id)
        if annotator is None:
            return jsonify({"error": f"Unknown annotator_id: {annotator_id}"}), 400

        try:
            active_store = multi_user_manager.get_store(annotator.annotator_id)
        except ValidationError as exc:
            return jsonify({"error": f"Failed to initialize annotator session: {exc}"}), 400
        except (OSError, json.JSONDecodeError) as exc:
            return jsonify({"error": f"Failed to read annotator session: {exc}"}), 400

        response = jsonify(
            {
                "enabled": True,
                "current_annotator_id": annotator.annotator_id,
                "current_annotator_name": annotator.name,
                "current_annotator_role": annotator.role,
                "video_count": len(active_store.session_data),
            }
        )
        response.set_cookie(ANNOTATOR_COOKIE_NAME, annotator.annotator_id, samesite="Lax")
        return response

    @app.get("/api/meta")
    def api_meta() -> Response:
        active_store, annotator, error_response = _get_active_store()
        if error_response is not None:
            return error_response
        if active_store is None:
            return jsonify({"error": "Annotation store is not initialized"}), 500

        videos = active_store.get_meta()
        if model_review_store is not None:
            for idx, video_meta in enumerate(videos):
                video_meta.update(model_review_store.build_meta(active_store, idx))
        else:
            for video_meta in videos:
                video_meta.update(
                    {
                        "review_verdict": "unreviewed",
                        "review_issue_count": 0,
                        "review_uncertain_count": 0,
                        "review_summary": "",
                    }
                )

        load_warnings: List[str] = []
        if multi_user_manager is not None:
            load_warnings.extend(
                [f"manifest: {warning}" for warning in multi_user_manager.manifest.load_warnings]
            )
        load_warnings.extend(list(active_store.load_warnings))
        if model_review_store is not None:
            load_warnings.extend(
                [f"model_review: {warning}" for warning in model_review_store.load_warnings]
            )

        return jsonify(
            {
                "videos": videos,
                "skill_options": active_store.skill_options,
                "action_text_options": active_store.action_text_options,
                "input_json": str(active_store.input_path),
                "output_json": str(active_store.output_path),
                "load_warnings": load_warnings,
                "annotation_statuses": _annotation_statuses(active_store),
                "annotator_id": annotator.annotator_id if annotator is not None else "",
                "annotator_name": annotator.name if annotator is not None else "",
                "annotator_role": annotator.role if annotator is not None else "",
                "app_mode": app_mode,
                "model_review_json": str(model_review_store.review_path) if model_review_store is not None else "",
            }
        )

    @app.post("/api/input/switch")
    def api_switch_input() -> Response:
        if multi_user_manager is not None:
            return jsonify({"error": "Input switching is disabled in multi-user mode"}), 409

        nonlocal store
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be JSON"}), 400

        input_json = str(payload.get("input_json", "")).strip()
        output_json = str(payload.get("output_json", "")).strip()
        if not input_json:
            return jsonify({"error": "input_json is required"}), 400

        input_path = Path(input_json).resolve()
        if not input_path.exists():
            return jsonify({"error": f"Input JSON not found: {input_path}"}), 400

        output_path, output_is_default = resolve_output_path(input_path, output_json)
        if output_is_default and not output_path.exists():
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.touch(exist_ok=True)
            except OSError as exc:
                return jsonify({"error": f"Failed to create default output file: {exc}"}), 500
        if store.runtime_config is None:
            return jsonify({"error": "Runtime config is not initialized"}), 500

        try:
            new_store = AnnotationStore.load(
                input_path=input_path,
                output_path=output_path,
                runtime_config=store.runtime_config,
            )
        except ValidationError as exc:
            return jsonify({"error": f"Failed to load input JSON: {exc}"}), 400
        except (OSError, json.JSONDecodeError) as exc:
            return jsonify({"error": f"Failed to read input JSON: {exc}"}), 400

        store = new_store
        return jsonify(
            {
                "video_count": len(store.session_data),
                "input_json": str(store.input_path),
                "output_json": str(store.output_path),
                "load_warnings": store.load_warnings,
            }
        )

    @app.get("/api/video/<int:index>")
    def api_video(index: int) -> Response:
        active_store, _annotator, error_response = _get_active_store()
        if error_response is not None:
            return error_response
        if active_store is None:
            return jsonify({"error": "Annotation store is not initialized"}), 500

        include_frame_map_raw: Optional[str] = request.args.get("include_frame_map")
        include_frame_map = str(include_frame_map_raw or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        try:
            payload = active_store.get_video(index, include_frame_map=include_frame_map)
        except IndexError as exc:
            logger.warning("api_video index error: index=%s error=%s", index, exc)
            return jsonify({"error": str(exc)}), 404
        except KeyError as exc:
            logger.warning("api_video key error: index=%s error=%s", index, exc)
            return jsonify({"error": str(exc)}), 404
        except FileNotFoundError as exc:
            logger.warning("api_video file error: index=%s error=%s", index, exc)
            return jsonify({"error": str(exc)}), 404
        except ValidationError as exc:
            logger.warning("api_video validation error: index=%s error=%s", index, exc)
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            logger.error("api_video runtime error: index=%s error=%s", index, exc)
            return jsonify({"error": str(exc)}), 500
        payload["app_mode"] = app_mode
        payload["model_review"] = (
            model_review_store.get_review(active_store, index)
            if model_review_store is not None
            else None
        )
        payload["annotation_status"] = _status_for_index(active_store, index)
        return jsonify(payload)

    @app.post("/api/audit/event")
    def api_audit_event() -> Response:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be JSON"}), 400

        active_store, annotator, error_response = _get_active_store()
        if error_response is not None:
            return error_response
        if active_store is None:
            return jsonify({"error": "Annotation store is not initialized"}), 500

        event_type = str(payload.get("event_type", "")).strip()
        if event_type not in AUDIT_BEHAVIOR_EVENT_TYPES:
            return jsonify({"error": "event_type must be item_open or item_close"}), 400
        try:
            index = int(payload.get("index"))
        except (TypeError, ValueError):
            return jsonify({"error": "index is required"}), 400
        if index < 0 or index >= len(active_store.session_data):
            return jsonify({"error": f"index out of range: {index}"}), 404

        try:
            active_ms = max(0, int(payload.get("active_ms") or 0))
        except (TypeError, ValueError):
            active_ms = 0
        try:
            lead_time_ms = max(0, int(payload.get("lead_time_ms") or 0))
        except (TypeError, ValueError):
            lead_time_ms = 0

        item_payload = _item_audit_payload(active_store._get_item(index))
        item_hash = _annotation_payload_hash(item_payload)
        audit_event = _append_audit_event(
            active_store=active_store,
            event_type=event_type,
            annotator=annotator,
            index=index,
            before_hash=item_hash,
            after_hash=item_hash,
            status="success",
            change_summary={
                "reason": str(payload.get("reason", ""))[:120],
                "active_ms": active_ms,
                "lead_time_ms": lead_time_ms,
                "total_changes": 0,
            },
            active_ms=active_ms,
            lead_time_ms=lead_time_ms,
        )
        return jsonify(
            {
                "ok": True,
                "event_id": audit_event.get("event_id", ""),
                "annotation_status": _status_for_index(active_store, index),
            }
        )

    @app.post("/api/video/<int:index>/annotation")
    def api_update_annotation(index: int) -> Response:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be JSON"}), 400

        active_store, annotator, error_response = _get_active_store()
        if error_response is not None:
            return error_response
        if active_store is None:
            return jsonify({"error": "Annotation store is not initialized"}), 500

        before_payload: Dict[str, Any] = {}
        try:
            before_payload = _item_audit_payload(active_store._get_item(index))
        except (IndexError, TypeError, ValueError):
            before_payload = {}
        before_hash = _annotation_payload_hash(before_payload) if before_payload else _item_audit_hash(active_store, index)
        audit_context = _audit_context_from_payload(payload)
        event_type = _audit_event_type("annotation_update", annotator)
        try:
            if multi_user_manager is not None and annotator is not None:
                updated = multi_user_manager.update_annotation(annotator.annotator_id, index, payload)
            else:
                updated = active_store.update_annotation(index, payload)
            after_payload = {
                "tasks": updated.get("tasks", []),
                "action_config": updated.get("action_config", []),
            }
            after_hash = _annotation_payload_hash(after_payload)
            diff = annotation_field_diff(before_payload, after_payload) if before_payload else {}
            change_summary = summarize_annotation_diff(diff)
            audit_event = _append_audit_event(
                active_store=active_store,
                event_type=event_type,
                annotator=annotator,
                index=index,
                before_hash=before_hash,
                after_hash=after_hash,
                status="success",
                diff=diff,
                change_summary=change_summary,
                active_ms=int(audit_context.get("active_ms", 0)),
                lead_time_ms=int(audit_context.get("lead_time_ms", 0)),
            )
            updated["annotation_status"] = _status_for_index(active_store, index)
            updated["audit_event_id"] = audit_event.get("event_id", "")
            logger.info(
                "annotation_update annotator_id=%s role=%s index=%s remote_addr=%s dirty_count=%s tasks_count=%s segments_count=%s",
                _annotator_log_id(annotator),
                _annotator_log_role(annotator),
                index,
                _request_remote_addr(),
                _dirty_count(active_store),
                len(updated.get("tasks", [])) if isinstance(updated.get("tasks"), list) else 0,
                len(updated.get("action_config", [])) if isinstance(updated.get("action_config"), list) else 0,
            )
        except IndexError as exc:
            logger.warning(
                "annotation_update_failed annotator_id=%s role=%s index=%s remote_addr=%s error=%s",
                _annotator_log_id(annotator),
                _annotator_log_role(annotator),
                index,
                _request_remote_addr(),
                exc,
            )
            _append_audit_event(
                active_store=active_store,
                event_type=event_type,
                annotator=annotator,
                index=index,
                before_hash=before_hash,
                after_hash="",
                status="failure",
                error=str(exc),
                change_summary={"total_changes": 0},
                active_ms=int(audit_context.get("active_ms", 0)),
                lead_time_ms=int(audit_context.get("lead_time_ms", 0)),
            )
            return jsonify({"error": str(exc)}), 404
        except ValidationError as exc:
            logger.warning(
                "annotation_update_failed annotator_id=%s role=%s index=%s remote_addr=%s error=%s",
                _annotator_log_id(annotator),
                _annotator_log_role(annotator),
                index,
                _request_remote_addr(),
                exc,
            )
            _append_audit_event(
                active_store=active_store,
                event_type=event_type,
                annotator=annotator,
                index=index,
                before_hash=before_hash,
                after_hash="",
                status="failure",
                error=str(exc),
                change_summary={"total_changes": 0},
                active_ms=int(audit_context.get("active_ms", 0)),
                lead_time_ms=int(audit_context.get("lead_time_ms", 0)),
            )
            response: Dict[str, Any] = {"error": str(exc)}
            if exc.field_errors:
                response["field_errors"] = exc.field_errors
            return jsonify(response), 400

        return jsonify(updated)

    @app.post("/api/video/<int:index>/discard")
    def api_update_discard(index: int) -> Response:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be JSON"}), 400

        active_store, annotator, error_response = _get_active_store()
        if error_response is not None:
            return error_response
        if active_store is None:
            return jsonify({"error": "Annotation store is not initialized"}), 500

        before_payload: Dict[str, Any] = {}
        try:
            before_payload = _item_audit_payload(active_store._get_item(index))
        except IndexError as exc:
            return jsonify({"error": str(exc)}), 404
        before_hash = _annotation_payload_hash(before_payload)
        audit_context = _audit_context_from_payload(payload)
        try:
            if multi_user_manager is not None and annotator is not None:
                updated = multi_user_manager.update_discard_and_save(
                    annotator.annotator_id,
                    index,
                    payload,
                )
            else:
                updated = active_store.update_discard_and_save(
                    index,
                    payload,
                    _annotator_log_id(annotator),
                )
            after_payload = _item_audit_payload(active_store._get_item(index))
            after_hash = _annotation_payload_hash(after_payload)
            before_discard = {
                key: copy.deepcopy(before_payload.get(key))
                for key in DISCARD_FIELDS
                if key in before_payload
            }
            after_discard = {
                key: copy.deepcopy(after_payload.get(key))
                for key in DISCARD_FIELDS
                if key in after_payload
            }
            audit_event = _append_audit_event(
                active_store=active_store,
                event_type="discard_update",
                annotator=annotator,
                index=index,
                before_hash=before_hash,
                after_hash=after_hash,
                status="success",
                diff={"discard": {"before": before_discard, "after": after_discard}},
                change_summary={
                    "total_changes": 1 if before_discard != after_discard else 0,
                    "discarded": bool(updated.get("discarded", False)),
                    "discard_reason": str(updated.get("discard_reason", "")),
                },
                active_ms=int(audit_context.get("active_ms", 0)),
                lead_time_ms=int(audit_context.get("lead_time_ms", 0)),
            )
            updated["annotation_status"] = _status_for_index(active_store, index)
            updated["audit_event_id"] = audit_event.get("event_id", "")
            return jsonify(updated)
        except ValidationError as exc:
            error_status = 404 if "index out of range" in str(exc).lower() else 400
            _append_audit_event(
                active_store=active_store,
                event_type="discard_update",
                annotator=annotator,
                index=index,
                before_hash=before_hash,
                after_hash="",
                status="failure",
                error=str(exc),
                change_summary={"total_changes": 0},
                active_ms=int(audit_context.get("active_ms", 0)),
                lead_time_ms=int(audit_context.get("lead_time_ms", 0)),
            )
            return jsonify({"error": str(exc)}), error_status
        except OSError as exc:
            logger.exception("discard_update save failed: index=%s", index)
            return jsonify({"error": f"Failed to save discard status: {exc}"}), 500

    @app.post("/api/save")
    def api_save() -> Response:
        payload = request.get_json(silent=True)
        audit_context = _audit_context_from_payload(payload if isinstance(payload, dict) else {})
        active_store, annotator, error_response = _get_active_store()
        if error_response is not None:
            return error_response
        if active_store is None:
            return jsonify({"error": "Annotation store is not initialized"}), 500

        dirty_count_before = _dirty_count(active_store)
        dirty_indices_before = sorted(
            idx for idx in getattr(active_store, "dirty_indices", set())
            if isinstance(idx, int)
        )
        before_hash = _store_audit_hash(active_store)
        event_type = _audit_event_type("annotation_save", annotator)
        try:
            if multi_user_manager is not None and annotator is not None:
                result = multi_user_manager.save(annotator.annotator_id)
            else:
                result = active_store.save()
            _append_audit_event(
                active_store=active_store,
                event_type=event_type,
                annotator=annotator,
                index=None,
                before_hash=before_hash,
                after_hash=_store_audit_hash(active_store),
                status="success",
                change_summary={
                    "dirty_indices": dirty_indices_before,
                    "dirty_items_saved": result.get("dirty_items_saved", 0),
                    "total_changes": 0,
                },
                active_ms=int(audit_context.get("active_ms", 0)),
                lead_time_ms=int(audit_context.get("lead_time_ms", 0)),
            )
            result["annotation_statuses"] = _annotation_statuses(active_store)
            logger.info(
                "annotation_save annotator_id=%s role=%s remote_addr=%s dirty_count_before=%s dirty_items_saved=%s saved_path=%s video_count=%s",
                _annotator_log_id(annotator),
                _annotator_log_role(annotator),
                _request_remote_addr(),
                dirty_count_before,
                result.get("dirty_items_saved", 0),
                result.get("saved_path", ""),
                result.get("video_count", 0),
            )
        except ValidationError as exc:
            logger.warning(
                "annotation_save_failed annotator_id=%s role=%s remote_addr=%s dirty_count_before=%s error=%s",
                _annotator_log_id(annotator),
                _annotator_log_role(annotator),
                _request_remote_addr(),
                dirty_count_before,
                exc,
            )
            _append_audit_event(
                active_store=active_store,
                event_type=event_type,
                annotator=annotator,
                index=None,
                before_hash=before_hash,
                after_hash="",
                status="failure",
                error=str(exc),
                change_summary={
                    "dirty_indices": dirty_indices_before,
                    "dirty_items_saved": 0,
                    "field_errors": exc.field_errors,
                    "total_changes": 0,
                },
                active_ms=int(audit_context.get("active_ms", 0)),
                lead_time_ms=int(audit_context.get("lead_time_ms", 0)),
            )
            response: Dict[str, Any] = {"error": str(exc)}
            if exc.field_errors:
                response["field_errors"] = exc.field_errors
            return jsonify(response), 400
        return jsonify(result)

    def _stream_video_response(index: int, view_key: Optional[str] = None) -> Response:
        active_store, _annotator, error_response = _get_active_store()
        if error_response is not None:
            return error_response
        if active_store is None:
            return jsonify({"error": "Annotation store is not initialized"}), 500

        range_header = request.headers.get("Range")
        try:
            source_kind, _video_source, video_path, remote_parts = active_store.resolve_video_source(
                index,
                view_key=view_key,
            )
        except IndexError as exc:
            logger.warning("api_stream index error: index=%s view=%s error=%s", index, view_key, exc)
            return jsonify({"error": str(exc)}), 404
        except KeyError as exc:
            logger.warning("api_stream view key error: index=%s view=%s error=%s", index, view_key, exc)
            return jsonify({"error": str(exc)}), 404
        except FileNotFoundError as exc:
            logger.warning("api_stream source not found: index=%s view=%s error=%s", index, view_key, exc)
            return jsonify({"error": str(exc)}), 404
        except RuntimeError as exc:
            logger.error("api_stream runtime error: index=%s view=%s error=%s", index, view_key, exc)
            return jsonify({"error": str(exc)}), 500

        if source_kind == "remote":
            if remote_parts is None:
                return jsonify({"error": "Invalid remote stream source"}), 500
            try:
                stream_iter, status_code, headers, content_type = active_store.open_remote_stream(
                    remote_parts=remote_parts,
                    range_header=range_header,
                )
            except ValidationError:
                logger.warning(
                    "api_stream invalid range: index=%s view=%s range=%s",
                    index,
                    view_key,
                    range_header,
                )
                return jsonify({"error": "Invalid Range header"}), 416
            except FileNotFoundError as exc:
                logger.warning("api_stream remote not found: index=%s view=%s error=%s", index, view_key, exc)
                return jsonify({"error": str(exc)}), 404
            except RuntimeError as exc:
                logger.error("api_stream remote runtime error: index=%s view=%s error=%s", index, view_key, exc)
                return jsonify({"error": str(exc)}), 500

            response = Response(stream_iter, status=status_code, mimetype=content_type)
            for key, value in headers.items():
                response.headers[key] = value
            return response

        if video_path is None:
            logger.error("api_stream local invalid source: index=%s view=%s", index, view_key)
            return jsonify({"error": "Invalid local stream source"}), 500

        if not video_path.is_file():
            logger.warning(
                "api_stream local file missing: index=%s view=%s path=%s",
                index,
                view_key,
                video_path,
            )
            return jsonify({"error": f"Video file not found: {video_path}"}), 404

        file_size = video_path.stat().st_size
        mime_type, _ = mimetypes.guess_type(video_path.name)
        content_type = mime_type or "video/mp4"

        if not range_header:
            def full_file_stream() -> Iterator[bytes]:
                with video_path.open("rb") as file:
                    while True:
                        chunk = file.read(1024 * 1024)
                        if not chunk:
                            break
                        yield chunk

            response = Response(full_file_stream(), status=200, mimetype=content_type)
            response.headers["Accept-Ranges"] = "bytes"
            response.headers["Content-Length"] = str(file_size)
            response.headers["Cache-Control"] = "no-store"
            return response

        try:
            start, end = parse_range_header(range_header, file_size)
        except (ValidationError, ValueError):
            return jsonify({"error": "Invalid Range header"}), 416

        chunk_size = end - start + 1

        def partial_stream() -> Iterator[bytes]:
            with video_path.open("rb") as file:
                file.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    read_size = min(1024 * 1024, remaining)
                    data = file.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        response = Response(partial_stream(), status=206, mimetype=content_type)

        response.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        response.headers["Accept-Ranges"] = "bytes"
        response.headers["Content-Length"] = str(chunk_size)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/stream/<int:index>")
    def api_stream_video(index: int) -> Response:
        return _stream_video_response(index=index, view_key=None)

    @app.get("/api/stream/<int:index>/<path:view_key>")
    def api_stream_video_view(index: int, view_key: str) -> Response:
        return _stream_video_response(index=index, view_key=view_key)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video annotation inspection UI")
    parser.add_argument(
        "--input-json",
        default="",
        help="Path to input JSON array/JSONL",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Path to output JSON/JSONL (default: <input_dir>/annotations.jsonl)",
    )
    parser.add_argument(
        "--multi-user-manifest",
        default="",
        help="Path to multi-user collaboration manifest JSON",
    )
    parser.add_argument(
        "--mode",
        default=APP_MODE_ANNOTATION,
        choices=sorted(APP_MODES),
        help="Application mode. Use model-review to overlay offline model review results.",
    )
    parser.add_argument(
        "--model-review-json",
        default="",
        help="Path to offline model review JSONL generated by scripts/model_review_score.py",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Flask host")
    parser.add_argument("--port", default=7860, type=int, help="Flask port")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to annotation_ui config.ini",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Enable Flask debug mode (default: enabled)",
    )
    parser.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="Disable Flask debug mode",
    )
    parser.set_defaults(debug=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    manifest_raw = str(args.multi_user_manifest or "").strip()
    app_mode = str(args.mode or APP_MODE_ANNOTATION).strip()
    model_review_raw = str(args.model_review_json or "").strip()

    try:
        runtime_config = RuntimeConfig.load(config_path)
    except ValidationError as exc:
        raise SystemExit(f"Failed to load config: {exc}") from exc

    if app_mode == APP_MODE_MODEL_REVIEW and not model_review_raw:
        raise SystemExit("--model-review-json is required when --mode model-review")

    model_review_store: Optional[ModelReviewStore] = None
    if model_review_raw:
        try:
            model_review_store = ModelReviewStore.load(Path(model_review_raw).resolve())
        except ValidationError as exc:
            raise SystemExit(f"Failed to load model review JSONL: {exc}") from exc

    if manifest_raw:
        manifest_path = Path(manifest_raw).resolve()
        try:
            multi_user_manifest = MultiUserManifest.load(manifest_path)
        except ValidationError as exc:
            raise SystemExit(f"Failed to load multi-user manifest: {exc}") from exc

        manager = MultiUserStoreManager(
            manifest=multi_user_manifest,
            runtime_config=runtime_config,
        )
        app = create_app(
            store=None,
            multi_user_manager=manager,
            model_review_store=model_review_store,
            app_mode=app_mode,
        )
        app.run(host=args.host, port=args.port, debug=args.debug)
        return

    if not args.input_json:
        raise SystemExit("--input-json or --multi-user-manifest is required")
    input_path = Path(args.input_json).resolve()
    output_path, output_is_default = resolve_output_path(input_path, args.output_json)

    if not input_path.exists():
        raise SystemExit(f"Input JSON not found: {input_path}")

    if output_is_default and not output_path.exists():
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch(exist_ok=True)
        except OSError as exc:
            raise SystemExit(f"Failed to create default output file: {exc}") from exc

    try:
        store = AnnotationStore.load(
            input_path=input_path,
            output_path=output_path,
            runtime_config=runtime_config,
        )
    except ValidationError as exc:
        raise SystemExit(f"Failed to initialize annotation store: {exc}") from exc

    try:
        store.warmup_playback()
    except Exception as exc:
        logger.warning("playback warmup crashed: %s", exc)

    app = create_app(
        store=store,
        model_review_store=model_review_store,
        app_mode=app_mode,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

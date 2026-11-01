#!/usr/bin/env python3
"""Model-free signal extraction for rule-based trajectory segmentation.

Only robot state/action columns and optional OpenCV frame-change statistics are
used. Annotation-like columns, including ``subtask_index``, are rejected before
Arrow reads so target boundaries cannot enter the feature cache.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
import uuid
from dataclasses import dataclass, field, replace
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

petrel_site_packages = os.environ.get("PETREL_CLIENT_SITE_PACKAGES")
if petrel_site_packages:
    sys.path.append(petrel_site_packages)
try:
    from petrel_client.client import Client
except (ImportError, OSError):  # pragma: no cover - optional for local tests.
    Client = None  # type: ignore[assignment]

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from roboatom.physextend.features.embodiment import EmbodimentProfile
from roboatom.physextend.features.feature_mapping import (
    AUTO_MAPPING_DIGEST,
    FeatureDimensionRole,
    FeatureMapConfig,
    mapping_digest,
    resolve_feature_roles,
    role_indices,
    schema_gripper_indices,
)
from roboatom.physextend.features.storage import is_rclone_remote, read_rclone_bytes

try:
    import av
except (ImportError, OSError):  # pragma: no cover - optional on CPU workers.
    av = None  # type: ignore[assignment]

try:
    import cv2
except (ImportError, OSError):  # pragma: no cover - optional on CPU workers.
    cv2 = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
REMOTE_PATH_RE = re.compile(r"^(?:[^/:\s]+:)?s3://")
ANNOTATION_NAME_RE = re.compile(r"(?:subtask|segment|annotation|label|ground.?truth|boundary)", re.IGNORECASE)
SIGNAL_NAME_RE = re.compile(
    r"(?:state|action|joint|motor|gripper|end_effector|eef|pose|position|velocity|effort|torque)",
    re.IGNORECASE,
)
DEFAULT_PETREL_CONF = "~/petreloss.conf"
CACHE_SCHEMA_VERSION = 5
SUPPORTED_CACHE_SCHEMA_VERSIONS = frozenset({4, CACHE_SCHEMA_VERSION})
_PETREL_CLIENT: Any | None = None


@dataclass(frozen=True)
class SignalEvent:
    """One stable, model-free signal transition."""

    frame: int
    kind: str
    strength: float
    channel: str


@dataclass(frozen=True)
class FeatureBundle:
    """Canonical per-frame rule features for one trajectory."""

    length: int
    activity: np.ndarray
    pause: np.ndarray
    turn: np.ndarray
    gripper_delta: np.ndarray
    gripper_level: np.ndarray
    visual_motion: np.ndarray
    events: tuple[SignalEvent, ...]
    source_columns: tuple[str, ...]
    state_gripper_level: np.ndarray | None = None
    state_gripper_delta: np.ndarray | None = None
    action_gripper_level: np.ndarray | None = None
    action_gripper_delta: np.ndarray | None = None
    action_state_error: np.ndarray | None = None
    embodiment: Mapping[str, Any] = field(default_factory=dict)
    feature_mapping: tuple[Mapping[str, Any], ...] = ()
    feature_mapping_digest: str = AUTO_MAPPING_DIGEST
    feature_mapping_config: str | None = None

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("Feature length must be positive")
        for name in (
            "activity",
            "pause",
            "turn",
            "gripper_delta",
            "gripper_level",
            "visual_motion",
        ):
            values = np.asarray(getattr(self, name))
            if values.shape != (self.length,):
                raise ValueError(f"{name} has shape {values.shape}, expected {(self.length,)}")
        for name in (
            "state_gripper_level",
            "state_gripper_delta",
            "action_gripper_level",
            "action_gripper_delta",
            "action_state_error",
        ):
            raw_values = getattr(self, name)
            if raw_values is None:
                continue
            values = np.asarray(raw_values)
            if values.shape != (self.length,):
                raise ValueError(
                    f"{name} has shape {values.shape}, expected {(self.length,)}"
                )


def _is_remote(path: str) -> bool:
    return REMOTE_PATH_RE.match(path) is not None


def _petrel_client() -> Any:
    global _PETREL_CLIENT
    if _PETREL_CLIENT is None:
        if Client is None:
            raise RuntimeError("petrel_client is required for cluster_hoss:s3 paths")
        conf_path = os.path.expanduser(os.environ.get("PETREL_CONF", DEFAULT_PETREL_CONF))
        _PETREL_CLIENT = Client(conf_path, enable_mc=False)
    return _PETREL_CLIENT


def _read_bytes(path: str) -> bytes:
    if is_rclone_remote(path):
        return read_rclone_bytes(path)
    if not _is_remote(path):
        return Path(path).read_bytes()
    try:
        payload = _petrel_client().get(path)
    except Exception as error:
        raise RuntimeError(f"Failed to read object storage path: {path}") from error
    if payload is None:
        raise FileNotFoundError(path)
    if isinstance(payload, bytearray):
        return bytes(payload)
    if not isinstance(payload, bytes):
        raise TypeError(f"Expected bytes for {path}, got {type(payload).__name__}")
    return payload


def _open_parquet(path: str) -> pq.ParquetFile:
    if _is_remote(path) or is_rclone_remote(path):
        return pq.ParquetFile(BytesIO(_read_bytes(path)))
    return pq.ParquetFile(path)


def _is_numeric_arrow_type(data_type: pa.DataType) -> bool:
    if pa.types.is_integer(data_type) or pa.types.is_floating(data_type):
        return True
    if pa.types.is_list(data_type) or pa.types.is_large_list(data_type) or pa.types.is_fixed_size_list(data_type):
        return pa.types.is_integer(data_type.value_type) or pa.types.is_floating(data_type.value_type)
    return False


def select_signal_columns(schema: pa.Schema) -> list[str]:
    """Select model-free numeric robot signals while rejecting label columns.

    Args:
        schema: Arrow parquet schema.

    Returns:
        Deterministically ordered safe column names.
    """
    selected: list[str] = []
    for column_field in schema:
        name = column_field.name
        if ANNOTATION_NAME_RE.search(name) or not SIGNAL_NAME_RE.search(name):
            continue
        if _is_numeric_arrow_type(column_field.type):
            selected.append(name)
    return sorted(selected)


def _column_matrix(table: pa.Table, column_name: str) -> np.ndarray:
    values = np.asarray(table[column_name].to_pylist(), dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"Unsupported signal shape for {column_name}: {values.shape}")
    if not np.all(np.isfinite(values)):
        finite = np.where(np.isfinite(values), values, np.nan)
        medians = np.nanmedian(finite, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        values = np.where(np.isfinite(values), values, medians)
    return values.astype(np.float32)


def _resample(values: np.ndarray, length: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape[0] == length:
        return array
    if array.shape[0] == 0:
        return np.zeros((length,) + array.shape[1:], dtype=np.float32)
    source = np.linspace(0.0, 1.0, array.shape[0], dtype=np.float32)
    target = np.linspace(0.0, 1.0, length, dtype=np.float32)
    if array.ndim == 1:
        return np.interp(target, source, array).astype(np.float32)
    return np.stack([np.interp(target, source, array[:, index]) for index in range(array.shape[1])], axis=1).astype(
        np.float32
    )


def robust_unit(values: np.ndarray, lower: float = 5.0, upper: float = 95.0) -> np.ndarray:
    """Map a scalar signal robustly into ``[0, 1]``."""
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return array
    low, high = np.percentile(array, [lower, upper])
    scale = max(float(high - low), 1e-6)
    return np.clip((array - low) / scale, 0.0, 1.0).astype(np.float32)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Return an edge-padded centered moving average."""
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if window <= 1 or array.size <= 1:
        return array.copy()
    actual_window = min(int(window), array.size)
    left = actual_window // 2
    right = actual_window - left - 1
    kernel = np.ones(actual_window, dtype=np.float32) / actual_window
    return np.convolve(np.pad(array, (left, right), mode="edge"), kernel, mode="valid").astype(np.float32)


def _matrix_motion(matrix: np.ndarray) -> np.ndarray:
    delta = np.diff(matrix, axis=0, prepend=matrix[[0]])
    scale = np.percentile(np.abs(delta), 90.0, axis=0)
    active = scale > 1e-7
    if not np.any(active):
        return np.zeros(matrix.shape[0], dtype=np.float32)
    normalized = delta[:, active] / scale[active]
    return robust_unit(np.sqrt(np.mean(np.square(normalized), axis=1)))


def _matrix_turn(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 3:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    delta = np.diff(matrix, axis=0, prepend=matrix[[0]])
    scale = np.percentile(np.abs(delta), 90.0, axis=0)
    active = scale > 1e-7
    if not np.any(active):
        return np.zeros(matrix.shape[0], dtype=np.float32)
    velocity = delta[:, active] / scale[active]
    previous = np.vstack([velocity[[0]], velocity[:-1]])
    denominator = np.linalg.norm(velocity, axis=1) * np.linalg.norm(previous, axis=1)
    cosine = np.sum(velocity * previous, axis=1) / np.maximum(denominator, 1e-6)
    magnitude = np.minimum(np.linalg.norm(velocity, axis=1), np.linalg.norm(previous, axis=1))
    return moving_average(robust_unit(np.clip(1.0 - cosine, 0.0, 2.0) * magnitude), 5)


def _aggregate(features: Sequence[np.ndarray], length: int) -> np.ndarray:
    if not features:
        return np.zeros(length, dtype=np.float32)
    matrix = np.stack([_resample(feature, length) for feature in features], axis=1)
    return np.percentile(matrix, 75.0, axis=1).astype(np.float32)


def _gripper_series(
    matrices: Sequence[np.ndarray], length: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Aggregate optional gripper matrices into robust level and delta."""
    if not matrices:
        return None, None
    levels = [
        robust_unit(matrix[:, index])
        for matrix in matrices
        for index in range(matrix.shape[1])
    ]
    level = _aggregate(levels, length)
    deltas = [moving_average(np.diff(item, prepend=item[0]), 3) for item in levels]
    delta_matrix = np.stack(deltas, axis=1)
    strongest = np.argmax(np.abs(delta_matrix), axis=1)
    delta = delta_matrix[np.arange(length), strongest]
    return level.astype(np.float32), delta.astype(np.float32)


def detect_signal_events(
    gripper_delta: np.ndarray,
    gripper_level: np.ndarray,
    activity: np.ndarray,
    minimum_distance: int = 6,
) -> tuple[SignalEvent, ...]:
    """Detect stable gripper changes, activity peaks, pauses, and turns."""
    length = gripper_delta.size
    candidates: list[SignalEvent] = []
    signals = {
        "gripper": robust_unit(np.abs(gripper_delta)),
        "activity": robust_unit(activity),
        "pause": 1.0 - robust_unit(activity),
    }
    thresholds = {name: float(np.percentile(values, 85.0)) for name, values in signals.items()}
    for name, values in signals.items():
        for frame in range(1, max(length - 1, 1)):
            is_peak = (
                values[frame] >= thresholds[name]
                and values[frame] >= values[frame - 1]
                and values[frame] >= values[frame + 1]
            )
            if is_peak:
                kind = name
                if name == "gripper":
                    kind = "gripper_increase" if gripper_delta[frame] >= 0 else "gripper_decrease"
                candidates.append(SignalEvent(frame, kind, float(values[frame]), name))
    selected: list[SignalEvent] = []
    for event in sorted(candidates, key=lambda item: (-item.strength, item.frame, item.kind)):
        is_duplicate = any(
            existing.kind == event.kind and abs(existing.frame - event.frame) < minimum_distance
            for existing in selected
        )
        if is_duplicate:
            continue
        selected.append(event)
    selected.sort(key=lambda item: (item.frame, item.kind))
    return tuple(selected)


def _append_frame_change(
    image: np.ndarray,
    frame_index: int,
    previous: np.ndarray | None,
    frame_numbers: list[int],
    changes: list[float],
) -> np.ndarray:
    resized = cv2.resize(image, (96, 64), interpolation=cv2.INTER_AREA)
    change = 0.0 if previous is None else float(np.mean(cv2.absdiff(resized, previous)))
    frame_numbers.append(frame_index)
    changes.append(change)
    return resized


def _decode_av_changes(video_path: str, stride: int) -> tuple[list[int], list[float]]:
    payload: bytes | None = _read_bytes(video_path) if _is_remote(video_path) else None
    source: Any = BytesIO(payload) if payload is not None else video_path
    frame_numbers: list[int] = []
    changes: list[float] = []
    previous: np.ndarray | None = None
    with av.open(source) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index % stride:
                continue
            previous = _append_frame_change(
                frame.to_ndarray(format="gray"),
                frame_index,
                previous,
                frame_numbers,
                changes,
            )
    return frame_numbers, changes


def _decode_opencv_changes(video_path: str, stride: int) -> tuple[list[int], list[float]]:
    temporary_path: str | None = None
    source_path = video_path
    if _is_remote(video_path):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temporary:
            temporary.write(_read_bytes(video_path))
            temporary_path = temporary.name
        source_path = temporary_path
    capture = cv2.VideoCapture(source_path)
    frame_numbers: list[int] = []
    changes: list[float] = []
    previous: np.ndarray | None = None
    frame_index = 0
    try:
        if not capture.isOpened():
            raise OSError(f"OpenCV could not open {video_path}")
        while True:
            has_frame, image = capture.read()
            if not has_frame:
                break
            if frame_index % stride == 0:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                previous = _append_frame_change(gray, frame_index, previous, frame_numbers, changes)
            frame_index += 1
    finally:
        capture.release()
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)
    return frame_numbers, changes


def _visual_motion(video_paths: Sequence[str], length: int, stride: int = 3) -> np.ndarray:
    if not video_paths or cv2 is None:
        return np.zeros(length, dtype=np.float32)
    per_view: list[np.ndarray] = []
    for video_path in video_paths:
        try:
            frame_numbers, changes = (
                _decode_av_changes(video_path, stride)
                if av is not None
                else _decode_opencv_changes(video_path, stride)
            )
        except (OSError, ValueError, RuntimeError) as error:
            logger.warning("Unable to decode model-free visual motion from %s: %s", video_path, error)
            continue
        if len(changes) >= 2:
            sampled = np.interp(
                np.arange(length),
                np.asarray(frame_numbers),
                np.asarray(changes),
            ).astype(np.float32)
            per_view.append(robust_unit(sampled))
    return _aggregate(per_view, length)


def _source_sensor_features(signals_path: str) -> Mapping[str, Any]:
    """Load info.json feature names when the path is a dataset chunk.

    Paths without ``/data/chunk-`` keep the historical column-name mapping.
    The schema lookup lives with decoding and is imported lazily to avoid a cycle.
    """
    if "/data/chunk-" not in signals_path:
        return {}
    try:
        from roboatom.physextend.decoders.seeded_decoder import sensor_schema
    except (ImportError, OSError) as error:
        logger.warning("Sensor schema unavailable for %s; using column/config mapping: %s", signals_path, error)
        return {}

    try:
        return sensor_schema(signals_path)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        logger.warning("Sensor schema read failed for %s; using column/config mapping: %s", signals_path, error)
        return {}


def _with_schema_grippers(
    roles: tuple[FeatureDimensionRole, ...],
    description: Mapping[str, Any],
) -> tuple[tuple[FeatureDimensionRole, ...], tuple[int, ...]]:
    """Add info.json gripper indices the column-name rules did not already claim.

    Custom roles keep the family written in the feature map. Motion continues to
    use the roles returned by ``resolve_feature_roles`` so validated activity
    does not move. Schema indices already labeled non-gripper by that map are
    not openings, even when the dimension name contains ``gripper``.
    """
    if not description:
        return roles, ()
    width = len(roles)
    schema_indices = schema_gripper_indices(description, width)
    if not schema_indices:
        return roles, ()
    claimed = set(role_indices(roles, "gripper"))
    rewritten: list[FeatureDimensionRole] = []
    for role in roles:
        if role.index in schema_indices and role.index not in claimed and role.origin != "config":
            rewritten.append(replace(role, family="gripper", quantity="position"))
        else:
            rewritten.append(role)
    return tuple(rewritten), schema_indices


def extract_features(
    signals_path: str,
    expected_length: int,
    video_paths: Sequence[str] = (),
    include_visual: bool = False,
    embodiment: Mapping[str, Any] | None = None,
    dataset: str | None = None,
    feature_map_config: FeatureMapConfig | None = None,
    source_uri: str | None = None,
) -> FeatureBundle:
    """Extract deterministic state/action and optional OpenCV features.

    Args:
        signals_path: Local or object-storage parquet path.
        expected_length: Manifest frame count.
        video_paths: Optional camera videos.
        include_visual: Whether to decode model-free frame differences.
        embodiment: Optional metadata-only robot profile from the manifest.
        dataset: Optional dataset identifier used only for feature-role lookup.
        feature_map_config: Optional explicit semantic mapping configuration.
        source_uri: Original dataset URI when ``signals_path`` is a temporary
            local copy; info.json schema is looked up from it.

    Returns:
        Canonical feature bundle.
    """
    if expected_length <= 0:
        raise ValueError("expected_length must be positive")
    parquet_file = _open_parquet(signals_path)
    columns = select_signal_columns(parquet_file.schema_arrow)
    if not columns:
        raise ValueError(f"No safe numeric robot signals in {signals_path}")
    table = parquet_file.read(columns=columns)
    motion_features: list[np.ndarray] = []
    turn_features: list[np.ndarray] = []
    gripper_matrices: list[np.ndarray] = []
    state_gripper_matrices: list[np.ndarray] = []
    action_gripper_matrices: list[np.ndarray] = []
    feature_mapping: list[Mapping[str, Any]] = []
    sensor_features = _source_sensor_features(source_uri or signals_path)
    for column in columns:
        matrix = _resample(_column_matrix(table, column), expected_length)
        column_roles = resolve_feature_roles(
            column,
            matrix.shape[1],
            dataset=dataset,
            config=feature_map_config,
        )
        roles, schema_indices = _with_schema_grippers(
            column_roles, sensor_features.get(column, {})
        )
        feature_mapping.extend(role.to_dict() for role in roles)
        config_non_gripper = {
            role.index
            for role in roles
            if role.origin == "config" and role.family != "gripper"
        }
        gripper_indices = tuple(
            sorted(
                (set(role_indices(roles, "gripper")) | set(schema_indices))
                - config_non_gripper
            )
        )
        state_gripper_indices = tuple(
            role.index
            for role in roles
            if role.index in gripper_indices and role.source == "state"
        )
        action_gripper_indices = tuple(
            role.index
            for role in roles
            if role.index in gripper_indices and role.source == "action"
        )
        motion_indices = tuple(
            role.index for role in column_roles if role.family != "gripper"
        )
        if gripper_indices:
            gripper_matrices.append(matrix[:, gripper_indices])
        if state_gripper_indices:
            state_gripper_matrices.append(matrix[:, state_gripper_indices])
        if action_gripper_indices:
            action_gripper_matrices.append(matrix[:, action_gripper_indices])
        if motion_indices:
            motion_matrix = matrix[:, motion_indices]
            motion_features.append(_matrix_motion(motion_matrix))
            turn_features.append(_matrix_turn(motion_matrix))
    activity = moving_average(_aggregate(motion_features, expected_length), 5)
    turn = moving_average(_aggregate(turn_features, expected_length), 5)
    gripper_level_optional, gripper_delta_optional = _gripper_series(
        gripper_matrices, expected_length
    )
    state_level, state_delta = _gripper_series(
        state_gripper_matrices, expected_length
    )
    action_level, action_delta = _gripper_series(
        action_gripper_matrices, expected_length
    )
    if gripper_level_optional is None or gripper_delta_optional is None:
        gripper_level = np.zeros(expected_length, dtype=np.float32)
        gripper_delta = np.zeros(expected_length, dtype=np.float32)
    else:
        gripper_level = gripper_level_optional
        gripper_delta = gripper_delta_optional
    action_state_error = (
        np.abs(action_level - state_level).astype(np.float32)
        if action_level is not None and state_level is not None
        else None
    )
    visual_motion = (
        _visual_motion(video_paths, expected_length)
        if include_visual
        else np.zeros(expected_length, dtype=np.float32)
    )
    combined_activity = np.maximum(activity, 0.7 * visual_motion)
    pause = 1.0 - robust_unit(moving_average(combined_activity, 11))
    events = detect_signal_events(gripper_delta, gripper_level, combined_activity)
    return FeatureBundle(
        length=expected_length,
        activity=combined_activity.astype(np.float32),
        pause=pause.astype(np.float32),
        turn=turn.astype(np.float32),
        gripper_delta=gripper_delta.astype(np.float32),
        gripper_level=gripper_level.astype(np.float32),
        visual_motion=visual_motion.astype(np.float32),
        events=events,
        source_columns=tuple(columns),
        state_gripper_level=state_level,
        state_gripper_delta=state_delta,
        action_gripper_level=action_level,
        action_gripper_delta=action_delta,
        action_state_error=action_state_error,
        embodiment=EmbodimentProfile.from_columns(columns, embodiment).to_dict(),
        feature_mapping=tuple(feature_mapping),
        feature_mapping_digest=mapping_digest(feature_map_config),
        feature_mapping_config=(
            feature_map_config.source_path if feature_map_config is not None else None
        ),
    )


def save_feature_bundle(path: Path, bundle: FeatureBundle) -> None:
    """Atomically persist a feature bundle as a non-pickle NPZ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "length": bundle.length,
        "events": [event.__dict__ for event in bundle.events],
        "source_columns": list(bundle.source_columns),
        "feature_mapping": [dict(role) for role in bundle.feature_mapping],
        "feature_mapping_digest": bundle.feature_mapping_digest,
        "feature_mapping_config": bundle.feature_mapping_config,
        "optional_arrays": [
            name
            for name in (
                "state_gripper_level",
                "state_gripper_delta",
                "action_gripper_level",
                "action_gripper_delta",
                "action_state_error",
            )
            if getattr(bundle, name) is not None
        ],
        "embodiment": dict(bundle.embodiment),
    }
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npz")
    np.savez_compressed(
        temporary_path,
        activity=bundle.activity,
        pause=bundle.pause,
        turn=bundle.turn,
        gripper_delta=bundle.gripper_delta,
        gripper_level=bundle.gripper_level,
        visual_motion=bundle.visual_motion,
        state_gripper_level=(
            bundle.state_gripper_level
            if bundle.state_gripper_level is not None
            else np.zeros(bundle.length, dtype=np.float32)
        ),
        state_gripper_delta=(
            bundle.state_gripper_delta
            if bundle.state_gripper_delta is not None
            else np.zeros(bundle.length, dtype=np.float32)
        ),
        action_gripper_level=(
            bundle.action_gripper_level
            if bundle.action_gripper_level is not None
            else np.zeros(bundle.length, dtype=np.float32)
        ),
        action_gripper_delta=(
            bundle.action_gripper_delta
            if bundle.action_gripper_delta is not None
            else np.zeros(bundle.length, dtype=np.float32)
        ),
        action_state_error=(
            bundle.action_state_error
            if bundle.action_state_error is not None
            else np.zeros(bundle.length, dtype=np.float32)
        ),
        metadata=np.asarray(json.dumps(metadata)),
    )
    temporary_path.replace(path)


def load_feature_bundle(
    path: Path, expected_feature_mapping_digest: str | None = None
) -> FeatureBundle:
    """Load and validate a cached feature bundle.

    Args:
        path: Supported feature-cache schema path.
        expected_feature_mapping_digest: Optional explicit mapping identity.

    Returns:
        Validated feature bundle.

    Raises:
        ValueError: If the schema or requested mapping identity is incompatible.
    """
    with np.load(path, allow_pickle=False) as values:
        metadata = json.loads(str(values["metadata"].item()))
        schema_version = int(metadata.get("schema_version", 0))
        if schema_version not in SUPPORTED_CACHE_SCHEMA_VERSIONS:
            raise ValueError(
                f"Feature cache {path} has schema version "
                f"{schema_version}, expected one of {sorted(SUPPORTED_CACHE_SCHEMA_VERSIONS)}"
            )
        stored_mapping_digest = str(
            metadata.get("feature_mapping_digest", AUTO_MAPPING_DIGEST)
        )
        if (
            expected_feature_mapping_digest is not None
            and stored_mapping_digest != expected_feature_mapping_digest
        ):
            raise ValueError(
                f"Feature cache {path} uses mapping {stored_mapping_digest}, "
                f"expected {expected_feature_mapping_digest}"
            )
        optional_arrays = set(metadata.get("optional_arrays", []))
        return FeatureBundle(
            length=int(metadata["length"]),
            activity=values["activity"].astype(np.float32),
            pause=values["pause"].astype(np.float32),
            turn=values["turn"].astype(np.float32),
            gripper_delta=values["gripper_delta"].astype(np.float32),
            gripper_level=values["gripper_level"].astype(np.float32),
            visual_motion=values["visual_motion"].astype(np.float32),
            events=tuple(SignalEvent(**event) for event in metadata["events"]),
            source_columns=tuple(metadata["source_columns"]),
            state_gripper_level=(
                values["state_gripper_level"].astype(np.float32)
                if "state_gripper_level" in optional_arrays
                else None
            ),
            state_gripper_delta=(
                values["state_gripper_delta"].astype(np.float32)
                if "state_gripper_delta" in optional_arrays
                else None
            ),
            action_gripper_level=(
                values["action_gripper_level"].astype(np.float32)
                if "action_gripper_level" in optional_arrays
                else None
            ),
            action_gripper_delta=(
                values["action_gripper_delta"].astype(np.float32)
                if "action_gripper_delta" in optional_arrays
                else None
            ),
            action_state_error=(
                values["action_state_error"].astype(np.float32)
                if "action_state_error" in optional_arrays
                else None
            ),
            embodiment=dict(metadata.get("embodiment", {})),
            feature_mapping=tuple(
                dict(role) for role in metadata.get("feature_mapping", [])
            ),
            feature_mapping_digest=stored_mapping_digest,
            feature_mapping_config=metadata.get("feature_mapping_config"),
        )

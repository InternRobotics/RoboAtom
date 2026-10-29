"""Semantic roles for boundary-free robot signal columns.

The parquet safety selector remains the single source of truth for which raw
columns may be read.  This module only explains selected numeric dimensions so
downstream feature extraction can identify gripper state/action channels.  A
configuration is optional; unmatched dimensions use deterministic name-based
rules and are never dropped.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast

FeatureSource = Literal["state", "action", "unknown"]
FeatureFamily = Literal["gripper", "joint", "eef_pose", "base", "other_motion"]
FeatureSide = Literal["left", "right", "both", "unknown"]
FeatureQuantity = Literal[
    "position", "velocity", "effort", "torque", "current", "delta", "unknown"
]

VALID_SOURCES = frozenset({"state", "action", "unknown"})
VALID_FAMILIES = frozenset({"gripper", "joint", "eef_pose", "base", "other_motion"})
VALID_SIDES = frozenset({"left", "right", "both", "unknown"})
VALID_QUANTITIES = frozenset(
    {"position", "velocity", "effort", "torque", "current", "delta", "unknown"}
)
CONFIG_SCHEMA_VERSION = 1
AUTO_MAPPING_DIGEST = "auto-v1"

ACTION_RE = re.compile(r"(?:^|[._/])actions?(?:[._/]|$)", re.IGNORECASE)
STATE_RE = re.compile(
    r"(?:^|[._/])(?:observation[._/])?states?(?:[._/]|$)", re.IGNORECASE
)
GRIPPER_RE = re.compile(
    r"(?:gripper|finger|end_effector_(?:left|right)_position)", re.IGNORECASE
)
JOINT_RE = re.compile(r"(?:joint|motor)", re.IGNORECASE)
EEF_RE = re.compile(r"(?:end_effector|eef|effector|tcp|cartesian)", re.IGNORECASE)
BASE_RE = re.compile(r"(?:base|chassis|mobile|wheel|robot[._/]position)", re.IGNORECASE)
LEFT_RE = re.compile(r"(?:^|[._/])left(?:[._/]|$)", re.IGNORECASE)
RIGHT_RE = re.compile(r"(?:^|[._/])right(?:[._/]|$)", re.IGNORECASE)


@dataclass(frozen=True)
class FeatureDimensionRole:
    """Semantic role assigned to one numeric column dimension."""

    column: str
    index: int
    source: FeatureSource
    family: FeatureFamily
    side: FeatureSide
    quantity: FeatureQuantity
    origin: Literal["config", "auto"]
    confidence: float
    is_ambiguous: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe audit representation."""
        return asdict(self)


@dataclass(frozen=True)
class FeatureMapRule:
    """One explicit role assignment for selected dimensions of a column."""

    column: str
    indices: str | tuple[int, ...] | Mapping[str, int]
    source: FeatureSource
    family: FeatureFamily
    side: FeatureSide
    quantity: FeatureQuantity
    expected_width: int | None = None


@dataclass(frozen=True)
class FeatureMapConfig:
    """Validated dataset-specific feature-role configuration."""

    datasets: Mapping[str, tuple[FeatureMapRule, ...]]
    digest: str
    source_path: str | None = None

    def rules_for(self, dataset: str | None, column: str) -> tuple[FeatureMapRule, ...]:
        """Return exact-column rules for a dataset."""
        if not dataset:
            return ()
        return tuple(
            rule for rule in self.datasets.get(dataset, ()) if rule.column == column
        )


def _validate_choice(name: str, value: Any, allowed: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}")
    return value


def _parse_indices(value: Any) -> str | tuple[int, ...] | Mapping[str, int]:
    if value is None or value == "all":
        return "all"
    if isinstance(value, list):
        if not value or any(not isinstance(index, int) or isinstance(index, bool) for index in value):
            raise ValueError("indices list must contain at least one integer")
        if len(set(value)) != len(value):
            raise ValueError("indices list contains duplicates")
        return tuple(value)
    if isinstance(value, Mapping):
        allowed = {"start", "stop", "step"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown slice keys: {sorted(unknown)}")
        parsed: dict[str, int] = {}
        for key, raw in value.items():
            if not isinstance(raw, int) or isinstance(raw, bool):
                raise ValueError(f"indices.{key} must be an integer")
            parsed[str(key)] = raw
        if "stop" not in parsed:
            raise ValueError("Slice indices require stop")
        if parsed.get("step", 1) <= 0:
            raise ValueError("Slice step must be positive")
        return parsed
    raise ValueError("indices must be 'all', an integer list, or a slice object")


def _parse_rule(raw: Any) -> FeatureMapRule:
    if not isinstance(raw, Mapping):
        raise ValueError("Every feature mapping rule must be an object")
    allowed_keys = {
        "column",
        "expected_width",
        "indices",
        "source",
        "family",
        "side",
        "quantity",
    }
    unknown_keys = set(raw) - allowed_keys
    if unknown_keys:
        raise ValueError(f"Unknown feature mapping rule keys: {sorted(unknown_keys)}")
    column = raw.get("column")
    if not isinstance(column, str) or not column.strip():
        raise ValueError("Feature mapping rule column must be non-empty")
    expected_width = raw.get("expected_width")
    if expected_width is not None and (
        not isinstance(expected_width, int)
        or isinstance(expected_width, bool)
        or expected_width <= 0
    ):
        raise ValueError("expected_width must be a positive integer")
    return FeatureMapRule(
        column=column,
        indices=_parse_indices(raw.get("indices", "all")),
        source=cast(
            FeatureSource,
            _validate_choice("source", raw.get("source"), VALID_SOURCES),
        ),
        family=cast(
            FeatureFamily,
            _validate_choice("family", raw.get("family"), VALID_FAMILIES),
        ),
        side=cast(
            FeatureSide,
            _validate_choice("side", raw.get("side", "unknown"), VALID_SIDES),
        ),
        quantity=cast(
            FeatureQuantity,
            _validate_choice(
                "quantity", raw.get("quantity", "unknown"), VALID_QUANTITIES
            ),
        ),
        expected_width=expected_width,
    )


def load_feature_map_config(path: Path | None) -> FeatureMapConfig | None:
    """Load and validate an optional JSON feature-role configuration.

    Args:
        path: JSON path, or ``None`` to use automatic mapping only.

    Returns:
        Validated configuration, or ``None``.

    Raises:
        ValueError: If the JSON shape or a rule is invalid.
    """
    if path is None:
        return None
    try:
        raw_text = path.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to load feature mapping config {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Feature mapping config root must be an object")
    unknown_root_keys = set(payload) - {"schema_version", "datasets"}
    if unknown_root_keys:
        raise ValueError(
            f"Unknown feature mapping config keys: {sorted(unknown_root_keys)}"
        )
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Feature mapping schema_version must be {CONFIG_SCHEMA_VERSION}"
        )
    raw_datasets = payload.get("datasets")
    if not isinstance(raw_datasets, Mapping):
        raise ValueError("Feature mapping config datasets must be an object")
    datasets: dict[str, tuple[FeatureMapRule, ...]] = {}
    for dataset, raw_rules in raw_datasets.items():
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("Dataset names must be non-empty strings")
        if not isinstance(raw_rules, list):
            raise ValueError(f"Dataset {dataset} rules must be a list")
        datasets[dataset] = tuple(_parse_rule(rule) for rule in raw_rules)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return FeatureMapConfig(datasets=datasets, digest=digest, source_path=str(path))


_NAMED_EE_GRIPPERS = frozenset({"ee.l_g", "ee.r_g"})


def dimension_names(names: Any, width: int) -> list[Any]:
    """Return one entry per vector dimension, using the historical name lookup.

    Dict values keep the list whose length matches the vector. A single extra
    list wrapper is unwrapped so names stored as ``[[...]]`` are visible.
    """
    if isinstance(names, Mapping):
        names = next(
            (
                value
                for value in names.values()
                if isinstance(value, list) and len(value) == width
            ),
            [],
        )
    if (
        isinstance(names, list)
        and len(names) == 1
        and isinstance(names[0], list)
        and len(names[0]) == width
    ):
        names = names[0]
    return list(names or [])


def is_gripper_opening_name(name: Any) -> bool:
    """Return whether one documented dimension name is a gripper opening."""
    text = str(name).lower()
    return "gripper" in text or text in _NAMED_EE_GRIPPERS


def schema_gripper_indices(description: Mapping[str, Any], width: int) -> tuple[int, ...]:
    """Recognize grippers from source metadata, including scalar effector openings.

    This is the info.json rule shared by feature extraction and decoding.
    Explicit feature-map rows are applied separately and take priority.
    """
    names = dimension_names(description.get("names", {}), width)
    indexes = {
        index
        for index, name in enumerate(names)
        if is_gripper_opening_name(name) and index < width
    }
    for name, field in description.get("field_descriptions", {}).items():
        opening = "gripper" in name.lower() or (
            "effector/position" in name.lower() and int(field.get("dimensions", 0)) == 1
        )
        if opening:
            indexes.update(
                int(index) for index in field.get("indices", []) if 0 <= int(index) < width
            )
    return tuple(sorted(indexes))


def _indices_for_rule(rule: FeatureMapRule, width: int) -> tuple[int, ...]:
    if rule.expected_width is not None and rule.expected_width != width:
        raise ValueError(
            f"Configured width for {rule.column} is {rule.expected_width}, observed {width}"
        )
    if rule.indices == "all":
        return tuple(range(width))
    if isinstance(rule.indices, tuple):
        indices = rule.indices
    else:
        indices = tuple(
            range(
                int(rule.indices.get("start", 0)),
                int(rule.indices["stop"]),
                int(rule.indices.get("step", 1)),
            )
        )
    if not indices:
        raise ValueError(f"Configured indices for {rule.column} select no dimensions")
    invalid = [index for index in indices if index < 0 or index >= width]
    if invalid:
        raise ValueError(
            f"Configured indices for {rule.column} are out of range for width {width}: {invalid}"
        )
    return indices


def _auto_source(column: str) -> FeatureSource:
    if STATE_RE.search(column):
        return "state"
    if ACTION_RE.search(column):
        return "action"
    return "unknown"


def _auto_family(column: str) -> FeatureFamily:
    if GRIPPER_RE.search(column):
        return "gripper"
    if JOINT_RE.search(column):
        return "joint"
    if EEF_RE.search(column):
        return "eef_pose"
    if BASE_RE.search(column):
        return "base"
    return "other_motion"


def _auto_side(column: str) -> FeatureSide:
    has_left = LEFT_RE.search(column) is not None
    has_right = RIGHT_RE.search(column) is not None
    if has_left and has_right:
        return "both"
    if has_left:
        return "left"
    if has_right:
        return "right"
    return "unknown"


def _auto_quantity(column: str) -> FeatureQuantity:
    lowered = column.lower()
    for quantity in ("velocity", "effort", "torque", "current", "delta", "position"):
        if quantity in lowered:
            return cast(FeatureQuantity, quantity)
    if "pose" in lowered:
        return "position"
    return "unknown"


def resolve_feature_roles(
    column: str,
    width: int,
    dataset: str | None = None,
    config: FeatureMapConfig | None = None,
) -> tuple[FeatureDimensionRole, ...]:
    """Resolve every dimension through explicit rules then automatic fallback.

    Args:
        column: Selected parquet field name.
        width: Numeric vector width after Arrow conversion.
        dataset: Optional manifest dataset identifier.
        config: Optional explicit mapping configuration.

    Returns:
        One role per dimension in ascending index order.

    Raises:
        ValueError: If configured rules overlap or reference invalid dimensions.
    """
    if width <= 0:
        raise ValueError("Feature column width must be positive")
    roles: dict[int, FeatureDimensionRole] = {}
    for rule in config.rules_for(dataset, column) if config is not None else ():
        for index in _indices_for_rule(rule, width):
            if index in roles:
                raise ValueError(
                    f"Feature mapping rules overlap for {dataset}:{column}[{index}]"
                )
            roles[index] = FeatureDimensionRole(
                column=column,
                index=index,
                source=rule.source,
                family=rule.family,
                side=rule.side,
                quantity=rule.quantity,
                origin="config",
                confidence=1.0,
                is_ambiguous=False,
            )
    auto_source = _auto_source(column)
    auto_family = _auto_family(column)
    auto_side = _auto_side(column)
    auto_quantity = _auto_quantity(column)
    is_packed = width > 1 and auto_family == "other_motion"
    for index in range(width):
        if index in roles:
            continue
        roles[index] = FeatureDimensionRole(
            column=column,
            index=index,
            source=auto_source,
            family=auto_family,
            side=auto_side,
            quantity=auto_quantity,
            origin="auto",
            confidence=0.35 if is_packed else 0.8,
            is_ambiguous=is_packed or auto_source == "unknown",
        )
    return tuple(roles[index] for index in range(width))


def mapping_digest(config: FeatureMapConfig | None) -> str:
    """Return the stable cache identity for explicit or automatic mapping."""
    return config.digest if config is not None else AUTO_MAPPING_DIGEST


def role_indices(
    roles: Sequence[FeatureDimensionRole],
    family: FeatureFamily,
    source: FeatureSource | None = None,
) -> tuple[int, ...]:
    """Return indices matching one semantic family and optional source."""
    return tuple(
        role.index
        for role in roles
        if role.family == family and (source is None or role.source == source)
    )

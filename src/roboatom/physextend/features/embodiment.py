"""Dataset-independent robot embodiment profiles for trajectory decoding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping, Sequence


UNKNOWN_ROBOT_TYPE = "unknown"
LEFT_RE = re.compile(r"(?:^|[._/])left(?:[._/]|$)", re.IGNORECASE)
RIGHT_RE = re.compile(r"(?:^|[._/])right(?:[._/]|$)", re.IGNORECASE)
GRIPPER_RE = re.compile(
    r"(?:gripper|finger|end_effector_(?:left|right)_position)", re.IGNORECASE
)
ACTION_RE = re.compile(r"(?:^|[._/])actions?(?:[._/]|$)", re.IGNORECASE)
STATE_RE = re.compile(
    r"(?:^|[._/])(?:observation[._/])?states?(?:[._/]|$)",
    re.IGNORECASE,
)
JOINT_RE = re.compile(r"(?:joint|motor)", re.IGNORECASE)
EEF_RE = re.compile(r"(?:end_effector|eef|tcp|cartesian)", re.IGNORECASE)
BASE_RE = re.compile(r"(?:base|chassis|mobile|wheel)", re.IGNORECASE)


def normalize_robot_type(value: Any) -> str:
    """Return a stable robot type read only from explicit metadata."""
    normalized = re.sub(
        r"[^a-z0-9]+", "_", str(value or "").strip().lower()
    ).strip("_")
    return normalized or UNKNOWN_ROBOT_TYPE


@dataclass(frozen=True)
class EmbodimentProfile:
    """Observable embodiment properties used for hierarchical pooling."""

    robot_type: str = UNKNOWN_ROBOT_TYPE
    arm_count: int = 0
    gripper_count: int = 0
    has_joint_state: bool = False
    has_eef_pose: bool = False
    has_action_state_pair: bool = False
    has_mobile_base: bool = False
    fps: float = 0.0

    def __post_init__(self) -> None:
        if self.arm_count < 0 or self.gripper_count < 0:
            raise ValueError("arm_count and gripper_count must be non-negative")
        if self.fps < 0.0:
            raise ValueError("fps must be non-negative")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None
    ) -> "EmbodimentProfile":
        """Build a normalized profile from manifest or source metadata."""
        metadata = value or {}
        return cls(
            robot_type=normalize_robot_type(
                metadata.get("robot_type")
                or metadata.get("robot_name")
                or metadata.get("embodiment")
            ),
            arm_count=max(0, int(metadata.get("arm_count") or 0)),
            gripper_count=max(0, int(metadata.get("gripper_count") or 0)),
            has_joint_state=bool(metadata.get("has_joint_state", False)),
            has_eef_pose=bool(metadata.get("has_eef_pose", False)),
            has_action_state_pair=bool(
                metadata.get("has_action_state_pair", False)
            ),
            has_mobile_base=bool(metadata.get("has_mobile_base", False)),
            fps=max(0.0, float(metadata.get("fps") or 0.0)),
        )

    @classmethod
    def from_columns(
        cls,
        columns: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
    ) -> "EmbodimentProfile":
        """Enrich explicit metadata with boundary-free schema properties."""
        base = cls.from_mapping(metadata)
        names = tuple(str(column) for column in columns)
        has_left = any(LEFT_RE.search(name) for name in names)
        has_right = any(RIGHT_RE.search(name) for name in names)
        arm_count = 2 if has_left and has_right else int(bool(names))
        gripper_names = [name for name in names if GRIPPER_RE.search(name)]
        gripper_left = any(LEFT_RE.search(name) for name in gripper_names)
        gripper_right = any(RIGHT_RE.search(name) for name in gripper_names)
        gripper_count = (
            2 if gripper_left and gripper_right else int(bool(gripper_names))
        )
        has_gripper_action = any(
            GRIPPER_RE.search(name) and ACTION_RE.search(name) for name in names
        )
        has_gripper_state = any(
            GRIPPER_RE.search(name) and STATE_RE.search(name) for name in names
        )
        return cls(
            robot_type=base.robot_type,
            arm_count=max(base.arm_count, arm_count),
            gripper_count=max(base.gripper_count, gripper_count),
            has_joint_state=base.has_joint_state
            or any(
                JOINT_RE.search(name) and STATE_RE.search(name) for name in names
            ),
            has_eef_pose=base.has_eef_pose
            or any(EEF_RE.search(name) for name in names),
            has_action_state_pair=base.has_action_state_pair
            or (has_gripper_action and has_gripper_state),
            has_mobile_base=base.has_mobile_base
            or any(BASE_RE.search(name) for name in names),
            fps=base.fps,
        )

    @property
    def structure_key(self) -> str:
        """Return a deterministic group key without dataset identity."""
        return ":".join(
            (
                f"arms={self.arm_count or 'unknown'}",
                f"grippers={self.gripper_count}",
                f"joint={int(self.has_joint_state)}",
                f"eef={int(self.has_eef_pose)}",
                f"pair={int(self.has_action_state_pair)}",
                f"base={int(self.has_mobile_base)}",
            )
        )

    @property
    def robot_key(self) -> str | None:
        """Return the robot child group when metadata is available."""
        if self.robot_type == UNKNOWN_ROBOT_TYPE:
            return None
        return f"{self.robot_type}|{self.structure_key}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the profile for manifests and diagnostics."""
        return asdict(self)

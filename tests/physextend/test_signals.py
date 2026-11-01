"""Tests for model-free trajectory signal extraction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from roboatom.physextend.features import signals as trajectory_signals
from roboatom.physextend.features.feature_mapping import load_feature_map_config
from roboatom.physextend.features.signals import (
    extract_features,
    load_feature_bundle,
    save_feature_bundle,
    select_signal_columns,
)


def _write_signals(path: Path, length: int = 100) -> None:
    frame = np.arange(length, dtype=np.float32)
    joint = np.stack([np.sin(frame / 10), np.cos(frame / 13)], axis=1)
    gripper = np.where(frame < 45, 0.0, 1.0).astype(np.float32)
    table = pa.table(
        {
            "observation.state": pa.array(joint.tolist(), type=pa.list_(pa.float32(), 2)),
            "states.gripper.position": gripper,
            "subtask_index": np.where(frame < 50, 0, 1),
            "frame_index": np.arange(length),
            "instruction": ["secret boundary label"] * length,
        }
    )
    pq.write_table(table, path)


def _write_action_state_signals(path: Path, length: int = 80) -> None:
    frame = np.arange(length, dtype=np.float32)
    state = np.where(frame < 35, 0.0, 1.0).astype(np.float32)
    action = np.where(frame < 30, 0.0, 1.0).astype(np.float32)
    table = pa.table(
        {
            "observation.states.left_gripper": state,
            "actions.left_gripper": action,
            "observation.states.left_joint": np.sin(frame / 8),
        }
    )
    pq.write_table(table, path)


def _write_feature_map(path: Path, rules: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps({"schema_version": 1, "datasets": {"demo": rules}}),
        encoding="utf-8",
    )


def test_select_signal_columns_excludes_annotation_and_indices(tmp_path: Path) -> None:
    parquet_path = tmp_path / "episode.parquet"
    _write_signals(parquet_path)
    schema = pq.ParquetFile(parquet_path).schema_arrow

    selected = select_signal_columns(schema)

    assert selected == ["observation.state", "states.gripper.position"]
    assert "subtask_index" not in selected


def test_schema_gripper_fills_packed_vector_and_keeps_motion(tmp_path: Path, monkeypatch) -> None:
    length = 80
    frame = np.arange(length, dtype=np.float32)
    state = np.column_stack(
        [
            np.sin(frame / 9),
            np.where(frame < 30, 0.2, 0.9),
            np.cos(frame / 7),
            np.where(frame < 50, 0.4, 1.0),
        ]
    ).astype(np.float32)
    parquet_path = tmp_path / "data" / "chunk-000" / "episode.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 4))}),
        parquet_path,
    )
    monkeypatch.setattr(
        trajectory_signals,
        "_source_sensor_features",
        lambda path: {"observation.state": {"names": [["joint_0", "ee.l_g", "ee.r_x", "ee.r_g"]]}},
    )

    features = extract_features(str(parquet_path), expected_length=length)

    selected = [
        (role["index"], role["family"])
        for role in features.feature_mapping
        if role["column"] == "observation.state" and role["family"] == "gripper"
    ]
    assert selected == [(1, "gripper"), (3, "gripper")]
    assert features.gripper_level.max() > features.gripper_level.min()
    assert features.activity.max() > features.activity.min()


def test_custom_feature_map_keeps_its_family_when_schema_names_gripper(tmp_path: Path, monkeypatch) -> None:
    length = 40
    frame = np.arange(length, dtype=np.float32)
    opening = np.column_stack([np.where(frame < 15, 0.0, 1.0), np.where(frame < 20, 1.0, 0.0)]).astype(np.float32)
    parquet_path = tmp_path / "episode.parquet"
    pq.write_table(
        pa.table({"actions.effector.position": pa.array(opening.tolist(), type=pa.list_(pa.float32(), 2))}),
        parquet_path,
    )
    config_path = tmp_path / "map.json"
    _write_feature_map(
        config_path,
        [
            {
                "column": "actions.effector.position",
                "indices": "all",
                "source": "action",
                "family": "eef_pose",
                "side": "unknown",
                "quantity": "position",
            }
        ],
    )
    monkeypatch.setattr(
        trajectory_signals,
        "_source_sensor_features",
        lambda path: {"actions.effector.position": {"names": {"motors": ["left_gripper", "right_gripper"]}}},
    )

    features = extract_features(
        str(parquet_path),
        expected_length=length,
        dataset="demo",
        feature_map_config=load_feature_map_config(config_path),
    )

    assert [role["family"] for role in features.feature_mapping] == ["eef_pose", "eef_pose"]
    assert features.gripper_level.max() == features.gripper_level.min()


def test_config_non_gripper_is_not_pulled_back_by_schema(tmp_path: Path, monkeypatch) -> None:
    length = 40
    frame = np.arange(length, dtype=np.float32)
    values = np.column_stack(
        [
            np.sin(frame / 6),
            np.where(frame < 18, 0.0, 1.0),
        ]
    ).astype(np.float32)
    parquet_path = tmp_path / "episode.parquet"
    pq.write_table(
        pa.table({"observation.state": pa.array(values.tolist(), type=pa.list_(pa.float32(), 2))}),
        parquet_path,
    )
    config_path = tmp_path / "map.json"
    _write_feature_map(
        config_path,
        [
            {
                "column": "observation.state",
                "indices": [0],
                "source": "state",
                "family": "joint",
                "side": "unknown",
                "quantity": "effort",
            }
        ],
    )
    monkeypatch.setattr(
        trajectory_signals,
        "_source_sensor_features",
        lambda path: {"observation.state": {"names": ["gripper_effort", "gripper"]}},
    )

    features = extract_features(
        str(parquet_path),
        expected_length=length,
        dataset="demo",
        feature_map_config=load_feature_map_config(config_path),
    )

    families = [(role["index"], role["family"], role["origin"]) for role in features.feature_mapping]
    assert families == [(0, "joint", "config"), (1, "gripper", "auto")]
    grippers = [role["index"] for role in features.feature_mapping if role["family"] == "gripper"]
    assert grippers == [1]
    assert features.gripper_level.max() > features.gripper_level.min()


def test_extract_features_detects_motion_gripper_and_pause(tmp_path: Path) -> None:
    parquet_path = tmp_path / "episode.parquet"
    _write_signals(parquet_path)

    features = extract_features(str(parquet_path), expected_length=100)

    assert features.length == 100
    assert features.activity.max() > features.activity.min()
    assert np.argmax(np.abs(features.gripper_delta)) in range(42, 49)
    assert any(event.kind.startswith("gripper_") for event in features.events)
    assert "subtask_index" not in features.source_columns


def test_cache_round_trip_uses_stable_non_pickle_format(tmp_path: Path) -> None:
    parquet_path = tmp_path / "episode.parquet"
    _write_signals(parquet_path)
    original = extract_features(str(parquet_path), expected_length=100)
    output_path = tmp_path / "cache" / "features.npz"

    save_feature_bundle(output_path, original)
    restored = load_feature_bundle(output_path)

    np.testing.assert_allclose(restored.activity, original.activity)
    np.testing.assert_allclose(restored.gripper_delta, original.gripper_delta)
    assert restored.events == original.events
    assert restored.source_columns == original.source_columns
    assert restored.feature_mapping == original.feature_mapping
    assert restored.feature_mapping_digest == original.feature_mapping_digest


def test_action_state_gripper_features_and_embodiment_round_trip(tmp_path: Path) -> None:
    parquet_path = tmp_path / "paired.parquet"
    _write_action_state_signals(parquet_path)

    original = extract_features(
        str(parquet_path),
        expected_length=80,
        embodiment={"robot_type": "Test Bot", "fps": 30},
    )
    output_path = tmp_path / "cache" / "features.npz"
    save_feature_bundle(output_path, original)
    restored = load_feature_bundle(output_path)

    assert restored.state_gripper_level is not None
    assert restored.action_gripper_level is not None
    assert restored.action_state_error is not None
    assert float(restored.action_state_error[32]) > 0.5
    assert restored.embodiment["robot_type"] == "test_bot"
    assert restored.embodiment["has_action_state_pair"] is True
    assert restored.embodiment["has_joint_state"] is True


def test_extract_features_rejects_leakage_only_parquet(tmp_path: Path) -> None:
    parquet_path = tmp_path / "leaky.parquet"
    pq.write_table(pa.table({"subtask_index": [0, 0, 1], "frame_index": [0, 1, 2]}), parquet_path)

    with pytest.raises(ValueError, match="No safe numeric robot signals"):
        extract_features(str(parquet_path), expected_length=3)


def test_explicit_mapping_splits_packed_gripper_from_motion(tmp_path: Path) -> None:
    parquet_path = tmp_path / "packed.parquet"
    frame = np.arange(60, dtype=np.float32)
    action = np.stack(
        [np.sin(frame / 8), np.where(frame < 25, 0.0, 1.0), np.cos(frame / 7)],
        axis=1,
    )
    pq.write_table(
        pa.table({"action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 3))}),
        parquet_path,
    )
    config_path = tmp_path / "feature-map.json"
    _write_feature_map(
        config_path,
        [
            {
                "column": "action",
                "expected_width": 3,
                "indices": [1],
                "source": "action",
                "family": "gripper",
                "side": "unknown",
                "quantity": "position",
            }
        ],
    )
    config = load_feature_map_config(config_path)

    features = extract_features(
        str(parquet_path),
        expected_length=60,
        dataset="demo",
        feature_map_config=config,
    )

    assert features.action_gripper_level is not None
    assert np.argmax(np.abs(features.gripper_delta)) in range(23, 28)
    assert features.activity.max() > features.activity.min()
    mapped = [role for role in features.feature_mapping if role["origin"] == "config"]
    automatic = [role for role in features.feature_mapping if role["origin"] == "auto"]
    assert [role["index"] for role in mapped] == [1]
    assert [role["index"] for role in automatic] == [0, 2]
    assert all(role["is_ambiguous"] for role in automatic)
    assert features.feature_mapping_digest == config.digest


def test_explicit_mapping_rejects_overlap_and_out_of_range(tmp_path: Path) -> None:
    parquet_path = tmp_path / "packed.parquet"
    pq.write_table(
        pa.table(
            {
                "action": pa.array(
                    [[0.0, 1.0], [1.0, 0.0]], type=pa.list_(pa.float32(), 2)
                )
            }
        ),
        parquet_path,
    )
    base = {
        "column": "action",
        "source": "action",
        "family": "gripper",
        "side": "unknown",
        "quantity": "position",
    }
    overlap_path = tmp_path / "overlap.json"
    _write_feature_map(
        overlap_path,
        [{**base, "indices": [0]}, {**base, "indices": [0, 1]}],
    )
    with pytest.raises(ValueError, match="overlap"):
        extract_features(
            str(parquet_path),
            expected_length=2,
            dataset="demo",
            feature_map_config=load_feature_map_config(overlap_path),
        )

    range_path = tmp_path / "range.json"
    _write_feature_map(range_path, [{**base, "indices": [2]}])
    with pytest.raises(ValueError, match="out of range"):
        extract_features(
            str(parquet_path),
            expected_length=2,
            dataset="demo",
            feature_map_config=load_feature_map_config(range_path),
        )


def test_feature_mapping_config_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.json"
    _write_feature_map(
        config_path,
        [
            {
                "column": "action",
                "indices": "all",
                "source": "action",
                "family": "gripper",
                "side": "unknown",
                "quantity": "position",
                "familly": "typo",
            }
        ],
    )

    with pytest.raises(ValueError, match="Unknown feature mapping rule keys"):
        load_feature_map_config(config_path)


def test_cache_rejects_wrong_feature_mapping_digest(tmp_path: Path) -> None:
    parquet_path = tmp_path / "episode.parquet"
    _write_signals(parquet_path)
    bundle = extract_features(str(parquet_path), expected_length=100)
    output_path = tmp_path / "cache" / "features.npz"
    save_feature_bundle(output_path, bundle)

    with pytest.raises(ValueError, match="uses mapping"):
        load_feature_bundle(output_path, expected_feature_mapping_digest="different")


def test_open_parquet_reads_rclone_remote_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parquet_path = tmp_path / "remote.parquet"
    _write_signals(parquet_path, length=12)
    payload = parquet_path.read_bytes()
    monkeypatch.setattr(trajectory_signals, "read_rclone_bytes", lambda path: payload)

    parquet_file = trajectory_signals._open_parquet("poss:bucket/remote.parquet")

    assert parquet_file.metadata.num_rows == 12

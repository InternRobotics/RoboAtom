"""Run isolation, schema caching and workspace ownership checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboatom.physextend.decoders import seeded_decoder
from roboatom.physextend.extend import _prepare_workdir
from roboatom.physextend.features.extraction import feature_paths
from roboatom.physextend.refine.methods import router_profiles
from roboatom.physextend.runtime import current_run, run_cache, run_context


def test_context_restored_after_nested_failure(tmp_path: Path) -> None:
    with run_context(tmp_path / "outer") as outer:
        outer.episodes["episode"] = {"length": 20}
        with pytest.raises(ValueError, match="failed"):
            with run_context(tmp_path / "inner") as inner:
                assert inner.episodes == {}
                raise ValueError("failed")
        assert current_run() is outer
        assert outer.episodes["episode"]["length"] == 20

    with pytest.raises(RuntimeError, match="No active"):
        current_run()


def test_memoization_does_not_cross_invocations(tmp_path: Path) -> None:
    calls: list[str] = []

    @run_cache(maxsize=2)
    def read_length(uri: str) -> int:
        calls.append(uri)
        return current_run().episodes[uri]["length"]

    for length in (20, 40):
        with run_context(tmp_path) as context:
            context.episodes["same-uri"] = {"length": length}
            assert read_length("same-uri") == length
            assert read_length("same-uri") == length

    assert calls == ["same-uri", "same-uri"]


def test_reused_workspace_reloads_router_metadata(tmp_path: Path) -> None:
    audit = tmp_path / "audit"
    audit.mkdir()
    for robot in ("franka", "ur"):
        (audit / "embodiments.json").write_text(
            json.dumps({"same-uri": {"robot_type": robot}}), encoding="utf-8"
        )
        with run_context(tmp_path):
            assert router_profiles()["same-uri"]["robot_type"] == robot


def test_schema_cache_is_local_and_refreshed_between_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    schema = {"observation.state": {"names": ["left_gripper"]}}

    def read_schema(uri: str) -> bytes:
        calls.append(uri)
        return json.dumps({"features": schema}).encode()

    monkeypatch.setattr(seeded_decoder, "_read_bytes", read_schema)
    uri = "remote:dataset/data/chunk-000/episode.parquet"
    for name in ("first", "second"):
        workspace = tmp_path / name
        with run_context(workspace):
            assert seeded_decoder.sensor_schema(uri) == schema
            assert seeded_decoder.sensor_schema(uri) == schema
        files = list((workspace / "sensor_schema").glob("*.json"))
        assert len(files) == 1
        assert json.loads(files[0].read_text())["features"] == schema
        schema = {"observation.state": {"names": ["right_gripper"]}}

    assert calls == ["remote:dataset/meta/info.json"] * 2
    assert not (tmp_path / "experiments").exists()


def test_feature_paths_follow_active_workspace(tmp_path: Path) -> None:
    episode = {"data_uri": "remote:dataset/episode.parquet", "length": 30}
    for name in ("first", "second"):
        workspace = tmp_path / name
        with run_context(workspace):
            assert all(path.parent == workspace / "features" for path in feature_paths(episode))


@pytest.mark.parametrize("marker", [None, "foreign-owner\n"])
def test_unrelated_workspace_is_preserved(tmp_path: Path, marker: str | None) -> None:
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    if marker is not None:
        (tmp_path / ".roboatom_physextend_workspace").write_text(marker, encoding="utf-8")

    with pytest.raises(ValueError, match="unrelated directory"):
        _prepare_workdir(tmp_path, [])

    assert sentinel.read_text() == "keep"


def test_owned_workspace_rebuilds_caches(tmp_path: Path) -> None:
    workspace = tmp_path / "run"
    _prepare_workdir(workspace, [])
    stale = workspace / "features" / "stale.npz"
    stale.write_bytes(b"stale")

    _prepare_workdir(workspace, [])

    assert not stale.exists()
    assert len(list((workspace / "recipes").glob("*.json"))) == 28

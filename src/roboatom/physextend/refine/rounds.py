"""Sequential extend trials with immutable inputs and embodiment-aware review."""

from __future__ import annotations

from roboatom.physextend.runtime import current_run

from collections import defaultdict
import copy
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np


from roboatom.physextend.refine import sequential as seq
from roboatom.physextend.refine import evidence as ev
from roboatom.physextend.decoders.gripper_cycle import normalize_channel

LOG = logging.getLogger(__name__)


def rows(path: Path) -> list[dict[str, Any]]:
    return seq.read_rows(path)


def save(path: Path, value: Any) -> None:
    seq.save(path, value)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize() -> list[dict[str, Any]]:
    """Load validated inputs and register only this run's episode metadata."""
    items = [seq.input_row(row) for row in rows(current_run().workdir / 'snapshot/inference.jsonl')]
    current_run().episodes.update({
        episode['data_uri']: episode
        for row in items for episode in [row['target'], *row['references']]
    })
    return items


def robot_name(value: str) -> str:
    value = value.lower().strip()
    if value in {"", "unknown", "unresolved"}:
        return "unknown"
    return {"arx_lift2": "lift2"}.get(value, value)


def robot_from_source_path(uri: str) -> str:
    """Read explicit robot collection tokens; never use task descriptions."""
    import re

    root = uri.lower().split("/data/")[0]
    patterns = [
        ("franka_fr3_dual", r"/franka_fr3_dual[^/]*/"),
        ("agilex_mobile", r"/agilex_mobile/"),
        ("franka", r"/franka(?:_[13]rgb)?/"),
        ("agilex", r"/agilex(?:_[13]rgb)?/"),
        ("ur", r"/ur(?:[35])?(?:_[13]rgb)?/"),
        ("tienkung", r"/tienkung[^/]*/"),
        ("ark", r"/ark/"),
    ]
    for model, pattern in patterns:
        if re.search(pattern, root + "/"):
            return model
    return "unknown"


def build_profiles() -> None:
    """Extract only source metadata; never copy target action annotations."""
    items = initialize()
    # Inference uses only explicit embodiment metadata, never target labels.
    metadata_file = current_run().workdir / "snapshot/embodiment_metadata.jsonl"
    if not metadata_file.is_file():
        raise FileNotFoundError(metadata_file)
    truth = {r["benchmark_id"]: r for r in rows(metadata_file)}
    root_models: dict[str, set[str]] = defaultdict(set)
    for value in truth.values():
        model = robot_name(value["review_provenance"]["record"].get("embodiment", ""))
        if model != "unknown":
            root_models[value["data_uri"].split("/data/")[0]].add(model)
    inventory = {e["data_uri"]: e for r in items for e in [r["target"], *r["references"]]}
    result = {}
    for index, (uri, episode) in enumerate(inventory.items()):
        record = truth.get(episode["benchmark_id"], {}).get("review_provenance", {}).get("record", {})
        model = robot_name(record.get("embodiment", ""))
        evidence = record.get("embodiment_evidence", "not_available")
        source = uri.split("/data/")[0]
        if model == "unknown" and len(root_models[source]) == 1:
            model = next(iter(root_models[source]))
            evidence = "same_exact_source_root_as_audited_record" if model != "unknown" else "not_available"
        if model == "unknown":
            model = robot_from_source_path(uri)
            evidence = "explicit_robot_collection_token_in_source_uri" if model != "unknown" else evidence
        feature, raw, metadata = seq.paths(episode)
        columns, channel_names, grippers = [], [], {}
        schema_uri = uri.split("/data/chunk-")[0] + "/meta/info.json"
        schema_path = current_run().workdir / "sensor_schema" / (seq.stable_hash(schema_uri) + ".json")
        schema = json.loads(schema_path.read_text()).get("features", {}) if schema_path.exists() else {}
        if metadata.exists():
            columns = json.loads(metadata.read_text())["columns"]
            channel_names = [n for c in columns for n in ev.flatten_names(schema.get(c, {}).get("names", []))]
            grippers = ev.schema_grippers(uri, episode["dataset"], episode["length"], True)
        names = columns + channel_names
        sides = {ev.side_of(name) for name in names} - {"unspecified"}
        arm_count = 2 if sides == {"left", "right"} else 0
        active = {
            name: float(np.percentile(value, 95) - np.percentile(value, 5))
            for name, value in grippers.items()
            if normalize_channel(value) is not None
        }
        result[uri] = {
            "benchmark_id": episode["benchmark_id"],
            "dataset": episode["dataset"],
            "robot_type": model,
            "robot_evidence": evidence,
            "source_root": source,
            "source_record_sha256": record.get("source_sha256"),
            "schema_sha256": digest(schema_path) if schema_path.exists() else None,
            "arm_count": 0,
            "arm_count_evidence": "hardware_count_not_verified",
            "observed_arm_slots": arm_count,
            "arm_slots_evidence": "explicit_left_right_names" if arm_count else "unknown",
            "gripper_sides": sorted({name.rsplit("/", 1)[-1] for name in grippers}),
            "gripper_type": "unknown",
            "has_mobile_base_signal": any(
                any(t in c.lower() for t in ["base", "chassis", "robot.velocity"]) for c in columns
            ),
            "fps": episode["fps"],
            "columns": columns,
            "has_command": any("action" in c.lower() or "command" in c.lower() for c in columns),
            "has_state": any("state" in c.lower() for c in columns),
            "units": {c: schema[c].get("unit", schema[c].get("units", "unknown")) for c in columns if c in schema},
            "opening_channels": list(grippers),
            "active_opening_channels": active,
            "features_available": metadata.exists(),
        }
        if index % 250 == 0:
            LOG.info("Profiles %d/%d", index, len(inventory))
    save(current_run().workdir / "audit/embodiments.json", result)
    summary = defaultdict(lambda: {"episodes": 0, "tasks": set(), "active_gripper": 0, "no_reference": 0})
    for row in items:
        target = row["target"]
        profile = result[target["data_uri"]]
        key = target["dataset"] + "|" + profile["robot_type"]
        group = summary[key]
        group["episodes"] += 1
        group["tasks"].add(target["task_id"])
        group["active_gripper"] += bool(profile["active_opening_channels"])
        group["no_reference"] += not row["references"]
    save(current_run().workdir / "audit/profile_summary.json", {k: dict(v, tasks=len(v["tasks"])) for k, v in summary.items()})


def set_boundaries(parent: dict[str, Any], boundaries: np.ndarray) -> dict[str, Any]:
    output = copy.deepcopy(parent)
    points = [0, *[int(x) for x in boundaries], int(parent["frames"])]
    if any(b <= a for a, b in zip(points, points[1:])):
        raise ValueError("Non-positive proposed partition")
    for segment, start, end in zip(output["segments"], points, points[1:]):
        segment.update(start=start, end=end, start_frame=start, end_frame=end)
    return output


def canonical_frame_fields(prediction: dict[str, Any]) -> dict[str, Any]:
    """Keep legacy frame aliases consistent with authoritative frame fields."""
    output = copy.deepcopy(prediction)
    for segment in output["segments"]:
        if "start" in segment:
            segment["start"] = segment["start_frame"]
        if "end" in segment:
            segment["end"] = segment["end_frame"]
    return output


def validate_prediction(output: dict[str, Any], target: dict[str, Any]) -> None:
    points = [0]
    for segment in output["segments"]:
        if segment["start_frame"] != points[-1] or segment["end_frame"] <= points[-1]:
            raise ValueError("Non-contiguous or non-positive partition")
        points.append(segment["end_frame"])
    if points[-1] != target["length"]:
        raise ValueError("Prediction does not cover the whole target")


def apply_method(
    row: dict[str, Any], parent: dict[str, Any], pairs: list[Any], config: dict[str, Any], proposal: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Dispatch only methods introduced after their predecessor was reviewed."""
    from roboatom.physextend.refine.methods import apply

    return apply(row, parent, pairs, config, proposal)

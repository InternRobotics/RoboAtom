"""Prepare and run one extend-only baseline; subsequent rounds require review."""

from __future__ import annotations

from roboatom.physextend.runtime import DATA, current_run

import hashlib
import json
import logging
from pathlib import Path
from typing import Any


from roboatom.physextend.features import extraction as feature_runner
from roboatom.physextend.decoders import candidates as candidate_runner
from roboatom.physextend.decoders import seeded_decoder as seeded_decoder
from roboatom.physextend.features.records import stable_hash, physical_id

LOG = logging.getLogger(__name__)


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read frozen records."""
    return [json.loads(line) for line in path.open() if line.strip()]


def save(path: Path, value: Any) -> None:
    """Persist a structured checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def paths(episode: dict[str, Any]) -> tuple[Path, Path, Path]:
    """Return the shared feature paths for this invocation."""
    return feature_runner.feature_paths(episode)


def adapt(episode: dict[str, Any]) -> dict[str, Any]:
    """Attach decoder metadata without introducing query annotations."""
    return dict(
        episode,
        canonical_task_id=episode["task_id"],
        record_id=episode["benchmark_id"],
        signals_path=episode["data_uri"],
    )


def input_row(item: dict[str, Any]) -> dict[str, Any]:
    """Validate extend origin and target/reference disjointness."""
    target = adapt(item["target"])
    references = [adapt(seed) for seed in item["seeds"]]
    if any(
        key in target
        for key in ["segments", "action_config", "boundaries", "annotations"]
    ):
        raise ValueError("Target GT leaked into inference input")
    if target["source_component"] != "extend":
        raise ValueError("Non-extend target rejected")
    for seed in references:
        if seed["source_component"] != "extend" or seed["task_id"] != target["task_id"]:
            raise ValueError("Illegal reference origin/task")
        if physical_id(seed) == physical_id(target):
            raise ValueError("Self reference rejected")
    signatures = {
        tuple(seg["skill"] for seg in seed["segments"]) for seed in references
    }
    return {
        "target": target,
        "references": references,
        "seeds": references,
        "record_id": target["benchmark_id"],
        "episode_index": target["benchmark_id"],
        "canonical_task_id": target["task_id"],
        "protocol": "cross_episode_seeded",
        "split": "extend_exploratory",
        "support_mode": "mixed" if len(signatures) > 1 else "same",
    }


def pair_path(row: dict[str, Any], seed: dict[str, Any]) -> Path:
    """Bind candidates to exact permitted inputs and base configuration."""
    identity = stable_hash(
        [
            row["target"],
            seed,
            hashlib.sha256(
                (
                    DATA / "configs/base.json"
                ).read_bytes()
            ).hexdigest(),
            "1c91ba6a01268810554384135e31e2afb280cbd2",
        ]
    )
    return current_run().workdir / "candidates" / (identity + ".json")


def prepare_pair(payload: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    """Generate one reference prediction without access to query GT."""
    row, seed = payload
    path = pair_path(row, seed)
    if path.exists():
        return {"path": str(path), "status": json.loads(path.read_text())["status"]}
    config = json.loads(
        (DATA / "configs/base.json").read_text()
    )
    allowed = dict(
        row,
        seed=seed,
        transcript=[
            {"skill": segment["skill"], "action_text": segment.get("action_text", "")}
            for segment in seed["segments"]
        ],
    )
    try:
        prediction, diagnostic = feature_runner.predict_row(allowed, config)
        bundle = feature_runner.load_features(str(paths(row["target"])[0]))
        candidate = {
            "status": "ok",
            "prediction": prediction,
            "diagnostic": diagnostic,
            "seed_record_id": seed["record_id"],
            "seed_physical_id": seed["episode_key"],
            "seed_signature": [segment["skill"] for segment in seed["segments"]],
            "target_events": [
                {
                    "frame": event.frame,
                    "channel": event.channel,
                    "strength": event.strength,
                }
                for event in bundle.events
            ],
            **candidate_runner.segment_evidence(row, seed, prediction),
        }
    except (
        ValueError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        IndexError,
    ) as error:
        candidate = {
            "status": "failed",
            "seed_record_id": seed["record_id"],
            "error": f"{type(error).__name__}: {error}",
        }
    save(path, candidate)
    return {"path": str(path), "status": candidate["status"]}


def prepare() -> None:
    """Warm only selected extend signals and independent reference candidates."""
    rows = [input_row(item) for item in read_rows(current_run().workdir / "snapshot/inference.jsonl")]
    episodes = {
        episode["data_uri"]: episode
        for row in rows
        if row["references"]
        for episode in [row["target"], *row["references"]]
    }
    status = []
    for index, result in enumerate(
        map(feature_runner.extract_one, episodes.values()), 1
    ):
        status.append(result)
        if index % 50 == 0:
            LOG.info("Features %d/%d", index, len(episodes))
    save(current_run().workdir / "feature_status.json", status)
    pairs = [(row, seed) for row in rows for seed in row["references"]]
    completed = []
    for index, result in enumerate(map(prepare_pair, pairs), 1):
        completed.append(result)
        if index % 25 == 0:
            LOG.info(
                "Candidates %d/%d failed=%d",
                index,
                len(pairs),
                sum(item["status"] == "failed" for item in completed),
            )
    save(current_run().workdir / "candidate_status.json", completed)

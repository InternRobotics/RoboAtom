"""Portable entry point for physical-signal-guided annotation extension."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
from typing import Any

from .runtime import DATA, current_run, run_context
from .selection import _validate_reference_coverage, select_references
from ..validation.data import canonical_row, read_rows, validate_rows, write_rows


FINAL_RECIPE = "agilex_short_segment_guard"
RECOMPUTE_METHODS = {"weak_channel_filter"}


def _source_row(row: dict[str, Any], data_root: Path, is_reference: bool) -> dict[str, Any]:
    """Convert a public manifest row to the original PhysExtend inference schema.

    Args:
        row: Canonical public row with closed-inclusive segments.
        data_root: Directory for relative sensor paths.
        is_reference: Whether to include the reference transcript.

    Returns:
        A PhysExtend episode descriptor with absolute sensor path.
    """
    source = str(row["signals_path"])
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", source):
        uri = source
    else:
        path = Path(source).expanduser()
        path = path if path.is_absolute() else data_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        uri = str(path)
    result = {
        "benchmark_id": row["episode_id"],
        "episode_key": row["episode_id"],
        "task_id": row["task_id"],
        "dataset": row.get("dataset", "custom"),
        "data_uri": uri,
        "physical_id": uri,
        "length": row["num_frames"],
        "fps": float(row["fps"]),
        "source_component": "extend",
        "frame_interval": "half_open",
        "embodiment": row.get("embodiment", "unknown"),
    }
    if is_reference:
        result["segments"] = [
            {
                "start": segment["start_frame"],
                "end": segment["end_frame"] + 1,
                "skill": str(segment["skill"]).strip().lower().replace("-", "_").replace(" ", "_"),
                "action_text": str(segment.get("subtask", "")),
            }
            for segment in row["segments"]
        ]
    return result


def _prepare_workdir(workdir: Path, rows: list[dict[str, Any]]) -> None:
    """Create an isolated PhysExtend workspace with immutable bundled recipes."""
    if workdir.exists():
        marker = workdir / ".roboatom_physextend_workspace"
        if not marker.is_file() or marker.read_text(encoding="utf-8") != "roboatom-physextend\n":
            raise ValueError(f"Refusing to replace an unrelated directory: {workdir}")
        shutil.rmtree(workdir)
    for name in ("snapshot", "features", "candidates", "audit"):
        (workdir / name).mkdir(parents=True, exist_ok=True)
    shutil.copytree(DATA / "recipes", workdir / "recipes")
    (workdir / ".roboatom_physextend_workspace").write_text("roboatom-physextend\n", encoding="utf-8")
    write_rows(workdir / "snapshot" / "inference.jsonl", rows)
    metadata = [
        {
            "benchmark_id": episode["benchmark_id"],
            "data_uri": episode["data_uri"],
            "review_provenance": {"record": {"embodiment": episode["embodiment"]}},
        }
        for item in rows
        for episode in [item["target"], *item["seeds"]]
    ]
    write_rows(workdir / "snapshot" / "embodiment_metadata.jsonl", metadata)


def _infer_one(
    full_row: dict[str, Any], recipes: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate PhysExtend's exact recursive recipe graph for one target."""
    from .refine import rounds, sequential

    count = min(4, len(full_row["references"]))
    references = full_row["references"][:count]
    row = dict(full_row, references=references, seeds=references)
    pairs = [
        (seed, json.loads(sequential.pair_path(full_row, seed).read_text()))
        for seed in references
    ]
    config_path = DATA / "configs" / ("single.json" if count == 1 else "multi.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cache: dict[str, dict[str, Any]] = {}
    step_diagnostics: dict[str, dict[str, Any]] = {}
    active: set[str] = set()
    context = current_run()
    original_provider = context.prediction_provider

    def compute(name: str) -> dict[str, Any]:
        if name in cache:
            return cache[name]
        if name in active:
            raise ValueError(f"Cyclic PhysExtend recipe: {name}")
        active.add(name)
        proposal = recipes[name]
        parent = {} if proposal["method"] in RECOMPUTE_METHODS else compute(proposal["parent"])
        output, diagnostic = rounds.apply_method(row, parent, pairs, config, proposal)
        output = rounds.canonical_frame_fields(output)
        rounds.validate_prediction(output, row["target"])
        cache[name] = output
        step_diagnostics[name] = diagnostic
        active.remove(name)
        return output

    def provider(name: str, requested_count: int) -> dict[str, dict[str, Any]]:
        if requested_count != count:
            raise ValueError("PhysExtend recipe requested a different reference count")
        return {row["record_id"]: compute(name)}

    try:
        context.prediction_provider = provider
        prediction = compute(FINAL_RECIPE)
    finally:
        context.prediction_provider = original_provider
    return prediction, {
        "recipe": FINAL_RECIPE,
        "references": [seed["benchmark_id"] for seed in references],
        "evaluated_steps": list(cache),
        "step_diagnostics": step_diagnostics,
    }


def run(args: Any) -> None:
    """Extend public JSONL manifests using the bundled final recipe."""
    references = read_rows(args.references)
    targets = read_rows(args.input)
    validate_rows(references, require_segments=True)
    validate_rows(targets, require_segments=False)
    canonical_references = [canonical_row(row) for row in references]
    canonical_targets = [canonical_row(row) for row in targets]
    for reference in canonical_references:
        _validate_reference_coverage(reference)
    if {row["episode_id"] for row in canonical_references} & {row["episode_id"] for row in canonical_targets}:
        raise ValueError("Reference and target episode IDs must be disjoint")
    data_root = args.data_root.resolve()
    items = []
    for target in canonical_targets:
        selected = select_references(target, canonical_references)
        if args.route == "single":
            selected = selected[:1]
        items.append({
            "target": _source_row(target, data_root, False),
            "seeds": [_source_row(reference, data_root, True) for reference in selected],
        })
    output = args.output.resolve()
    workdir = output.parent / (output.stem + "_physextend_work")
    _prepare_workdir(workdir, items)
    with run_context(workdir):
        from .refine import rounds, sequential
        sequential.prepare()
        rounds.build_profiles()
        recipes = {
            path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in (workdir / "recipes").glob("*.json")
        }
        prepared = rounds.initialize()
        predictions, diagnostics = [], []
        for row in prepared:
            prediction, diagnostic = _infer_one(row, recipes)
            target = row["target"]
            closed = {
                "episode_id": target["benchmark_id"],
                "task_id": target["task_id"],
                "fps": target["fps"],
                "num_frames": target["length"],
                "frame_interval": "closed",
                "segments": [
                    {
                        "start_frame": segment["start_frame"],
                        "end_frame": segment["end_frame"] - 1,
                        "skill": segment["skill"],
                        "subtask": segment.get("action_text", ""),
                    }
                    for segment in prediction["segments"]
                ],
            }
            validate_rows([closed], require_segments=True)
            predictions.append(closed)
            diagnostics.append({"episode_id": target["benchmark_id"], **diagnostic})
        write_rows(output, predictions)
        write_rows(output.with_suffix(".diagnostics.jsonl"), diagnostics)

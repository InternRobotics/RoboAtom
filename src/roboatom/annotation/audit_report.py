#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roboatom.annotation.app import MultiUserManifest, audit_sidecar_path, status_sidecar_path  # noqa: E402


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                events.append(
                    {
                        "event_type": "invalid_audit_line",
                        "status": "failure",
                        "error": f"invalid JSON at line {line_no}",
                    }
                )
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def load_status_counts(path: Path) -> Dict[str, int]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"invalid_status_file": 1}
    items = raw.get("items") if isinstance(raw, dict) else []
    if not isinstance(items, list):
        return {}
    counts: Counter[str] = Counter()
    for item in items:
        if isinstance(item, dict):
            counts[str(item.get("status") or "unknown")] += 1
    return dict(sorted(counts.items()))


def summarize_events(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    annotators: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "events": 0,
            "update_success": 0,
            "save_success": 0,
            "failure": 0,
            "modified_items": set(),
            "field_changes": Counter(),
            "boundary_adjustments": 0,
            "text_changes": 0,
            "skill_changes": 0,
            "active_ms_values": [],
            "reviewed_items": set(),
        }
    )
    event_counts: Counter[str] = Counter()
    hash_chain_breaks = 0
    previous_hash = ""

    for event in events:
        event_type = str(event.get("event_type") or "unknown")
        status = str(event.get("status") or "")
        annotator_id = str(event.get("annotator_id") or "unknown")
        role = str(event.get("role") or "")
        event_counts[event_type] += 1

        current_hash = str(event.get("event_hash") or "")
        expected_prev = str(event.get("prev_event_hash") or "")
        if previous_hash and expected_prev and expected_prev != previous_hash:
            hash_chain_breaks += 1
        if current_hash:
            previous_hash = current_hash

        row = annotators[annotator_id]
        row["events"] += 1
        if status == "failure":
            row["failure"] += 1

        local_index = event.get("local_index", event.get("index"))
        summary = event.get("change_summary") if isinstance(event.get("change_summary"), dict) else {}
        dirty_indices = summary.get("dirty_indices") if isinstance(summary.get("dirty_indices"), list) else []
        target_indices = []
        if isinstance(local_index, int):
            target_indices.append(local_index)
        for value in dirty_indices:
            try:
                target_indices.append(int(value))
            except (TypeError, ValueError):
                continue

        if event_type in {"annotation_update", "review_update"} and status == "success":
            row["update_success"] += 1
            for index in target_indices:
                row["modified_items"].add(index)
            fields_changed = summary.get("fields_changed") if isinstance(summary.get("fields_changed"), dict) else {}
            for field_name, count in fields_changed.items():
                try:
                    safe_count = int(count)
                except (TypeError, ValueError):
                    safe_count = 0
                row["field_changes"][str(field_name)] += safe_count
            row["boundary_adjustments"] += int(summary.get("boundary_changes") or 0)
            row["text_changes"] += int(summary.get("text_changes") or 0)
            row["skill_changes"] += int(summary.get("skill_changes") or 0)

        if event_type in {"annotation_save", "review_save"} and status == "success":
            row["save_success"] += 1
            if event_type == "review_save" or role == "reviewer":
                for index in target_indices:
                    row["reviewed_items"].add(index)

        if event_type == "item_close":
            try:
                row["active_ms_values"].append(max(0, int(event.get("active_ms") or summary.get("active_ms") or 0)))
            except (TypeError, ValueError):
                pass

    result: Dict[str, Any] = {}
    for annotator_id, row in annotators.items():
        active_values = row.pop("active_ms_values")
        modified_items = row.pop("modified_items")
        reviewed_items = row.pop("reviewed_items")
        field_changes = row.pop("field_changes")
        active_avg = (sum(active_values) / len(active_values)) if active_values else 0
        result[annotator_id] = {
            **row,
            "modified_items": len(modified_items),
            "reviewed_items": len(reviewed_items),
            "field_changes": dict(sorted(field_changes.items())),
            "average_active_ms": round(active_avg, 2),
        }

    return {
        "event_counts": dict(sorted(event_counts.items())),
        "hash_chain_breaks": hash_chain_breaks,
        "annotators": dict(sorted(result.items())),
    }


def summarize_output(output_path: Path) -> Dict[str, Any]:
    output = output_path.expanduser().resolve()
    audit_path = audit_sidecar_path(output)
    status_path = status_sidecar_path(output)
    events = load_jsonl(audit_path)
    summary = summarize_events(events)
    return {
        "output": str(output),
        "audit_path": str(audit_path),
        "status_path": str(status_path),
        "events": len(events),
        "status_counts": load_status_counts(status_path),
        **summary,
    }


def merge_totals(outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    event_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    annotators: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "events": 0,
            "update_success": 0,
            "save_success": 0,
            "failure": 0,
            "modified_items": 0,
            "reviewed_items": 0,
            "boundary_adjustments": 0,
            "text_changes": 0,
            "skill_changes": 0,
            "field_changes": Counter(),
        }
    )
    for output in outputs:
        event_counts.update(output.get("event_counts", {}))
        status_counts.update(output.get("status_counts", {}))
        for annotator_id, row in output.get("annotators", {}).items():
            target = annotators[annotator_id]
            for key in (
                "events",
                "update_success",
                "save_success",
                "failure",
                "modified_items",
                "reviewed_items",
                "boundary_adjustments",
                "text_changes",
                "skill_changes",
            ):
                target[key] += int(row.get(key) or 0)
            target["field_changes"].update(row.get("field_changes", {}))

    return {
        "event_counts": dict(sorted(event_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "annotators": {
            annotator_id: {
                **{key: value for key, value in row.items() if key != "field_changes"},
                "field_changes": dict(sorted(row["field_changes"].items())),
            }
            for annotator_id, row in sorted(annotators.items())
        },
    }


def output_paths_from_manifest(manifest_path: Path) -> List[Path]:
    manifest = MultiUserManifest.load(manifest_path.expanduser().resolve())
    return [annotator.output_path for annotator in manifest.annotators]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SubtaskAnnotator audit/status sidecar files.")
    parser.add_argument("--output", action="append", default=[], help="Annotation output JSON/JSONL path")
    parser.add_argument("--manifest", help="Multi-user manifest path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = [Path(path) for path in args.output]
    if args.manifest:
        output_paths.extend(output_paths_from_manifest(Path(args.manifest)))
    if not output_paths:
        raise SystemExit("Provide --output PATH or --manifest PATH")

    outputs = [summarize_output(path) for path in output_paths]
    payload = {
        "outputs": outputs,
        "totals": merge_totals(outputs),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

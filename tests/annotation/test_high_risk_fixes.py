import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from roboatom.annotation.app import (
    AnnotationStore,
    MultiUserManifest,
    MultiUserStoreManager,
    RuntimeConfig,
    ValidationError,
    create_app,
    merge_multi_user_outputs,
    prepare_multi_user_seed_files,
    resolve_output_path,
)


ROOT = Path(__file__).resolve().parents[2] / "src/roboatom/annotation"


def make_item(boundary=10, task_text="Sort items."):
    return {
        "path": "/tmp/nonexistent-video.mp4",
        "frames": 20,
        "fps": 10,
        "tasks": [task_text],
        "action_config": [
            {
                "seg_id": 0,
                "start_frame": 0,
                "end_frame": boundary,
                "action_text": "Picks up the item.",
                "skill": "Pick",
            },
            {
                "seg_id": 1,
                "start_frame": boundary,
                "end_frame": 20,
                "action_text": "Places the item.",
                "skill": "Place",
            },
        ],
    }


def write_jsonl(path, items):
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
        encoding="utf-8",
    )


def write_runtime_config(root):
    config_path = root / "config.ini"
    config_path.write_text(
        "\n".join(
            [
                "[video]",
                "s3_mode = hybrid",
                "",
                "[local]",
                "access_key = test",
                "secret_key = test",
                "endpoint = http://127.0.0.1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def write_manifest(root, *, master_items=None, start_index=0, count=1, include_reviewer=False):
    master_path = root / "master.jsonl"
    seed_path = root / "seed_ann.jsonl"
    output_path = root / "ann.jsonl"
    reviewer_output_path = root / "reviewer.jsonl"
    merged_path = root / "merged.jsonl"
    manifest_path = root / "manifest.json"

    write_jsonl(master_path, master_items or [make_item(8)])
    write_jsonl(seed_path, [make_item(8)])
    write_jsonl(output_path, [make_item(12)])

    annotators = [
        {
            "id": "ann",
            "start_index": start_index,
            "count": count,
            "seed_input_json": str(seed_path),
            "output_json": str(output_path),
        }
    ]
    if include_reviewer:
        write_jsonl(reviewer_output_path, [make_item(12)])
        annotators.append(
            {
                "id": "reviewer",
                "role": "reviewer",
                "output_json": str(reviewer_output_path),
            }
        )

    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "master_input_json": str(master_path),
                "merged_output_json": str(merged_path),
                "annotators": annotators,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest_path, seed_path, output_path, reviewer_output_path, merged_path


class HighRiskFixesTest(unittest.TestCase):
    def test_chapter_label_uses_text_content_instead_of_action_inner_html(self):
        source = (ROOT / "static/js/main.js").read_text(encoding="utf-8")

        self.assertIn("label.textContent =", source)
        self.assertIn("chapter.appendChild(label);", source)
        self.assertNotIn(
            'chapter.innerHTML = `<span class="chapter-seg-label">${segment.seg_id}: ${truncate(segment.action_text',
            source,
        )

    def test_corrupt_annotator_output_fails_closed_without_seed_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = write_runtime_config(root)
            manifest_path, _seed_path, output_path, _reviewer_output, _merged = write_manifest(root)
            output_path.write_text("{broken jsonl\n", encoding="utf-8")

            runtime_config = RuntimeConfig.load(config_path)
            manifest = MultiUserManifest.load(manifest_path)
            manager = MultiUserStoreManager(manifest=manifest, runtime_config=runtime_config)

            with self.assertRaisesRegex(ValidationError, "cannot be restored"):
                manager.get_store("ann")

    def test_missing_annotator_output_still_initializes_from_seed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = write_runtime_config(root)
            manifest_path, _seed_path, output_path, _reviewer_output, _merged = write_manifest(root)
            output_path.unlink()

            runtime_config = RuntimeConfig.load(config_path)
            manifest = MultiUserManifest.load(manifest_path)
            manager = MultiUserStoreManager(manifest=manifest, runtime_config=runtime_config)
            store = manager.get_store("ann")

        self.assertEqual(store.session_data[0]["action_config"][0]["end_frame"], 8)

    def test_corrupt_reviewer_output_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = write_runtime_config(root)
            manifest_path, _seed_path, _output_path, reviewer_output, _merged = write_manifest(
                root,
                include_reviewer=True,
            )
            reviewer_output.write_text("{broken jsonl\n", encoding="utf-8")

            runtime_config = RuntimeConfig.load(config_path)
            manifest = MultiUserManifest.load(manifest_path)
            manager = MultiUserStoreManager(manifest=manifest, runtime_config=runtime_config)

            with self.assertRaisesRegex(ValidationError, "cannot be restored"):
                manager.get_store("reviewer")

    def test_prepare_seed_skips_existing_by_default_and_force_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path, seed_path, _output_path, _reviewer_output, _merged = write_manifest(root)
            seed_path.write_text("existing seed\n", encoding="utf-8")
            manifest = MultiUserManifest.load(manifest_path)

            skipped = prepare_multi_user_seed_files(manifest)
            self.assertEqual(skipped, [])
            self.assertEqual(seed_path.read_text(encoding="utf-8"), "existing seed\n")

            written = prepare_multi_user_seed_files(manifest, force=True)
            self.assertEqual(written, [seed_path])
            self.assertNotEqual(seed_path.read_text(encoding="utf-8"), "existing seed\n")

    def test_manifest_assignment_gap_warns_without_blocking(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = write_runtime_config(root)
            manifest_path, _seed_path, _output_path, _reviewer_output, _merged = write_manifest(
                root,
                master_items=[make_item(8), make_item(9)],
                start_index=1,
                count=1,
            )

            manifest = MultiUserManifest.load(manifest_path)
            runtime_config = RuntimeConfig.load(config_path)
            manager = MultiUserStoreManager(manifest=manifest, runtime_config=runtime_config)
            client = create_app(store=None, multi_user_manager=manager).test_client()
            self.assertEqual(
                client.post("/api/collab/select", json={"annotator_id": "ann"}).status_code,
                200,
            )
            meta_response = client.get("/api/meta")
            meta_payload = meta_response.get_json()

        self.assertTrue(any("assignment gap" in warning for warning in manifest.load_warnings))
        self.assertEqual(meta_response.status_code, 200)
        self.assertTrue(
            any("assignment gap" in warning for warning in meta_payload["load_warnings"])
        )

    def test_merge_rejects_invalid_output_and_does_not_write_merged_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = write_runtime_config(root)
            manifest_path, _seed_path, output_path, _reviewer_output, merged_path = write_manifest(root)
            output_path.write_text("{broken jsonl\n", encoding="utf-8")
            runtime_config = RuntimeConfig.load(config_path)
            manifest = MultiUserManifest.load(manifest_path)

            with self.assertRaisesRegex(ValidationError, "output_json is invalid"):
                merge_multi_user_outputs(manifest, runtime_config=runtime_config)

            self.assertFalse(merged_path.exists())

    def test_default_output_path_matches_input_format_extension(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            json_path, json_default = resolve_output_path(root / "input.json", "")
            jsonl_path, jsonl_default = resolve_output_path(root / "input.jsonl", "")

        self.assertTrue(json_default)
        self.assertEqual(json_path.name, "annotations.json")
        self.assertTrue(jsonl_default)
        self.assertEqual(jsonl_path.name, "annotations.jsonl")

    def test_annotation_update_and_save_append_audit_events(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = write_runtime_config(root)
            input_path = root / "input.jsonl"
            output_path = root / "out.jsonl"
            write_jsonl(input_path, [make_item(8)])

            runtime_config = RuntimeConfig.load(config_path)
            store = AnnotationStore.load(
                input_path=input_path,
                output_path=output_path,
                runtime_config=runtime_config,
            )
            client = create_app(store=store).test_client()

            updated_item = make_item(12, task_text="Sort items carefully.")
            updated_item["action_config"][0]["action_text"] = "Grabs the item."
            updated_item["action_config"][1]["skill"] = "Move"
            response = client.post(
                "/api/video/0/annotation",
                json={
                    "tasks": updated_item["tasks"],
                    "action_config": updated_item["action_config"],
                    "_audit": {"active_ms": 1234, "lead_time_ms": 1234},
                },
            )
            self.assertEqual(response.status_code, 200)

            failed = client.post(
                "/api/video/0/annotation",
                json={"tasks": [], "action_config": []},
            )
            self.assertEqual(failed.status_code, 400)

            saved = client.post("/api/save", json={"_audit": {"active_ms": 1500, "lead_time_ms": 1500}})
            self.assertEqual(saved.status_code, 200)

            audit_path = output_path.with_name(f"{output_path.name}.audit.jsonl")
            audit_events = [
                json.loads(line)
                for line in audit_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            status_path = output_path.with_name(f"{output_path.name}.status.json")
            status_snapshot = json.loads(status_path.read_text(encoding="utf-8"))
            meta = client.get("/api/meta").get_json()
            report_run = subprocess.run(
                [sys.executable, "-m", "roboatom.annotation.audit_report", "--output", str(output_path)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            audit_report = json.loads(report_run.stdout)

        self.assertEqual(
            [(event["event_type"], event["status"]) for event in audit_events],
            [
                ("annotation_update", "success"),
                ("annotation_update", "failure"),
                ("annotation_save", "success"),
            ],
        )
        for event in audit_events:
            self.assertIn("event_id", event)
            self.assertIn("before_hash", event)
            self.assertIn("after_hash", event)
            self.assertIn("created_at", event)
            self.assertEqual(event["schema_version"], 2)
            self.assertIn("request_id", event)
            self.assertIn("session_id", event)
            self.assertIn("item_key", event)
            self.assertIn("local_index", event)
            self.assertIn("global_index", event)
            self.assertIn("sample_id", event)
            self.assertIn("user_agent", event)
            self.assertIn("revision_before", event)
            self.assertIn("revision_after", event)
            self.assertIn("diff", event)
            self.assertIn("change_summary", event)
            self.assertIn("prev_event_hash", event)
            self.assertIn("event_hash", event)
            self.assertNotIn("before_item", event)
            self.assertNotIn("after_item", event)

        first_event = audit_events[0]
        self.assertEqual(first_event["active_ms"], 1234)
        self.assertEqual(first_event["diff"]["tasks"]["before"], ["Sort items."])
        self.assertEqual(first_event["diff"]["tasks"]["after"], ["Sort items carefully."])
        self.assertEqual(first_event["change_summary"]["fields_changed"]["action_text"], 1)
        self.assertEqual(first_event["change_summary"]["fields_changed"]["end_frame"], 1)
        self.assertEqual(first_event["change_summary"]["fields_changed"]["start_frame"], 1)
        self.assertEqual(first_event["change_summary"]["fields_changed"]["skill"], 1)
        self.assertEqual(audit_events[1]["prev_event_hash"], first_event["event_hash"])
        self.assertEqual(status_snapshot["items"][0]["status"], "saved")
        self.assertTrue(status_snapshot["items"][0]["modified"])
        self.assertGreater(status_snapshot["items"][0]["change_count"], 0)
        self.assertEqual(meta["annotation_statuses"][0]["status"], "saved")
        self.assertGreater(
            audit_report["outputs"][0]["annotators"]["single"]["text_changes"],
            0,
        )

    def test_legacy_output_without_audit_is_reported_in_meta(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = write_runtime_config(root)
            input_path = root / "input.jsonl"
            output_path = root / "out.jsonl"
            write_jsonl(input_path, [make_item(8)])
            write_jsonl(output_path, [make_item(8)])

            runtime_config = RuntimeConfig.load(config_path)
            store = AnnotationStore.load(
                input_path=input_path,
                output_path=output_path,
                runtime_config=runtime_config,
            )
            client = create_app(store=store).test_client()
            meta = client.get("/api/meta")

        self.assertEqual(meta.status_code, 200)
        self.assertEqual(
            meta.get_json()["annotation_statuses"][0]["status"],
            "legacy_output_without_audit",
        )

    def test_item_open_and_close_audit_events_update_status_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = write_runtime_config(root)
            input_path = root / "input.jsonl"
            output_path = root / "out.jsonl"
            write_jsonl(input_path, [make_item(8)])

            runtime_config = RuntimeConfig.load(config_path)
            store = AnnotationStore.load(
                input_path=input_path,
                output_path=output_path,
                runtime_config=runtime_config,
            )
            client = create_app(store=store).test_client()
            self.assertEqual(
                client.post("/api/audit/event", json={"event_type": "item_open", "index": 0}).status_code,
                200,
            )
            self.assertEqual(
                client.post(
                    "/api/audit/event",
                    json={"event_type": "item_close", "index": 0, "active_ms": 42},
                ).status_code,
                200,
            )

            audit_path = output_path.with_name(f"{output_path.name}.audit.jsonl")
            events = [
                json.loads(line)
                for line in audit_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            status_path = output_path.with_name(f"{output_path.name}.status.json")
            status_snapshot = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual([event["event_type"] for event in events], ["item_open", "item_close"])
        self.assertEqual(status_snapshot["items"][0]["status"], "opened")
        self.assertEqual(status_snapshot["items"][0]["active_ms"], 42)

    def test_reviewer_events_use_review_event_types_and_reviewed_status(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = write_runtime_config(root)
            manifest_path, _seed_path, _output_path, reviewer_output, _merged = write_manifest(
                root,
                include_reviewer=True,
            )
            runtime_config = RuntimeConfig.load(config_path)
            manager = MultiUserStoreManager(
                manifest=MultiUserManifest.load(manifest_path),
                runtime_config=runtime_config,
            )
            client = create_app(store=None, multi_user_manager=manager).test_client()
            self.assertEqual(
                client.post("/api/collab/select", json={"annotator_id": "reviewer"}).status_code,
                200,
            )

            updated_item = make_item(13, task_text="Review items.")
            self.assertEqual(
                client.post(
                    "/api/video/0/annotation",
                    json={
                        "tasks": updated_item["tasks"],
                        "action_config": updated_item["action_config"],
                    },
                ).status_code,
                200,
            )
            self.assertEqual(client.post("/api/save", json={}).status_code, 200)

            audit_path = reviewer_output.with_name(f"{reviewer_output.name}.audit.jsonl")
            events = [
                json.loads(line)
                for line in audit_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            status_path = reviewer_output.with_name(f"{reviewer_output.name}.status.json")
            status_snapshot = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(
            [(event["event_type"], event["role"], event["status"]) for event in events],
            [
                ("review_update", "reviewer", "success"),
                ("review_save", "reviewer", "success"),
            ],
        )
        self.assertEqual(status_snapshot["items"][0]["status"], "reviewed")


if __name__ == "__main__":
    unittest.main()

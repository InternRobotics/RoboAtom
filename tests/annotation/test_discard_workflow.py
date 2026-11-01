import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from roboatom.annotation.app import (
    AnnotationStore,
    MultiUserStoreManager,
    REVIEW_BACKUP_FIELD,
    RuntimeConfig,
    ValidationError,
    create_app,
)


def make_item():
    return {
        "episode_index": "sample-1",
        "path": "/tmp/nonexistent.mp4",
        "frames": 20,
        "fps": 10,
        "tasks": ["Sort items."],
        "action_config": [
            {
                "seg_id": 0,
                "start_frame": 0,
                "end_frame": 20,
                "action_text": "Pick and place the item.",
                "skill": "Pick",
            }
        ],
    }


def build_store(root: Path) -> AnnotationStore:
    input_path = root / "input.jsonl"
    output_path = root / "output.jsonl"
    config_path = root / "config.ini"
    input_path.write_text(json.dumps(make_item()) + "\n", encoding="utf-8")
    config_path.write_text(
        "[video]\ns3_mode = hybrid\n\n[local]\n"
        "access_key = test\nsecret_key = test\nendpoint = http://127.0.0.1\n",
        encoding="utf-8",
    )
    return AnnotationStore.load(input_path, output_path, RuntimeConfig.load(config_path))


class DiscardWorkflowTest(unittest.TestCase):
    def test_api_discard_modify_and_restore_are_saved(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = build_store(Path(tmp_dir))
            client = create_app(store).test_client()

            response = client.post(
                "/api/video/0/discard",
                json={"discarded": True, "discard_reason": "unclear", "discard_note": "blurred"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.get_json()["discarded"])
            saved = json.loads(store.output_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(saved["discard_reason"], "unclear")
            self.assertEqual(saved["discard_note"], "blurred")
            self.assertIn("discarded_at", saved)

            response = client.post(
                "/api/video/0/discard",
                json={"discarded": True, "discard_reason": "duplicate", "discard_note": ""},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["discard_reason"], "duplicate")

            blocked = client.post(
                "/api/video/0/annotation",
                json={"tasks": make_item()["tasks"], "action_config": make_item()["action_config"]},
            )
            self.assertEqual(blocked.status_code, 400)

            restored = client.post("/api/video/0/discard", json={"discarded": False})
            self.assertEqual(restored.status_code, 200)
            saved = json.loads(store.output_path.read_text(encoding="utf-8").splitlines()[0])
            for field in ("discarded", "discard_reason", "discard_note", "discarded_by", "discarded_at"):
                self.assertNotIn(field, saved)

    def test_save_failure_rolls_back_memory_and_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = build_store(Path(tmp_dir))
            before = json.loads(json.dumps(store.session_data[0]))
            with patch.object(store, "_write_output_atomically", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    store.update_discard_and_save(
                        0,
                        {"discarded": True, "discard_reason": "video_unplayable"},
                        "annotator_01",
                    )
            self.assertEqual(store.session_data[0], before)
            self.assertFalse(store.dirty_indices)
            self.assertFalse(store.output_path.exists())

    def test_discard_reason_is_optional(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = build_store(Path(tmp_dir))
            result = store.update_discard_and_save(0, {"discarded": True}, "annotator_01")
            self.assertTrue(result["discarded"])
            saved = json.loads(store.output_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(saved["discarded"])
            self.assertNotIn("discard_reason", saved)
            self.assertNotIn("discard_note", saved)

    def test_reason_and_note_validation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = build_store(Path(tmp_dir))
            with self.assertRaises(ValidationError):
                store.update_discard_and_save(0, {"discarded": True, "discard_reason": "bad"}, "ann")
            with self.assertRaises(ValidationError):
                store.update_discard_and_save(
                    0,
                    {"discarded": True, "discard_reason": "other", "discard_note": "x" * 301},
                    "ann",
                )

    def test_reviewer_explicit_restore_overlays_only_unchanged_annotator(self):
        annotator_item = make_item()
        annotator_item.update({"discarded": True, "discard_reason": "unclear"})
        reviewer_item = make_item()
        reviewer_item[REVIEW_BACKUP_FIELD] = json.loads(json.dumps(annotator_item))

        self.assertTrue(MultiUserStoreManager._annotations_match(annotator_item, reviewer_item[REVIEW_BACKUP_FIELD]))
        merged = json.loads(json.dumps(annotator_item))
        MultiUserStoreManager._apply_reviewer_overlay(merged, reviewer_item)
        self.assertNotIn("discarded", merged)

        annotator_item["discard_reason"] = "duplicate"
        self.assertFalse(MultiUserStoreManager._annotations_match(annotator_item, reviewer_item[REVIEW_BACKUP_FIELD]))


if __name__ == "__main__":
    unittest.main()

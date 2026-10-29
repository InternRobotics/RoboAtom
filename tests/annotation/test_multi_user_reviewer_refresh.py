import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from roboatom.annotation.app import (
    MultiUserManifest,
    MultiUserStoreManager,
    REVIEW_BACKUP_FIELD,
    RuntimeConfig,
    ValidationError,
    merge_multi_user_outputs,
)


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


def touch_future(path):
    future = time.time() + 2
    os.utime(path, (future, future))


class MultiUserReviewerRefreshTest(unittest.TestCase):
    def build_manager(self, tmp_dir, annotator_item, reviewer_item=None):
        root = Path(tmp_dir)
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

        master_path = root / "master.jsonl"
        seed_path = root / "seed_ann.jsonl"
        annotator_output = root / "ann.jsonl"
        reviewer_output = root / "reviewer.jsonl"
        manifest_path = root / "manifest.json"

        write_jsonl(master_path, [make_item(8)])
        write_jsonl(seed_path, [make_item(8)])
        write_jsonl(annotator_output, [annotator_item])
        if reviewer_item is not None:
            write_jsonl(reviewer_output, [reviewer_item])

        manifest_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "master_input_json": str(master_path),
                    "merged_output_json": str(root / "merged.jsonl"),
                    "annotators": [
                        {
                            "id": "ann",
                            "start_index": 0,
                            "count": 1,
                            "seed_input_json": str(seed_path),
                            "output_json": str(annotator_output),
                        },
                        {
                            "id": "reviewer",
                            "name": "Reviewer",
                            "role": "reviewer",
                            "output_json": str(reviewer_output),
                        },
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        runtime_config = RuntimeConfig.load(config_path)
        manifest = MultiUserManifest.load(manifest_path)
        manager = MultiUserStoreManager(
            manifest=manifest,
            runtime_config=runtime_config,
        )
        return manager, annotator_output, reviewer_output

    def test_reviewer_ignores_plain_stale_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager, _annotator_output, _reviewer_output = self.build_manager(
                tmp_dir,
                annotator_item=make_item(12),
                reviewer_item=make_item(6),
            )

            reviewer_store = manager.get_store("reviewer")

        self.assertEqual(
            reviewer_store.session_data[0]["action_config"][0]["end_frame"],
            12,
        )
        self.assertNotIn(REVIEW_BACKUP_FIELD, reviewer_store.session_data[0])

    def test_reviewer_overlay_is_skipped_after_annotator_rework(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            reviewed_item = make_item(14)
            reviewed_item[REVIEW_BACKUP_FIELD] = {
                "tasks": make_item(10)["tasks"],
                "action_config": make_item(10)["action_config"],
            }
            manager, annotator_output, _reviewer_output = self.build_manager(
                tmp_dir,
                annotator_item=make_item(10),
                reviewer_item=reviewed_item,
            )

            reviewer_store = manager.get_store("reviewer")
            self.assertEqual(
                reviewer_store.session_data[0]["action_config"][0]["end_frame"],
                14,
            )

            write_jsonl(annotator_output, [make_item(12)])
            touch_future(annotator_output)
            refreshed_store = manager.get_store("reviewer")

        self.assertEqual(
            refreshed_store.session_data[0]["action_config"][0]["end_frame"],
            12,
        )
        self.assertNotIn(REVIEW_BACKUP_FIELD, refreshed_store.session_data[0])

    def test_repeated_reviewer_save_preserves_original_backup(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_annotator_item = make_item(10)
            reviewed_item = make_item(14)
            reviewed_item[REVIEW_BACKUP_FIELD] = {
                "tasks": original_annotator_item["tasks"],
                "action_config": original_annotator_item["action_config"],
            }
            manager, _annotator_output, reviewer_output = self.build_manager(
                tmp_dir,
                annotator_item=original_annotator_item,
                reviewer_item=reviewed_item,
            )

            updated_item = make_item(16)
            manager.update_annotation(
                "reviewer",
                0,
                {
                    "tasks": updated_item["tasks"],
                    "action_config": updated_item["action_config"],
                },
            )
            manager.save("reviewer")

            saved = json.loads(reviewer_output.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(saved["action_config"][0]["end_frame"], 16)
        self.assertEqual(
            saved[REVIEW_BACKUP_FIELD]["action_config"][0]["end_frame"],
            10,
        )

    def test_merge_applies_reviewer_overlay_when_annotator_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            original_annotator_item = make_item(10)
            reviewed_item = make_item(14)
            reviewed_item[REVIEW_BACKUP_FIELD] = {
                "tasks": original_annotator_item["tasks"],
                "action_config": original_annotator_item["action_config"],
            }
            manager, _annotator_output, _reviewer_output = self.build_manager(
                tmp_dir,
                annotator_item=original_annotator_item,
                reviewer_item=reviewed_item,
            )

            merged_path = merge_multi_user_outputs(
                manager.manifest,
                runtime_config=manager.runtime_config,
            )
            merged_item = json.loads(merged_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(merged_item["action_config"][0]["end_frame"], 14)
        self.assertNotIn(REVIEW_BACKUP_FIELD, merged_item)

    def test_merge_skips_reviewer_overlay_after_annotator_rework(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            reviewed_item = make_item(14)
            reviewed_item[REVIEW_BACKUP_FIELD] = {
                "tasks": make_item(10)["tasks"],
                "action_config": make_item(10)["action_config"],
            }
            manager, _annotator_output, _reviewer_output = self.build_manager(
                tmp_dir,
                annotator_item=make_item(12),
                reviewer_item=reviewed_item,
            )

            merged_path = merge_multi_user_outputs(
                manager.manifest,
                runtime_config=manager.runtime_config,
            )
            merged_item = json.loads(merged_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(merged_item["action_config"][0]["end_frame"], 12)
        self.assertNotIn(REVIEW_BACKUP_FIELD, merged_item)

    def test_merge_rejects_corrupt_reviewer_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager, _annotator_output, reviewer_output = self.build_manager(
                tmp_dir,
                annotator_item=make_item(10),
                reviewer_item=make_item(14),
            )
            reviewer_output.write_text("{broken jsonl\n", encoding="utf-8")

            with self.assertRaisesRegex(ValidationError, "reviewer 'reviewer' output_json is invalid"):
                merge_multi_user_outputs(
                    manager.manifest,
                    runtime_config=manager.runtime_config,
                )

        self.assertFalse(manager.manifest.merged_output_path.exists())


if __name__ == "__main__":
    unittest.main()

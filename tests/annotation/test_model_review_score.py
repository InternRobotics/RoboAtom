import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2] / "src/roboatom/annotation"
sys.path.insert(0, str(ROOT / "scripts"))

from roboatom.annotation import review as scorer  # noqa: E402


def make_segments():
    return [
        {
            "seg_id": 0,
            "start_frame": 0,
            "end_frame": 99,
            "action_text": "Picks up the red can.",
            "skill": "Pick",
        },
        {
            "seg_id": 1,
            "start_frame": 100,
            "end_frame": 199,
            "action_text": "Places the red can into the blue bin.",
            "skill": "Place",
        },
    ]


class ModelReviewScoreTest(unittest.TestCase):
    def test_iter_annotation_rows_keeps_companion_metadata(self):
        annotation = {
            "length": 20,
            "tasks": ["Sort objects."],
            "action_config": [
                {
                    "seg_id": 0,
                    "start_frame": 0,
                    "end_frame": 19,
                    "action_text": "Picks up the object.",
                    "skill": "Pick",
                }
            ],
        }
        companion = {
            "official_task_instruction": "sort objects",
            "ground_truth_inventory": {"core_objects": ["object"]},
            "evaluation": {"overall_score": 0},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "input.jsonl"
            path.write_text(json.dumps([annotation, companion]) + "\n", encoding="utf-8")

            rows = list(scorer.iter_annotation_records(path))
            compatible_rows = list(scorer.iter_annotation_rows(path))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], 1)
        self.assertEqual(rows[0][1], annotation)
        self.assertEqual(rows[0][2], companion)
        self.assertEqual(compatible_rows, [(1, annotation)])

    def test_episode_manifest_has_global_inside_and_boundary_roles(self):
        manifest = scorer.build_multi_view_frame_manifest(
            segments=make_segments(),
            total_frames=200,
            view_keys=["head"],
            max_global_frames=5,
            boundary_offsets=[-10, 0, 10],
            include_global=True,
        )

        roles = {item["role"] for item in manifest}
        self.assertIn("inside_25", roles)
        self.assertIn("inside_50", roles)
        self.assertIn("inside_75", roles)
        self.assertIn("boundary_0_1+0", roles)
        self.assertTrue(any(role.startswith("global_") for role in roles))
        self.assertEqual({item["view_key"] for item in manifest}, {"head"})
        self.assertEqual(
            len({(item["view_key"], item["frame"]) for item in manifest}),
            len(manifest),
        )

    def test_fallback_manifest_is_view_scoped_without_global_frames(self):
        manifest = scorer.build_multi_view_frame_manifest(
            segments=make_segments(),
            total_frames=200,
            view_keys=["hand_right", "hand_left"],
            max_global_frames=48,
            boundary_offsets=[0],
            include_global=False,
        )

        self.assertEqual({item["view_key"] for item in manifest}, {"hand_right", "hand_left"})
        self.assertFalse(any(str(item["role"]).startswith("global_") for item in manifest))
        self.assertEqual(
            len({(item["view_key"], item["frame"]) for item in manifest}),
            len(manifest),
        )

    def test_prompt_describes_episode_level_description_and_segmentation_review(self):
        manifest = scorer.build_multi_view_frame_manifest(
            segments=make_segments(),
            total_frames=200,
            view_keys=["head"],
            max_global_frames=3,
            boundary_offsets=[0],
            include_global=True,
        )
        prompt = scorer.build_review_prompt(
            sample_id="episode_001",
            task_text="Sort garbage.",
            total_frames=200,
            fps=30.0,
            segments=make_segments(),
            frame_manifest=manifest,
            skill_options=["Pick", "Place"],
            available_view_keys=["head"],
            review_pass="primary",
            companion_metadata={
                "official_task_instruction": "sort garbage",
                "ground_truth_inventory": {"core_objects": ["red can"]},
                "evaluation": {"overall_score": 0},
            },
            allow_needs_more_views=True,
        )

        self.assertIn("Description accuracy", prompt)
        self.assertIn("Segmentation accuracy", prompt)
        self.assertIn("boundary cuts an atomic action event", prompt)
        self.assertIn('"Pick"', prompt)
        self.assertIn("needs_more_views", prompt)
        self.assertIn("ignore any old evaluation scores", prompt)
        self.assertNotIn("boundary_tolerance_frames", prompt)
        self.assertNotIn("overall_score", prompt)

    def test_view_fallback_trigger_conditions(self):
        segments = make_segments()
        correct = scorer.normalize_review_response(
            {"verdict": "correct", "issues": [], "needs_more_views": False},
            segments=segments,
            total_frames=200,
        )
        self.assertFalse(scorer.should_run_view_fallback({}, correct))

        explicit = scorer.normalize_review_response(
            {"verdict": "correct", "issues": [], "needs_more_views": True},
            segments=segments,
            total_frames=200,
        )
        self.assertTrue(scorer.should_run_view_fallback({"needs_more_views": True}, explicit))

        uncertain = scorer.normalize_review_response(
            {"verdict": "uncertain", "issues": []},
            segments=segments,
            total_frames=200,
        )
        self.assertTrue(scorer.should_run_view_fallback({}, uncertain))

        unclear = scorer.normalize_review_response(
            {
                "verdict": "uncertain",
                "issues": [
                    {
                        "seg_id": 0,
                        "issue_type": "unclear_visual",
                        "evidence_frames": [10],
                        "confidence": 0.3,
                    }
                ],
            },
            segments=segments,
            total_frames=200,
        )
        self.assertTrue(scorer.should_run_view_fallback({}, unclear))

    def test_non_stream_completion_text_with_mock_response(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"verdict":"correct","needs_more_views":false,"issues":[],"summary":"ok"}'
                    )
                )
            ],
            usage={"total_tokens": 10},
            _request_id="req-1",
        )
        completions = SimpleNamespace(create=lambda **_kwargs: response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        text = scorer.api_completion_text_with_retry(
            client,
            model="mock",
            messages=[],
            temperature=0.1,
            top_p=1.0,
            max_tokens=128,
            max_retries=1,
            stream=False,
            debug_meta={"sample_id": "sample", "frame_count": 0},
        )

        self.assertIn('"verdict":"correct"', text)

    def test_stream_completion_text_joins_chunks_and_skips_empty_events(self):
        events = [
            {},
            {"choices": []},
            {"choices": [{"delta": {}}]},
            {"choices": [{"delta": {"content": "```json\n{"}}]},
            {"choices": [{"delta": {"content": '"verdict":"correct","issues":[]'}}]},
            {"choices": [{"delta": {"content": ',"summary":"ok"}\n```'}}]},
        ]
        completions = SimpleNamespace(create=lambda **_kwargs: iter(events))
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        text = scorer.api_completion_text_with_retry(
            client,
            model="mock",
            messages=[],
            temperature=0.1,
            top_p=1.0,
            max_tokens=128,
            max_retries=1,
            stream=True,
            debug_meta={"sample_id": "sample", "frame_count": 0, "stream_log_chars": 5},
        )
        parsed = scorer.parse_json_object(text)

        self.assertEqual(parsed["verdict"], "correct")
        self.assertEqual(parsed["issues"], [])

    def test_make_client_passes_timeout_arguments(self):
        captured = {}

        def fake_timeout(timeout, *, connect):
            captured["timeout"] = timeout
            captured["connect"] = connect
            return "timeout-object"

        client_cls = Mock(return_value="http-client")
        fake_httpx = SimpleNamespace(Timeout=fake_timeout, Client=client_cls)
        with patch.object(scorer, "httpx", fake_httpx), patch.object(scorer, "OpenAI") as openai_cls:
            scorer.make_client(
                "https://example.test/v1",
                "sk-test",
                request_timeout=3000,
                connect_timeout=60,
            )

        self.assertEqual(captured, {"timeout": 3000.0, "connect": 60.0})
        client_cls.assert_called_once()
        openai_cls.assert_called_once()


if __name__ == "__main__":
    unittest.main()

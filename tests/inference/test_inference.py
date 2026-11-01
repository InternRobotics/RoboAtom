"""Checks for portable model inputs, camera layouts and HTTP inference."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from roboatom.inference import client
from roboatom.inference.prompt import camera_layout, user_prompt
from roboatom.inference.server import create_app
from roboatom.inference.video import compose_mosaic, sample_indices


@pytest.fixture
def benchmark_row() -> dict[str, Any]:
    return {
        "benchmark_id": "test-video", "length": 60, "fps": 30.0,
        "task": "Place the object.", "model_view_key": "head",
        "evaluation_views": ["head", "hand_left", "hand_right"],
        "views": {"head": "head.mp4", "hand_left": "left.mp4", "hand_right": "right.mp4"},
    }


@pytest.mark.parametrize(
    ("keys", "layout", "roles"),
    [
        (["head"], "single", ["primary"]),
        (["head", "wrist"], "vertical_stack", ["primary", "aux"]),
        (["head", "hand_right", "hand_left"], "primary_large_right_stack",
         ["primary", "left_wrist", "right_wrist"]),
        (["head", "hand_right", "overlook", "hand_left"], "grid_2x2",
         ["primary", "aux", "left_wrist", "right_wrist"]),
    ],
)
def test_camera_selection(
    benchmark_row: dict[str, Any], keys: list[str], layout: str, roles: list[str],
) -> None:
    benchmark_row.update(evaluation_views=keys, views={key: f"{key}.mp4" for key in keys})

    result = camera_layout(benchmark_row)

    assert result["layout"] == layout
    assert [view["role"] for view in result["views"]] == roles
    if "hand_left" in keys:
        assert next(view["path"] for view in result["views"] if view["role"] == "left_wrist") == "hand_left.mp4"


def test_prompt_uses_only_video_metadata(benchmark_row: dict[str, Any]) -> None:
    benchmark_row["segments"] = [{"subtask": "SECRET_GROUND_TRUTH"}]

    prompt = user_prompt(benchmark_row, "primary_large_right_stack")

    assert "total_frames: 60\nsource_fps: 30\n" in prompt
    assert "RIGHT_TOP=LEFT_WRIST" in prompt
    assert "[start_frame, end_frame]" in prompt
    assert "SECRET_GROUND_TRUTH" not in prompt


def test_sampling_includes_endpoints_and_caps_frames() -> None:
    assert sample_indices(60, 2.0) == [0, 19, 39, 59]
    assert sample_indices(1, 1 / 30) == [0]
    indices = sample_indices(30000, 1000.0)
    assert len(indices) == 448
    assert (indices[0], indices[-1]) == (0, 29999)


def test_mosaic_panel_positions() -> None:
    frames = {
        "primary": Image.new("RGB", (60, 40), "red"),
        "left_wrist": Image.new("RGB", (30, 20), "green"),
        "right_wrist": Image.new("RGB", (30, 20), "blue"),
    }

    result = compose_mosaic(frames, "primary_large_right_stack")

    assert result.size == (90, 40)
    assert result.getpixel((30, 20)) == (255, 0, 0)
    assert result.getpixel((75, 10)) == (0, 128, 0)
    assert result.getpixel((75, 30)) == (0, 0, 255)


def test_service_health_and_prediction(benchmark_row: dict[str, Any]) -> None:
    class FakeSegmenter:
        def predict(self, row: dict[str, Any]) -> dict[str, Any]:
            return {"episode_id": row["benchmark_id"], "segments": []}

    service = TestClient(create_app(FakeSegmenter()))

    assert service.get("/health").json() == {"status": "ok"}
    assert service.post("/predict", json=benchmark_row).json()["episode_id"] == "test-video"
    assert service.post("/predict", json={}).status_code == 422


def test_client_excludes_ground_truth(
    tmp_path: Path, benchmark_row: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_row["segments"] = [{"subtask": "SECRET_GROUND_TRUTH"}]
    input_path, output_path = tmp_path / "input.jsonl", tmp_path / "predictions.jsonl"
    input_path.write_text(json.dumps(benchmark_row) + "\n")
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"episode_id": "test-video", "segments": []})

    original_client = httpx.Client
    monkeypatch.setattr(client.httpx, "Client", lambda **kwargs: original_client(
        transport=httpx.MockTransport(respond), **kwargs,
    ))

    client.infer(input_path, output_path, "http://inference.test")

    assert len(requests) == 1
    assert "segments" not in requests[0]
    assert json.loads(output_path.read_text())["episode_id"] == "test-video"


def test_client_propagates_service_failure(
    tmp_path: Path, benchmark_row: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps(benchmark_row) + "\n")
    original_client = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(503))
    monkeypatch.setattr(client.httpx, "Client", lambda **kwargs: original_client(transport=transport, **kwargs))

    with pytest.raises(httpx.HTTPStatusError):
        client.infer(input_path, tmp_path / "predictions.jsonl", "http://inference.test")

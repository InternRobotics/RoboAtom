"""Call a RoboAtom-Anno service and write evaluator-compatible predictions."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx

from roboatom.validation.data import read_rows

logger = logging.getLogger(__name__)
INPUT_FIELDS = ("benchmark_id", "length", "fps", "task", "views", "evaluation_views", "model_view_key")


def infer(input_path: Path, output_path: Path, base_url: str) -> None:
    """Generate one prediction per input row through the HTTP service.

    Args:
        input_path: Benchmark input JSONL, without ground truth annotations.
        output_path: Prediction JSONL consumed by roboatom evaluate.
        base_url: HTTP service URL.

    Raises:
        FileExistsError: If the output file already exists.
        httpx.HTTPError: If a prediction request fails.
    """
    rows = read_rows(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        httpx.Client(base_url=base_url.rstrip("/"), timeout=1800.0, trust_env=False) as client,
        output_path.open("x", encoding="utf-8") as stream,
    ):
        for index, row in enumerate(rows, 1):
            response = client.post("/predict", json={key: row[key] for key in INPUT_FIELDS})
            response.raise_for_status()
            stream.write(json.dumps(response.json(), ensure_ascii=False) + "\n")
            stream.flush()
            logger.info("Predicted %d/%d: %s", index, len(rows), row["benchmark_id"])

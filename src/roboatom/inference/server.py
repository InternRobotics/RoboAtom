"""HTTP service for RoboAtom-Anno."""
from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException

logger = logging.getLogger(__name__)


def create_app(segmenter: Any) -> FastAPI:
    """Create a service around a loaded video segmenter.

    Args:
        segmenter: Object implementing predict(input_row).

    Returns:
        Application with health and prediction endpoints.
    """
    app = FastAPI(title="RoboAtom-Anno")
    lock = Lock()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    def predict(row: dict[str, Any]) -> dict[str, Any]:
        try:
            with lock:
                return segmenter.predict(row)
        except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app


def serve(model_path: Path, media_root: Path, host: str, port: int) -> None:
    """Load the model and run one HTTP inference worker.

    Args:
        model_path: Complete Hugging Face model directory.
        media_root: Root of the videos visible to the server.
        host: Bind address.
        port: Listening port.
    """
    import uvicorn

    from .model import Segmenter

    logger.info("Loading RoboAtom-Anno from %s", model_path)
    segmenter = Segmenter(model_path, media_root)
    uvicorn.run(create_app(segmenter), host=host, port=port)

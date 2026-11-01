"""Append full text LLM prompts to a log file for model_review_score*.py scripts."""

from __future__ import annotations

from pathlib import Path

RECORD_SEPARATOR = "====\n"


def resolve_prompt_log_path(raw: str | None) -> Path | None:
    text = str(raw or "").strip()
    if not text or text.lower() in ("none", "off", "false", "0"):
        return None
    return Path(text).expanduser().resolve()


def append_text_prompt(
    log_path: Path | None,
    *,
    script: str,
    line_no: int,
    sample_id: str,
    review_pass: str,
    prompt: str,
) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(RECORD_SEPARATOR)
        handle.write(
            f"PROMPT script={script} line_no={line_no} review_pass={review_pass}\n"
            f"sample_id={sample_id}\n"
        )
        handle.write(str(prompt))
        if not str(prompt).endswith("\n"):
            handle.write("\n")

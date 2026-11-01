"""Reference selection rules for same-task boundary extension."""
from __future__ import annotations
from typing import Any


def _reference_signature(row: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Keep action count, order, and per-step meaning in reference admission."""
    return tuple(
        (
            str(segment['skill']).strip().casefold().replace('-', '_').replace(' ', '_'),
            ' '.join(str(segment.get('subtask', '')).casefold().split()).rstrip('.!?。！？'),
        )
        for segment in row['segments']
    )


def _validate_reference_coverage(row: dict[str, Any]) -> None:
    """A reference must annotate every frame before it can supply a transcript."""
    expected = 0
    for segment in row['segments']:
        if segment['start_frame'] != expected:
            raise ValueError(f"Reference {row['episode_id']} has an unannotated frame gap")
        expected = segment['end_frame'] + 1
    if expected != row['num_frames']:
        raise ValueError(f"Reference {row['episode_id']} does not cover its final frame")


def select_references(
    target: dict[str, Any], references: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Abstain on ambiguous same-task transcripts instead of guessing a variant.

    The target has no labels, so sensor similarity alone cannot establish object
    count, destination, or operation order. A human-audited ``task_variant`` may
    narrow references; otherwise their full skill/text signatures must agree.
    """
    same_task = [row for row in references if row['task_id'] == target['task_id']]
    if not same_task:
        raise ValueError(f"No same-task reference for {target['episode_id']}")
    variant = target.get('task_variant')
    if variant is not None:
        if not isinstance(variant, str) or not variant.strip():
            raise ValueError(f"Target {target['episode_id']} has an invalid task_variant")
        same_task = [row for row in same_task if row.get('task_variant') == variant]
        if not same_task:
            raise ValueError(f"No reference for task_variant {variant!r} of {target['episode_id']}")
    elif any(row.get('task_variant') is not None for row in same_task):
        raise ValueError(
            f"Target {target['episode_id']} needs task_variant to select audited references"
        )
    signatures = {_reference_signature(row) for row in same_task}
    if len(signatures) != 1:
        raise ValueError(
            f"Ambiguous same-task references for {target['episode_id']}: "
            "skill sequence or subtask descriptions differ; assign a reviewed "
            "task_variant to the target and references"
        )
    return same_task

"""Offline Skill-F1@0.7 replica (MS-TCN matching) for screening hypotheses on the dev fold.

Not used for acceptance; registered rounds are scored by the frozen evaluator.
"""

from __future__ import annotations


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def counts(pred: list[tuple[str, float, float]], gold: list[tuple[str, float, float]]) -> tuple[int, int, int]:
    hit: set[int] = set()
    tp = fp = 0
    for label, start, end in pred:
        best, index = 0.0, None
        for j, (g_label, g_start, g_end) in enumerate(gold):
            if g_label != label:
                continue
            value = iou((start, end), (g_start, g_end))
            if value > best + 1e-9:
                best, index = value, j
        if index is not None and index not in hit and best + 1e-9 >= 0.7:
            tp += 1
            hit.add(index)
        else:
            fp += 1
    return tp, fp, len(gold) - len(hit)


def with_boundaries(labels: list[str], cuts: list[float], length: float) -> list[tuple[str, float, float]]:
    edges = [0.0, *cuts, float(length)]
    return [(label, edges[i], edges[i + 1]) for i, label in enumerate(labels)]


def f1(total: tuple[int, int, int]) -> float:
    tp, fp, fn = total
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0

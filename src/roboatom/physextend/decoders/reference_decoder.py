"""Annotated reference trajectory shared by the physical decoders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


from roboatom.physextend.features.signals import FeatureBundle


@dataclass(frozen=True)
class SeedTrajectory:
    """One annotated source trajectory and its boundary-free signal features.

    Args:
        episode_key: Stable source identifier used only for deterministic logs.
        segments: Annotated source segments matching the transcript.
        features: Model-free source signal features.
    """

    episode_key: str
    segments: tuple[Mapping[str, Any], ...]
    features: FeatureBundle

    def __post_init__(self) -> None:
        if not self.episode_key:
            raise ValueError("Seed episode_key must be non-empty")
        if not self.segments:
            raise ValueError("Seed segments must be non-empty")

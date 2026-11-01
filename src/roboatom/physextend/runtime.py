"""Run-local paths, episode metadata and memoization for PhysExtend."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

DATA = Path(__file__).resolve().parent / "data"
Parameters = ParamSpec("Parameters")
Result = TypeVar("Result")
PredictionProvider = Callable[[str, int], dict[str, dict[str, Any]]]


@dataclass
class RunContext:
    """State owned by one invocation, discarded when that invocation finishes."""

    workdir: Path
    episodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    prediction_provider: PredictionProvider | None = None
    caches: dict[Callable[..., Any], Any] = field(default_factory=dict)


_context: ContextVar[RunContext | None] = ContextVar("physextend_run", default=None)


def current_run() -> RunContext:
    """Return the active run, rejecting file access outside an invocation.

    Raises:
        RuntimeError: If no PhysExtend run is active.
    """
    context = _context.get()
    if context is None:
        raise RuntimeError("No active PhysExtend workspace")
    return context


@contextmanager
def run_context(workdir: Path) -> Iterator[RunContext]:
    """Bind an isolated workspace and restore the caller's context on exit."""
    context = RunContext(workdir)
    token = _context.set(context)
    try:
        yield context
    finally:
        _context.reset(token)


def run_cache(
    maxsize: int,
) -> Callable[[Callable[Parameters, Result]], Callable[Parameters, Result]]:
    """Memoize within a run; standalone decoder calls remain uncached."""
    def decorate(function: Callable[Parameters, Result]) -> Callable[Parameters, Result]:
        @wraps(function)
        def wrapped(*args: Parameters.args, **kwargs: Parameters.kwargs) -> Result:
            context = _context.get()
            if context is None:
                return function(*args, **kwargs)
            if function not in context.caches:
                context.caches[function] = lru_cache(maxsize=maxsize)(function)
            return context.caches[function](*args, **kwargs)

        return wrapped

    return decorate

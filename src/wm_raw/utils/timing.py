"""Timing utilities for profiling model loading and forward passes."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)


@contextmanager
def log_duration(
    description: str,
    *,
    level: int = logging.INFO,
    logger_instance: logging.Logger | None = None,
) -> Generator[None, None, None]:
    """Context manager that logs wall-clock duration of a block.

    Usage:
        with log_duration("Loading checkpoint"):
            load_checkpoint(...)
        # prints: "Loading checkpoint ... done (12.3s)"

    Args:
        description: Human-readable label for the timed block.
        level: Logging level (default INFO).
        logger_instance: Logger to use (default: wm_raw.utils.timing logger).
    """
    log = logger_instance or logger
    log.log(level, "%s ...", description)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        log.log(level, "%s ... done (%.1fs)", description, elapsed)


class Timer:
    """Lightweight timer for accumulating stage durations.

    Usage:
        timer = Timer()
        with timer.stage("build model"):
            model = WorldModel(config)
        with timer.stage("load checkpoint"):
            load_checkpoint(...)
        timer.summary()  # prints all stages
    """

    def __init__(self, logger_instance: logging.Logger | None = None) -> None:
        self._log = logger_instance or logger
        self._stages: list[tuple[str, float]] = []

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        """Time a named stage."""
        self._log.info("[timer] %s ...", name)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._stages.append((name, elapsed))
            self._log.info("[timer] %s ... done (%.1fs)", name, elapsed)

    def summary(self) -> str:
        """Return formatted summary of all stages."""
        lines = ["Timing summary:"]
        total = 0.0
        for name, elapsed in self._stages:
            lines.append(f"  {name:40s} {elapsed:6.1f}s")
            total += elapsed
        lines.append(f"  {'TOTAL':40s} {total:6.1f}s")
        msg = "\n".join(lines)
        self._log.info(msg)
        return msg

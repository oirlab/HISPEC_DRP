"""
High-resolution wall timing for structured logs.

Call sites use :func:`perf_clock` and :func:`log_wall_s` without ``import time`` in
their modules; monotonic clocks live here only.
"""
from __future__ import annotations

import logging
from time import perf_counter

__all__ = ["log_wall_s", "perf_clock", "wall_seconds"]


def perf_clock() -> float:
    """Return a monotonic reference time (seconds); pair with :func:`log_wall_s`."""

    return perf_counter()


def wall_seconds(t0: float) -> float:
    """Elapsed seconds since ``t0`` from :func:`perf_clock`."""

    return perf_counter() - t0


def log_wall_s(
    logger: logging.Logger,
    t0: float,
    fmt: str,
    *fmt_args,
    level: int = logging.INFO,
) -> None:
    """
    Log ``fmt`` with ``fmt_args`` plus ``wall_s=<elapsed>``.

    Example: ``log_wall_s(logger, t0, "STEP foo bar=%s", 42)``.
    """

    dt = wall_seconds(t0)
    msg = fmt.rstrip() + " wall_s=%.3f"
    if fmt_args:
        logger.log(level, msg, *fmt_args, dt)
    else:
        logger.log(level, msg, dt)

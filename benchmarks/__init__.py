"""
Benchmark / simulation entry points for HISPEC DRP (ramps, rate maps, end-to-end runs).

Heavy helpers live in :mod:`benchmarks.end_to_end_simulation`. They are **not**
imported eagerly here so that ``python -m benchmarks.end_to_end_simulation`` does not
preload that module when Python imports this package (which would trigger runpy’s
“found in sys.modules…” :exc:`RuntimeWarning`).
"""

from __future__ import annotations

import importlib
from typing import Any
from .run_caldb import *


def __getattr__(name: str) -> Any:
    """Lazily re-export public names from :mod:`benchmarks.end_to_end_simulation`."""
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(".end_to_end_simulation", __name__)
    if hasattr(mod, name):
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

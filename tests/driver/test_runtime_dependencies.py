"""Regression tests for runtime dependency versions and importability."""

from __future__ import annotations

import importlib.metadata
import re

import pytest


def _parse_version(version: str) -> tuple[int, ...]:
    """Parse a PEP 440-ish version into a comparable int tuple (major.minor.micro)."""
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", version)
    if not match:
        raise ValueError(f"Unparseable version string: {version!r}")
    return tuple(int(part or 0) for part in match.groups())


def test_numba_meets_minimum_version():
    """numba < 0.66 fails to import against NumPy 2.4+ (ImportError at hispecdrp import)."""
    version = importlib.metadata.version("numba")
    assert _parse_version(version) >= (0, 66, 0), (
        f"numba {version} is too old; need >= 0.66.0 for NumPy 2.4+ compatibility "
        "(see pyproject.toml)."
    )


def test_numba_imports_with_installed_numpy():
    """Importing numba must succeed with the currently installed NumPy."""
    import numba  # noqa: F401
    import numpy as np

    # Sanity: numba's own check would have raised already; keep the versions visible.
    assert hasattr(numba, "__version__")
    assert _parse_version(np.__version__) >= (2, 0, 0)


def test_barycorrpy_is_installed_and_importable():
    """barycorrpy is required by barycenter_correction (pulled in via hispecdrp.primitives)."""
    version = importlib.metadata.version("barycorrpy")
    assert _parse_version(version) >= (0, 4, 4), (
        f"barycorrpy {version} is too old; need >= 0.4.4 (see pyproject.toml)."
    )
    import barycorrpy  # noqa: F401


def test_hispecdrp_imports_without_missing_runtime_deps():
    """``import hispecdrp`` must not fail on missing numba/barycorrpy (or version mismatch)."""
    import hispecdrp  # noqa: F401
    from hispecdrp.primitives.barycenter_correction import barycenter_corr  # noqa: F401


def test_pyproject_pins_numba_and_barycorrpy():
    """Keep pyproject.toml pins in sync with the tests above."""
    from pathlib import Path

    from tests.utils import repo_root
    text = (repo_root() / "pyproject.toml").read_text()
    assert re.search(r'"numba>=0\.66', text), "pyproject.toml must pin numba>=0.66"
    assert re.search(r'"barycorrpy>=0\.4\.4"', text), (
        "pyproject.toml must pin barycorrpy>=0.4.4"
    )

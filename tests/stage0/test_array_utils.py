"""Fast unit tests for pure-Python array utilities (no Numba cold-compile tax)."""

from __future__ import annotations

import numpy as np
import pytest

from hispecdrp.primitives.utils.array_utils import (
    nanmedian_axis0_ignore_empty_columns,
    solve_chebyshev2d_ols,
)


def test_nanmedian_axis0_ignore_empty_columns():
    arr = np.array(
        [
            [1.0, np.nan, 3.0],
            [2.0, np.nan, 5.0],
            [np.nan, np.nan, 7.0],
        ]
    )
    out = nanmedian_axis0_ignore_empty_columns(arr)
    np.testing.assert_allclose(out, [1.5, np.nan, 5.0], equal_nan=True)

    with pytest.raises(ValueError, match="2-D"):
        nanmedian_axis0_ignore_empty_columns(np.arange(3.0))


def test_solve_chebyshev2d_ols_recovers_constant():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 40)
    y = rng.uniform(-1, 1, 40)
    z = np.full_like(x, 2.5)
    coeffs = solve_chebyshev2d_ols(x, y, z, deg_x=1, deg_y=1)
    assert coeffs.shape == (2, 2)
    assert coeffs[0, 0] == pytest.approx(2.5, abs=1e-10)

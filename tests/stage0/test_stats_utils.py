"""Fast unit tests for pure-Python stats helpers (avoid Numba cold-compile)."""

from __future__ import annotations

import numpy as np

from hispecdrp.primitives.utils.stats import get_err_from_posterior


def test_get_err_from_posterior_returns_finite_interval():
    x = np.linspace(-2.0, 2.0, 401)
    posterior = np.exp(-0.5 * ((x - 0.3) / 0.4) ** 2)
    mode, left, right = get_err_from_posterior(x, posterior)
    assert np.isfinite(mode) and left > 0 and right > 0

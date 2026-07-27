"""Ramp-fit products must mark non-finite rate/rate_err as DO_NOT_USE."""

from __future__ import annotations

import numpy as np

from hispecdrp.datamodels.dqflags import DQ_FLAGS
from hispecdrp.primitives.ramp_fitting.ramp_fit_primitive import flag_nonfinite_rate_pixels


def test_flag_nonfinite_rate_err_sets_do_not_use():
    """NaN rate_err (e.g. empty inter-order pixels) and NaN rate → DO_NOT_USE."""
    ny, nx = 4, 5
    rate = np.zeros((ny, nx), dtype=np.float32)
    rate_err = np.full((ny, nx), 0.1, dtype=np.float32)
    rate[1, 1] = 0.0
    rate_err[1, 1] = np.nan
    rate[2, 2] = np.nan
    rate_err[2, 2] = 0.2
    dq = np.zeros((ny, nx), dtype=np.uint32)

    out_dq, n_bad = flag_nonfinite_rate_pixels(rate, rate_err, dq)
    do_not_use = int(DQ_FLAGS["DO_NOT_USE"])
    assert n_bad == 2
    assert out_dq[1, 1] & do_not_use
    assert out_dq[2, 2] & do_not_use
    assert out_dq[0, 0] == 0
    # Original dq left unchanged when flags are written to a copy.
    assert dq[1, 1] == 0


def test_flag_nonfinite_rate_noop_when_all_finite():
    rate = np.ones((2, 2), dtype=np.float32)
    rate_err = np.full((2, 2), 0.1, dtype=np.float32)
    dq = np.zeros((2, 2), dtype=np.uint32)
    out_dq, n_bad = flag_nonfinite_rate_pixels(rate, rate_err, dq)
    assert n_bad == 0
    assert np.array_equal(out_dq, dq)

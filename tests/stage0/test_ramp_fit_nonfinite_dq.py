"""Ramp-fit products: finite rate_err for empty pixels; DO_NOT_USE only for bad rate."""

from __future__ import annotations

import numpy as np

from hispecdrp.datamodels.dqflags import DQ_FLAGS
from hispecdrp.primitives.ramp_fitting.ramp_fit_primitive import (
    flag_nonfinite_rate_pixels,
    sanitize_ramp_fit_rate_err,
)


def test_sanitize_fills_nan_rate_err_on_finite_rate():
    """Empty pixels (rate=0, NaN err) get finite rate_err=0, not DO_NOT_USE."""
    ny, nx = 4, 5
    rate = np.zeros((ny, nx), dtype=np.float32)
    rate_err = np.full((ny, nx), 0.1, dtype=np.float32)
    rate[1, 1] = 0.0
    rate_err[1, 1] = np.nan
    rate[2, 2] = np.nan
    rate_err[2, 2] = 0.2
    dq = np.zeros((ny, nx), dtype=np.uint32)

    out_err, out_dq, n_filled, n_flagged = sanitize_ramp_fit_rate_err(
        rate, rate_err, dq
    )
    do_not_use = int(DQ_FLAGS["DO_NOT_USE"])
    assert n_filled == 1
    assert n_flagged == 1
    assert np.isfinite(out_err[1, 1]) and out_err[1, 1] == 0.0
    assert (out_dq[1, 1] & do_not_use) == 0
    assert out_dq[2, 2] & do_not_use
    assert out_err[2, 2] == 0.0
    assert dq[1, 1] == 0  # original unchanged


def test_sanitize_uses_rnoise_floor_when_available():
    rate = np.array([[0.0]], dtype=np.float32)
    rate_err = np.array([[np.nan]], dtype=np.float32)
    dq = np.zeros((1, 1), dtype=np.uint32)
    vr = np.array([[0.25]], dtype=np.float32)
    out_err, out_dq, n_filled, n_flagged = sanitize_ramp_fit_rate_err(
        rate, rate_err, dq, rate_var_rnoise=vr
    )
    assert n_filled == 1 and n_flagged == 0
    assert out_err[0, 0] == np.float32(0.5)
    assert out_dq[0, 0] == 0


def test_sanitize_noop_when_all_finite():
    rate = np.ones((2, 2), dtype=np.float32)
    rate_err = np.full((2, 2), 0.1, dtype=np.float32)
    dq = np.zeros((2, 2), dtype=np.uint32)
    out_err, out_dq, n_filled, n_flagged = sanitize_ramp_fit_rate_err(
        rate, rate_err, dq
    )
    assert n_filled == 0 and n_flagged == 0
    assert np.array_equal(out_err, rate_err)
    assert np.array_equal(out_dq, dq)


def test_flag_nonfinite_wrapper_only_flags_bad_rate():
    """Legacy wrapper no longer DO_NOT_USE-flags NaN err on finite rate."""
    rate = np.zeros((2, 2), dtype=np.float32)
    rate_err = np.full((2, 2), np.nan, dtype=np.float32)
    rate[0, 0] = np.nan
    dq = np.zeros((2, 2), dtype=np.uint32)
    out_dq, n_bad = flag_nonfinite_rate_pixels(rate, rate_err, dq)
    do_not_use = int(DQ_FLAGS["DO_NOT_USE"])
    assert n_bad == 1
    assert out_dq[0, 0] & do_not_use
    assert (out_dq[1, 1] & do_not_use) == 0

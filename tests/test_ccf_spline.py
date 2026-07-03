"""Tests for cubic-spline mask CCF utilities."""
import numpy as np

from hispecdrp.primitives.utils.ccf.ccf_spline import (
    compute_ccf_spline,
    cubic_spline_build,
    cubic_spline_eval_scalar,
)
from hispecdrp.utils.physical_constants import C_M_S


def test_cubic_spline_reproduces_linear_function():
    x = np.linspace(0.0, 1.0, 20, dtype=np.float64)
    y = 2.5 * x + 1.0
    xk, a, b, c, d = cubic_spline_build(x, y)
    for xq in np.linspace(x[0], x[-1], 37):
        got = cubic_spline_eval_scalar(xk, a, b, c, d, float(xq))
        assert abs(got - (2.5 * xq + 1.0)) < 1e-10


def test_compute_ccf_spline_peak_near_zero_for_rest_aligned_line():
    wave = np.linspace(0.65, 0.66, 200, dtype=np.float64)
    line_um = 0.655
    spec = np.exp(-0.5 * ((wave - line_um) / 0.00015) ** 2)
    spec_err = np.full_like(wave, 0.01)
    spec_dq = np.zeros(wave.shape[0], dtype=np.int32)

    vel_grid, ccf, ccf_err = compute_ccf_spline(
        wave,
        spec,
        spec_err,
        spec_dq,
        np.array([line_um], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        vel_window=20_000.0,
        vel_step=100.0,
        zb=0.0,
        vsys=0.0,
    )

    peak_v = float(vel_grid[int(np.nanargmax(ccf))])
    assert abs(peak_v) < 500.0
    assert np.nanmax(ccf) > 0.5
    assert np.all(np.isfinite(ccf_err[np.isfinite(ccf)]))


def test_compute_ccf_spline_requires_two_good_pixels():
    wave = np.array([0.65, 0.66], dtype=np.float64)
    spec = np.array([1.0, 1.0], dtype=np.float64)
    spec_err = np.full(2, 0.01)
    spec_dq = np.array([0, 1], dtype=np.int32)

    vel_grid, ccf, ccf_err = compute_ccf_spline(
        wave,
        spec,
        spec_err,
        spec_dq,
        np.array([0.655], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        vel_window=10_000.0,
        vel_step=1_000.0,
    )
    assert np.all(np.isnan(ccf))
    assert np.all(np.isnan(ccf_err))


def test_compute_ccf_spline_shifts_with_injected_velocity():
    wave = np.linspace(0.64, 0.67, 400, dtype=np.float64)
    spec = np.exp(-0.5 * ((wave - 0.655) / 0.0003) ** 2)
    spec_err = np.full_like(wave, 0.01)
    spec_dq = np.zeros(wave.shape[0], dtype=np.int32)
    line_um = 0.655

    vel_inj = 5_000.0
    sh = 1.0 + vel_inj / C_M_S
    spec_shifted = np.interp(wave, wave / sh, spec, left=0.0, right=0.0)

    vel_grid, ccf, _ = compute_ccf_spline(
        wave,
        spec_shifted,
        spec_err,
        spec_dq,
        np.array([line_um], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        vel_window=30_000.0,
        vel_step=200.0,
    )
    peak_v = float(vel_grid[int(np.nanargmax(ccf))])
    assert abs(peak_v + vel_inj) < 800.0

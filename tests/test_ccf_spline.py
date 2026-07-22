"""Tests for cubic-spline mask CCF utilities."""
import numpy as np

from hispecdrp.primitives.utils.ccf.ccf_spline import compute_ccf_spline
from hispecdrp.primitives.utils.ccf.cubic_spline import (
    cubic_spline_build,
    cubic_spline_eval_scalar_arrays,
)
from hispecdrp.utils.physical_constants import C_M_S


def test_cubic_spline_reproduces_linear_function():
    x = np.linspace(0.0, 1.0, 20, dtype=np.float64)
    y = 2.5 * x + 1.0
    xk, a, b, c, d = cubic_spline_build(x, y)
    for xq in np.linspace(x[0], x[-1], 37):
        got = cubic_spline_eval_scalar_arrays(xk, a, b, c, d, float(xq))
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


def test_compute_ccf_spline_fixed_line_set_normalizer():
    """Line set frozen at line_set_velocity; W does not jump when a line leaves."""
    wave = np.linspace(1.0, 1.01, 300, dtype=np.float64)
    # Continuum + two absorption lines near opposite ends of the order.
    lam_a, lam_b = 1.0015, 1.0085
    spec = np.ones_like(wave)
    spec -= 0.4 * np.exp(-0.5 * ((wave - lam_a) / 0.00008) ** 2)
    spec -= 0.4 * np.exp(-0.5 * ((wave - lam_b) / 0.00008) ** 2)
    spec_err = np.full_like(wave, 0.01)
    spec_dq = np.zeros(wave.shape[0], dtype=np.int32)
    centers = np.array([lam_a, lam_b], dtype=np.float64)
    weights = np.array([1.0, 2.0], dtype=np.float64)

    # At RV=0 both lines land; choose a velocity where the redder line leaves.
    v_leave = C_M_S * (wave[-1] / lam_b - 1.0) + 500.0
    vel_grid = np.array([0.0, v_leave], dtype=np.float64)

    _, ccf0, err0 = compute_ccf_spline(
        wave,
        spec,
        spec_err,
        spec_dq,
        centers,
        weights,
        vel_grid=vel_grid,
        line_set_velocity=0.0,
        use_variance=True,
    )
    assert np.all(np.isfinite(ccf0))
    assert np.all(np.isfinite(err0))
    # At v_leave the redder fixed-set line is out → F=0 for that term, but W is
    # still 1+2, so the CCF must differ from a renormalized single-line mean.
    _, ccf_one, _ = compute_ccf_spline(
        wave,
        spec,
        spec_err,
        spec_dq,
        np.array([lam_a], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        vel_grid=np.array([v_leave], dtype=np.float64),
        line_set_velocity=0.0,
        use_variance=True,
    )
    assert abs(float(ccf0[1]) - float(ccf_one[0])) > 1e-6

    # No freeze (scout mode): renormalized single-line mean matches.
    _, ccf_dyn, _ = compute_ccf_spline(
        wave,
        spec,
        spec_err,
        spec_dq,
        centers,
        weights,
        vel_grid=np.array([v_leave], dtype=np.float64),
        line_set_velocity=None,
        use_variance=True,
    )
    assert abs(float(ccf_dyn[0]) - float(ccf_one[0])) < 1e-10

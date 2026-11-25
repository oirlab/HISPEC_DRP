"""Infinite-SNR recovery of one Gaussian line on a single order."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np

from hispecdrp.simulations.gaussian_line_rv import (
    plot_gaussian_line_profile,
    plot_gaussian_line_rv_recovery,
    recover_gaussian_line,
    run_gaussian_line_rv_grid,
    single_order_wave_grid,
)
from hispecdrp.utils.physical_constants import C_M_S

# 1e-11 m/s. Native pixel samples and a free continuum recover the line at
# float64 precision; the largest bias on the grid is a few times 1e-13 m/s.
_MAX_ABS_BIAS_M_S = 1.0e-11


def test_gaussian_line_recovered_at_every_injected_rv(tmp_path):
    """StellarRV returns the injected RV for one perfect Gaussian line."""
    result = run_gaussian_line_rv_grid()
    assert result.injected_m_s.size == 41
    bias = result.bias_m_s
    assert np.all(np.isfinite(bias))
    assert np.all(np.isfinite(result.rv_err_m_s))
    assert np.all(result.rv_err_m_s > 0.0)
    assert np.all(result.rv_err_m_s < 0.01)
    assert float(np.max(np.abs(bias))) < _MAX_ABS_BIAS_M_S
    assert result.v_step_m_s == 1.0
    assert result.v_half_window_m_s == 20_000.0
    assert result.mask_ccf_pixel_interp_order == 8
    assert result.mask_ccf_resample is False
    assert result.mask_ccf_fit_baseline_free is True

    # One pixel of velocity. The centroid stays on the injected RV at every phase.
    wave, lam = single_order_wave_grid()
    pix = float(np.median(np.abs(np.diff(C_M_S * np.log(wave)))))
    for phase in (0.0, 0.25, 0.5, 0.75):
        rv = phase * pix
        rec, err = recover_gaussian_line(rv, wave_um=wave, lam_rest_um=lam)
        assert abs(rec - rv) < _MAX_ABS_BIAS_M_S
        assert np.isfinite(err) and err > 0.0

    # A 2 km/s prior offset must not be echoed back as the answer.
    offset = run_gaussian_line_rv_grid(
        np.array([0.0, 537.0, 10_000.0]),
        prior_offset_m_s=2_000.0,
    )
    off_bias = offset.bias_m_s
    assert float(np.max(np.abs(off_bias))) < _MAX_ABS_BIAS_M_S
    echoed = np.abs(offset.recovered_m_s - (offset.injected_m_s + 2_000.0))
    assert float(np.min(echoed)) > 1_000.0

    recovery = tmp_path / "gaussian_line_rv_recovery.png"
    profile = tmp_path / "gaussian_line_profile.png"
    plot_gaussian_line_rv_recovery(result, recovery)
    plot_gaussian_line_profile(profile)
    assert recovery.is_file() and recovery.stat().st_size > 0
    assert profile.is_file() and profile.stat().st_size > 0

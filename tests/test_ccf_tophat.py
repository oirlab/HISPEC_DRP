"""Tests for top-hat mask CCF utilities."""
import numpy as np

from hispecdrp.primitives.utils.ccf.ccf_tophat import compute_ccf_tophat
from hispecdrp.utils.physical_constants import C_M_S


def test_compute_ccf_tophat_peak_near_zero_for_rest_aligned_mask():
    wave = np.linspace(0.65, 0.66, 200, dtype=np.float64)
    spec = np.zeros_like(wave)
    spec_err = np.full_like(wave, 0.01)
    spec_dq = np.zeros(wave.shape[0], dtype=np.int32)

    line_um = 0.655
    width_m_s = 30_000.0
    half = 0.5 * width_m_s / C_M_S
    spec[(wave >= line_um * (1.0 - half)) & (wave <= line_um * (1.0 + half))] = 1.0

    vel_grid, ccf, ccf_err = compute_ccf_tophat(
        wave,
        spec,
        spec_err,
        spec_dq,
        np.array([line_um], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        tophat_width=width_m_s,
        vel_window=20_000.0,
        vel_step=100.0,
        zb=0.0,
        vsys=0.0,
    )

    idx0 = int(np.argmin(np.abs(vel_grid)))
    assert ccf[idx0] > 0.99
    assert ccf[idx0] >= np.nanmax(ccf) - 1e-6
    assert np.all(np.isfinite(ccf_err[np.isfinite(ccf)]))


def test_compute_ccf_tophat_skips_bad_dq_pixels():
    wave = np.linspace(0.65, 0.66, 50, dtype=np.float64)
    spec = np.ones_like(wave)
    spec_err = np.full_like(wave, 0.01)
    spec_dq = np.zeros(wave.shape[0], dtype=np.int32)
    spec_dq[25] = 1

    vel_grid, ccf, _ = compute_ccf_tophat(
        wave,
        spec,
        spec_err,
        spec_dq,
        np.array([0.655], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        tophat_width=30_000.0,
        vel_window=10_000.0,
        vel_step=1_000.0,
    )
    assert np.all(np.isnan(ccf)) or not np.any(np.isfinite(ccf))

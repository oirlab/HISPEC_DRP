"""Empirical ProcessLFC LSF → pixel-σ mapping for ccfopt mask builds."""

from __future__ import annotations

import numpy as np

from hispecdrp.simulations.lsf import sigma_pix_from_gaussian_lsf_model


class _FakeLSF:
    def __init__(self, fwhm_wave: np.ndarray):
        self.lsf_params_fib1 = fwhm_wave


def test_sigma_pix_clamps_zero_dispersion_edges():
    n_ord, n_pix = 2, 64
    wave = np.linspace(1.2, 1.25, n_pix)[None, :] * np.ones((n_ord, 1))
    wave[0, 0] = wave[0, 1]  # flat edge → zero |dλ/dpix|
    fwhm = np.full((n_ord, n_pix), 8.0e-6)
    lsf = _FakeLSF(fwhm)
    sig = sigma_pix_from_gaussian_lsf_model(lsf, wave, fiber=1)
    assert sig.shape == wave.shape
    assert np.all(np.isfinite(sig))
    assert float(np.nanmax(sig)) <= 20.0
    assert float(np.nanmin(sig)) >= 0.25


def test_sigma_pix_slices_full_band_lsf_to_order_subset():
    from hispecdrp.simulations.lsf import _slice_lsf_fwhm_to_wave

    n_full, n_pix = 5, 32
    wave_one = np.linspace(1.20, 1.21, n_pix)[None, :]
    fwhm = np.arange(1, n_full + 1, dtype=np.float64)[:, None] * np.ones((1, n_pix))
    # Request echelle order 112 with LSF starting at 110 → row index 2.
    sliced = _slice_lsf_fwhm_to_wave(
        fwhm,
        wave_one,
        fiber=1,
        order_start=112,
        lsf_order_start=110,
    )
    assert sliced.shape == wave_one.shape
    assert np.allclose(sliced, fwhm[2:3])

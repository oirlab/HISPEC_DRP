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

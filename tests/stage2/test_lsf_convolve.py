"""Unit tests for HDF→1D LSF convolution utilities."""

from __future__ import annotations

import numpy as np

from hispecdrp.primitives.lsf.lsf_convolve import (
    convolve_order_spectrum_gaussian_lsf,
    lsf_1d_from_psf_stamp,
)


def test_lsf_1d_from_psf_stamp_normalizes_and_centers():
    oversample = 4.0
    h, w = 21, 25
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = 0.5 * (h - 1), 0.5 * (w - 1)
    stamp = np.exp(-0.5 * (((yy - cy) / 2.0) ** 2 + ((xx - cx) / 3.0) ** 2))
    offs, ker = lsf_1d_from_psf_stamp(stamp, oversample=oversample)
    assert offs.size == ker.size
    assert abs(float(np.sum(ker)) - 1.0) < 1e-12
    # Second moment finite and positive
    mu = float(np.sum(offs * ker))
    var = float(np.sum(ker * (offs - mu) ** 2))
    assert var > 0.0
    assert abs(mu) < 0.2  # near stamp center


def test_convolve_order_spectrum_conserves_flat_flux():
    wave = np.linspace(1.0, 1.05, 401)
    flux = np.ones_like(wave)
    sigma = np.full_like(wave, 1.5)
    out = convolve_order_spectrum_gaussian_lsf(wave, flux, sigma)
    # Interior should remain ~1 after Gaussian convolution of a flat spectrum.
    assert abs(float(np.nanmedian(out[20:-20])) - 1.0) < 1e-3


def test_convolve_deepens_line_when_comparing_to_delta_like_core():
    """Broader LSF shallows a narrow absorption line (depth_ratio < 1 vs native)."""
    wave = np.linspace(1.0, 1.02, 801)
    x = np.arange(wave.size, dtype=np.float64)
    native = 1.0 - 0.8 * np.exp(-0.5 * ((x - 400) / 0.5) ** 2)
    sigma = np.full_like(wave, 4.0)
    lsf = convolve_order_spectrum_gaussian_lsf(wave, native, sigma)
    depth_n = 1.0 - float(np.min(native))
    depth_l = 1.0 - float(np.min(lsf))
    assert depth_l < 0.95 * depth_n


def test_convolve_order_spectrum_accepts_decreasing_wave_with_plateau():
    wave = np.linspace(1.05, 1.0, 401)
    wave[-1] = wave[-2]
    flux = np.ones_like(wave)
    sigma = np.full_like(wave, 1.5)
    out = convolve_order_spectrum_gaussian_lsf(wave, flux, sigma)
    assert np.all(np.isfinite(out))
    assert abs(float(np.nanmedian(out[20:-20])) - 1.0) < 1e-3


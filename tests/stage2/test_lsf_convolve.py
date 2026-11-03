"""Unit tests for generic LSF convolution utilities."""

from __future__ import annotations

import numpy as np
import pytest

from hispecdrp.primitives.lsf.lsf_convolve import (
    convolve_l2_with_gaussian_lsf,
    convolve_numba_1d_chromatic,
    convolve_order_spectrum_gaussian_lsf,
    lsf_1d_from_psf_stamp,
    lsf_library_from_psf_fiber_block,
    sigma_pix_from_lsf_library,
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


def test_lsf_convolve_source_has_no_simulations_import():
    import ast
    from pathlib import Path

    src = Path(lsf_1d_from_psf_stamp.__code__.co_filename).read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "simulations" not in alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert "simulations" not in node.module


def test_lsf_library_and_sigma_from_fiber_block():
    oversample = 4.0
    h, w = 11, 15
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = 0.5 * (h - 1), 0.5 * (w - 1)
    stamp = np.exp(-0.5 * (((yy - cy) / 1.5) ** 2 + ((xx - cx) / 2.0) ** 2))
    block = {
        110: {
            "psf_stamps": [stamp, stamp],
            "psf_wavelengths": np.array([1.0, 1.1]),
            "psf_data_spacing": 1.0 / oversample,
        }
    }
    lib = lsf_library_from_psf_fiber_block(block, pixelsize=1.0)
    assert 110 in lib
    assert lib[110]["wavelengths"].size == 2
    wave = np.linspace(1.0, 1.1, 8)
    sig = sigma_pix_from_lsf_library(lib, 110, wave)
    assert sig.shape == wave.shape
    assert np.all(np.isfinite(sig))
    assert np.all(sig > 0)
    missing = sigma_pix_from_lsf_library(lib, 999, wave)
    assert np.allclose(missing, 1.0)


def test_convolve_l2_with_gaussian_lsf_matches_1d():
    wave = np.linspace(1.0, 1.05, 101)
    flux = np.ones_like(wave)
    sigma = np.full_like(wave, 1.5)
    wave_2d = np.vstack([wave, wave])
    flux_2d = np.vstack([flux, 2.0 * flux])
    sigma_2d = np.vstack([sigma, sigma])
    out = convolve_l2_with_gaussian_lsf(wave_2d, flux_2d, sigma_2d)
    one = convolve_order_spectrum_gaussian_lsf(wave, flux, sigma)
    assert out.shape == flux_2d.shape
    assert np.allclose(out[0], one)
    assert np.allclose(out[1], 2.0 * one, rtol=1e-6, atol=1e-6)


def test_make_kernel_grid_accepts_negative_fwhm():
    """HISPEC WAVE decreases along the detector, so signed FWHM_WAVE is < 0."""
    from hispecdrp.datamodels.gausslsf import _make_gaussians_chromatic, _make_kernel_grid
    from hispecdrp.primitives.lsf.lsf_convolve import convolve_numba_1d_chromatic

    dw = 2.8e-6
    fwhms = np.full(32, -8.2e-6)
    kg = _make_kernel_grid(dw, fwhms, 5.0)
    assert kg.size >= 1
    assert kg.size % 2 == 1
    kernels = _make_gaussians_chromatic(kg, fwhms)
    assert kernels.shape == (32, kg.size)
    spec = np.ones(32)
    out = convolve_numba_1d_chromatic(spec, kernels)
    assert out.shape == spec.shape
    assert np.all(np.isfinite(out))


def test_convolve_numba_1d_chromatic_identity_delta_kernel():
    spec = np.linspace(1.0, 2.0, 21)
    kernels = np.zeros((spec.size, 5), dtype=np.float32)
    kernels[:, 2] = 1.0
    out = convolve_numba_1d_chromatic(spec, kernels)
    assert np.allclose(out, spec)


def test_fit_gausslsf_chebyshev2d_rejects_empty_table():
    from hispecdrp.primitives.lsf.lsf_gauss_primitive import fit_gausslsf_chebyshev2d

    empty = np.array([])
    with pytest.raises(ValueError, match="No usable LFC FWHM"):
        fit_gausslsf_chebyshev2d(
            empty,
            empty,
            empty,
            deg_intra_order=3,
            deg_inter_order=3,
            max_iters=2,
            sigma_cut=4.0,
        )


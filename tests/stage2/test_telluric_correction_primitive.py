"""Tests for ``TelluricCorrectionPrimitive``.
"""
import time

import numpy as np

from hispecdrp import datamodels, TelluricCorrectionPrimitive
from hispecdrp.primitives.lsf.lsf_convolve import convolve_numba_1d_chromatic
from hispecdrp.primitives.telluric_correction.telluric_models import (
    _convolve_telluric_with_gaussian_lsf,
)

from tests.utils import repo_root


def _constant_fwhm_lsf_model(wave_2d, *, order_start, order_end, fwhm_um, fiber=1):
    lsf_model = datamodels.GaussianLSFModel(
        meta={
            "order_start": order_start,
            "order_end": order_end,
            "spectrograph": "BSPEC",
            "instrument_name": "HISPEC",
        }
    )
    lsf_model.meta.order_start = order_start
    lsf_model.meta.order_end = order_end
    setattr(lsf_model, f"WAVE_FIB{fiber}", np.asarray(wave_2d, dtype=np.float64))
    setattr(lsf_model, f"CHEB_COEFFS_FIB{fiber}", np.array([[float(fwhm_um)]], dtype=np.float64))
    n_pix = int(np.asarray(wave_2d).shape[1])
    getattr(lsf_model, f"cheb_norm_fib{fiber}").add_row(
        {
            "FIBER": str(fiber),
            "PIXEL_NORM_MIN": 0.0,
            "PIXEL_NORM_MAX": float(n_pix - 1),
            "ORDER_NORM_MIN": int(order_start),
            "ORDER_NORM_MAX": int(order_end),
        }
    )
    return lsf_model


def test_telluric_correction_primitive():
    """Exercise TelluricCorrectionPrimitive.call on a compact L1 + LSF stack."""
    start_time = time.time()

    data_dir = repo_root() / "hispecdrp" / "data" / "atmosphere"
    telluric_basemodel_filepath = data_dir / "mktrans_zm_10_10.dat"

    telluric_data = np.loadtxt(telluric_basemodel_filepath)
    telluric_wavelength = telluric_data[:, 0]  # microns
    telluric_trans = telluric_data[:, 1]

    n_orders, n_pix = 4, 512
    order_start, order_end = 120, 123
    sci_flux = 100.0
    fwhm_um = 1.5 / 9.0e4
    science_model = datamodels.L1Model(
        meta={
            "order_start": order_start,
            "order_end": order_end,
            "spectrograph": "BSPEC",
            "instrument_name": "HISPEC",
        }
    )
    science_model.meta.order_start = order_start
    science_model.meta.order_end = order_end
    science_model.meta.object_fiber1 = "SCIENCE"
    science_model.meta.airmass_start = 1.1
    science_model.meta.airmass_end = 1.11

    avg_airmass = np.nanmean(
        [science_model.meta.airmass_start, science_model.meta.airmass_end]
    )

    wv = np.empty((n_orders, n_pix), dtype=np.float64)
    for i, order in enumerate(range(order_start, order_end + 1)):
        lam_c = 147.0 / float(order)
        wv[i] = np.linspace(lam_c * 0.99, lam_c * 1.01, n_pix)

    lsf_model = _constant_fwhm_lsf_model(
        wv, order_start=order_start, order_end=order_end, fwhm_um=fwhm_um, fiber=1
    )

    for fiber in range(1, 4):
        interp_flux = np.ones((n_orders, n_pix), dtype=np.float32) * sci_flux
        if fiber == 1:
            for i, order in enumerate(range(order_start, order_end + 1)):
                tel_conv = _convolve_telluric_with_gaussian_lsf(
                    telluric_wavelength,
                    telluric_trans**avg_airmass,
                    wv[i],
                    lsf_model,
                    fiber_number=1,
                    order=order,
                )
                interp_flux[i] *= tel_conv
        setattr(science_model, f"WAVE_FIB{fiber}", wv)
        setattr(science_model, f"SPEC_UNCAL_FIB{fiber}", interp_flux.astype(np.float32))
        setattr(
            science_model,
            f"SPEC_UNCAL_ERR_FIB{fiber}",
            np.ones((n_orders, n_pix)) * np.sqrt(sci_flux),
        )
        setattr(
            science_model,
            f"SPEC_UNCAL_VAR_POISSON_FIB{fiber}",
            np.zeros((n_orders, n_pix)),
        )
        setattr(
            science_model,
            f"SPEC_UNCAL_VAR_RNOISE_FIB{fiber}",
            np.zeros((n_orders, n_pix)),
        )
        setattr(
            science_model,
            f"SPEC_UNCAL_DQ_FIB{fiber}",
            np.zeros((n_orders, n_pix), dtype=np.uint32),
        )

    telluric_corrected = TelluricCorrectionPrimitive.call(
        science_model, lsf=lsf_model, show_preview=False, save_preview=False
    )

    step_time = time.time()
    print(f"Telluric correction, elapsed: {step_time - start_time:.2f}s")

    spec = np.asarray(getattr(telluric_corrected, "SPEC_FIB1"), dtype=float)
    rel = np.abs((spec - sci_flux) / sci_flux)
    interior = rel[:, 16:-16]
    assert np.nanmedian(interior) < 1e-4
    assert float(np.mean(interior < 1e-3)) > 0.95


def test_make_kernels_and_convolve_numba_1d_chromatic():
    wave = np.linspace(1.20, 1.21, 401)
    flux = np.ones_like(wave)
    flux[198:204] = 0.0
    lsf = _constant_fwhm_lsf_model(
        wave[None, :],
        order_start=120,
        order_end=120,
        fwhm_um=1.0e-4,
        fiber=1,
    )
    kernel_grid, kernels = lsf.make_kernels(
        wave=wave, fiber=1, order=120, kernel_size_sigma=5.0
    )
    assert kernel_grid.ndim == 1
    assert kernels.shape == (wave.size, kernel_grid.size)
    assert kernels.shape[1] % 2 == 1
    assert np.allclose(kernels.sum(axis=1), 1.0, atol=1e-5)
    fluxc = convolve_numba_1d_chromatic(flux, kernels)
    assert fluxc.shape == flux.shape
    assert float(np.min(fluxc)) > float(np.min(flux))

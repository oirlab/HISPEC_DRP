"""Tests for ``TelluricCorrectionPrimitive``.
"""
import numpy as np
from hispecdrp import datamodels, TelluricCorrectionPrimitive
from hispecdrp.primitives.telluric_correction.telluric_models import _convolve_and_resample
import time
from pathlib import Path

from tests.utils import repo_root


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
    variable_lsf = np.linspace(1.5 / 9e4 / 2.355, 1.5 / 1.1e5 / 2.355, n_pix)

    for fiber in range(1, 4):
        wv = np.empty((n_orders, n_pix), dtype=np.float32)
        interp_flux = np.ones((n_orders, n_pix), dtype=np.float32) * sci_flux
        for i, order in enumerate(range(order_start, order_end + 1)):
            lam_c = 147.0 / float(order)
            wv[i] = np.linspace(lam_c * 0.99, lam_c * 1.01, n_pix, dtype=np.float32)
            tel_conv = _convolve_and_resample(
                telluric_wavelength,
                telluric_trans**avg_airmass,
                wv[i],
                variable_lsf,
            )
            interp_flux[i] *= tel_conv
        setattr(science_model, f"WAVE_FIB{fiber}", wv)
        setattr(science_model, f"SPEC_UNCAL_FIB{fiber}", interp_flux)
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
            f"SPEC_DQ_FIB{fiber}",
            np.zeros((n_orders, n_pix), dtype=np.uint32),
        )

    lsf_model = datamodels.GaussianLSFModel()
    lsf_params = np.empty((n_orders, n_pix, 3))
    lsf_params[:, :, 0] = 0.0
    lsf_params[:, :, 1] = variable_lsf
    lsf_params[:, :, 2] = 0.0
    setattr(lsf_model, "LSF_PARAMS_FIB1", lsf_params)

    telluric_corrected = TelluricCorrectionPrimitive.call(
        science_model, lsf=lsf_model, show_preview=False, save_preview=False
    )

    step_time = time.time()
    print(f"Telluric correction, elapsed: {step_time - start_time:.2f}s")

    assert ((getattr(telluric_corrected, "SPEC_FIB1") - sci_flux) / sci_flux).all() < 1e-4

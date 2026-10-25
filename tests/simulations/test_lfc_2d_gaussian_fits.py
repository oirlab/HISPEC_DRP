"""Tests for astropy Gaussian2D LFC spot fitting helpers."""

from __future__ import annotations

import numpy as np
import pytest

from hispecdrp.simulations.e2e_pipeline import (
    _canonicalize_ellipse_ab_pa,
    fit_lfc_spots_astropy_gaussian2d,
    lfc_fitparams_from_2d_gaussian_fits,
    paint_elliptical_gaussian_comb,
)


def test_canonicalize_ellipse_swaps_axes():
    a, b, pa = _canonicalize_ellipse_ab_pa(0.5, 1.2, 0.1)
    assert a >= b
    assert abs(a - 1.2) < 1e-12
    assert abs(b - 0.5) < 1e-12


def test_fit_lfc_spots_astropy_gaussian2d_recovers_centroids():
    line_list = {
        129: np.array([1.1300, 1.1310, 1.1320]),
    }
    a_true, b_true, pa_true = 1.2, 0.7, 0.4
    rate = paint_elliptical_gaussian_comb(
        line_list,
        spectrograph="BSPEC",
        fiber=2,
        a_pix=a_true,
        b_pix=b_true,
        pa_rad=pa_true,
        peak_value=1.0e6,
    )
    fits = fit_lfc_spots_astropy_gaussian2d(
        rate,
        line_list,
        spectrograph="BSPEC",
        fiber=2,
        half_win=10,
    )
    assert len(fits) == 3
    assert all(f["success"] for f in fits)
    drow = np.array([f["row_fit"] - f["row_expected"] for f in fits])
    dcol = np.array([f["col_fit"] - f["col_expected"] for f in fits])
    assert np.nanmax(np.abs(drow)) < 0.05
    assert np.nanmax(np.abs(dcol)) < 0.05
    a_fit = np.array([f["a_fit"] for f in fits])
    b_fit = np.array([f["b_fit"] for f in fits])
    assert np.nanmedian(np.abs(a_fit - a_true)) < 0.1
    assert np.nanmedian(np.abs(b_fit - b_true)) < 0.1


def test_lfc_fitparams_from_2d_gaussian_fits_schema():
    fits = [
        {
            "order": 129,
            "lam_true_um": 1.13,
            "row_expected": 2000.0,
            "col_expected": 2280.0,
            "row_fit": 2000.1,
            "col_fit": 2280.05,
            "a_fit": 1.5,
            "b_fit": 0.8,
            "pa_fit_rad": 0.3,
            "amp_fit": 1.0e5,
            "row_err": 0.01,
            "col_err": 0.01,
            "success": True,
            "residual_rms": 10.0,
            "n_pix": 100,
        },
        {
            "order": 129,
            "lam_true_um": 1.131,
            "row_expected": 2100.0,
            "col_expected": 2280.0,
            "row_fit": np.nan,
            "col_fit": np.nan,
            "a_fit": np.nan,
            "b_fit": np.nan,
            "pa_fit_rad": np.nan,
            "amp_fit": np.nan,
            "row_err": np.nan,
            "col_err": np.nan,
            "success": False,
            "residual_rms": np.nan,
            "n_pix": 0,
        },
    ]
    # nu0 / delta_nu in Hz (HISPECConfig convention)
    tbl = lfc_fitparams_from_2d_gaussian_fits(
        fits, nu0=5.22e9, delta_nu=20e9
    )
    assert len(tbl) == 1
    for col in (
        "MU_PIX",
        "WAVE",
        "ORDER",
        "MODE_NUM",
        "AMPLITUDE",
        "FWHM_PIX",
        "MU_ERR_WAVE",
    ):
        assert col in tbl.colnames
    assert np.isfinite(tbl["MU_ERR_WAVE"][0])
    assert tbl["MU_ERR_WAVE"][0] > 0
    assert abs(tbl["MU_PIX"][0] - 2000.1) < 1e-9

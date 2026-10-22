"""Failed LFC Gaussian fits must not crash ``FitLFCLinesPrimitive``."""

from __future__ import annotations

import numpy as np
import scipy.constants

from hispecdrp import FitLFCLinesPrimitive, datamodels
from hispecdrp.primitives.line_fitting.fit_lfc_lines_primitive import (
    _drop_duplicate_lfc_modes,
    _unusable_lfc_line_fit_reason,
)
from hispecdrp.primitives.line_fitting.fit_lines import (
    GaussianLineFitResult,
    fit_line_gauss,
)


def _fit(**kwargs) -> GaussianLineFitResult:
    base = dict(
        amplitude=200.0,
        amplitude_err=5.0,
        mu=100.0,
        mu_err=0.05,
        fwhm=3.5,
        fwhm_err=0.08,
        background_coeffs=np.array([0.0]),
        background_coeffs_err=np.array([0.0]),
        redchi2=1.0,
    )
    base.update(kwargs)
    return GaussianLineFitResult(**base)


def test_drop_duplicate_lfc_modes_keeps_strongest():
    from astropy.table import QTable

    table = QTable(
        {
            "ORDER": [118, 118, 118, 119],
            "MODE_NUM": [2464, 2464, 2465, 2400],
            "MU_PIX": [268.1, 277.9, 313.4, 100.0],
            "AMPLITUDE": [900.0, 400.0, 800.0, 700.0],
        }
    )
    out = _drop_duplicate_lfc_modes(table)
    assert len(out) == 3
    kept = {(int(o), int(m)): float(mu) for o, m, mu in
            zip(out["ORDER"], out["MODE_NUM"], out["MU_PIX"])}
    # Stronger 268.1 fit wins over the 277.9 duplicate of mode 2464.
    assert kept[(118, 2464)] == 268.1
    assert kept[(118, 2465)] == 313.4
    assert kept[(119, 2400)] == 100.0


def test_unusable_lfc_line_fit_skips_collapsed_and_bound_hits():
    bounds = (2.0, 6.0)
    assert _unusable_lfc_line_fit_reason(_fit(), fwhm_bounds=bounds) is None
    assert _unusable_lfc_line_fit_reason(_fit(mu=np.nan), fwhm_bounds=bounds)
    assert _unusable_lfc_line_fit_reason(_fit(amplitude=0.0), fwhm_bounds=bounds)
    assert _unusable_lfc_line_fit_reason(_fit(fwhm=6.0), fwhm_bounds=bounds)
    assert _unusable_lfc_line_fit_reason(_fit(fwhm_err=25.0), fwhm_bounds=bounds)


def test_fit_line_gauss_masks_zero_and_nan_errors():
    x = np.arange(21, dtype=float)
    y = np.exp(-0.5 * ((x - 10.0) / 1.2) ** 2)
    y_err = np.full_like(y, 0.05)
    y_err[0] = 0.0
    y_err[1] = np.nan
    dq = np.zeros(21, dtype=np.uint32)
    out = fit_line_gauss(
        x, y, y_err, dq, mu_guess=10.0, fwhm_bounds=(1.0, 8.0)
    )
    assert np.isfinite(out.mu)
    assert abs(out.mu - 10.0) < 0.5


def test_fit_lfc_lines_skips_nan_centroids():
    n_pix = 256
    pix = np.arange(n_pix, dtype=float)
    wave = np.linspace(1.50, 1.48, n_pix)  # HISPEC-like decreasing WAVE
    spec = np.zeros(n_pix, dtype=np.float32)
    spec_err = np.full(n_pix, 0.05, dtype=np.float32)
    # Interior peaks, including the first/last detected in the order.
    centers = (40.0, 80.0, 160.0, 220.0)
    for mu in centers:
        spec += 200.0 * np.exp(-0.5 * ((pix - mu) / 1.1) ** 2).astype(np.float32)
    spec_err[10:25] = 0.0
    spec_err[30:40] = np.nan

    l2 = datamodels.L2Model(
        meta={
            "instrument_name": "HISPEC",
            "spectrograph": "BSPEC",
            "order_start": 110,
            "order_end": 110,
        }
    )
    l2[2].spec = spec[None, :].copy()
    l2[2].spec_err = spec_err[None, :].copy()
    l2[2].spec_dq = np.zeros((1, n_pix), dtype=np.uint32)
    l2[2].wave = wave[None, :].copy()

    nu0 = scipy.constants.c / float(wave[80]) * 1.0e6
    delta_nu = 1.0e10
    out = FitLFCLinesPrimitive.call(
        l2,
        do_orders=[110],
        do_fibers=["2"],
        lfc_nu0=nu0,
        lfc_delta_nu=delta_nu,
        min_line_spacing=20,
        save_preview=False,
        show_preview=False,
        save_result=False,
    )
    table = out.lfc_fitparams_fib2
    assert len(table) >= 1
    assert np.all(np.isfinite(np.asarray(table["MU_PIX"])))
    assert np.all(np.asarray(table["FWHM_WAVE"], dtype=float) > 0.0)

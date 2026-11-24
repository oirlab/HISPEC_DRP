"""Optional per-order cubic-spline WAVE model for ``CalculateChebyshev2DWLSPrimitive``."""

from __future__ import annotations

import numpy as np
import pytest

from hispecdrp import datamodels
from hispecdrp.primitives.wavesol.wavesol_cheb2d_primitive import (
    DEFAULT_DEG_INTRA_ORDER,
    SOURCE_CONFIG,
    WAVE_MODEL_CHEBYSHEV2D,
    WAVE_MODEL_CUBIC_SPLINE_1D,
    WAVE_MODELS,
    CalculateChebyshev2DWLSPrimitive,
    auto_deg_inter_order,
    fit_wavesol_chebyshev2d,
    fit_wavesol_spline_1d,
)


def _line_table_model(n_pix: int = 128):
    pix = np.arange(n_pix, dtype=float)
    wave0 = 1.20 + 1.0e-5 * (pix - 60.0) + 8.0e-10 * (pix - 60.0) ** 3
    wave1 = 1.22 + 1.1e-5 * (pix - 60.0) + 7.0e-10 * (pix - 60.0) ** 3
    truth = np.vstack([wave0, wave1])

    l2 = datamodels.L2Model(
        meta={
            "instrument_name": "HISPEC",
            "spectrograph": "BSPEC",
            "order_start": 110,
            "order_end": 111,
        }
    )
    l2[2].spec = np.ones((2, n_pix), dtype=np.float32)
    l2[2].spec_err = np.full((2, n_pix), 0.05, dtype=np.float32)
    l2[2].spec_dq = np.zeros((2, n_pix), dtype=np.uint32)
    l2[2].wave = truth.astype(np.float64)

    model = datamodels.LFCFitParamsModel.from_datamodels(
        l2, include_data=True, ancestor_rule="none"
    )
    table = model.lfc_fitparams_fib2
    xs = np.linspace(12.0, 110.0, 12)
    for oi, order in enumerate((110, 111)):
        for j, x in enumerate(xs):
            y = float(np.interp(x, pix, truth[oi]))
            table.add_row(
                {
                    "ORDER": order,
                    "AMPLITUDE": 100.0,
                    "AMPLITUDE_ERR": 1.0,
                    "MU_PIX": x,
                    "MU_ERR_PIX": 0.01,
                    "MU_ERR_WAVE": 1.0e-7,
                    "FWHM_PIX": 3.0,
                    "FWHM_ERR_PIX": 0.1,
                    "FWHM_WAVE": 1.0e-4,
                    "FWHM_ERR_WAVE": 1.0e-6,
                    "MODE_NUM": j,
                    "WAVE": y,
                }
            )
    return model, truth, xs


def test_default_wave_model_is_chebyshev2d_for_every_source():
    for name, cfg in SOURCE_CONFIG.items():
        assert cfg["defaults"]["wave_model"] in WAVE_MODELS, name
    assert SOURCE_CONFIG["lfc"]["defaults"]["wave_model"] == WAVE_MODEL_CHEBYSHEV2D


def test_fit_wavesol_spline_1d_reproduces_line_table():
    pixels = np.array([10.0, 30.0, 50.0, 70.0, 90.0, 110.0])
    waves = 1.5 + 1.0e-5 * (pixels - 60.0) + 5.0e-10 * (pixels - 60.0) ** 3
    orders = np.full(pixels.size, 110)
    pix = np.arange(128, dtype=float)
    out = fit_wavesol_spline_1d(pixels, orders, waves, pix, np.array([110]))
    at_lines = np.interp(pixels, pix, out["wave_grid"][0])
    assert np.allclose(at_lines, waves, rtol=0, atol=1.0e-12)
    assert out["wave_model"] == WAVE_MODEL_CUBIC_SPLINE_1D


def test_cubic_spline_1d_option_fills_wave_fib(tmp_path):
    model, truth, xs = _line_table_model()
    out = CalculateChebyshev2DWLSPrimitive.call(
        model,
        wave_model=WAVE_MODEL_CUBIC_SPLINE_1D,
        do_fibers=["2"],
        fit_orders=[110, 111],
        output_dir=tmp_path,
        save_result=False,
        save_preview=False,
        show_preview=False,
    )
    from scipy.interpolate import CubicSpline

    wave = np.asarray(out.wave_fib2, dtype=float)
    assert wave.shape == truth.shape
    pix = np.arange(truth.shape[1], dtype=float)
    for oi in (0, 1):
        ok = np.isfinite(wave[oi])
        fn = CubicSpline(pix[ok], wave[oi, ok], extrapolate=False)
        expect = np.interp(xs, pix, truth[oi])
        assert np.allclose(fn(xs), expect, rtol=0, atol=1.0e-12)
    interior = (pix >= 12) & (pix <= 110)
    assert np.all(np.isfinite(wave[:, interior]))


def test_spline_flags_boundary_intervals_unreliable(tmp_path):
    model, _truth, _xs = _line_table_model()
    flag = int(datamodels.DQ_FLAGS["UNRELIABLE_WS"])
    out = CalculateChebyshev2DWLSPrimitive.call(
        model,
        wave_model=WAVE_MODEL_CUBIC_SPLINE_1D,
        do_fibers=["2"],
        fit_orders=[110, 111],
        output_dir=tmp_path,
        save_result=False,
        save_preview=False,
        show_preview=False,
    )
    dq = np.asarray(out.spec_dq_fib2)
    # Lines at 12..110 in steps of ~8.9 pix: first interval ~12-21 and
    # last interval ~101-110 are boundary segments and must be flagged,
    # while mid-order pixels must not.
    assert dq[0, 15] & flag
    assert dq[0, 105] & flag
    assert not dq[0, 60] & flag

    cheb = CalculateChebyshev2DWLSPrimitive.call(
        model,
        deg_intra_order=3,
        deg_inter_order=1,
        sigma_cut=float("inf"),  # noise-free data: don't clip on round-off residuals
        do_fibers=["2"],
        fit_orders=[110, 111],
        output_dir=tmp_path,
        save_result=False,
        save_preview=False,
        show_preview=False,
    )
    dq_cheb = np.asarray(cheb.spec_dq_fib2)
    assert not dq_cheb[0, 15] & flag


def test_default_wave_model_still_writes_chebyshev_wave(tmp_path):
    model, truth, _xs = _line_table_model()
    out = CalculateChebyshev2DWLSPrimitive.call(
        model,
        deg_intra_order=3,
        deg_inter_order=1,
        sigma_cut=float("inf"),  # noise-free data: don't clip on round-off residuals
        do_fibers=["2"],
        fit_orders=[110, 111],
        output_dir=tmp_path,
        save_result=False,
        save_preview=False,
        show_preview=False,
    )
    wave = np.asarray(out.wave_fib2, dtype=float)
    assert wave.shape == truth.shape
    assert np.any(np.isfinite(wave))
    assert getattr(out, "cheb_coeffs_fib2").shape == (4, 2)


def test_auto_deg_inter_order_caps_with_minimum_not_axis():
    assert auto_deg_inter_order(1) == 0
    assert auto_deg_inter_order(2) == 1
    assert auto_deg_inter_order(10) == 6
    assert auto_deg_inter_order(40) == 10


def test_default_degrees_fit_two_orders(tmp_path):
    model, truth, _xs = _line_table_model()
    out = CalculateChebyshev2DWLSPrimitive.call(
        model,
        do_fibers=["2"],
        fit_orders=[110, 111],
        output_dir=tmp_path,
        save_result=False,
        save_preview=False,
        show_preview=False,
    )
    coeffs = np.asarray(out.cheb_coeffs_fib2)
    assert coeffs.shape == (DEFAULT_DEG_INTRA_ORDER + 1, 2)
    assert np.any(np.isfinite(out.wave_fib2))


def _many_order_line_table(n_orders=8, n_pix=128, n_lines=16):
    order_start = 110
    orders = np.arange(order_start, order_start + n_orders)
    pix = np.arange(n_pix, dtype=float)
    truth = np.vstack(
        [
            1.20 + 0.01 * i + 1.0e-5 * (pix - 60.0) + 8.0e-10 * (pix - 60.0) ** 3
            for i in range(n_orders)
        ]
    )
    l2 = datamodels.L2Model(
        meta={
            "instrument_name": "HISPEC",
            "spectrograph": "BSPEC",
            "order_start": int(orders[0]),
            "order_end": int(orders[-1]),
        }
    )
    l2[2].spec = np.ones((n_orders, n_pix), dtype=np.float32)
    l2[2].spec_err = np.full((n_orders, n_pix), 0.05, dtype=np.float32)
    l2[2].spec_dq = np.zeros((n_orders, n_pix), dtype=np.uint32)
    l2[2].wave = truth.astype(np.float64)
    model = datamodels.LFCFitParamsModel.from_datamodels(
        l2, include_data=True, ancestor_rule="none"
    )
    table = model.lfc_fitparams_fib2
    xs = np.linspace(12.0, 110.0, n_lines)
    for oi, order in enumerate(orders):
        for j, x in enumerate(xs):
            y = float(np.interp(x, pix, truth[oi]))
            table.add_row(
                {
                    "ORDER": int(order),
                    "AMPLITUDE": 100.0,
                    "AMPLITUDE_ERR": 1.0,
                    "MU_PIX": x,
                    "MU_ERR_PIX": 0.01,
                    "MU_ERR_WAVE": 1.0e-7,
                    "FWHM_PIX": 3.0,
                    "FWHM_ERR_PIX": 0.1,
                    "FWHM_WAVE": 1.0e-4,
                    "FWHM_ERR_WAVE": 1.0e-6,
                    "MODE_NUM": j,
                    "WAVE": y,
                }
            )
    return model, orders


def test_default_auto_inter_order_with_many_orders(tmp_path):
    """Regression for ``np.min(..., max_degree)`` parsed as ``axis=max_degree``."""
    model, orders = _many_order_line_table()
    out = CalculateChebyshev2DWLSPrimitive.call(
        model,
        do_fibers=["2"],
        fit_orders=list(orders),
        output_dir=tmp_path,
        save_result=False,
        save_preview=False,
        show_preview=False,
    )
    expect_inter = auto_deg_inter_order(len(orders))
    coeffs = np.asarray(out.cheb_coeffs_fib2)
    assert coeffs.shape == (DEFAULT_DEG_INTRA_ORDER + 1, expect_inter + 1)
    assert np.any(np.isfinite(out.wave_fib2))

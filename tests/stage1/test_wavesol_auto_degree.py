"""Regression tests for default Chebyshev degree selection."""

from __future__ import annotations

import numpy as np

from hispecdrp import datamodels
from hispecdrp.primitives.wavesol.wavesol_cheb2d_primitive import (
    DEFAULT_DEG_INTRA_ORDER,
    CalculateChebyshev2DWLSPrimitive,
    auto_deg_inter_order,
)


def test_auto_deg_inter_order_caps_with_minimum_not_axis():
    assert auto_deg_inter_order(1) == 0
    assert auto_deg_inter_order(2) == 1
    assert auto_deg_inter_order(10) == 6
    assert auto_deg_inter_order(40) == 10


def test_auto_deg_inter_order_few_orders():
    """1-5 orders use the table, then the log rule; the cap applies to both."""
    assert [auto_deg_inter_order(Norders) for Norders in range(1, 7)] == [0, 1, 2, 3, 3, 5]
    assert [auto_deg_inter_order(Norders, anchor_degree=4, max_degree=6) for Norders in range(1, 8)] == [0, 1, 2, 3, 3, 3, 3]
    assert auto_deg_inter_order(5, max_degree=2) == 2


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


def test_default_auto_degrees_do_not_treat_max_degree_as_axis(tmp_path):
    """``np.min(..., max_degree)`` was parsed as ``axis=max_degree``."""
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

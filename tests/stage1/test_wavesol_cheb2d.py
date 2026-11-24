"""Basic smoke tests for ``CalculateChebyshev2DWLSPrimitive``: it runs on each source type."""

from __future__ import annotations

import numpy as np

from hispecdrp import datamodels
from hispecdrp.primitives.line_fitting.fit_lines import LINE_DQ_FLAGS
from hispecdrp.primitives.wavesol.wavesol_cheb2d_primitive import (
    CalculateChebyshev2DWLSPrimitive,
)

ORDER_START = 110
N_ORDERS = 6
N_PIX = 512
N_LINES = 20
MU_ERR_WAVE = 1.0e-7


def _line_table(model_cls, table_attr, seed: int = 0):
    """Fit-params model whose fiber-2 table holds noisy lines on a smooth wavelength grid.

    The line wavelengths carry noise at the level of ``MU_ERR_WAVE``; with exact values the
    sigma clipping would act on round-off and could reject every line.
    """
    rng = np.random.default_rng(seed)
    orders = np.arange(ORDER_START, ORDER_START + N_ORDERS)
    pix = np.arange(N_PIX, dtype=float)
    truth = np.vstack(
        [
            1.20 + 0.01 * i + 2.0e-5 * (pix - N_PIX / 2) + 1.0e-11 * (pix - N_PIX / 2) ** 3
            for i in range(N_ORDERS)
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
    l2[2].spec = np.ones((N_ORDERS, N_PIX), dtype=np.float32)
    l2[2].spec_err = np.full((N_ORDERS, N_PIX), 0.05, dtype=np.float32)
    l2[2].spec_dq = np.zeros((N_ORDERS, N_PIX), dtype=np.uint32)
    l2[2].wave = truth

    model = model_cls.from_datamodels(l2, include_data=True, ancestor_rule="none")
    table = getattr(model, table_attr)
    xs = np.linspace(20.0, N_PIX - 20.0, N_LINES)
    extra = {"LINE_DQ": 0} if "LINE_DQ" in table.colnames else {}  # LFC tables have no LINE_DQ
    for oi, order in enumerate(orders):
        for x in xs:
            table.add_row(
                {
                    **extra,
                    "ORDER": int(order),
                    "AMPLITUDE": 100.0,
                    "AMPLITUDE_ERR": 1.0,
                    "MU_PIX": x,
                    "MU_ERR_PIX": 0.01,
                    "MU_ERR_WAVE": MU_ERR_WAVE,
                    "FWHM_PIX": 3.0,
                    "FWHM_ERR_PIX": 0.1,
                    "FWHM_WAVE": 1.0e-4,
                    "FWHM_ERR_WAVE": 1.0e-6,
                    "WAVE": float(np.interp(x, pix, truth[oi]) + rng.normal(0.0, MU_ERR_WAVE)),
                }
            )
    return model, list(orders)


def _run(model, fit_orders, tmp_path):
    return CalculateChebyshev2DWLSPrimitive.call(
        model,
        do_fibers=["2"],
        fit_orders=fit_orders,
        output_dir=tmp_path,
        save_result=False,
        save_preview=False,
        show_preview=False,
        return_instance=True,
    )


def _assert_basic_output(out, prim):
    wave = np.asarray(out.wave_fib2, dtype=float)
    assert wave.shape == (N_ORDERS, N_PIX)
    assert np.all(np.isfinite(wave[:, 20:N_PIX - 20]))
    assert np.asarray(out.cheb_coeffs_fib2).ndim == 2
    assert len(prim._output["fit_results"]) == 1


def test_cheb2d_runs_on_lfc_table(tmp_path):
    model, fit_orders = _line_table(datamodels.LFCFitParamsModel, "lfc_fitparams_fib2")
    out, prim = _run(model, fit_orders, tmp_path)

    assert isinstance(out, datamodels.LFCWavesolModel)
    assert prim._output["source"] == "lfc"
    _assert_basic_output(out, prim)


def test_cheb2d_runs_on_uranium_table(tmp_path):
    model, fit_orders = _line_table(datamodels.UraniumFitParamsModel, "uranium_fitparams_fib2")
    out, prim = _run(model, fit_orders, tmp_path)

    assert isinstance(out, datamodels.UraniumWavesolModel)
    assert prim._output["source"] == "uranium"
    _assert_basic_output(out, prim)


def test_cheb2d_respects_line_dq(tmp_path):
    model, fit_orders = _line_table(datamodels.UraniumFitParamsModel, "uranium_fitparams_fib2")
    table = model.uranium_fitparams_fib2
    flagged = np.arange(0, len(table), 7)
    table["LINE_DQ"][flagged] = LINE_DQ_FLAGS["POOR_FIT"]

    out, prim = _run(model, fit_orders, tmp_path)

    _assert_basic_output(out, prim)
    n_unflagged = len(table) - flagged.size
    assert 0 < len(prim._output["fit_results"][0]["good"]) <= n_unflagged

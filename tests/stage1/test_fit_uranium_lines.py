"""Basic smoke test for ``FitUraniumLinesPrimitive``: it runs and returns a populated table."""

from __future__ import annotations

import numpy as np

from hispecdrp import datamodels
from hispecdrp.primitives.line_fitting.fit_uranium_lines_primitive import (
    LINELIST_DIR,
    FitUraniumLinesPrimitive,
    _read_linelist,
)

ORDER_START = 110
N_ORDERS = 2
N_PIX = 2048


def _synthetic_uranium_l2(seed: int = 0):
    """L2 model whose fiber 2 holds a noisy synthetic uranium lamp spectrum.

    Each order covers a 0.01 um slice of Y band with a linear wavelength grid, and every
    line-list line inside it is drawn as a Gaussian (FWHM 3 px) scaled by its relative flux.
    """
    rng = np.random.default_rng(seed)
    lamp_wvl, lamp_flux = _read_linelist(LINELIST_DIR / "UNe_lines_nir.txt")

    pix = np.arange(N_PIX, dtype=float)
    sigma_pix = 3.0 / 2.3548
    noise = 1.0
    wave = np.empty((N_ORDERS, N_PIX))
    spec = np.empty((N_ORDERS, N_PIX))
    for oi in range(N_ORDERS):
        w0 = 1.05 + 0.01 * oi
        wave[oi] = w0 + (0.01 / (N_PIX - 1)) * pix
        flux = np.full(N_PIX, 10.0)
        in_order = (lamp_wvl > wave[oi, 0]) & (lamp_wvl < wave[oi, -1])
        for wl, fl in zip(lamp_wvl[in_order], lamp_flux[in_order]):
            x0 = np.interp(wl, wave[oi], pix)
            flux += 10.0 * fl * np.exp(-0.5 * ((pix - x0) / sigma_pix) ** 2)
        spec[oi] = flux + rng.normal(0.0, noise, N_PIX)

    l2 = datamodels.L2Model(
        meta={
            "instrument_name": "HISPEC",
            "spectrograph": "BSPEC",
            "order_start": ORDER_START,
            "order_end": ORDER_START + N_ORDERS - 1,
        }
    )
    l2[2].spec = spec.astype(np.float32)
    l2[2].spec_err = np.full((N_ORDERS, N_PIX), noise, dtype=np.float32)
    l2[2].spec_dq = np.zeros((N_ORDERS, N_PIX), dtype=np.uint32)
    l2[2].wave = wave
    return l2


def test_fit_uranium_lines_runs(tmp_path):
    fit_orders = list(range(ORDER_START, ORDER_START + N_ORDERS))
    out = FitUraniumLinesPrimitive.call(
        _synthetic_uranium_l2(),
        do_fibers=["2"],
        do_orders=fit_orders,
        output_dir=tmp_path,
        save_result=False,
        save_preview=False,
        show_preview=False,
    )

    assert isinstance(out, datamodels.UraniumFitParamsModel)
    table = out.uranium_fitparams_fib2
    assert len(table) > 0
    for col in ("ORDER", "MU_PIX", "WAVE", "MU_ERR_WAVE", "LINE_DQ"):
        assert col in table.colnames
    assert np.all(np.isin(np.asarray(table["ORDER"]), fit_orders))
    assert np.any(np.asarray(table["LINE_DQ"]) == 0)

"""
One-off / dev helper: sweep ``ccf_parabolic_vertex_n_samples`` on the lte049 BSPEC
synthetic setup and report |bias| vs injected RV.

Run from repo root::

  python tests/_scan_btsettl_parabolic_vertex_n_samples.py

Requires BT-Settl template + lte049 mask (same as test_fit_ccf_kpf_style_btsettl_bspec_all_chunks_parallel).
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np

# Repo root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from hispecdrp import datamodels  # noqa: E402
from hispecdrp.primitives import StellarRVPrimitive  # noqa: E402

# Import test module for shared constants/helpers
import tests.test_stellar_rv as ts  # noqa: E402

C_KM_S = ts.C_KM_S
INPUT_RV_KM_S = ts.INPUT_RV_KM_S
_WINDOW_UM = ts.WINDOW_UM
_TEST_ECHELLE_N_PIX = ts._TEST_ECHELLE_N_PIX
_TEST_TEMPLATE_CCF_V_STEP_M_S = ts._TEST_TEMPLATE_CCF_V_STEP_M_S


def _build_l2_and_orders():
    mas = ts._lte049_btsettl_paired_mask_path()
    if not os.path.isfile(mas):
        raise FileNotFoundError(mas)
    path = ts._lte049_btsettl_spec_h5_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    wl_min, wl_max = ts._BTSETTL_WL_RANGE_UM
    w_rest, flux_rest = ts._load_btsettl_sorted_micron(path, wl_min, wl_max)
    template_ln = np.log(w_rest)
    orders = ts._bspec_btsettl_ccf_mask_kpf_order_ids()
    n_orders = len(orders)
    n_pix = _TEST_ECHELLE_N_PIX
    wave_2d = np.zeros((n_orders, n_pix), dtype=np.float64)
    spec_2d = np.zeros((n_orders, n_pix), dtype=np.float64)
    spec_err_2d = np.zeros((n_orders, n_pix), dtype=np.float64)
    dq_2d = np.zeros((n_orders, n_pix), dtype=np.uint32)
    for oi, order in enumerate(orders):
        m = int(order)
        wave_ord, _lam_c = ts._order_wavelength_grid(m, n_pix)
        wave_2d[oi] = wave_ord
        spec_2d[oi] = np.interp(
            wave_ord / (1.0 + INPUT_RV_KM_S / C_KM_S),
            w_rest,
            flux_rest,
        ).astype(np.float64)
        spec_err_2d[oi] = ts._bspec_simulated_flux_err(spec_2d[oi])
    do_not_use = int(datamodels.DQ_FLAGS["DO_NOT_USE"])
    do_not_use_u = np.uint32(do_not_use)
    for oi in range(n_orders):
        spec_2d[oi, 0] = 0.0
        spec_2d[oi, -1] = 0.0
    bad_flux = ~np.isfinite(spec_2d) | (spec_2d <= 0.0)
    dq_2d[bad_flux] |= do_not_use_u

    order_start, order_end = int(min(orders)), int(max(orders))
    rvccf_l2 = datamodels.L2Model(
        meta={
            "order_start": order_start,
            "order_end": order_end,
            "spectrograph": "BSPEC",
            "instrument_name": "HISPEC",
        }
    )
    rvccf_l2.meta.order_start = order_start
    rvccf_l2.meta.order_end = order_end
    rvccf_l2.spec_fib1 = spec_2d.astype(np.float32)
    rvccf_l2.spec_err_fib1 = spec_err_2d.astype(np.float32)
    rvccf_l2.wave_fib1 = wave_2d
    rvccf_l2.spec_dq_fib1 = dq_2d
    return rvccf_l2, orders, mas, float(_WINDOW_UM)


def bias_m_s(n_samples: int, rvccf_l2, orders, mas_lte049, window_um: float, tmp: str) -> float:
    inj = INPUT_RV_KM_S * 1000.0
    prim = StellarRVPrimitive(
        algorithm="ccf_mask_kpf",
        mask_path=mas_lte049,
        do_fibers=["1"],
        do_orders=orders,
        window_size_um=window_um,
        v_step_coarse=_TEST_TEMPLATE_CCF_V_STEP_M_S,
        mask_auto_hybrid_scout=False,
        outlier_sigma=4.0,
        max_cores=6,
        save_preview=False,
        show_preview=False,
        output_dir=tmp,
        suffix=f"scan_n{n_samples}",
        ccf_parabolic_vertex_n_samples=int(n_samples),
    )
    out = prim.apply(rvccf_l2, do_orders=orders)
    avg = getattr(out, "rvccfavg_fib1", None)
    if avg is None or len(avg) == 0:
        return float("nan")
    return float(avg["RV"][0]) - inj


def main() -> None:
    rvccf_l2, orders, mas, window_um = _build_l2_and_orders()
    inj = INPUT_RV_KM_S * 1000.0
    odd_ns = [n for n in range(3, 72, 2)]
    rows: list[tuple[int, float, float]] = []
    best_n: int | None = None
    best_abs = float("inf")
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="btsettl_parabolic_scan_") as tmp:
        for n in odd_ns:
            t1 = time.perf_counter()
            b = bias_m_s(n, rvccf_l2, orders, mas, window_um, tmp)
            dt = time.perf_counter() - t1
            ab = abs(b)
            rows.append((n, b, dt))
            if ab < best_abs:
                best_abs = ab
                best_n = n
    wall = time.perf_counter() - t0
    print(f"Injected RV (m/s): {inj:.3f}  orders={orders}  wall_total={wall:.1f}s\n")
    print(f"{'n_samples':>10}  {'bias_m_s':>12}  {'|bias|':>10}  {'sec':>8}")
    for n, b, dt in rows:
        print(f"{n:10d}  {b:12.4f}  {abs(b):10.4f}  {dt:8.2f}")
    print()
    if best_n is not None:
        best_bias = next(b for n, b, _ in rows if n == best_n)
        print(f"Minimum |bias|: n={best_n}  bias={best_bias:.4f} m/s")
        under = [(n, b) for n, b, _ in rows if abs(b) < 1.0]
        if under:
            print(f"n values with |bias| < 1 m/s: {[n for n, _ in under]}")
        else:
            print("WARNING: no n in 3..71 achieved |bias| < 1 m/s")


if __name__ == "__main__":
    main()

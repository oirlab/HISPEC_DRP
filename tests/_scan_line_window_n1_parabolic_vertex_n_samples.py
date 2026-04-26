"""
Sweep ``ccf_parabolic_vertex_n_samples`` for the **line-window n_lines=1** lte049 BSPEC
setup (same geometry as ``test_fit_ccf_kpf_style_btsettl_bspec_mask_kpf_line_window_n1_parallel``).

Reports per-chunk bias for ORDER=139 CHUNK_ID=0 (segment 1/24), RMS |bias| over all chunks,
and RVCCFAVG bias vs injected RV.

Run from repo root with the ``hispec`` env::

  conda run -n hispec python tests/_scan_line_window_n1_parabolic_vertex_n_samples.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from hispecdrp import datamodels  # noqa: E402
from hispecdrp.primitives import StellarRVPrimitive  # noqa: E402
from hispecdrp.primitives.utils.ccf_utils import load_mask  # noqa: E402

import tests.test_stellar_rv as ts  # noqa: E402


def _build_line_window_l2_orders_mask():
    mas_lte049 = ts._lte049_btsettl_paired_mask_path()
    if not os.path.isfile(mas_lte049):
        raise FileNotFoundError(mas_lte049)
    path = ts._lte049_btsettl_spec_h5_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    wl_min, wl_max = ts._BTSETTL_WL_RANGE_UM
    w_rest, flux_rest = ts._load_btsettl_sorted_micron(path, wl_min, wl_max)
    orders = ts._bspec_btsettl_ccf_mask_kpf_order_ids()
    input_rv_m_s = ts.INPUT_RV_KM_S * 1000.0
    n_orders = len(orders)
    n_pix = ts._TEST_ECHELLE_N_PIX
    wave_2d = np.zeros((n_orders, n_pix), dtype=np.float64)
    spec_2d = np.zeros((n_orders, n_pix), dtype=np.float64)
    spec_err_2d = np.zeros((n_orders, n_pix), dtype=np.float64)
    dq_2d = np.zeros((n_orders, n_pix), dtype=np.uint32)
    for oi, order in enumerate(orders):
        m = int(order)
        wave_ord, _lam_c = ts._order_wavelength_grid(m, n_pix)
        wave_2d[oi] = wave_ord
        spec_2d[oi] = np.interp(
            wave_ord / (1.0 + ts.INPUT_RV_KM_S / ts.C_KM_S),
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
    load_mask(mas_lte049, mask_width_um=None)  # fail fast if mask missing
    return rvccf_l2, orders, mas_lte049, float(input_rv_m_s)


def _metrics_for_n(
    n_samples: int,
    rvccf_l2,
    orders,
    mas_lte049: str,
    input_rv_m_s: float,
    tmp_dir: str,
) -> tuple[float, float, float, float]:
    """Return (bias_chunk0_m_s, rms_abs_bias_chunks_m_s, bias_avg_m_s, mad_bias_chunks_m_s)."""
    v_coarse_half_m_s = float(ts._BTSETTL_LINE_WINDOW_N1_V_COARSE_HALF_SPAN_KM_S) * 1000.0
    v_min = float(input_rv_m_s - v_coarse_half_m_s)
    v_max = float(input_rv_m_s + v_coarse_half_m_s)
    window_um = float(ts._BTSETTL_LINE_WINDOW_N1_WINDOW_UM)
    prim = StellarRVPrimitive(
        algorithm="ccf_mask_kpf",
        mask_path=mas_lte049,
        do_fibers=["1"],
        do_orders=orders,
        window_size_um=window_um,
        window_line_by_line=True,
        ccf_input_velocity=float(input_rv_m_s),
        v_min_coarse=v_min,
        v_max_coarse=v_max,
        v_step_coarse=float(ts._BTSETTL_LINE_WINDOW_N1_V_STEP_COARSE),
        mask_auto_hybrid_scout=False,
        ccf_parabolic_vertex_n_samples=int(n_samples),
        mask_ccf_pixel_interp_order=ts._mask_ccf_pixel_interp_order_from_env(),
        outlier_sigma=4.0,
        max_cores=6,
        save_preview=False,
        show_preview=False,
        output_dir=tmp_dir,
        suffix=f"scan_lw_n{n_samples}",
    )
    out = prim.apply(rvccf_l2, do_orders=orders)
    tbl = getattr(out, "rvccf_fib1", None)
    avg = getattr(out, "rvccfavg_fib1", None)
    if tbl is None or len(tbl) == 0 or avg is None or len(avg) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    ordv = np.asarray(tbl["ORDER"], dtype=np.int64)
    cid = np.asarray(tbl["CHUNK_ID"], dtype=np.int64)
    rvv = np.asarray(tbl["RV"], dtype=np.float64)
    m0 = (ordv == 139) & (cid == 0)
    b0 = float(rvv[m0][0] - input_rv_m_s) if np.any(m0) else float("nan")
    biases = rvv - float(input_rv_m_s)
    rms = float(np.sqrt(np.mean(biases**2)))
    mad = float(np.median(np.abs(biases - np.median(biases))))
    bavg = float(avg["RV"][0]) - float(input_rv_m_s)
    return b0, rms, bavg, mad


def main() -> None:
    import tempfile
    import time

    # Align with ``test_fit_ccf_kpf_style_btsettl_bspec_mask_kpf_line_window_n1_parallel`` when
    # the shell leaves HISPEC_DRP_MASK_CCF_PIXEL_INTERP_ORDER unset (otherwise match env).
    os.environ.setdefault("HISPEC_DRP_MASK_CCF_PIXEL_INTERP_ORDER", "3")

    rvccf_l2, orders, mas, inj = _build_line_window_l2_orders_mask()
    # Odd n from 11 to 151 (extend past default 31)
    odd_ns = [n for n in range(11, 152, 2)]
    rows: list[tuple[int, float, float, float, float, float]] = []
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="lw_n1_parabolic_scan_") as tmp:
        for n in odd_ns:
            t1 = time.perf_counter()
            b0, rms, bavg, mad = _metrics_for_n(n, rvccf_l2, orders, mas, inj, tmp)
            dt = time.perf_counter() - t1
            rows.append((n, b0, rms, bavg, mad, dt))
    wall = time.perf_counter() - t0
    print(
        f"Injected RV (m/s): {inj:.3f}  orders={orders}  "
        f"pio={ts._mask_ccf_pixel_interp_order_from_env()}  wall_total={wall:.1f}s\n"
    )
    hdr = (
        f"{'n':>5}  {'b0_m_s':>10}  {'rms|bias|':>12}  {'b_avg_m_s':>12}  {'mad_bias':>10}  {'s':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    best = (None, float("inf"))  # (n, rms)
    best0 = (None, float("inf"))  # (n, abs b0)
    for n, b0, rms, bavg, mad, dt in rows:
        print(
            f"{n:5d}  {b0:10.3f}  {rms:12.3f}  {bavg:12.3f}  {mad:10.3f}  {dt:6.2f}"
        )
        if np.isfinite(rms) and rms < best[1]:
            best = (n, rms)
        if np.isfinite(b0) and abs(b0) < best0[1]:
            best0 = (n, abs(b0))
    print()
    print(f"Minimum RMS chunk bias: n={best[0]}  RMS={best[1]:.3f} m/s")
    print(f"Minimum |bias| chunk 0: n={best0[0]}  |b0|={best0[1]:.3f} m/s")


if __name__ == "__main__":
    main()

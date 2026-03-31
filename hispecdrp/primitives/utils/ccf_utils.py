"""
Cross-correlation (CCF) and radial-velocity utilities for stellar templates.

Provides weighted-covariance CCF in :math:`\\ln\\lambda`, Gaussian fitting to
:math:`\\mathrm{CCF}(v)` via :func:`fit_ccf_rv` (Astropy ``Gaussian1D`` +
``LevMarLSQFitter``, two-pass windows, error heuristics labeled “KPF” in-code), a
Numba multi-order path, continuum normalization, median/MAD outlier rejection, and
inverse-variance chunk combination. **All velocities are in m s\\ :sup:`−1`**.

Notes
-----
The **template CCF** implementation is specific to HISPEC DRP. The **Gaussian fit**
machinery intentionally mirrors *published* behavior of Keck **KPF** PRV-style
two-pass Levenberg–Marquardt CCF fitting; this file is **not** a copy of KPF
pipeline source. **Mask-line** SPIROU/KPF-style CCF is implemented below
(:func:`run_ccf_mask_kpf_for_fiber`).

References
----------
.. [1] KPF-Pipeline radial_velocity ``src`` (mask-line tables and RV context;
       template path differs): https://github.com/Keck-DataReductionPipelines/KPF-Pipeline/tree/81609b0ac0ab6d12536fb771bfdd0112001340c3/modules/radial_velocity/src
.. [2] ``alg.py`` (masking algorithms in KPF):
       https://github.com/Keck-DataReductionPipelines/KPF-Pipeline/blob/81609b0ac0ab6d12536fb771bfdd0112001340c3/modules/radial_velocity/src/alg.py
.. [3] spirou-ccf (SPIRou spectra / CCF ecosystem):
       https://github.com/edermartioli/spirou-ccf
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass

import numpy as np
from astropy.table import Table
from astropy.modeling import fitting
from astropy.modeling.models import Gaussian1D
from scipy.ndimage import median_filter
from astropy.stats import mad_std
from pathlib import Path

from ...datamodels.dqflags import DQ_FLAGS
from ...datamodels.extensions import RVCCF_CCF_ELEM_DTYPE
from ...utils.parallelization_utils import (
    njit,
    numba_thread_scope,
    prange,
    resolve_numba_threads,
)

# --- Physical / grid defaults (StellarRVPrimitive uses _V_* for bulk processing) ---
# Speed of light in m/s: RV v maps to Δlnλ = v/c when shifting the template in log space.
C_M_S = 299792458.0

# Default CCF velocity search (m/s).
# Sub-step biases (e.g. ~20 m/s) can occur from the grid + Gaussian fit to the peak.
_V_MIN = -500000.0
_V_MAX = 500000.0
_V_STEP = 50.0


def resolve_ccf_broad_grid_m_s(
    v_min: float | None = None,
    v_max: float | None = None,
    v_step: float | None = None,
) -> tuple[float, float, float]:
    """
    Broad uniform CCF velocity grid (m/s) for stellar RV.

    When a value is ``None``, uses the module defaults ``_V_MIN``, ``_V_MAX``, ``_V_STEP``.

    Parameters
    ----------
    v_min, v_max, v_step : float or None
        Lower bound, upper bound, and step (m/s) for the coarse grid search.

    Returns
    -------
    v_min_m_s, v_max_m_s, v_step_m_s : float
    """
    vm = float(_V_MIN if v_min is None else v_min)
    vx = float(_V_MAX if v_max is None else v_max)
    vs = float(_V_STEP if v_step is None else v_step)
    if not np.isfinite(vm) or not np.isfinite(vx) or not np.isfinite(vs):
        raise ValueError("v_min, v_max, v_step must be finite when set (m/s)")
    if vs <= 0:
        raise ValueError("v_step must be finite and > 0 (m/s)")
    if vx <= vm:
        raise ValueError("v_max must be greater than v_min (m/s)")
    return vm, vx, vs


def resolve_ccf_velocity_grid(
    v_center_m_s: float | None,
    *,
    window_m_s: float = 300.0,
    step_m_s: float = 0.1,
) -> tuple[float, float, float]:
    """
    CCF search grid in m/s for stellar RV.

    When ``v_center_m_s`` is ``None``, returns the broad default
    ``(_V_MIN, _V_MAX, _V_STEP)``.

    When ``v_center_m_s`` is set, returns bounds for the **refined** leg of the
    hybrid search: velocities from ``center - window_m_s`` to ``center + window_m_s``
    (i.e. ``window_m_s`` is the **half-width** in m/s, so default 300 → ±300 m/s
    span around the center), sampled every ``step_m_s``.

    Parameters
    ----------
    v_center_m_s : float or None
        Center of the refined velocity window (m/s), or ``None`` for the full default grid.
    window_m_s : float, optional
        Half-width of the refined window (m/s) when ``v_center_m_s`` is set. Default 300 m/s
        (±300 m/s around the center).
    step_m_s : float, optional
        Velocity step (m/s) on the refined grid. Default 0.1 m/s (10 cm/s).

    Returns
    -------
    v_min_m_s, v_max_m_s, v_step_m_s : float
    """
    if v_center_m_s is None:
        return float(_V_MIN), float(_V_MAX), float(_V_STEP)
    vc = float(v_center_m_s)
    w = float(window_m_s)
    vs = float(step_m_s)
    if not np.isfinite(vc):
        raise ValueError("ccf_input_velocity must be finite when set (m/s)")
    if not np.isfinite(w) or w <= 0:
        raise ValueError(
            "ccf_velocity_window must be finite and > 0 when ccf_input_velocity is set (m/s)"
        )
    if not np.isfinite(vs) or vs <= 0:
        raise ValueError(
            "ccf_velocity_step must be finite and > 0 when ccf_input_velocity is set (m/s)"
        )
    v_min = vc - w
    v_max = vc + w
    if v_max <= v_min:
        raise ValueError("ccf velocity window produces an empty or invalid v range")
    return v_min, v_max, vs


def combine_hybrid_broad_refined_rv(
    rv_broad: float,
    err_broad: float,
    rv_refined: float,
    err_refined: float,
) -> tuple[float, float]:
    """
    Combine per-chunk RV estimates from a full broad grid search and a refined window search.

    Uses an inverse-variance weighted mean when both uncertainties are finite and
    positive; otherwise returns the refined estimate if valid, else the broad one.
    """
    ok_b = (
        np.isfinite(rv_broad)
        and np.isfinite(err_broad)
        and float(err_broad) > 0.0
    )
    ok_r = (
        np.isfinite(rv_refined)
        and np.isfinite(err_refined)
        and float(err_refined) > 0.0
    )
    if ok_b and ok_r:
        wb = 1.0 / (float(err_broad) ** 2)
        wr = 1.0 / (float(err_refined) ** 2)
        wsum = wb + wr
        mu = (wb * float(rv_broad) + wr * float(rv_refined)) / wsum
        sig = 1.0 / math.sqrt(wsum)
        return mu, sig
    if ok_r:
        return float(rv_refined), float(err_refined)
    if ok_b:
        return float(rv_broad), float(err_broad)
    return float("nan"), float("nan")


def effective_ccf_v_step_for_fit(v_grid: np.ndarray) -> float:
    """
    Representative velocity spacing (m/s) for Gaussian / fallback CCF fits on ``v_grid``.

    Uses the median positive difference so non-uniform (merged hybrid) grids get a
    sensible effective step near the peak.
    """
    vg = np.asarray(v_grid, dtype=np.float64).ravel()
    if vg.size < 2:
        return 1.0
    d = np.diff(vg)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return 1.0
    return float(np.nanmedian(d))


def build_hybrid_merged_velocity_grid(
    broad_min: float,
    broad_max: float,
    broad_step: float,
    center_m_s: float,
    half_width_m_s: float,
    refined_step_m_s: float,
) -> np.ndarray:
    """
    Single CCF velocity list: full ``[broad_min, broad_max]`` sampled at ``broad_step``,
    plus extra samples on ``[center ± half_width_m_s]`` every ``refined_step_m_s``.

    Sorted unique values (rounded) so the merged grid spans the entire broad
    range while increasing resolution around ``center_m_s``.
    """
    vb = float(broad_min)
    vx = float(broad_max)
    vsb = float(broad_step)
    vc = float(center_m_s)
    hw = float(half_width_m_s)
    vsr = float(refined_step_m_s)
    if not np.isfinite(vb) or not np.isfinite(vx) or vx <= vb:
        raise ValueError("build_hybrid_merged_velocity_grid: invalid broad bounds")
    if not np.isfinite(vsb) or vsb <= 0:
        raise ValueError("build_hybrid_merged_velocity_grid: broad_step must be finite and > 0")
    if not np.isfinite(vc) or not np.isfinite(hw) or hw <= 0:
        raise ValueError("build_hybrid_merged_velocity_grid: center and half-width must be finite, half-width > 0")
    if not np.isfinite(vsr) or vsr <= 0:
        raise ValueError("build_hybrid_merged_velocity_grid: refined_step must be finite and > 0")
    r_lo = vc - hw
    r_hi = vc + hw
    if r_hi <= r_lo:
        raise ValueError("build_hybrid_merged_velocity_grid: refined window is empty")
    broad = np.arange(vb, vx + 0.5 * vsb, vsb, dtype=np.float64)
    refined = np.arange(r_lo, r_hi + 0.5 * vsr, vsr, dtype=np.float64)
    merged = np.unique(np.round(np.concatenate((broad, refined)), decimals=6))
    return merged.astype(np.float64)


# Typical velocity scale per pixel for L2 (BSPEC/RSPEC) from wavelength sampling:
#   Δv = c * (Δλ/λ).  With λ_center = 147/order µm and pixel size Δλ = (147/order²)/4096 µm,
#   velocity per pixel = c / (order * 4096) m/s.
#   BSPEC (order 110–148): ~494–665 m/s per pixel.  RSPEC (order 60–97): ~755–1220 m/s per pixel.

logger = logging.getLogger(__name__)


def build_rvccf_chunk_supplement(
    *,
    chunk_id: np.ndarray,
    n_good: np.ndarray,
    v_min: float,
    v_max: float,
    v_step: float,
    ccf_2d: np.ndarray,
) -> Table:
    """
    Supplement table with raw CCF(v) per chunk (columns merged into ``RVCCF_FIB*``).

    Parameters
    ----------
    ccf_2d : np.ndarray
        Shape ``(n_chunk, n_vel)``; row ``i`` is the CCF for chunk ``i``.
    """
    n = int(ccf_2d.shape[0]) if ccf_2d.ndim >= 2 else 0
    if n == 0:
        return Table(
            names=["CHUNK_ID", "N_GOOD", "V_MIN", "V_MAX", "V_STEP", "N_VEL", "CCF"],
            dtype=[np.int32, np.int32, np.float64, np.float64, np.float64, np.int32, object],
        )
    if len(chunk_id) != n or len(n_good) != n:
        raise ValueError("build_rvccf_chunk_supplement: chunk_id / n_good length mismatch vs ccf_2d")
    n_vel = int(ccf_2d.shape[1])
    ccf_col = np.empty(n, dtype=object)
    for i in range(n):
        ccf_col[i] = np.asarray(ccf_2d[i], dtype=RVCCF_CCF_ELEM_DTYPE).ravel()
    v_min_f = float(v_min)
    v_max_f = float(v_max)
    v_step_f = float(v_step)
    return Table(
        {
            "CHUNK_ID": np.asarray(chunk_id, dtype=np.int32).ravel(),
            "N_GOOD": np.asarray(n_good, dtype=np.int32).ravel(),
            "V_MIN": np.full(n, v_min_f, dtype=np.float64),
            "V_MAX": np.full(n, v_max_f, dtype=np.float64),
            "V_STEP": np.full(n, v_step_f, dtype=np.float64),
            "N_VEL": np.full(n, n_vel, dtype=np.int32),
            "CCF": ccf_col,
        }
    )


def assemble_rvccf_fib_table(
    rvs: np.ndarray,
    rv_errs: np.ndarray,
    wavelengths: np.ndarray,
    orders: np.ndarray,
    inlier: np.ndarray,
    *,
    supplement: Table | None = None,
) -> Table:
    """
    Full per-chunk ``RVCCF_FIB*`` table: RV metrics plus optional CCF supplement columns.

    When ``supplement`` is None or has length 0, ``CCF`` rows are empty arrays
    (``RVCCF_CCF_ELEM_DTYPE``), ``N_GOOD`` is -1, and velocity metadata is NaN / ``N_VEL`` 0.
    """
    n = len(rvs)
    mbjd = np.zeros(n, dtype=np.float64)
    berv = np.zeros(n, dtype=np.float64)
    res = np.full(n, np.nan, dtype=np.float64)
    tell_abs = np.zeros(n, dtype=np.float64)
    bis = np.zeros(n, dtype=np.float64)
    rv_abs = np.zeros(n, dtype=np.float64)
    rv_abs_err = np.zeros(n, dtype=np.float64)
    order_col = np.asarray(orders, dtype=np.int64).ravel()
    if len(order_col) != n:
        order_col = np.full(n, -1, dtype=np.int64)
    order_i16 = np.clip(order_col, -32768, 32767).astype(np.int16)
    wl = np.asarray(wavelengths, dtype=np.float64)
    inlier_u8 = np.asarray(inlier, dtype=np.uint8)

    if supplement is None or len(supplement) == 0:
        chunk_id = np.arange(n, dtype=np.int32)
        n_good = np.full(n, -1, dtype=np.int32)
        v_min_a = np.full(n, np.nan, dtype=np.float64)
        v_max_a = np.full(n, np.nan, dtype=np.float64)
        v_step_a = np.full(n, np.nan, dtype=np.float64)
        n_vel = np.zeros(n, dtype=np.int32)
        ccf_col = np.empty(n, dtype=object)
        for i in range(n):
            ccf_col[i] = np.array([], dtype=RVCCF_CCF_ELEM_DTYPE)
    else:
        if len(supplement) != n:
            raise ValueError(
                f"assemble_rvccf_fib_table: supplement has {len(supplement)} rows, "
                f"expected {n} to match per-chunk RV arrays"
            )
        chunk_id = np.asarray(supplement["CHUNK_ID"], dtype=np.int32)
        n_good = np.asarray(supplement["N_GOOD"], dtype=np.int32)
        v_min_a = np.asarray(supplement["V_MIN"], dtype=np.float64)
        v_max_a = np.asarray(supplement["V_MAX"], dtype=np.float64)
        v_step_a = np.asarray(supplement["V_STEP"], dtype=np.float64)
        n_vel = np.asarray(supplement["N_VEL"], dtype=np.int32)
        ccf_col = np.empty(n, dtype=object)
        for i in range(n):
            raw = supplement["CCF"][i]
            if raw is None or (isinstance(raw, np.ndarray) and raw.size == 0):
                ccf_col[i] = np.array([], dtype=RVCCF_CCF_ELEM_DTYPE)
            else:
                ccf_col[i] = np.asarray(raw, dtype=RVCCF_CCF_ELEM_DTYPE).ravel()

    return Table(
        {
            "MBJDW": mbjd,
            "BERVW": berv,
            "RV": np.asarray(rvs, dtype=np.float64),
            "RV_ERR": np.asarray(rv_errs, dtype=np.float64),
            "ORDER": order_i16,
            "WAVELENGTH": wl,
            "RESOLUTION": res,
            "TELL_ABS": tell_abs,
            "BIS": bis,
            "RV_ABS": rv_abs,
            "RV_ABS_ERR": rv_abs_err,
            "INLIER": inlier_u8,
            "CHUNK_ID": chunk_id,
            "N_GOOD": n_good,
            "V_MIN": v_min_a,
            "V_MAX": v_max_a,
            "V_STEP": v_step_a,
            "N_VEL": n_vel,
            "CCF": ccf_col,
        }
    )


def flat_chunk_wl_lo_hi(
    wave_2d: np.ndarray,
    order_indices: np.ndarray,
    n_chunks_per_order: np.ndarray,
    chunk_start: np.ndarray,
    total_chunks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Wavelength bounds per flattened chunk index (matches Numba chunking in
    :func:`_process_all_orders_ccf_numba`).
    """
    wl_lo = np.full(total_chunks, np.nan, dtype=np.float64)
    wl_hi = np.full(total_chunks, np.nan, dtype=np.float64)
    for j in range(len(order_indices)):
        oi = int(order_indices[j])
        w = wave_2d[oi]
        w_min = float(np.nanmin(w))
        w_max = float(np.nanmax(w))
        n_chunks = int(n_chunks_per_order[j])
        if n_chunks < 1 or not np.isfinite(w_min) or not np.isfinite(w_max) or w_max <= w_min:
            continue
        start = int(chunk_start[j])
        for k in range(n_chunks):
            idx = start + k
            if idx >= total_chunks:
                break
            wl_lo[idx] = w_min + k * (w_max - w_min) / n_chunks
            wl_hi[idx] = w_min + (k + 1) * (w_max - w_min) / n_chunks
    return wl_lo, wl_hi


# --- CCF Gaussian fit (KPF RadialVelocityAlg.fit_ccf analog; internal km/s, public m/s) ---


def _gaussian_ccf_model(v: np.ndarray, amplitude: float, v0: float, sigma: float, baseline: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((v - v0) / np.maximum(sigma, 1e-10)) ** 2) + baseline


@dataclass
class CCFFitKpfConfig:
    """Two-pass Gaussian CCF fit; half-widths in km/s."""

    pass1_half_width_km_s: float = 500.0
    pass2_half_width_km_s: float = 25.0
    initial_stddev_km_s: float = 5.0
    min_points: int = 5
    emission_strength_ratio: float = 1.02
    cal_mask_methods: frozenset = frozenset({"thar", "lfc", "etalon"})


def dv_per_pixel_m_s(wl_um: float) -> float:
    """Echelle Δv per pixel (m/s): order ≈ 147/λ µm → c/(order·4096)."""
    if not np.isfinite(wl_um) or wl_um <= 0:
        return float("nan")
    return float(C_M_S / (max(147.0 / wl_um, 1.0) * 4096.0))


def fit_ccf_rv(
    v_grid_m_s: np.ndarray,
    ccf: np.ndarray,
    v_step_m_s: float,
    n_good: int | None = None,
    return_params: bool = False,
    *,
    mask_method: str | None = None,
    config: CCFFitKpfConfig | None = None,
    vel_span_m_s_per_pixel: float | None = None,
    two_pass: bool = True,
):
    """
    Fit a Gaussian model to a precomputed CCF and return RV ± error in m s⁻¹.

    Uses a robust emission/absorption profile guess, median-subtracted CCF in a
    sliding velocity window, and one or two Levenberg–Marquardt passes. Formal
    uncertainty may use a KPF-inspired ``ccf_error`` scaling when plausible, else
    covariance from the fitter, else a σ/√n fallback.

    Parameters
    ----------
    v_grid_m_s : np.ndarray
        Uniform velocity grid (m s⁻¹).
    ccf : np.ndarray
        CCF values on the same grid.
    v_step_m_s : float
        Grid spacing (m s⁻¹).
    n_good : int or None, optional
        Number of good spectral pixels in the chunk (for fallback error scaling).
    return_params : bool, optional
        If True, return ``(rv, rv_err, params_dict, diag_dict)``; else ``(rv, rv_err)``.
    mask_method : str or None, optional
        If in ``CCFFitKpfConfig.cal_mask_methods``, use absorption-style calibration
        peak logic.
    config : CCFFitKpfConfig or None, optional
        Override fit windows and thresholds.
    vel_span_m_s_per_pixel : float or None, optional
        Spectral sampling scale passed into error heuristics.
    two_pass : bool, optional
        If True (default), run a second fit narrowed around the first mean.

    Returns
    -------
    tuple
        If ``return_params`` is False: ``(rv_ms, rv_err_ms)``.
        If True: ``(rv_ms, rv_err_ms, params, diag)`` with fitted Gaussian parameters
        and diagnostic metadata.

    Notes
    -----
    Design parallels public KPF PRV CCF-fitting ideas; see module ``References``.
    """
    cfg = config or CCFFitKpfConfig()
    vmax = 2.0e5  # max plausible formal σ_RV (m/s); KPF sqrt(CCF) model breaks on flux² CCFs

    def _fail(msg: str):
        d = {
            "fit_ok": False,
            "used_covariance_for_error": False,
            "used_sigma_fallback": False,
            "used_kpf_ccf_error": False,
            "message": msg,
        }
        logger.warning("CCF fit: %s", msg)
        return (np.nan, np.nan, {}, d) if return_params else (np.nan, np.nan)

    vg = np.asarray(v_grid_m_s, np.float64).ravel()
    y0 = np.asarray(ccf, np.float64).ravel()
    ok = np.isfinite(vg) & np.isfinite(y0)
    if np.sum(ok) < cfg.min_points + 2:
        return _fail("insufficient valid points")
    vm, y = vg[ok], y0[ok]
    vk = vm / 1000.0
    mm = str(mask_method).lower() if mask_method else ""
    if mm in cfg.cal_mask_methods:
        profile, seed_ms = "absorption", float(vm[int(np.nanargmax(y))])
    else:
        med = float(np.median(y))
        imx, imn = int(np.nanargmax(y)), int(np.nanargmin(y))
        profile = (
            "emission"
            if (med - y[imn]) > cfg.emission_strength_ratio * max(y[imx] - med, 1e-30)
            else "absorption"
        )
        seed_ms = float(vm[imn if profile == "emission" else imx])
    rv_gk = seed_ms / 1000.0
    vstep_k = float(np.mean(np.diff(np.sort(vk)))) if len(vk) > 1 else float(v_step_m_s) / 1000.0
    vsp_k = (
        float(vel_span_m_s_per_pixel) / 1000.0
        if vel_span_m_s_per_pixel and np.isfinite(vel_span_m_s_per_pixel) and vel_span_m_s_per_pixel > 0
        else vstep_k
    )

    def _lm(mo: Gaussian1D, ft: fitting.LevMarLSQFitter):
        mk, sk = float(mo.mean.value), abs(float(mo.stddev.value))
        sm = None
        fi = getattr(ft, "fit_info", {}) or {}
        pc = fi.get("param_cov")
        if pc is None:
            pc = fi.get("cov_x")
        if pc is not None:
            try:
                pc = np.asarray(pc, np.float64)
                j = list(mo.param_names).index("mean")
                if pc.shape[0] > j and pc[j, j] > 0:
                    sm = float(math.sqrt(pc[j, j]))
            except Exception:
                pass
        return mk, sk, sm

    def _pass(center_km: float, hw_km: float, sd0: float, ft: fitting.LevMarLSQFitter):
        lo, hi = center_km - hw_km, center_km + hw_km
        sel = (vk >= lo) & (vk <= hi)
        if not np.any(sel):
            return None, np.array([]), float("nan"), float("nan"), 0.0
        gx = vk[sel]
        gy0 = y[sel]
        fin = np.isfinite(gx) & np.isfinite(gy0)
        gx, gy0 = gx[fin], gy0[fin]
        if gy0.size < cfg.min_points:
            return None, gx, float("nan"), float("nan"), 0.0
        bmed = float(np.nanmedian(gy0))
        gy = gy0 - bmed
        span = abs(float(np.nanmax(gy) - np.nanmin(gy))) * 100.0
        amp = (
            max(-1e7, float(np.nanmin(gy)) - span)
            if profile == "emission"
            else min(1e7, float(np.nanmax(gy)) + span)
        )
        try:
            g = ft(Gaussian1D(amplitude=amp, mean=center_km, stddev=sd0), gx, gy, maxiter=500)
            return g, gx, lo * 1000.0, hi * 1000.0, bmed
        except Exception:
            return None, gx, lo * 1000.0, hi * 1000.0, bmed

    def _kpf_err_km(fw_km: float, rv0_km: float) -> float:
        v = np.asarray(vk, np.float64).ravel()
        c = np.asarray(y, np.float64).ravel()
        m = np.isfinite(v) & np.isfinite(c)
        v, c = v[m], c[m]
        if v.size < 3 or fw_km <= 0:
            return 0.0
        dv = np.diff(np.sort(v))
        vs = float(np.mean(dv[(dv > 0) & np.isfinite(dv)])) if np.any(dv > 0) else 0.0
        if vs <= 0:
            return 0.0
        nsp = 1.0 if not (np.isfinite(vsp_k) and vsp_k > 0) else vs / vsp_k
        s2 = (v >= rv0_km - fw_km * 0.5) & (v <= rv0_km + fw_km * 0.5)
        vf, cf = v[s2], c[s2]
        if cf.size == 0 or np.any(cf < 0):
            return 0.0
        ns = np.sqrt(np.maximum(cf, 1e-30))
        g = np.gradient(cf, vf)
        top = float(np.sqrt(np.sum((g / ns) ** 2)))
        bot = float(np.sqrt(np.sum(cf)))
        if top <= 0 or bot <= 0:
            return 0.0
        return 1.0 / ((top / bot) * (nsp**0.5) * bot)

    def _ok_err(e: float, sig: float, vstep: float) -> bool:
        return (
            math.isfinite(e)
            and e > 0
            and e <= vmax
            and e <= 2000.0 * vstep
            and (sig <= 0 or e <= 50.0 * sig)
        )

    ft1 = fitting.LevMarLSQFitter()
    g1, gx1, vlo1, vhi1, med1 = _pass(rv_gk, cfg.pass1_half_width_km_s, cfg.initial_stddev_km_s, ft1)
    if g1 is None or gx1.size < cfg.min_points:
        return _fail("first-pass fit failed")
    m1k, _, _ = _lm(g1, ft1)
    if not (float(gx1.min()) <= m1k <= float(gx1.max())):
        return _fail("mean outside window")
    best_g, best_ft, best_gx, best_med, best_vlo, best_vhi = g1, ft1, gx1, med1, vlo1, vhi1
    pw = 1
    if two_pass:
        ft2 = fitting.LevMarLSQFitter()
        g2, gx2, vlo2, vhi2, med2 = _pass(
            m1k, cfg.pass2_half_width_km_s, float(np.clip(cfg.initial_stddev_km_s, 0.05, 200.0)), ft2
        )
        if g2 is not None and gx2.size >= cfg.min_points:
            m2k, _, _ = _lm(g2, ft2)
            r2 = m2k * 1000.0
            if vlo2 <= r2 <= vhi2 and float(gx2.min()) <= m2k <= float(gx2.max()):
                best_g, best_ft, best_gx, best_med, best_vlo, best_vhi = g2, ft2, gx2, med2, vlo2, vhi2
                pw = 2

    mk, sk, sigc = _lm(best_g, best_ft)
    rv_ms, sig_ms = float(mk * 1000.0), float(sk * 1000.0)
    fw_km = 2.0 * (cfg.pass2_half_width_km_s if pw == 2 else cfg.pass1_half_width_km_s)
    ek = _kpf_err_km(fw_km, mk)
    ek_ms = float(ek * 1000.0)
    vstep = float(v_step_m_s)
    used_kpf = ek > 0 and np.isfinite(ek) and _ok_err(ek_ms, sig_ms, vstep)
    used_cov = used_fb = False
    if used_kpf:
        rv_err_ms, msg = ek_ms, "KPF ccf_error"
    elif (
        sigc is not None
        and np.isfinite(sigc)
        and sigc > 0
        and _ok_err(float(sigc * 1000.0), sig_ms, vstep)
    ):
        rv_err_ms, used_cov, msg = float(sigc * 1000.0), True, "LevMar cov"
    else:
        n_fit = int(best_gx.size)
        rv_err_ms = float(sig_ms / max(np.sqrt(n_fit / 50.0), 1e-6))
        if n_good is not None and n_good > 50:
            rv_err_ms = max(rv_err_ms, float(sig_ms / np.sqrt(n_good / 50.0)))
        used_fb = True
        if ek > 0 and np.isfinite(ek):
            msg = "σ/√n (KPF ccf_error implausible)"
        elif sigc is not None and np.isfinite(float(sigc)) and float(sigc) > 0:
            msg = "σ/√n (LevMar cov implausible)"
        else:
            msg = "σ/√n fallback"

    params = {
        "amplitude": float(best_g.amplitude.value),
        "v0": rv_ms,
        "sigma": sig_ms,
        "baseline": float(best_med),
    }
    diag = {
        "fit_ok": True,
        "used_covariance_for_error": used_cov,
        "used_sigma_fallback": used_fb,
        "used_kpf_ccf_error": used_kpf,
        "message": msg,
        "fit_v_lo": float(best_vlo),
        "fit_v_hi": float(best_vhi),
        "profile": profile,
        "kpf_pass_used": pw,
        "ccf_dir_style": "up" if profile == "absorption" else "down",
    }
    return (rv_ms, rv_err_ms, params, diag) if return_params else (rv_ms, rv_err_ms)


# =============================================================================
# Per-chunk CCF (Numba) + Python Gaussian fit for many orders/chunks at once
# =============================================================================


def _ccf_rv_from_ccf(
    v_grid: np.ndarray,
    ccf: np.ndarray,
    v_step: float,
    n_good: int,
    vel_span_m_s_per_pixel: float | None = None,
) -> tuple[float, float]:
    """Gaussian fit to CCF; on failure use grid peak and HWHM-based error."""
    ccf = np.asarray(ccf, dtype=np.float64)
    n_vel = len(ccf)
    if n_good < 10 or n_vel < 3 or not np.any(np.isfinite(ccf)):
        return np.nan, np.nan
    ccf_max = float(np.nanmax(ccf))
    ccf_min = float(np.nanmin(ccf))
    i_max = int(np.nanargmax(ccf))
    rv = float(v_grid[i_max])
    if not np.isfinite(ccf_max):
        return np.nan, np.nan
    if ccf_max <= ccf_min or not np.isfinite(ccf_min):
        return rv, float(v_step * 2.0)

    vsp = (
        float(vel_span_m_s_per_pixel)
        if vel_span_m_s_per_pixel is not None and np.isfinite(vel_span_m_s_per_pixel) and vel_span_m_s_per_pixel > 0
        else None
    )
    rv_fit, err_fit = fit_ccf_rv(
        v_grid, ccf, v_step, n_good=n_good, vel_span_m_s_per_pixel=vsp
    )
    if np.isfinite(rv_fit) and np.isfinite(err_fit) and err_fit > 0.0:
        return float(rv_fit), float(err_fit)

    half = (ccf_max + ccf_min) / 2.0
    below = np.where(ccf[:i_max] <= half)[0]
    above = np.where(ccf[i_max:] <= half)[0]
    left = int(below[-1]) if len(below) else 0
    right = i_max + int(above[0]) if len(above) else n_vel - 1
    hwhm = (right - left) * v_step * 0.5
    return rv, float(max(hwhm / np.sqrt(max(n_good, 1) / 50.0), v_step))


@njit(cache=True)
def _ccf_chunk_ccf_numba(
    ln_lam: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    good: np.ndarray,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    v_min: float,
    v_max: float,
    v_step: float,
    ccf_out: np.ndarray,
) -> int:
    """
    Weighted cross-correlation CCF(v) for one chunk. Writes ``ccf_out`` (length
    n_vel); returns count of good spectral pixels (for error scaling).
    """
    n_good = 0
    for i in range(len(good)):
        if good[i]:
            n_good += 1
    if n_good < 10:
        return n_good

    n_pix = len(ln_lam)
    # Inverse-variance weights (same as 1/spec_err² with floor); reused for every v.
    inv_w = np.empty(n_pix)
    for i in range(n_pix):
        if good[i] and np.isfinite(spec_err[i]) and spec_err[i] > 0:
            inv_w[i] = 1.0 / max(spec_err[i] * spec_err[i], 1e-30)
        else:
            inv_w[i] = 0.0

    n_vel = len(ccf_out)
    for i in range(n_vel):
        ccf_out[i] = np.nan

    # One CCF sample per trial velocity: shift template in lnλ, interpolate to data pixels.
    for iv in range(n_vel):
        v = v_min + iv * v_step
        delta_ln = v / C_M_S
        t_ln_shifted = template_ln + delta_ln
        t_at_data = np.interp(ln_lam, t_ln_shifted, template_flux)
        # Weighted means of data and template at this v (same pixels as masked_ccf_rv).
        sum_w = 0.0
        sum_d = 0.0
        sum_t = 0.0
        n_use = 0
        for i in range(n_pix):
            if (
                good[i]
                and inv_w[i] > 0.0
                and np.isfinite(t_at_data[i])
                and t_at_data[i] != 0.0
            ):
                w = inv_w[i]
                sum_w += w
                sum_d += spec[i] * w
                sum_t += t_at_data[i] * w
                n_use += 1
        if n_use < 5:
            continue
        d_mean = sum_d / sum_w
        t_mean = sum_t / sum_w
        # Weighted covariance (mean-subtracted); same definition as Python masked_ccf_rv.
        cov = 0.0
        for i in range(n_pix):
            if (
                good[i]
                and inv_w[i] > 0.0
                and np.isfinite(t_at_data[i])
                and t_at_data[i] != 0.0
            ):
                w = inv_w[i]
                cov += (spec[i] - d_mean) * (t_at_data[i] - t_mean) * w
        ccf_out[iv] = cov / sum_w

    return n_good


@njit(cache=True)
def _ccf_chunk_ccf_numba_vgrid(
    ln_lam: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    good: np.ndarray,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    v_grid: np.ndarray,
    ccf_out: np.ndarray,
) -> int:
    """Same as :func:`_ccf_chunk_ccf_numba` but samples at arbitrary ``v_grid`` (m/s)."""
    n_good = 0
    for i in range(len(good)):
        if good[i]:
            n_good += 1
    if n_good < 10:
        return n_good

    n_pix = len(ln_lam)
    inv_w = np.empty(n_pix)
    for i in range(n_pix):
        if good[i] and np.isfinite(spec_err[i]) and spec_err[i] > 0:
            inv_w[i] = 1.0 / max(spec_err[i] * spec_err[i], 1e-30)
        else:
            inv_w[i] = 0.0

    n_vel = len(ccf_out)
    for i in range(n_vel):
        ccf_out[i] = np.nan

    for iv in range(n_vel):
        v = float(v_grid[iv])
        delta_ln = v / C_M_S
        t_ln_shifted = template_ln + delta_ln
        t_at_data = np.interp(ln_lam, t_ln_shifted, template_flux)
        sum_w = 0.0
        sum_d = 0.0
        sum_t = 0.0
        n_use = 0
        for i in range(n_pix):
            if (
                good[i]
                and inv_w[i] > 0.0
                and np.isfinite(t_at_data[i])
                and t_at_data[i] != 0.0
            ):
                w = inv_w[i]
                sum_w += w
                sum_d += spec[i] * w
                sum_t += t_at_data[i] * w
                n_use += 1
        if n_use < 5:
            continue
        d_mean = sum_d / sum_w
        t_mean = sum_t / sum_w
        cov = 0.0
        for i in range(n_pix):
            if (
                good[i]
                and inv_w[i] > 0.0
                and np.isfinite(t_at_data[i])
                and t_at_data[i] != 0.0
            ):
                w = inv_w[i]
                cov += (spec[i] - d_mean) * (t_at_data[i] - t_mean) * w
        ccf_out[iv] = cov / sum_w

    return n_good


@njit(parallel=True, cache=True)
def _process_all_orders_ccf_numba(
    wave_2d: np.ndarray,
    spec_2d: np.ndarray,
    spec_err_2d: np.ndarray,
    dq_2d: np.ndarray,
    do_not_use: int,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    window_size_um: float,
    order_indices: np.ndarray,
    n_chunks_per_order: np.ndarray,
    chunk_start: np.ndarray,
    v_min: float,
    v_max: float,
    v_step: float,
    n_vel: int,
    all_ccf: np.ndarray,
    all_n_good: np.ndarray,
    all_wl: np.ndarray,
) -> None:
    """
    Parallel per-order CCF fill: writes ``all_ccf[chunk, :]``, ``all_n_good[chunk]``,
    ``all_wl[chunk]``. RV fit in Python via :func:`fit_ccf_rv`.
    """
    n_processed = len(order_indices)

    for idx in prange(n_processed):
        oi = order_indices[idx]
        w = wave_2d[oi]
        s = spec_2d[oi]
        e = spec_err_2d[oi]
        d = dq_2d[oi]
        n_pix = len(w)
        # Pixel usable for CCF: not DQ-bad, finite flux and positive error.
        mask = np.empty(n_pix, dtype=np.bool_)
        for i in range(n_pix):
            mask[i] = (d[i] & do_not_use) != 0
        good = np.empty(n_pix, dtype=np.bool_)
        for i in range(n_pix):
            good[i] = not mask[i] and np.isfinite(s[i]) and np.isfinite(e[i]) and e[i] > 0

        w_min = np.nanmin(w)
        w_max = np.nanmax(w)
        ln_min = np.log(w_min) - 0.01
        ln_max = np.log(w_max) + 0.01
        n_tpl = 0
        for i in range(len(template_ln)):
            if template_ln[i] >= ln_min and template_ln[i] <= ln_max:
                n_tpl += 1
        if n_tpl < 20:
            continue  # skip order: template would be too sparse after clip
        t_ln = np.empty(n_tpl)
        t_flux = np.empty(n_tpl)
        j = 0
        for i in range(len(template_ln)):
            if template_ln[i] >= ln_min and template_ln[i] <= ln_max:
                t_ln[j] = template_ln[i]
                t_flux[j] = template_flux[i]
                j += 1

        ccf_row = np.empty(n_vel)
        n_chunks = n_chunks_per_order[idx]
        start = chunk_start[idx]
        # Wavelength bins along the order; each bin gets one CCF vector.
        for k in range(n_chunks):
            lam_lo = w_min + k * (w_max - w_min) / n_chunks
            lam_hi = w_min + (k + 1) * (w_max - w_min) / n_chunks
            in_chunk = np.empty(n_pix, dtype=np.bool_)
            chunk_len = 0
            for i in range(n_pix):
                in_chunk[i] = w[i] >= lam_lo and w[i] < lam_hi
                if in_chunk[i]:
                    chunk_len += 1
            wl_c = 0.5 * (lam_lo + lam_hi)
            all_wl[start + k] = wl_c
            if chunk_len < 20:
                all_n_good[start + k] = 0
                continue
            ln_lam_c = np.empty(chunk_len)
            spec_c = np.empty(chunk_len)
            spec_err_c = np.empty(chunk_len)
            good_c = np.empty(chunk_len, dtype=np.bool_)
            j = 0
            for i in range(n_pix):
                if in_chunk[i]:
                    ln_lam_c[j] = np.log(w[i])
                    spec_c[j] = s[i]
                    spec_err_c[j] = e[i]
                    good_c[j] = good[i]
                    j += 1
            ng = _ccf_chunk_ccf_numba(
                ln_lam_c, spec_c, spec_err_c, good_c,
                t_ln, t_flux, v_min, v_max, v_step,
                ccf_row,
            )
            all_n_good[start + k] = ng
            for iv in range(n_vel):
                all_ccf[start + k, iv] = ccf_row[iv]


@njit(parallel=True, cache=True)
def _process_all_orders_ccf_numba_vgrid(
    wave_2d: np.ndarray,
    spec_2d: np.ndarray,
    spec_err_2d: np.ndarray,
    dq_2d: np.ndarray,
    do_not_use: int,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    window_size_um: float,
    order_indices: np.ndarray,
    n_chunks_per_order: np.ndarray,
    chunk_start: np.ndarray,
    v_grid: np.ndarray,
    n_vel: int,
    all_ccf: np.ndarray,
    all_n_good: np.ndarray,
    all_wl: np.ndarray,
) -> None:
    """Same as :func:`_process_all_orders_ccf_numba` but with an explicit ``v_grid`` (m/s)."""
    n_processed = len(order_indices)

    for idx in prange(n_processed):
        oi = order_indices[idx]
        w = wave_2d[oi]
        s = spec_2d[oi]
        e = spec_err_2d[oi]
        d = dq_2d[oi]
        n_pix = len(w)
        mask = np.empty(n_pix, dtype=np.bool_)
        for i in range(n_pix):
            mask[i] = (d[i] & do_not_use) != 0
        good = np.empty(n_pix, dtype=np.bool_)
        for i in range(n_pix):
            good[i] = not mask[i] and np.isfinite(s[i]) and np.isfinite(e[i]) and e[i] > 0

        w_min = np.nanmin(w)
        w_max = np.nanmax(w)
        ln_min = np.log(w_min) - 0.01
        ln_max = np.log(w_max) + 0.01
        n_tpl = 0
        for i in range(len(template_ln)):
            if template_ln[i] >= ln_min and template_ln[i] <= ln_max:
                n_tpl += 1
        if n_tpl < 20:
            continue
        t_ln = np.empty(n_tpl)
        t_flux = np.empty(n_tpl)
        j = 0
        for i in range(len(template_ln)):
            if template_ln[i] >= ln_min and template_ln[i] <= ln_max:
                t_ln[j] = template_ln[i]
                t_flux[j] = template_flux[i]
                j += 1

        ccf_row = np.empty(n_vel)
        n_chunks = n_chunks_per_order[idx]
        start = chunk_start[idx]
        for k in range(n_chunks):
            lam_lo = w_min + k * (w_max - w_min) / n_chunks
            lam_hi = w_min + (k + 1) * (w_max - w_min) / n_chunks
            in_chunk = np.empty(n_pix, dtype=np.bool_)
            chunk_len = 0
            for i in range(n_pix):
                in_chunk[i] = w[i] >= lam_lo and w[i] < lam_hi
                if in_chunk[i]:
                    chunk_len += 1
            wl_c = 0.5 * (lam_lo + lam_hi)
            all_wl[start + k] = wl_c
            if chunk_len < 20:
                all_n_good[start + k] = 0
                continue
            ln_lam_c = np.empty(chunk_len)
            spec_c = np.empty(chunk_len)
            spec_err_c = np.empty(chunk_len)
            good_c = np.empty(chunk_len, dtype=np.bool_)
            j = 0
            for i in range(n_pix):
                if in_chunk[i]:
                    ln_lam_c[j] = np.log(w[i])
                    spec_c[j] = s[i]
                    spec_err_c[j] = e[i]
                    good_c[j] = good[i]
                    j += 1
            ng = _ccf_chunk_ccf_numba_vgrid(
                ln_lam_c, spec_c, spec_err_c, good_c,
                t_ln, t_flux, v_grid,
                ccf_row,
            )
            all_n_good[start + k] = ng
            for iv in range(n_vel):
                all_ccf[start + k, iv] = ccf_row[iv]


def _process_all_orders_rvs_numba_single_grid(
    wave_2d: np.ndarray,
    spec_2d: np.ndarray,
    spec_err_2d: np.ndarray,
    dq_2d: np.ndarray,
    do_not_use: int,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    window_size_um: float,
    order_indices: np.ndarray,
    n_chunks_per_order: np.ndarray,
    chunk_start: np.ndarray,
    v_min: float,
    v_max: float,
    v_step: float,
    *,
    return_ccf: bool = False,
    numba_threads: int | None = None,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Single uniform velocity grid: Numba CCF fill + per-chunk Gaussian RV."""
    n_processed = len(order_indices)
    total_chunks = int(chunk_start[n_processed - 1] + n_chunks_per_order[n_processed - 1])
    n_vel = int((v_max - v_min) / v_step) + 1
    # Phase A — Numba fills CCF matrix (memory: total_chunks × n_vel).
    all_ccf = np.full((total_chunks, n_vel), np.nan, dtype=np.float64)
    all_n_good = np.zeros(total_chunks, dtype=np.int32)
    all_wl = np.empty(total_chunks, dtype=np.float64)
    all_wl[:] = np.nan
    all_rvs = np.empty(total_chunks, dtype=np.float64)
    all_rvs[:] = np.nan
    all_rv_errs = np.empty(total_chunks, dtype=np.float64)
    all_rv_errs[:] = np.nan

    nt = numba_threads if numba_threads is not None else 1
    logger.info(
        "CCF (Numba): starting CCF matrix fill total_chunks=%d n_vel=%d numba_threads=%d",
        total_chunks,
        n_vel,
        nt,
    )
    _process_all_orders_ccf_numba(
        wave_2d, spec_2d, spec_err_2d, dq_2d, do_not_use,
        template_ln, template_flux, window_size_um,
        order_indices, n_chunks_per_order, chunk_start,
        v_min, v_max, v_step, n_vel,
        all_ccf, all_n_good, all_wl,
    )
    logger.info(
        "CCF (Numba): CCF matrix fill complete total_chunks=%d n_vel=%d",
        total_chunks,
        n_vel,
    )

    # Phase B — Gaussian fit per chunk row in Python (cannot run inside Numba).
    v_grid = v_min + np.arange(n_vel, dtype=np.float64) * v_step
    logger.info(
        "CCF (Python): starting per-chunk Gaussian RV fits total_chunks=%d",
        total_chunks,
    )
    n_fit_rows = 0
    for i in range(total_chunks):
        if all_n_good[i] < 10:
            continue
        n_fit_rows += 1
        wl_c = float(all_wl[i])
        vspan = dv_per_pixel_m_s(wl_c)
        if not np.isfinite(vspan):
            vspan = None
        rv, rv_err = _ccf_rv_from_ccf(
            v_grid,
            all_ccf[i],
            v_step,
            int(all_n_good[i]),
            vel_span_m_s_per_pixel=vspan,
        )
        all_rvs[i] = rv
        all_rv_errs[i] = rv_err
    logger.info(
        "CCF (Python): per-chunk Gaussian fits complete rows_fitted=%d of total_chunks=%d",
        n_fit_rows,
        total_chunks,
    )

    if return_ccf:
        return all_rvs, all_rv_errs, all_wl, all_ccf, v_grid, all_n_good
    return all_rvs, all_rv_errs, all_wl


def _process_all_orders_rvs_numba_single_vgrid(
    wave_2d: np.ndarray,
    spec_2d: np.ndarray,
    spec_err_2d: np.ndarray,
    dq_2d: np.ndarray,
    do_not_use: int,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    window_size_um: float,
    order_indices: np.ndarray,
    n_chunks_per_order: np.ndarray,
    chunk_start: np.ndarray,
    v_grid: np.ndarray,
    *,
    return_ccf: bool = False,
    numba_threads: int | None = None,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Non-uniform ``v_grid`` (e.g. merged hybrid): Numba CCF fill + per-chunk Gaussian RV."""
    v_grid = np.asarray(v_grid, dtype=np.float64).ravel()
    n_vel = int(v_grid.shape[0])
    if n_vel < 2:
        raise ValueError("v_grid must contain at least two velocity samples (m/s)")
    v_step_fit = effective_ccf_v_step_for_fit(v_grid)

    n_processed = len(order_indices)
    total_chunks = int(chunk_start[n_processed - 1] + n_chunks_per_order[n_processed - 1])
    all_ccf = np.full((total_chunks, n_vel), np.nan, dtype=np.float64)
    all_n_good = np.zeros(total_chunks, dtype=np.int32)
    all_wl = np.empty(total_chunks, dtype=np.float64)
    all_wl[:] = np.nan
    all_rvs = np.empty(total_chunks, dtype=np.float64)
    all_rvs[:] = np.nan
    all_rv_errs = np.empty(total_chunks, dtype=np.float64)
    all_rv_errs[:] = np.nan

    nt = numba_threads if numba_threads is not None else 1
    logger.info(
        "CCF (Numba): starting CCF matrix fill (merged v_grid) total_chunks=%d n_vel=%d "
        "numba_threads=%d",
        total_chunks,
        n_vel,
        nt,
    )
    _process_all_orders_ccf_numba_vgrid(
        wave_2d,
        spec_2d,
        spec_err_2d,
        dq_2d,
        do_not_use,
        template_ln,
        template_flux,
        window_size_um,
        order_indices,
        n_chunks_per_order,
        chunk_start,
        v_grid,
        n_vel,
        all_ccf,
        all_n_good,
        all_wl,
    )
    logger.info(
        "CCF (Numba): CCF matrix fill complete total_chunks=%d n_vel=%d",
        total_chunks,
        n_vel,
    )

    logger.info(
        "CCF (Python): starting per-chunk Gaussian RV fits total_chunks=%d",
        total_chunks,
    )
    n_fit_rows = 0
    for i in range(total_chunks):
        if all_n_good[i] < 10:
            continue
        n_fit_rows += 1
        wl_c = float(all_wl[i])
        vspan = dv_per_pixel_m_s(wl_c)
        if not np.isfinite(vspan):
            vspan = None
        rv, rv_err = _ccf_rv_from_ccf(
            v_grid,
            all_ccf[i],
            v_step_fit,
            int(all_n_good[i]),
            vel_span_m_s_per_pixel=vspan,
        )
        all_rvs[i] = rv
        all_rv_errs[i] = rv_err
    logger.info(
        "CCF (Python): per-chunk Gaussian fits complete rows_fitted=%d of total_chunks=%d",
        n_fit_rows,
        total_chunks,
    )

    if return_ccf:
        return all_rvs, all_rv_errs, all_wl, all_ccf, v_grid, all_n_good
    return all_rvs, all_rv_errs, all_wl


def process_all_orders_rvs_numba(
    wave_2d: np.ndarray,
    spec_2d: np.ndarray,
    spec_err_2d: np.ndarray,
    dq_2d: np.ndarray,
    do_not_use: int,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    window_size_um: float,
    order_indices: np.ndarray,
    n_chunks_per_order: np.ndarray,
    chunk_start: np.ndarray,
    v_min: float,
    v_max: float,
    v_step: float,
    *,
    return_ccf: bool = False,
    numba_threads: int | None = None,
    hybrid_center_m_s: float | None = None,
    hybrid_half_width_m_s: float = 300.0,
    hybrid_step_m_s: float = 0.1,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """
    High-throughput per-chunk RVs: Numba CCF grid, then SciPy Gaussian per chunk.

    Workhorse behind :func:`process_orders_rvs_numba_parallel`.

    When ``hybrid_center_m_s`` is ``None``, uses a single grid ``(v_min, v_max, v_step)``.

    When ``hybrid_center_m_s`` is set, uses one merged velocity list spanning the broad
    range ``v_min`` … ``v_max`` at ``v_step``, with additional samples on
    ``center ± hybrid_half_width_m_s`` every ``hybrid_step_m_s`` (see
    :func:`build_hybrid_merged_velocity_grid`). A single CCF is built on that grid and
    per-chunk RVs are fit once. Optional CCF output uses the same merged ``v_grid``.
    """
    if hybrid_center_m_s is None:
        return _process_all_orders_rvs_numba_single_grid(
            wave_2d,
            spec_2d,
            spec_err_2d,
            dq_2d,
            do_not_use,
            template_ln,
            template_flux,
            window_size_um,
            order_indices,
            n_chunks_per_order,
            chunk_start,
            v_min,
            v_max,
            v_step,
            return_ccf=return_ccf,
            numba_threads=numba_threads,
        )

    vc = float(hybrid_center_m_s)
    hw = float(hybrid_half_width_m_s)
    vsf = float(hybrid_step_m_s)
    if not np.isfinite(vc) or not np.isfinite(hw) or hw <= 0:
        raise ValueError(
            "hybrid CCF: center and half-width must be finite and half-width > 0 (m/s)"
        )
    if not np.isfinite(vsf) or vsf <= 0:
        raise ValueError("hybrid CCF: refined step must be finite and > 0 (m/s)")
    if vc + hw <= vc - hw:
        raise ValueError("hybrid CCF: refined velocity window is empty")

    v_grid = build_hybrid_merged_velocity_grid(
        float(v_min),
        float(v_max),
        float(v_step),
        vc,
        hw,
        vsf,
    )
    logger.info(
        "CCF (Numba): hybrid merged grid — broad [%g, %g] step %g; denser [%g, %g] "
        "step %g → n_vel=%d",
        v_min,
        v_max,
        v_step,
        vc - hw,
        vc + hw,
        vsf,
        int(v_grid.shape[0]),
    )
    return _process_all_orders_rvs_numba_single_vgrid(
        wave_2d,
        spec_2d,
        spec_err_2d,
        dq_2d,
        do_not_use,
        template_ln,
        template_flux,
        window_size_um,
        order_indices,
        n_chunks_per_order,
        chunk_start,
        v_grid,
        return_ccf=return_ccf,
        numba_threads=numba_threads,
    )


def flat_chunk_echelle_orders(
    order_tasks: list,
    chunk_start: np.ndarray,
    n_flat: int,
) -> np.ndarray:
    """
    Echelle order number for each flattened chunk index (same packing as
    :func:`process_orders_rvs_numba_parallel`).

    ``chunk_start[j]`` is the global flat index of the first chunk of
    ``order_tasks[j]`` (``(order_row_index, echelle_order_number)``).
    """
    if n_flat <= 0 or not order_tasks:
        return np.array([], dtype=np.int64)
    chunk_start = np.asarray(chunk_start, dtype=np.int64)
    order_nums = np.array([int(t[1]) for t in order_tasks], dtype=np.int64)
    ii = np.arange(n_flat, dtype=np.int64)
    j = np.searchsorted(chunk_start, ii, side="right") - 1
    valid = (j >= 0) & (j < len(order_tasks))
    out = np.full(n_flat, -1, dtype=np.int64)
    out[valid] = order_nums[j[valid]]
    return out


def process_orders_rvs_numba_parallel(
    wave_2d: np.ndarray,
    spec_2d: np.ndarray,
    spec_err_2d: np.ndarray,
    dq_2d: np.ndarray,
    do_not_use: int,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    window_size_um: float,
    order_tasks: list,
    max_cores: int | None = None,
    *,
    return_ccf: bool = False,
    ccf_v_min: float | None = None,
    ccf_v_max: float | None = None,
    ccf_v_step: float | None = None,
    ccf_hybrid_center_m_s: float | None = None,
    ccf_hybrid_half_width_m_s: float = 300.0,
    ccf_hybrid_step_m_s: float = 0.1,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
):
    """
    Stellar-RV fast path: many orders × wavelength chunks in one call.

    Flattens ``order_tasks`` into chunk index arrays, runs
    :func:`process_all_orders_rvs_numba` (Numba CCF + Python Gaussian fit per chunk),
    and optionally sets Numba thread count via ``max_cores``.

    Parameters
    ----------
    wave_2d : np.ndarray, shape (n_orders, n_pix)
        Wavelength in micron.
    spec_2d, spec_err_2d : np.ndarray
        Flux and uncertainty per order.
    dq_2d : np.ndarray, uint32
        Data quality flags.
    do_not_use : int
        Bit mask; pixels with ``(dq & do_not_use) != 0`` are excluded.
    template_ln, template_flux : np.ndarray
        Full stellar template.
    window_size_um : float
        Wavelength span per chunk (micron).
    order_tasks : list of tuple
        ``(order_row_index, order_number)`` for each order to include.
    max_cores : int, optional
        Numba parallel thread count for the order-parallel CCF loop. ``None`` or
        ``0`` uses all logical CPUs (see :func:`~hispecdrp.utils.parallelization_utils.resolve_numba_threads`).
        ``1`` forces single-threaded execution (reproducible, slower).
    ccf_v_min, ccf_v_max, ccf_v_step : float, optional
        Single-grid CCF velocity range (m/s). Ignored when ``ccf_hybrid_center_m_s``
        is set. Defaults are ``_V_MIN``, ``_V_MAX``, ``_V_STEP``.
    ccf_hybrid_center_m_s : float or None, optional
        When set, uses a single merged velocity grid (full ``_V_*`` range plus denser
        sampling around ``center ± ccf_hybrid_half_width_m_s``); see
        :func:`build_hybrid_merged_velocity_grid`.
    ccf_hybrid_half_width_m_s, ccf_hybrid_step_m_s : float, optional
        Refined half-width (m/s) and step (m/s); defaults match
        :class:`~hispecdrp.primitives.stellar_rv.stellar_rv_primitive.StellarRVPrimitive`.

    Returns
    -------
    all_rvs : np.ndarray
        Per-chunk RV (m/s), 1-D flattened over all orders.
    all_rv_errs : np.ndarray
        Per-chunk uncertainty (m/s).
    all_wl : np.ndarray
        Chunk center wavelength (micron).
    n_chunks_per_order, chunk_start : np.ndarray
        Chunk layout (always returned). Use with :func:`flat_chunk_echelle_orders`
        to map flat chunk index → echelle order number.
    all_ccf, v_grid, order_indices, all_n_good : np.ndarray, optional
        Extra arrays when ``return_ccf`` is True (for preview / RVCCF supplement columns).

    Notes
    -----
    **Performance:** Phase A (Numba ``prange`` over orders) fills
    ``total_chunks × n_vel`` with ``n_vel ≈ (v_max - v_min) / v_step``. Phase B
    runs :func:`fit_ccf_rv` per chunk in a serial Python loop. Dominant levers:
    ``numba_threads`` (``max_cores``) for Phase A, and optional coarser
    ``ccf_v_step`` if acceptable for the use case.
    """
    if not order_tasks:
        empty = (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
        z64 = np.array([], dtype=np.int64)
        if return_ccf:
            return empty + (
                np.zeros((0, 0), dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.int32),
                z64,
                z64,
                z64,
            )
        return empty + (z64, z64)
    wave_2d = np.asarray(wave_2d, dtype=np.float64)
    spec_2d = np.asarray(spec_2d, dtype=np.float64)
    spec_err_2d = np.asarray(spec_err_2d, dtype=np.float64)
    dq_2d = np.asarray(dq_2d, dtype=np.uint32)
    template_ln = np.asarray(template_ln, dtype=np.float64)
    template_flux = np.asarray(template_flux, dtype=np.float64)

    order_indices = np.array([oi for oi, _ in order_tasks], dtype=np.int64)
    n_chunks_per_order = np.array([
        max(1, int((np.nanmax(wave_2d[oi]) - np.nanmin(wave_2d[oi])) / window_size_um))
        for oi, _ in order_tasks
    ], dtype=np.int64)
    chunk_start = np.zeros(len(order_indices), dtype=np.int64)
    for i in range(1, len(order_indices)):
        chunk_start[i] = chunk_start[i - 1] + n_chunks_per_order[i - 1]

    nthr = resolve_numba_threads(max_cores)
    vm = float(_V_MIN if ccf_v_min is None else ccf_v_min)
    vx = float(_V_MAX if ccf_v_max is None else ccf_v_max)
    vs = float(_V_STEP if ccf_v_step is None else ccf_v_step)
    hyc = ccf_hybrid_center_m_s
    hybrid = hyc is not None and np.isfinite(float(hyc))
    if not hybrid:
        if not np.isfinite(vm) or not np.isfinite(vx) or not np.isfinite(vs) or vs <= 0:
            raise ValueError(
                "ccf_v_min, ccf_v_max must be finite and ccf_v_step must be finite and > 0"
            )
        if vx <= vm:
            raise ValueError("ccf_v_max must be greater than ccf_v_min")
    total_chunks_layout = int(chunk_start[-1] + n_chunks_per_order[-1])
    if hybrid:
        logger.info(
            "CCF: process_orders_rvs_numba_parallel HYBRID mode n_order_tasks=%d "
            "total_chunks=%d window_um=%.4f return_ccf=%s numba_threads=%d "
            "center=%.2f m/s half_width=%.2f refined_step=%.4f",
            len(order_tasks),
            total_chunks_layout,
            window_size_um,
            return_ccf,
            nthr,
            float(hyc),
            float(ccf_hybrid_half_width_m_s),
            float(ccf_hybrid_step_m_s),
        )
    else:
        logger.info(
            "CCF: process_orders_rvs_numba_parallel n_order_tasks=%d total_chunks=%d "
            "window_um=%.4f return_ccf=%s numba_threads=%d "
            "v_grid_m_s=[%.0f, %.0f] step=%.1f",
            len(order_tasks),
            total_chunks_layout,
            window_size_um,
            return_ccf,
            nthr,
            vm,
            vx,
            vs,
        )
    # Flattened (order, chunk) RV list; parallelized over orders inside Numba.
    with numba_thread_scope(max_cores):
        if hybrid:
            kw_h = dict(
                hybrid_center_m_s=float(hyc),
                hybrid_half_width_m_s=float(ccf_hybrid_half_width_m_s),
                hybrid_step_m_s=float(ccf_hybrid_step_m_s),
            )
        else:
            kw_h = {}
        if return_ccf:
            all_rvs, all_rv_errs, all_wl, all_ccf, v_grid, all_n_good = process_all_orders_rvs_numba(
                wave_2d,
                spec_2d,
                spec_err_2d,
                dq_2d,
                do_not_use,
                template_ln,
                template_flux,
                window_size_um,
                order_indices,
                n_chunks_per_order,
                chunk_start,
                vm,
                vx,
                vs,
                return_ccf=True,
                numba_threads=nthr,
                **kw_h,
            )
            return (
                all_rvs,
                all_rv_errs,
                all_wl,
                all_ccf,
                v_grid,
                all_n_good,
                n_chunks_per_order,
                chunk_start,
                order_indices,
            )
        all_rvs, all_rv_errs, all_wl = process_all_orders_rvs_numba(
            wave_2d,
            spec_2d,
            spec_err_2d,
            dq_2d,
            do_not_use,
            template_ln,
            template_flux,
            window_size_um,
            order_indices,
            n_chunks_per_order,
            chunk_start,
            vm,
            vx,
            vs,
            return_ccf=False,
            numba_threads=nthr,
            **kw_h,
        )
        return all_rvs, all_rv_errs, all_wl, n_chunks_per_order, chunk_start


def finalize_ccf_preview_dict(
    order_slots: list[dict],
    all_ccf_rows: list[np.ndarray],
    v_grid: np.ndarray,
    v_min: float,
    v_max: float,
    v_step: float,
) -> dict:
    """
    Build the ``_ccf_preview_by_fiber`` structure from per-order CCF 2-D arrays.

    ``order_slots`` entries must already include per-order mean CCF rows; this
    prepends the global mean slot (mean over all chunk rows).
    """
    if not all_ccf_rows:
        return {
            "v_step": float(v_step),
            "v_min": float(v_min),
            "v_max": float(v_max),
            "slots": [],
        }
    stacked = np.vstack(all_ccf_rows)
    combined = np.nanmean(stacked, axis=0)
    mean_slot = {
        "kind": "mean",
        "order": None,
        "v": np.asarray(v_grid, dtype=np.float64),
        "ccf": np.asarray(combined, dtype=np.float64),
        "wl_lo": float("nan"),
        "wl_hi": float("nan"),
    }
    slots = [mean_slot] + order_slots
    vs = float(v_step)
    logger.info(
        "CCF preview: finalize_ccf_preview_dict fitting Gaussians n_slots=%d (mean + orders)",
        len(slots),
    )
    for sl in slots:
        wl = sl.get("wl_lo", np.nan)
        if not np.isfinite(wl):
            wl = sl.get("wl_hi", np.nan)
        vsp = dv_per_pixel_m_s(float(wl))
        rv, rv_err, params, diag = fit_ccf_rv(
            np.asarray(sl["v"], np.float64),
            np.asarray(sl["ccf"], np.float64),
            vs,
            return_params=True,
            vel_span_m_s_per_pixel=vsp if np.isfinite(vsp) else None,
        )
        sl["kpf_fit"] = {"rv": rv, "rv_err": rv_err, "params": params, "diagnostics": diag}
    logger.info(
        "CCF preview: finalize_ccf_preview_dict Gaussian fits done n_slots=%d",
        len(slots),
    )
    return {
        "v_step": float(v_step),
        "v_min": float(v_min),
        "v_max": float(v_max),
        "slots": slots,
    }


def ccf_preview_dict_from_numba_flat(
    order_tasks: list[tuple[int, int]],
    order_indices: np.ndarray,
    n_chunks_per_order: np.ndarray,
    chunk_start: np.ndarray,
    all_ccf: np.ndarray,
    v_grid: np.ndarray,
    wave_2d: np.ndarray,
    v_min: float,
    v_max: float,
    v_step: float,
) -> dict | None:
    """
    Build CCF preview dict from flattened Numba ``all_ccf`` (same layout as
    :func:`process_orders_rvs_numba_parallel` with ``return_ccf=True``).
    """
    if not order_tasks or len(v_grid) == 0:
        return None
    order_slots: list[dict] = []
    all_rows: list[np.ndarray] = []
    for idx in range(len(order_tasks)):
        oi = int(order_indices[idx])
        order_num = int(order_tasks[idx][1])
        start = int(chunk_start[idx])
        nch = int(n_chunks_per_order[idx])
        block = all_ccf[start : start + nch]
        if block.size == 0:
            continue
        ccf_m = np.nanmean(block, axis=0)
        w = wave_2d[oi]
        wl_lo, wl_hi = float(np.nanmin(w)), float(np.nanmax(w))
        order_slots.append(
            {
                "kind": "order",
                "order": order_num,
                "v": np.asarray(v_grid, dtype=np.float64),
                "ccf": np.asarray(ccf_m, dtype=np.float64),
                "wl_lo": wl_lo,
                "wl_hi": wl_hi,
            }
        )
        all_rows.append(block)
    if not order_slots:
        return None
    logger.info(
        "CCF preview: ccf_preview_dict_from_numba_flat n_orders=%d (mean CCF rows + finalize)",
        len(order_slots),
    )
    return finalize_ccf_preview_dict(order_slots, all_rows, v_grid, v_min, v_max, v_step)


def ccf_preview_dict_from_spectra_orders(
    order_tasks: list[tuple[int, int]],
    wave: np.ndarray,
    spec: np.ndarray,
    err: np.ndarray,
    dq: np.ndarray,
    do_not_use: int,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    window_size_um: float,
    v_min: float,
    v_max: float,
    v_step: float,
) -> dict | None:
    """
    Pure-Python CCF preview (recomputes chunk CCFs). Used when Numba is unavailable
    or for ``debug_ccf`` + preview where no cached Numba stack exists.
    """
    if not order_tasks:
        return None
    logger.info(
        "CCF preview: ccf_preview_dict_from_spectra_orders starting n_order_tasks=%d "
        "window_um=%.4f v_grid=[%.0f, %.0f] m/s step=%.1f",
        len(order_tasks),
        window_size_um,
        v_min,
        v_max,
        v_step,
    )
    order_slots: list[dict] = []
    all_rows: list[np.ndarray] = []
    v_grid_ref = None
    for oi, order in order_tasks:
        w = np.asarray(wave[oi], dtype=np.float64)
        s = np.asarray(spec[oi], dtype=np.float64).copy()
        e = np.asarray(err[oi] if err.ndim >= 2 else err, dtype=np.float64)
        d = np.asarray(dq[oi] if dq.ndim >= 2 else dq, dtype=np.uint32)
        mask = (d & do_not_use) != 0
        w_min, w_max = np.nanmin(w), np.nanmax(w)
        ln_min, ln_max = np.log(w_min), np.log(w_max)
        in_range = (template_ln >= ln_min - 0.01) & (template_ln <= ln_max + 0.01)
        if np.sum(in_range) < 20:
            continue
        t_ln = template_ln[in_range]
        t_flux = template_flux[in_range]
        _rvs, _rv_errs, _centers, v_grid, ccf_2d = rv_per_chunks(
            w,
            s,
            e,
            mask,
            t_ln,
            t_flux,
            window_size_um,
            v_min=v_min,
            v_max=v_max,
            v_step=v_step,
            return_ccf=True,
        )
        if v_grid_ref is None:
            v_grid_ref = v_grid
        elif v_grid_ref.shape != v_grid.shape or not np.allclose(v_grid_ref, v_grid):
            continue
        ccf_m = np.nanmean(ccf_2d, axis=0)
        order_slots.append(
            {
                "kind": "order",
                "order": int(order),
                "v": np.asarray(v_grid, dtype=np.float64),
                "ccf": np.asarray(ccf_m, dtype=np.float64),
                "wl_lo": float(np.nanmin(w)),
                "wl_hi": float(np.nanmax(w)),
            }
        )
        all_rows.append(ccf_2d)
    if not order_slots or v_grid_ref is None:
        logger.info(
            "CCF preview: ccf_preview_dict_from_spectra_orders no valid orders after CCF pass"
        )
        return None
    logger.info(
        "CCF preview: ccf_preview_dict_from_spectra_orders built %d order slots; calling finalize",
        len(order_slots),
    )
    return finalize_ccf_preview_dict(order_slots, all_rows, v_grid_ref, v_min, v_max, v_step)


# =============================================================================
# Continuum normalization (optional pre-step before CCF on extracted orders)
# =============================================================================


def continuum_normalize(
    flux: np.ndarray,
    wavelength: np.ndarray | None = None,
    lowpass_window: int | None = None,
    poly_order: int | None = None,
) -> np.ndarray:
    """
    Divide flux by an estimated continuum (median filter and/or polynomial).

    Used optionally before CCF in :func:`process_one_order_rvs` so line depths are
    less sensitive to slow spectral slope.

    Parameters
    ----------
    flux : np.ndarray
        1-D flux.
    wavelength : np.ndarray or None, optional
        Abscissa for polynomial continuum; if None, pixel index is used.
    lowpass_window : int or None, optional
        Odd median-filter window; even values are incremented. None skips filtering.
    poly_order : int or None, optional
        If set, fit continuum as a polynomial of this order (after median step if any).

    Returns
    -------
    np.ndarray
        ``flux / continuum``, same shape as ``flux``.

    Notes
    -----
    If both ``lowpass_window`` and ``poly_order`` are None, returns ``flux`` unchanged.
    """
    flux = np.asarray(flux, dtype=np.float64)
    n = len(flux)
    if n == 0 or (lowpass_window is None and poly_order is None):
        return flux
    x = np.asarray(wavelength, dtype=np.float64) if wavelength is not None else np.arange(n, dtype=np.float64)
    if len(x) != n:
        x = np.arange(n, dtype=np.float64)

    continuum = np.full_like(flux, np.nan)
    if lowpass_window is not None and lowpass_window > 0:
        w = int(lowpass_window)
        if w % 2 == 0:
            w += 1
        w = min(w, n)
        if w >= 3:
            continuum = median_filter(flux, size=w, mode="nearest")
        else:
            continuum = np.where(np.isfinite(flux), flux, np.nan)
    else:
        continuum = np.where(np.isfinite(flux), flux, np.nan)

    if poly_order is not None and poly_order >= 0:
        ok = np.isfinite(x) & np.isfinite(continuum) & (continuum > 0)
        if np.sum(ok) > max(poly_order + 1, 5):
            try:
                p = np.polyfit(x[ok], continuum[ok], poly_order)
                continuum = np.polyval(p, x)
            except Exception:
                pass

    continuum = np.maximum(continuum, 1e-30)
    return np.where(np.isfinite(flux) & np.isfinite(continuum), flux / continuum, flux)


def _ln_lam(wave: np.ndarray) -> np.ndarray:
    """Natural log of wavelength (micron → log micron). Internal helper."""
    return np.log(np.asarray(wave, dtype=np.float64))


def _make_velocity_grid(v_min: float, v_max: float, v_step: float) -> np.ndarray:
    """Uniform RV trial grid in m/s (matches Numba chunk CCF grid spacing)."""
    return np.arange(v_min, v_max + v_step * 0.5, v_step, dtype=np.float64)


def _ccf_weighted_covariance_scan(
    ln_lam: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    good: np.ndarray,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    v_grid: np.ndarray,
) -> np.ndarray:
    """Python CCF(v); same statistic as ``_ccf_chunk_ccf_numba`` (fallback when Numba off)."""
    ccf = np.zeros_like(v_grid, dtype=np.float64)
    for i, v in enumerate(v_grid):
        delta_ln = v / C_M_S
        t_at_data = np.interp(
            ln_lam,
            template_ln + delta_ln,
            template_flux,
            left=0.0,
            right=0.0,
        )
        use = good & np.isfinite(t_at_data) & (t_at_data != 0)
        if np.sum(use) < 5:
            ccf[i] = np.nan
            continue
        d = spec[use]
        t = t_at_data[use]
        w = 1.0 / np.maximum(spec_err[use] ** 2, 1e-30)
        d_norm = d - np.average(d, weights=w)
        t_norm = t - np.average(t, weights=w)
        ccf[i] = np.average(d_norm * t_norm, weights=w)
    return ccf


# =============================================================================
# Masked CCF on a single spectrum (full order or chunk): build CCF, then fit
# =============================================================================


def _masked_ccf_rv_core(
    wave: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    mask: np.ndarray,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    v_min: float,
    v_max: float,
    v_step: float,
    *,
    diagnostics: bool,
    v_grid: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray, dict | None]:
    """
    Single CCF build + fit; optionally build diagnostics dict (no duplicate CCF work).

    Returns
    -------
    rv, rv_err, v_grid, ccf, diagnostics_or_none
    """
    good = ~mask & np.isfinite(spec) & np.isfinite(spec_err) & (spec_err > 0)
    n_good = int(np.sum(good))
    wl_min = float(np.nanmin(wave))
    wl_max = float(np.nanmax(wave))
    spec_mean = float(np.nanmean(spec[good])) if n_good > 0 else np.nan
    spec_std = float(np.nanstd(spec[good])) if n_good > 1 else 0.0
    empty_diag = {
        "ccf_max": np.nan,
        "ccf_min": np.nan,
        "n_good": n_good,
        "n_use_at_peak": 0,
        "spec_mean": spec_mean,
        "spec_std": spec_std,
        "template_mean_at_peak": np.nan,
        "template_std_at_peak": np.nan,
        "wl_min": wl_min,
        "wl_max": wl_max,
    }

    if v_grid is not None:
        v_grid = np.asarray(v_grid, dtype=np.float64).ravel()
    else:
        v_grid = _make_velocity_grid(v_min, v_max, v_step)
    if n_good < 10:
        nan_ccf = np.full_like(v_grid, np.nan)
        if diagnostics:
            return np.nan, np.nan, v_grid, nan_ccf, empty_diag
        return np.nan, np.nan, v_grid, nan_ccf, None

    ln_lam = _ln_lam(wave)
    ccf = np.full(len(v_grid), np.nan, dtype=np.float64)
    v_step_fit = effective_ccf_v_step_for_fit(v_grid)
    if (
        len(v_grid) >= 2
        and np.allclose(np.diff(v_grid), v_grid[1] - v_grid[0], rtol=1e-5, atol=1e-3)
    ):
        _ccf_chunk_ccf_numba(
            ln_lam,
            spec,
            spec_err,
            good,
            template_ln,
            template_flux,
            float(v_grid[0]),
            float(v_grid[-1]),
            float(v_grid[1] - v_grid[0]),
            ccf,
        )
    else:
        ccf[:] = _ccf_weighted_covariance_scan(
            ln_lam, spec, spec_err, good, template_ln, template_flux, v_grid,
        )
    if not np.any(np.isfinite(ccf)):
        if diagnostics:
            return np.nan, np.nan, v_grid, ccf, empty_diag
        return np.nan, np.nan, v_grid, ccf, None

    rv, rv_err = _ccf_rv_from_ccf(v_grid, ccf, v_step_fit, n_good)
    diag_out: dict | None = None
    if diagnostics:
        ccf_max = float(np.nanmax(ccf))
        ccf_min = float(np.nanmin(ccf))
        delta_ln_peak = rv / C_M_S
        template_ln_shifted = template_ln + delta_ln_peak
        t_at_data = np.interp(
            ln_lam, template_ln_shifted, template_flux, left=0.0, right=0.0
        )
        use_peak = good & np.isfinite(t_at_data) & (t_at_data != 0)
        n_use_at_peak = int(np.sum(use_peak))
        t_mean = float(np.mean(t_at_data[use_peak])) if n_use_at_peak > 0 else np.nan
        t_std = float(np.std(t_at_data[use_peak])) if n_use_at_peak > 1 else 0.0
        diag_out = {
            "ccf_max": ccf_max,
            "ccf_min": ccf_min,
            "n_good": n_good,
            "n_use_at_peak": n_use_at_peak,
            "spec_mean": spec_mean,
            "spec_std": spec_std,
            "template_mean_at_peak": t_mean,
            "template_std_at_peak": t_std,
            "wl_min": wl_min,
            "wl_max": wl_max,
        }
    return rv, rv_err, v_grid, ccf, diag_out


def masked_ccf_rv(
    wave: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    mask: np.ndarray,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    v_min: float = -500000.0,
    v_max: float = 500000.0,
    v_step: float = 500.0,
) -> tuple[float, float]:
    """
    Radial velocity from a single masked spectrum via log-λ CCF and Gaussian peak fit.

    Builds a weighted covariance CCF on a velocity grid, then calls
    :func:`fit_ccf_rv` (sub-grid RV); on failure uses the grid peak and a
    HWHM-based error.  See module docstring for the CCF definition.

    Parameters
    ----------
    wave : np.ndarray
        Wavelength per pixel, micron.
    spec : np.ndarray
        Flux (same length as ``wave``).
    spec_err : np.ndarray
        Flux uncertainty (positive, finite where used).
    mask : np.ndarray
        Boolean; ``True`` excludes the pixel from the CCF.
    template_ln : np.ndarray
        Template ln(wavelength), same units as ``np.log(wave)``.
    template_flux : np.ndarray
        Template flux on ``template_ln``.
    v_min, v_max, v_step : float, optional
        CCF velocity range and sampling step (m/s). Defaults match broad stellar search.

    Returns
    -------
    rv : float
        Radial velocity (m/s); ``nan`` if too few good pixels or no finite CCF.
    rv_err : float
        Uncertainty (m/s) from the Gaussian fit covariance or HWHM fallback.

    See Also
    --------
    masked_ccf_rv_full : Same CCF plus returns ``v_grid`` and ``ccf`` for plotting.
    rv_per_chunks : Runs this per wavelength chunk along an order.
    """
    rv, rv_err, _vg, _ccf, _d = _masked_ccf_rv_core(
        wave, spec, spec_err, mask, template_ln, template_flux,
        v_min, v_max, v_step, diagnostics=False,
    )
    return rv, rv_err


def masked_ccf_rv_full(
    wave: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    mask: np.ndarray,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    v_min: float = -500000.0,
    v_max: float = 500000.0,
    v_step: float = 500.0,
    *,
    v_grid: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Same physics as :func:`masked_ccf_rv`, plus the CCF array for diagnostics/plots.

    Parameters
    ----------
    wave, spec, spec_err, mask, template_ln, template_flux
        Same as :func:`masked_ccf_rv`.
    v_min, v_max, v_step : float, optional
        Velocity grid in m/s. Unused for building the trial list when ``v_grid`` is passed.
    v_grid : np.ndarray or None, optional
        Explicit velocity samples (m/s), e.g. from :func:`build_hybrid_merged_velocity_grid`.

    Returns
    -------
    rv : float
        Radial velocity (m/s).
    rv_err : float
        Uncertainty (m/s).
    v_grid : np.ndarray
        Trial velocities (m/s), shape ``(n_vel,)``.
    ccf : np.ndarray
        CCF amplitude at each ``v_grid`` point (flux² scale); ``nan`` where invalid.

    See Also
    --------
    rv_per_chunks : Per-chunk CCFs along an order (``return_ccf=True``).
    """
    rv, rv_err, v_grid, ccf, _d = _masked_ccf_rv_core(
        wave,
        spec,
        spec_err,
        mask,
        template_ln,
        template_flux,
        v_min,
        v_max,
        v_step,
        diagnostics=False,
        v_grid=v_grid,
    )
    return rv, rv_err, v_grid, ccf


def masked_ccf_rv_diagnostics(
    wave: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    mask: np.ndarray,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    v_min: float = -500000.0,
    v_max: float = 500000.0,
    v_step: float = 500.0,
) -> tuple[float, float, dict]:
    """
    RV from masked CCF plus a diagnostics dict for logging and debug plots.

    Uses the same CCF and Gaussian/HWHM path as :func:`masked_ccf_rv`.  After the
    fit, evaluates the template shifted by the returned ``rv`` and reports template
    statistics at the data pixels.

    Parameters
    ----------
    wave, spec, spec_err, mask, template_ln, template_flux
        Same as :func:`masked_ccf_rv`.
    v_min, v_max, v_step : float, optional
        CCF grid (m/s).

    Returns
    -------
    rv : float
        Radial velocity (m/s).
    rv_err : float
        Uncertainty (m/s).
    diagnostics : dict
        ``ccf_max``, ``ccf_min``, ``n_good``, ``n_use_at_peak``, ``spec_mean``,
        ``spec_std``, ``template_mean_at_peak``, ``template_std_at_peak``,
        ``wl_min``, ``wl_max``.

    See Also
    --------
    masked_ccf_rv : RV only.
    """
    rv, rv_err, _vg, _ccf, diag = _masked_ccf_rv_core(
        wave, spec, spec_err, mask, template_ln, template_flux,
        v_min, v_max, v_step, diagnostics=True,
    )
    assert diag is not None
    return rv, rv_err, diag


# =============================================================================
# Split spectrum into wavelength chunks; each chunk → masked_ccf_rv
# =============================================================================


def rv_per_chunks(
    wave: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    mask: np.ndarray,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    window_size_um: float,
    v_min: float = -500000.0,
    v_max: float = 500000.0,
    v_step: float = 500.0,
    *,
    return_ccf: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """
    Split one echelle spectrum into wavelength chunks; one RV per chunk.

    Each chunk with ≥20 pixels uses :func:`_masked_ccf_rv_core` once (Numba CCF when
    available). Always returns chunk center wavelengths; with ``return_ccf=True``
    also returns ``v_grid`` and ``ccf_2d``.
    """
    w_min, w_max = np.nanmin(wave), np.nanmax(wave)
    n_chunk = max(1, int((w_max - w_min) / window_size_um))
    rvs = np.full(n_chunk, np.nan, dtype=np.float64)
    rv_errs = np.full(n_chunk, np.nan, dtype=np.float64)
    v_grid = _make_velocity_grid(v_min, v_max, v_step)
    ccf_2d = (
        np.full((n_chunk, len(v_grid)), np.nan, dtype=np.float64) if return_ccf else None
    )
    chunk_centers = np.full(n_chunk, np.nan, dtype=np.float64)

    for k in range(n_chunk):
        lam_lo = w_min + k * (w_max - w_min) / n_chunk
        lam_hi = w_min + (k + 1) * (w_max - w_min) / n_chunk
        in_chunk = (wave >= lam_lo) & (wave < lam_hi)
        chunk_centers[k] = 0.5 * (lam_lo + lam_hi)
        if np.sum(in_chunk) < 20:
            continue
        rv, rv_err, vg, ccf, _ = _masked_ccf_rv_core(
            wave[in_chunk],
            spec[in_chunk],
            spec_err[in_chunk],
            mask[in_chunk],
            template_ln,
            template_flux,
            v_min,
            v_max,
            v_step,
            diagnostics=False,
        )
        rvs[k] = rv
        rv_errs[k] = rv_err
        if return_ccf and ccf_2d is not None and vg.shape == v_grid.shape and np.allclose(vg, v_grid):
            ccf_2d[k, :] = ccf

    if return_ccf:
        assert ccf_2d is not None
        return rvs, rv_errs, chunk_centers, v_grid, ccf_2d
    return rvs, rv_errs, chunk_centers


# =============================================================================
# Single-order helper + multi-chunk CCF export (preview / diagnostics)
# =============================================================================


def process_one_order_rvs(
    wave: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    dq: np.ndarray,
    do_not_use: int,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    window_size_um: float,
    do_continuum_normalize: bool = False,
    continuum_lowpass_window: int | None = None,
    continuum_poly_order: int | None = None,
    *,
    return_ccf: bool = False,
    ccf_v_min: float | None = None,
    ccf_v_max: float | None = None,
    ccf_v_step: float | None = None,
    ccf_hybrid_center_m_s: float | None = None,
    ccf_hybrid_half_width_m_s: float = 300.0,
    ccf_hybrid_step_m_s: float = 0.1,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """
    One echelle order: clip template, optional continuum norm, chunk RVs.

    Serial-friendly (picklable) path used when not using the Numba multi-order
    driver.  Returns empty arrays if the template has fewer than 20 points in the
    order wavelength range. When ``ccf_hybrid_center_m_s`` is set, delegates to
    :func:`process_orders_rvs_numba_parallel` (broad + refined hybrid).

    Parameters
    ----------
    wave, spec, spec_err : np.ndarray
        1-D wavelength (micron), flux, uncertainty.
    dq : np.ndarray, uint32
        Data quality; pixels with ``(dq & do_not_use) != 0`` are masked.
    do_not_use : int
        DQ bitmask for bad pixels.
    template_ln, template_flux : np.ndarray
        Full-template arrays (log micron, flux).
    window_size_um : float
        Wavelength chunk width (micron).
    do_continuum_normalize : bool, optional
        If True and a continuum option is set, normalize flux before CCF.
    continuum_lowpass_window, continuum_poly_order : int or None, optional
        Passed to :func:`continuum_normalize`.
    return_ccf : bool, optional
        If True, also return ``v_grid`` and per-chunk CCF ``ccf_2d`` (uses
        ``ccf_v_min`` / ``ccf_v_max`` / ``ccf_v_step`` or defaults ``_V_*``).
    ccf_v_min, ccf_v_max, ccf_v_step : float or None, optional
        Velocity grid when ``return_ccf`` is True.

    Returns
    -------
    rvs : np.ndarray
        Per-chunk RV (m/s).
    rv_errs : np.ndarray
        Per-chunk uncertainties (m/s).
    chunk_centers : np.ndarray
        Chunk center wavelengths (micron).
    v_grid, ccf_2d : np.ndarray, optional
        Returned only when ``return_ccf`` is True.

    See Also
    --------
    process_orders_rvs_numba_parallel : Faster path for many orders at once.
    """
    w = np.asarray(wave, dtype=np.float64)
    s = np.asarray(spec, dtype=np.float64).copy()
    e = np.asarray(spec_err, dtype=np.float64)
    mask = (np.asarray(dq, dtype=np.uint32) & do_not_use) != 0

    if do_continuum_normalize and (continuum_lowpass_window is not None or continuum_poly_order is not None):
        s = continuum_normalize(
            s,
            wavelength=w,
            lowpass_window=continuum_lowpass_window,
            poly_order=continuum_poly_order,
        )

    w_min, w_max = np.nanmin(w), np.nanmax(w)
    ln_min, ln_max = np.log(w_min), np.log(w_max)
    in_range = (template_ln >= ln_min - 0.01) & (template_ln <= ln_max + 0.01)
    if np.sum(in_range) < 20:
        empty3 = (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
        if return_ccf:
            vm = _V_MIN if ccf_v_min is None else ccf_v_min
            vx = _V_MAX if ccf_v_max is None else ccf_v_max
            vs = _V_STEP if ccf_v_step is None else ccf_v_step
            vg = _make_velocity_grid(vm, vx, vs)
            return empty3 + (vg, np.zeros((0, len(vg)), dtype=np.float64))
        return empty3

    t_ln = template_ln[in_range]
    t_flux = template_flux[in_range]

    vm = _V_MIN if ccf_v_min is None else ccf_v_min
    vx = _V_MAX if ccf_v_max is None else ccf_v_max
    vs = _V_STEP if ccf_v_step is None else ccf_v_step

    if ccf_hybrid_center_m_s is not None:
        dq_1d = np.where(mask, np.uint32(do_not_use), np.uint32(0))
        wave_2d = w.reshape(1, -1)
        spec_2d = s.reshape(1, -1)
        err_2d = e.reshape(1, -1)
        dq_2d = dq_1d.reshape(1, -1)
        out = process_orders_rvs_numba_parallel(
            wave_2d,
            spec_2d,
            err_2d,
            dq_2d,
            do_not_use,
            t_ln,
            t_flux,
            window_size_um,
            [(0, 0)],
            max_cores=1,
            return_ccf=return_ccf,
            ccf_v_min=vm,
            ccf_v_max=vx,
            ccf_v_step=vs,
            ccf_hybrid_center_m_s=float(ccf_hybrid_center_m_s),
            ccf_hybrid_half_width_m_s=float(ccf_hybrid_half_width_m_s),
            ccf_hybrid_step_m_s=float(ccf_hybrid_step_m_s),
        )
        if return_ccf:
            arv, are, awl, accf, vg, ang, _nc, _cs, _oi = out
            return arv, are, awl, vg, accf
        arv, are, awl, _nc, _cs = out
        return arv, are, awl

    return rv_per_chunks(
        w,
        s,
        e,
        mask,
        t_ln,
        t_flux,
        window_size_um,
        v_min=vm,
        v_max=vx,
        v_step=vs,
        return_ccf=return_ccf,
    )


# =============================================================================
# Combine many chunk RVs: robust rejection then inverse-variance mean
# (used after stellar-template or mask-shift CCF per-chunk RVs)
# =============================================================================


def combine_rvs_median_mad(
    rvs: np.ndarray,
    rv_errs: np.ndarray,
    outlier_sigma: float = 4.0,
) -> tuple[np.ndarray, float, float]:
    """
    Robust outlier mask on per-chunk RVs (median ± ``outlier_sigma`` × MAD).

    Only chunks with finite ``rvs`` and positive ``rv_errs`` are considered for the
    median/MAD; invalid chunks are marked not inlier.

    Parameters
    ----------
    rvs, rv_errs : np.ndarray
        Same length; per-chunk RV (m/s) and reported error.
    outlier_sigma : float, optional
        Threshold in robust standard-deviation units (default 4).

    Returns
    -------
    inlier : np.ndarray of bool
        ``True`` for chunks to keep for weighted averaging.
    median_rv : float
        Median of valid RVs.
    mad_rv : float
        :func:`astropy.stats.mad_std` of valid RVs (falls back to ``std`` if needed).

    See Also
    --------
    weighted_mean_rv : Combine inliers by inverse variance.
    """
    # Only chunks with finite RV and positive reported error participate.
    valid = np.isfinite(rvs) & np.isfinite(rv_errs) & (rv_errs > 0)
    if not np.any(valid):
        return np.zeros_like(rvs, dtype=bool), np.nan, np.nan
    rv_valid = rvs[valid]
    median_rv = np.nanmedian(rv_valid)
    mad_rv = mad_std(rv_valid, ignore_nan=True)
    if mad_rv <= 0:
        mad_rv = np.nanstd(rv_valid)
    if not np.isfinite(mad_rv) or mad_rv <= 0:
        mad_rv = 1.0
    # Inlier if within outlier_sigma * MAD of the chunk median (symmetric rejection).
    threshold = outlier_sigma * mad_rv
    inlier = valid & (np.abs(rvs - median_rv) <= threshold)
    return inlier, float(median_rv), float(mad_rv)


def weighted_mean_rv(
    rvs: np.ndarray,
    rv_errs: np.ndarray,
    inlier: np.ndarray,
) -> tuple[float, float]:
    """
    Inverse-variance weighted mean RV and error on the mean.

    Weights are ``1 / rv_err**2`` for chunks where ``inlier`` is True and errors
    are finite and positive.

    Parameters
    ----------
    rvs, rv_errs : np.ndarray
        Per-chunk values (m/s).
    inlier : np.ndarray of bool
        Typically from :func:`combine_rvs_median_mad`.

    Returns
    -------
    rv_mean : float
        Combined RV (m/s).
    rv_std : float
        Standard error of the weighted mean, ``1/sqrt(sum_i w_i)`` (m/s).

    See Also
    --------
    combine_rvs_median_mad : Upstream outlier rejection.
    """
    use = inlier & np.isfinite(rvs) & np.isfinite(rv_errs) & (rv_errs > 0)
    if not np.any(use):
        return np.nan, np.nan
    w = 1.0 / (np.asarray(rv_errs, dtype=np.float64) ** 2)
    w = np.where(use, w, 0.0)
    if np.sum(w) <= 0:
        return np.nan, np.nan
    rv_mean = np.average(rvs, weights=w)
    # Error on weighted mean (Gaussian errors, uncorrelated chunks): 1/sqrt(sum w_i).
    var_mean = 1.0 / np.sum(w)
    rv_std = np.sqrt(var_mean)
    return float(rv_mean), float(rv_std)


# =============================================================================
# Mask-shift line CCF (SPIROU .mas / KPF two-column)
# =============================================================================

@njit(cache=True, parallel=True)
def _calc_mask_ccf_chunks_numba(
    left_edges_2d: np.ndarray,
    right_edges_2d: np.ndarray,
    pixel_width_2d: np.ndarray,
    s_use_2d: np.ndarray,
    sn_2d: np.ndarray,
    n_pix_arr: np.ndarray,
    v_grid_m_s: np.ndarray,
    zb: float,
    line_start_2d: np.ndarray,
    line_end_2d: np.ndarray,
    line_w_2d: np.ndarray,
    n_line_arr: np.ndarray,
) -> np.ndarray:
    """
    Chunk-batched mask CCF(v) with parallelism over chunks.

    Input arrays are padded to fixed widths; ``n_pix_arr`` and ``n_line_arr`` give
    valid lengths for each chunk row.
    """
    n_chunk = left_edges_2d.shape[0]
    n_v = v_grid_m_s.shape[0]
    out = np.full((n_chunk, n_v), np.nan, dtype=np.float64)
    zb1 = 1.0 + zb

    for ic in prange(n_chunk):
        n_pix = int(n_pix_arr[ic])
        n_ln = int(n_line_arr[ic])
        if n_pix <= 0 or n_ln <= 0:
            continue
        # Strict chunk policy: only include a shifted mask segment when it is
        # fully contained in this chunk's wavelength support; clipped/partial
        # edge overlaps are excluded.
        chunk_lo = left_edges_2d[ic, 0]
        chunk_hi = right_edges_2d[ic, n_pix - 1]
        for i_v in range(n_v):
            sh = (1.0 + v_grid_m_s[i_v] / C_M_S) / zb1
            mask_sp = np.zeros(n_pix, dtype=np.float64)
            for j in range(n_ln):
                ls = line_start_2d[ic, j] * sh
                le = line_end_2d[ic, j] * sh
                if not (np.isfinite(ls) and np.isfinite(le)) or le <= ls:
                    continue
                if ls < chunk_lo or le > chunk_hi:
                    continue
                cxp = np.searchsorted(right_edges_2d[ic, :n_pix], ls, side="left")
                cxn = np.searchsorted(left_edges_2d[ic, :n_pix], le, side="right")
                if cxn <= 0 or cxp >= n_pix - 1:
                    continue
                lo = cxp if cxp > 0 else 0
                hi = cxn if cxn < n_pix else n_pix
                if hi <= lo:
                    continue
                for n in range(lo, hi):
                    le_n = left_edges_2d[ic, n]
                    ri = right_edges_2d[ic, n]
                    if le_n <= le and ri >= ls:
                        ws = le_n if le_n > ls else ls
                        we = ri if ri < le else le
                        ov = we - ws
                        pwi = pixel_width_2d[ic, n]
                        if ov > 0.0 and np.isfinite(pwi) and pwi > 0.0:
                            mask_sp[n] += line_w_2d[ic, j] * (ov / pwi)
            acc = 0.0
            for n in range(n_pix):
                acc += s_use_2d[ic, n] * mask_sp[n] * sn_2d[ic, n]
            out[ic, i_v] = acc
    return out


@dataclass(frozen=True)
class MaskLines:
    """
    Mask line segments in wavelength space (microns).

    All arrays are 1-D and equal length after validation in ``__post_init__``.

    Attributes
    ----------
    start_um : np.ndarray
        Segment start wavelength (µm), same frame as L2 ``WAVE_*`` (typically vacuum).
    end_um : np.ndarray
        Segment end wavelength (µm).
    weight : np.ndarray
        Per-segment weight used in the mask CCF sum.
    """

    start_um: np.ndarray
    end_um: np.ndarray
    weight: np.ndarray

    def __post_init__(self) -> None:
        start_um = np.asarray(self.start_um, dtype=np.float64)
        end_um = np.asarray(self.end_um, dtype=np.float64)
        weight = np.asarray(self.weight, dtype=np.float64)
        if start_um.shape != end_um.shape or start_um.shape != weight.shape:
            raise ValueError("MaskLines arrays must have matching shapes.")
        if start_um.ndim != 1:
            raise ValueError("MaskLines arrays must be 1D.")

        object.__setattr__(self, "start_um", start_um)
        object.__setattr__(self, "end_um", end_um)
        object.__setattr__(self, "weight", weight)


def resolve_mask_path(mask_path: str | None) -> str:
    """
    Resolve a user mask path or the packaged default to an absolute path.

    Parameters
    ----------
    mask_path : str or None
        User path to a ``.mas`` or two-column mask. If ``None``, uses
        ``hispecdrp/data/masks/M2_filt_allRV.mas`` (SPIROU-style default).

    Returns
    -------
    str
        Absolute filesystem path.

    Raises
    ------
    FileNotFoundError
        If the default packaged mask is missing and ``mask_path`` was ``None``.

    Notes
    -----
    Default mask provenance: spirou-ccf ``ccf_masks`` on GitHub; see this module’s
    ``References`` section and ``hispecdrp/data/masks/README.md``.
    """

    if mask_path:
        p = Path(mask_path).expanduser()
        return str(p if p.is_absolute() else (Path.cwd() / p).resolve())

    # Default packaged file: .../hispecdrp/data/masks/
    default_file = Path(__file__).resolve().parents[2] / "data" / "masks" / "M2_filt_allRV.mas"
    if not default_file.exists():
        raise FileNotFoundError(
            f"Default mask not found: {default_file}. "
            "Pass StellarRVPrimitive(mask_path=...) to use a custom mask."
        )
    return str(default_file)


def _infer_and_convert_to_micron(values: np.ndarray) -> np.ndarray:
    """
    Infer wavelength unit scale and convert to micron.

    Used for **KPF-style two-column** masks only. Heuristics (median of positive finite
    values):

    - median > 1e3 -> Angstrom (e.g. 8000 Å) -> /1e4
    - median > 10  -> nm (e.g. 1500 nm)      -> /1e3
    - otherwise assume already micron.

    SPIROU three-column ``.mas`` files are loaded by :func:`load_mask_spirou_mas`, which
    always treats the wavelength columns as **vacuum nanometers** (NIR values like 1136 nm
    must not use this function's ``median > 1e3 -> Å`` rule, which would map them to ~0.11 µm).
    """

    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v) & (v > 0)
    if not np.any(finite):
        return v
    med = float(np.nanmedian(v[finite]))
    if med > 1e3:
        return v / 1e4
    if med > 10:
        return v / 1e3
    return v


def load_mask_spirou_mas(mask_file: str) -> MaskLines:
    """
    Load a SPIROU-style three-column ``.mas`` mask file.

    Each data row: wavelength start, wavelength end, weight. Wavelength columns are
    **divided by 1000** to convert **nanometers → microns**. No additional air→vacuum
    correction is applied in code. Bundled HISPEC masks document **vacuum** nm in file
    headers; some upstream SPIROU/APERO masks may use **air** nm—in that case convert
    externally or pre-process the file.

    Comment lines beginning with ``#``, ``;``, or ``//`` are skipped.

    Parameters
    ----------
    mask_file : str
        Path to the ``.mas`` file.

    Returns
    -------
    MaskLines
        Segment boundaries and weights in microns (same spectral frame as the numbers
        in the file after the nm→µm scale change).

    See Also
    --------
    load_mask_kpf_two_column : KPF-style two-column masks.
    """

    try:
        data = np.loadtxt(mask_file, comments="#")
    except OSError as e:
        raise ValueError(f"Cannot read mask file: {mask_file}") from e
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"Need ≥3 columns (start_nm, end_nm, weight): {mask_file}")
    start = data[:, 0].astype(np.float64) / 1e3
    end = data[:, 1].astype(np.float64) / 1e3
    weight = data[:, 2].astype(np.float64)
    lo = np.minimum(start, end)
    hi = np.maximum(start, end)
    return MaskLines(start_um=lo, end_um=hi, weight=weight)


def load_mask_kpf_two_column(mask_file: str, *, mask_width_um: float | None) -> MaskLines:
    """
    Load a KPF-style two-column mask file.

    The file is interpreted as:
    - wavelength_center_or_start : column 0
    - weight                      : column 1

    Because the input does not include an explicit segment width, callers must
    provide ``mask_width_um`` to define a half-width around each wavelength.
    If mask_width_um is None, a conservative width based on median spacing is used.
    """

    try:
        data = np.loadtxt(mask_file, comments="#")
    except OSError as e:
        raise ValueError(f"Cannot read mask file: {mask_file}") from e
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"Need ≥2 columns (wavelength, weight): {mask_file}")
    w = data[:, 0].astype(np.float64)
    weight = data[:, 1].astype(np.float64)
    w = _infer_and_convert_to_micron(w)

    w_sorted = np.sort(w[np.isfinite(w)])
    if w_sorted.size >= 2:
        median_spacing = float(np.nanmedian(np.diff(w_sorted)))
    else:
        median_spacing = 1e-5

    # Default width: a small fraction of median spacing (keeps mask compact).
    if mask_width_um is None:
        mask_width_um = max(median_spacing * 0.2, 1e-6)

    half = 0.5 * float(mask_width_um)
    return MaskLines(start_um=w - half, end_um=w + half, weight=weight)


def load_mask(mask_path: str, *, mask_width_um: float | None = None) -> MaskLines:
    """
    Load a line mask from disk, dispatching on file extension.

    Parameters
    ----------
    mask_path : str
        Path to ``.mas`` (SPIROU-style) or ``.txt`` / ``.dat`` / ``.csv`` (KPF-style).
    mask_width_um : float or None, optional
        Half-width of each segment in µm for two-column masks only; ignored for
        ``.mas`` files that carry explicit start/end columns.

    Returns
    -------
    MaskLines
        Parsed segments in microns.

    Notes
    -----
    Unknown extensions try SPIROU parsing first, then KPF two-column.
    """
    p = Path(mask_path)
    suffix = p.suffix.lower()
    if suffix == ".mas":
        return load_mask_spirou_mas(mask_path)
    if suffix in (".txt", ".dat", ".csv"):
        return load_mask_kpf_two_column(mask_path, mask_width_um=mask_width_um)
    # Best-effort fallback: try SPIROU, then KPF 2-col.
    try:
        return load_mask_spirou_mas(mask_path)
    except Exception:
        return load_mask_kpf_two_column(mask_path, mask_width_um=mask_width_um)


def pixel_edges_from_wavelength(wave_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pixel bin edges for each wavelength pixel.

    Returns (left_edges_um, right_edges_um), each of shape (n_pixel,).
    """

    w = np.asarray(wave_um, dtype=np.float64).ravel()
    if w.size < 2:
        raise ValueError("Need at least 2 wavelength points to build pixel edges.")

    # KPF-style construction (midpoints between adjacent pixels, with end padding).
    x_pixel_wave_start = (w + np.roll(w, 1)) / 2.0
    x_pixel_wave_end = np.roll(x_pixel_wave_start, -1)
    x_pixel_wave_start[0] = w[0] - (w[1] - w[0]) / 2.0
    x_pixel_wave_end[-1] = w[-1] + (w[-1] - w[-2]) / 2.0
    left = x_pixel_wave_start
    right = x_pixel_wave_end
    return left, right


def restrict_mask_to_chunk(
    mask_lines: MaskLines,
    w_min_um: float,
    w_max_um: float,
    *,
    v_min_m_s: float,
    v_max_m_s: float,
    zb: float = 0.0,
) -> MaskLines:
    """
    Restrict mask lines to those that can overlap the chunk wavelength range
    within the configured velocity search limits.
    """

    # shift_factor = (1+v/c)/(1+zb)
    shift_low = (1.0 + v_min_m_s / C_M_S) / (1.0 + zb)
    shift_high = (1.0 + v_max_m_s / C_M_S) / (1.0 + zb)

    # A line overlaps [w_min_um, w_max_um] for some v in [v_min, v_max] if:
    # - its shifted end can reach above w_min
    # - its shifted start can reach below w_max
    keep = (mask_lines.end_um * shift_high >= w_min_um) & (mask_lines.start_um * shift_low <= w_max_um)
    if not np.any(keep):
        return MaskLines(
            start_um=np.array([], dtype=np.float64),
            end_um=np.array([], dtype=np.float64),
            weight=np.array([], dtype=np.float64),
        )
    return MaskLines(
        start_um=mask_lines.start_um[keep],
        end_um=mask_lines.end_um[keep],
        weight=mask_lines.weight[keep],
    )


@njit(cache=True)
def _nanmedian_numba_1d(x: np.ndarray) -> float:
    vals = np.empty(x.size, dtype=np.float64)
    n = 0
    for i in range(x.size):
        xi = x[i]
        if np.isfinite(xi):
            vals[n] = xi
            n += 1
    if n == 0:
        return np.nan
    vv = np.sort(vals[:n])
    mid = n // 2
    if (n % 2) == 1:
        return float(vv[mid])
    return float(0.5 * (vv[mid - 1] + vv[mid]))


@njit(cache=True)
def _rv_from_ccf_mask_kpf_numba_single(
    v_grid_m_s: np.ndarray,
    ccf_raw: np.ndarray,
    n_good: int,
) -> tuple[float, float, int]:
    n = v_grid_m_s.size
    if ccf_raw.size != n:
        return np.nan, np.nan, 0

    ccf_min = np.inf
    ccf_max = -np.inf
    for i in range(n):
        yi = ccf_raw[i]
        if np.isfinite(yi):
            if yi < ccf_min:
                ccf_min = yi
            if yi > ccf_max:
                ccf_max = yi
    if not np.isfinite(ccf_min) or not np.isfinite(ccf_max):
        return np.nan, np.nan, 0

    baseline_raw = _nanmedian_numba_1d(ccf_raw)
    height = float(ccf_max - baseline_raw)
    depth = float(baseline_raw - ccf_min)
    sign = -1.0 if depth > height else 1.0
    flipped = 1 if sign < 0.0 else 0

    i_peak = -1
    y_peak = -np.inf
    for i in range(n):
        yi = sign * ccf_raw[i]
        if np.isfinite(yi) and yi > y_peak:
            y_peak = yi
            i_peak = i
    if i_peak < 0:
        return np.nan, np.nan, flipped

    v_peak = float(v_grid_m_s[i_peak])
    v_step_edge = float(abs(v_grid_m_s[1] - v_grid_m_s[0])) if n > 1 else 1.0
    if i_peak <= 0 or i_peak >= (n - 1):
        rv_err_edge = max(float(v_step_edge * 2.0), 1.0)
        return v_peak, rv_err_edge, flipped

    v0 = float(v_grid_m_s[i_peak - 1])
    v1 = float(v_grid_m_s[i_peak])
    v2 = float(v_grid_m_s[i_peak + 1])
    h_m = v1 - v0
    h_p = v2 - v1
    v_step_eff = 0.5 * (h_m + h_p)
    if h_m <= 0.0 or h_p <= 0.0:
        return v_peak, max(float(abs(v_step_eff) * 2.0), 1.0), flipped

    y_m1 = float(sign * ccf_raw[i_peak - 1])
    y_0 = float(sign * ccf_raw[i_peak])
    y_p1 = float(sign * ccf_raw[i_peak + 1])
    baseline = float(sign * baseline_raw)
    amp = y_0 - baseline
    if not np.isfinite(amp) or amp <= 0.0:
        return v_peak, max(float(abs(v_step_eff) * 2.0), 1.0), flipped

    # Parabola y = a*v^2 + b*v + c through (v0,y_m1), (v1,y_0), (v2,y_p1); vertex at -b/(2a).
    A = v1 * v1 - v0 * v0
    B = v1 - v0
    C = y_0 - y_m1
    D = v2 * v2 - v1 * v1
    E = v2 - v1
    F = y_p1 - y_0
    det = A * E - B * D
    if not np.isfinite(det) or abs(det) < 1e-30:
        return v_peak, max(float(abs(v_step_eff) * 2.0), 1.0), flipped
    a = (C * E - B * F) / det
    b = (A * F - C * D) / det
    if not np.isfinite(a) or abs(a) < 1e-30:
        return v_peak, max(float(abs(v_step_eff) * 2.0), 1.0), flipped
    rv = -b / (2.0 * a)
    if rv < min(v0, v2) or rv > max(v0, v2):
        rv = v_peak

    ypp = 2.0 * a
    if not np.isfinite(ypp) or ypp >= 0.0:
        sigma_v = float(abs(v_step_eff) * 2.0)
    else:
        sigma_v2 = max(float(-amp / ypp), (0.1 * float(abs(v_step_eff))) ** 2)
        sigma_v = float(np.sqrt(sigma_v2))
    n_scale = max(float(n_good), 1.0) / 50.0
    rv_err = float(max(sigma_v / np.sqrt(n_scale), float(abs(v_step_eff))))
    return float(rv), float(rv_err), flipped


@njit(parallel=True, cache=True)
def _rv_from_ccf_mask_kpf_rows_numba(
    v_grid_m_s: np.ndarray,
    ccf_rows: np.ndarray,
    n_good_arr: np.ndarray,
    out_rv: np.ndarray,
    out_rv_err: np.ndarray,
    out_flip: np.ndarray,
) -> None:
    n_chunk = ccf_rows.shape[0]
    for i in prange(n_chunk):
        rv, rv_err, flipped = _rv_from_ccf_mask_kpf_numba_single(
            v_grid_m_s,
            ccf_rows[i],
            int(n_good_arr[i]),
        )
        out_rv[i] = rv
        out_rv_err[i] = rv_err
        out_flip[i] = flipped


def rv_per_chunks_mask_kpf(
    wave_um: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    dq_bad_mask: np.ndarray,
    mask_lines: MaskLines,
    *,
    window_size_um: float,
    v_min_m_s: float = _V_MIN,
    v_max_m_s: float = _V_MAX,
    v_step_m_s: float = _V_STEP,
    v_grid_m_s: np.ndarray | None = None,
    mask_restrict_v_min_m_s: float | None = None,
    mask_restrict_v_max_m_s: float | None = None,
    zb: float = 0.0,
    max_cores: int | None = None,
    return_ccf: bool = False,
    min_good_pixels: int = 10,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Compute per-chunk RVs and optional CCF stacks for one echelle order.

    Splits the order into ``n_chunk`` wavelength bins of width set by
    ``window_size_um`` (derived from full order span), then computes chunk CCF rows
    in one Numba batch via :func:`_calc_mask_ccf_chunks_numba` (parallel over chunks),
    followed by a Numba chunk-parallel parabolic RV extraction. Returns flattened lists of
    chunk centers and RVs.

    Parameters
    ----------
    wave_um, spec, spec_err : np.ndarray
        Wavelength (µm), flux, and error; same length.
    dq_bad_mask : np.ndarray
        True where pixel should be excluded (e.g. DO_NOT_USE).
    mask_lines : MaskLines
        Full-order mask; restricted per chunk via :func:`restrict_mask_to_chunk`.
    window_size_um : float
        Target chunk width in µm (actual count adapts to order span).
    v_min_m_s, v_max_m_s, v_step_m_s : float, optional
        Velocity search grid (m s⁻¹). Ignored when ``v_grid_m_s`` is provided.
    v_grid_m_s : np.ndarray or None, optional
        Explicit velocity samples (m s⁻¹), e.g. from :func:`build_hybrid_merged_velocity_grid`.
        When set, ``v_min_m_s`` / ``v_max_m_s`` / ``v_step_m_s`` are not used to build the grid
        but still scope mask restriction unless ``mask_restrict_*`` override.
    zb : float, optional
        Barycentric factor passed to the CCF shift.
    max_cores : int or None, optional
        Numba thread count cap for chunk-parallel CCF evaluation.
    return_ccf : bool, optional
        If True, also return mean CCF rows per chunk (stacked for preview).
    min_good_pixels : int, optional
        Minimum unmasked finite pixels required per chunk.

    Returns
    -------
    rvs : np.ndarray
        Per-chunk RV (m s⁻¹).
    rv_errs : np.ndarray
        Per-chunk RV uncertainty.
    centers : np.ndarray
        Chunk center wavelength (µm).
    v_grid : np.ndarray
        Velocity grid used for the CCF.
    ccf_2d : np.ndarray or None
        If ``return_ccf``, shape ``(n_chunks, n_v)`` of CCFs; else ``None``.
    wl_lo, wl_hi : np.ndarray
        Per-chunk wavelength bounds (µm), same length as ``rvs``.
    n_good : np.ndarray
        Per-chunk count of good pixels used in the mask CCF.
    """

    w = np.asarray(wave_um, dtype=np.float64).ravel()
    s = np.asarray(spec, dtype=np.float64).ravel()
    e = np.asarray(spec_err, dtype=np.float64).ravel()
    d = np.asarray(dq_bad_mask, dtype=bool).ravel()

    def _empty(v_grid_local: np.ndarray):
        ccf_out = np.empty((0, len(v_grid_local)), dtype=np.float64) if return_ccf else None
        z = np.array([], dtype=np.float64)
        zi = np.array([], dtype=np.int32)
        return (z, z, z, v_grid_local, ccf_out, z, z, zi)

    vr_lo = float(v_min_m_s if mask_restrict_v_min_m_s is None else mask_restrict_v_min_m_s)
    vr_hi = float(v_max_m_s if mask_restrict_v_max_m_s is None else mask_restrict_v_max_m_s)

    good_pix = ~d & np.isfinite(w) & np.isfinite(s) & np.isfinite(e)
    if v_grid_m_s is not None:
        v_grid = np.asarray(v_grid_m_s, dtype=np.float64).ravel()
    else:
        v_grid = np.arange(
            v_min_m_s,
            v_max_m_s + 0.5 * v_step_m_s,
            v_step_m_s,
            dtype=np.float64,
        )
    if np.sum(good_pix) < min_good_pixels:
        return _empty(v_grid)

    w_min, w_max = float(np.nanmin(w)), float(np.nanmax(w))
    if not np.isfinite(w_min) or not np.isfinite(w_max) or w_max <= w_min:
        return _empty(v_grid)

    n_chunk = max(1, int((w_max - w_min) / float(window_size_um)))

    all_rvs: list[float] = []
    all_rv_errs: list[float] = []
    all_centers: list[float] = []
    all_wl_lo: list[float] = []
    all_wl_hi: list[float] = []
    all_n_good_out: list[int] = []
    all_ccf_rows: list[np.ndarray] = []
    chunk_payloads: list[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            float,
            float,
            float,
            int,
        ]
    ] = []

    for k in range(n_chunk):
        lam_lo = w_min + k * (w_max - w_min) / n_chunk
        lam_hi = w_min + (k + 1) * (w_max - w_min) / n_chunk
        in_chunk = (w >= lam_lo) & (w < lam_hi) & np.isfinite(s) & np.isfinite(e) & (~d)
        if int(np.sum(in_chunk)) < min_good_pixels:
            continue

        wl_center = 0.5 * (lam_lo + lam_hi)

        w_c = w[in_chunk]
        s_c = s[in_chunk]
        e_c = e[in_chunk]

        # Restrict mask lines so we don't waste time on clearly non-overlapping
        # segments; expanded to cover the configured velocity range.
        mask_c = restrict_mask_to_chunk(
            mask_lines,
            float(np.nanmin(w_c)),
            float(np.nanmax(w_c)),
            v_min_m_s=vr_lo,
            v_max_m_s=vr_hi,
            zb=zb,
        )
        if mask_c.start_um.size == 0:
            continue

        left_edges, right_edges = pixel_edges_from_wavelength(w_c)
        pixel_width = right_edges - left_edges
        pixel_width = np.where(pixel_width > 0, pixel_width, np.nan)
        finite = np.isfinite(s_c) & np.isfinite(e_c) & (e_c > 0) & np.isfinite(pixel_width)
        if int(np.sum(finite)) < min_good_pixels:
            continue
        s_use = np.where(finite, s_c, 0.0).astype(np.float64)
        sn = np.where(finite, 1.0, 0.0).astype(np.float64)
        chunk_payloads.append(
            (
                left_edges.astype(np.float64),
                right_edges.astype(np.float64),
                pixel_width.astype(np.float64),
                s_use,
                sn,
                mask_c.start_um.astype(np.float64),
                mask_c.end_um.astype(np.float64),
                mask_c.weight.astype(np.float64),
                float(lam_lo),
                float(lam_hi),
                float(wl_center),
                int(np.sum(in_chunk)),
            )
        )

    if not chunk_payloads:
        return _empty(v_grid)

    n_chunk_valid = len(chunk_payloads)
    n_pix_max = max(len(t[0]) for t in chunk_payloads)
    n_line_max = max(len(t[5]) for t in chunk_payloads)
    left_2d = np.full((n_chunk_valid, n_pix_max), np.nan, dtype=np.float64)
    right_2d = np.full((n_chunk_valid, n_pix_max), np.nan, dtype=np.float64)
    pwidth_2d = np.full((n_chunk_valid, n_pix_max), np.nan, dtype=np.float64)
    s_use_2d = np.zeros((n_chunk_valid, n_pix_max), dtype=np.float64)
    sn_2d = np.zeros((n_chunk_valid, n_pix_max), dtype=np.float64)
    n_pix_arr = np.zeros(n_chunk_valid, dtype=np.int64)
    line_start_2d = np.full((n_chunk_valid, n_line_max), np.nan, dtype=np.float64)
    line_end_2d = np.full((n_chunk_valid, n_line_max), np.nan, dtype=np.float64)
    line_w_2d = np.zeros((n_chunk_valid, n_line_max), dtype=np.float64)
    n_line_arr = np.zeros(n_chunk_valid, dtype=np.int64)
    centers = np.zeros(n_chunk_valid, dtype=np.float64)
    n_good_arr = np.zeros(n_chunk_valid, dtype=np.int64)
    wl_lo_arr = np.zeros(n_chunk_valid, dtype=np.float64)
    wl_hi_arr = np.zeros(n_chunk_valid, dtype=np.float64)

    for i, payload in enumerate(chunk_payloads):
        left_i, right_i, pwi_i, s_i, sn_i, ls_i, le_i, lw_i, lam_lo_i, lam_hi_i, wl_i, ng_i = payload
        npix = len(left_i)
        nln = len(ls_i)
        left_2d[i, :npix] = left_i
        right_2d[i, :npix] = right_i
        pwidth_2d[i, :npix] = pwi_i
        s_use_2d[i, :npix] = s_i
        sn_2d[i, :npix] = sn_i
        n_pix_arr[i] = npix
        line_start_2d[i, :nln] = ls_i
        line_end_2d[i, :nln] = le_i
        line_w_2d[i, :nln] = lw_i
        n_line_arr[i] = nln
        centers[i] = wl_i
        n_good_arr[i] = ng_i
        wl_lo_arr[i] = lam_lo_i
        wl_hi_arr[i] = lam_hi_i

    with numba_thread_scope(max_cores):
        ccf_rows = _calc_mask_ccf_chunks_numba(
            left_2d,
            right_2d,
            pwidth_2d,
            s_use_2d,
            sn_2d,
            n_pix_arr,
            v_grid,
            float(zb),
            line_start_2d,
            line_end_2d,
            line_w_2d,
            n_line_arr,
        )
        rv_rows = np.empty(n_chunk_valid, dtype=np.float64)
        rv_err_rows = np.empty(n_chunk_valid, dtype=np.float64)
        flip_rows = np.zeros(n_chunk_valid, dtype=np.int64)
        _rv_from_ccf_mask_kpf_rows_numba(
            v_grid,
            ccf_rows,
            n_good_arr.astype(np.int64),
            rv_rows,
            rv_err_rows,
            flip_rows,
        )

    for i in range(n_chunk_valid):
        rv = float(rv_rows[i])
        rv_err = float(rv_err_rows[i])
        if not np.isfinite(rv):
            continue
        all_rvs.append(float(rv))
        all_rv_errs.append(float(rv_err) if np.isfinite(rv_err) and rv_err > 0 else np.nan)
        all_centers.append(float(centers[i]))
        all_wl_lo.append(float(wl_lo_arr[i]))
        all_wl_hi.append(float(wl_hi_arr[i]))
        all_n_good_out.append(int(n_good_arr[i]))
        if return_ccf:
            all_ccf_rows.append(-ccf_rows[i] if int(flip_rows[i]) == 1 else ccf_rows[i].copy())
    if not all_rvs:
        return _empty(v_grid)

    rvs_arr = np.asarray(all_rvs, dtype=np.float64)
    rv_errs_arr = np.asarray(all_rv_errs, dtype=np.float64)
    centers_arr = np.asarray(all_centers, dtype=np.float64)
    wl_lo_out = np.asarray(all_wl_lo, dtype=np.float64)
    wl_hi_out = np.asarray(all_wl_hi, dtype=np.float64)
    n_good_out = np.asarray(all_n_good_out, dtype=np.int32)
    if return_ccf:
        ccf_2d = np.vstack(all_ccf_rows).astype(np.float64) if all_ccf_rows else np.empty((0, len(v_grid)))
    else:
        ccf_2d = None
    return rvs_arr, rv_errs_arr, centers_arr, v_grid, ccf_2d, wl_lo_out, wl_hi_out, n_good_out


def run_ccf_mask_kpf_for_fiber(
    *,
    input_model,
    fiber: str,
    do_orders: list[int],
    order_vec: list[int],
    window_size_um: float,
    outlier_sigma: float,
    mask_path: str | None,
    mask_width_um: float | None = None,
    v_min_m_s: float = _V_MIN,
    v_max_m_s: float = _V_MAX,
    v_step_m_s: float = _V_STEP,
    ccf_hybrid_center_m_s: float | None = None,
    ccf_hybrid_half_width_m_s: float = 300.0,
    ccf_hybrid_step_m_s: float = 0.1,
    zb: float = 0.0,
    max_cores: int | None = None,
    want_preview: bool = False,
    debug_ccf: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    np.ndarray,
    dict | None,
    Table,
]:
    """
    Run mask-line CCF RV for one fiber (``algorithm="ccf_mask_kpf"``).

    Reads ``SPEC_FIB*``, ``SPEC_ERR_FIB*``, ``SPEC_DQ_FIB*``, ``WAVE_FIB*`` from
    ``input_model``, loads the mask (see :func:`resolve_mask_path`), and processes
    each requested echelle order with :func:`rv_per_chunks_mask_kpf`. Chunk RVs are
    combined with :func:`combine_rvs_median_mad` and :func:`weighted_mean_rv`.

    Parameters
    ----------
    input_model
        L2-like datamodel with fiber spectra and wavelengths.
    fiber : str
        Fiber label (e.g. ``"A"``) used in attribute names.
    do_orders : list of int
        Echelle orders to include.
    order_vec : list of int
        Full order indexing aligned with array axis 0.
    window_size_um : float
        Wavelength chunk width (µm).
    outlier_sigma : float
        Median/MAD outlier threshold for chunk RV combination.
    mask_path : str or None
        Path to mask file; ``None`` selects packaged default (see README).
    mask_width_um : float or None, optional
        Segment half-width for two-column masks only.
    v_min_m_s, v_max_m_s, v_step_m_s : float, optional
        **Broad** uniform velocity search (m s⁻¹). When ``ccf_hybrid_center_m_s`` is set,
        these define the coarse leg of a **merged** grid (see
        :func:`build_hybrid_merged_velocity_grid`); when unset, they define the sole CCF grid.
    ccf_hybrid_center_m_s, ccf_hybrid_half_width_m_s, ccf_hybrid_step_m_s : float or None
        When ``ccf_hybrid_center_m_s`` is finite, CCF is evaluated once on the merged velocity
        list (broad ``v_min_m_s`` … ``v_max_m_s`` at ``v_step_m_s`` plus dense samples around
        the center). When ``None`` / not finite, only the broad grid is used.
    zb : float, optional
        Barycentric factor for mask shift.
    max_cores : int or None, optional
        Numba thread count cap used for chunk-parallel mask CCF evaluation.
    want_preview : bool, optional
        If True, build order-averaged CCF data for Plotly preview.
    debug_ccf : bool, optional
        Extra logging per order.

    Returns
    -------
    all_wavelengths : np.ndarray
        Chunk center wavelengths (µm).
    all_rvs : np.ndarray
        Per-chunk RVs (m s⁻¹).
    all_rv_errs : np.ndarray
        Per-chunk uncertainties.
    inlier : np.ndarray
        Boolean mask after outlier rejection.
    rv_mean : float
        Inverse-variance weighted mean RV (m s⁻¹).
    rv_std : float
        Combined uncertainty (m s⁻¹).
    all_orders : np.ndarray
        Echelle order index per chunk (-1 if length mismatch).
    preview_detail : dict or None
        Payload for stellar RV preview, or None.
    rvccf_chunk_supplement : astropy.table.Table
        Columns ``CHUNK_ID``, ``N_GOOD``, ``V_MIN``, ``V_MAX``, ``V_STEP``, ``N_VEL``, ``CCF``
        merged into ``RVCCF_FIB*`` via :func:`assemble_rvccf_fib_table`.

    Notes
    -----
    Mask and algorithm provenance: module docstring ``References``;
    ``hispecdrp/data/masks/README.md``.
    """

    logger.info(
        "CCF mask_kpf: starting fiber=%s n_do_orders=%d window_um=%.4f want_preview=%s "
        "debug_ccf=%s",
        fiber,
        len(do_orders),
        window_size_um,
        want_preview,
        debug_ccf,
    )
    if not np.isfinite(window_size_um) or float(window_size_um) <= 0:
        raise ValueError("window_size_um must be finite and > 0 for ccf_mask_kpf")
    hybrid = bool(
        ccf_hybrid_center_m_s is not None
        and np.isfinite(float(ccf_hybrid_center_m_s))
    )
    if hybrid:
        logger.info(
            "CCF mask_kpf: HYBRID mode center=%.2f half_width=%.2f refined_step=%.4f (m/s)",
            float(ccf_hybrid_center_m_s),
            float(ccf_hybrid_half_width_m_s),
            float(ccf_hybrid_step_m_s),
        )
    else:
        logger.info(
            "CCF mask_kpf: velocity grid (m/s) v_min=%.2f v_max=%.2f step=%.4f",
            float(v_min_m_s),
            float(v_max_m_s),
            float(v_step_m_s),
        )

    spec_attr = f"SPEC_FIB{fiber}"
    err_attr = f"SPEC_ERR_FIB{fiber}"
    dq_attr = f"SPEC_DQ_FIB{fiber}"
    wave_attr = f"WAVE_FIB{fiber}"

    spec = getattr(input_model, spec_attr)
    err = getattr(input_model, err_attr) if hasattr(input_model, err_attr) else np.ones_like(spec)
    dq = getattr(input_model, dq_attr) if hasattr(input_model, dq_attr) else np.zeros_like(spec, dtype=np.uint32)
    wave = getattr(input_model, wave_attr)

    if err is None:
        err = np.ones_like(spec, dtype=np.float64)
    if dq is None:
        dq = np.zeros_like(spec, dtype=np.uint32)

    if hybrid:
        _h_vc = float(ccf_hybrid_center_m_s)
        _h_hw = float(ccf_hybrid_half_width_m_s)
        _h_vsf = float(ccf_hybrid_step_m_s)
        v_grid = build_hybrid_merged_velocity_grid(
            float(v_min_m_s),
            float(v_max_m_s),
            float(v_step_m_s),
            _h_vc,
            _h_hw,
            _h_vsf,
        )
    else:
        _h_vc = _h_hw = _h_vsf = float("nan")
        v_grid = np.arange(
            v_min_m_s,
            v_max_m_s + 0.5 * v_step_m_s,
            v_step_m_s,
            dtype=np.float64,
        )

    # Load mask once per fiber run (mask does not depend on order).
    resolved_mask = resolve_mask_path(mask_path)
    mask_lines = load_mask(resolved_mask, mask_width_um=mask_width_um)
    if mask_lines.start_um.size == 0:
        raise ValueError(f"Mask loaded no lines: {resolved_mask}")
    logger.info(
        "CCF mask_kpf: loaded mask path=%s n_line_segments=%d",
        resolved_mask,
        int(mask_lines.start_um.size),
    )

    # TODO: refine mask-CCF DQ filtering policy. Today we only gate on DO_NOT_USE;
    # review which spectral-quality bits are populated in L2 for this path and whether
    # additional flags (e.g. LOW_INST_TRANS, TELLURIC, OH_LINE) should be excluded.
    do_not_use = int(DQ_FLAGS.get("DO_NOT_USE", 1))
    # We treat any non-zero "do_not_use" bits as bad pixels.
    bad = (np.asarray(dq, dtype=np.uint32) & do_not_use) != 0

    order_tasks = [(oi, order) for oi, order in enumerate(order_vec) if order in do_orders]
    logger.info(
        "CCF mask_kpf: fiber=%s will process n_order_tasks=%d (echelle orders in do_orders)",
        fiber,
        len(order_tasks),
    )

    all_rvs: list[float] = []
    all_rv_errs: list[float] = []
    all_wavelengths: list[float] = []
    all_orders: list[int] = []

    # When True, collect per-order CCF stacks for Plotly preview (Gaussian refit in
    # finalize_ccf_preview_dict). Disable preview if unstable CCF shapes appear in your env.
    want_preview_effective = bool(want_preview)

    order_slots: list[dict] = []
    all_ccf_rows: list[np.ndarray] = []

    rvccf_rows_accum: list[np.ndarray] = []
    rvccf_n_good_accum: list[int] = []

    for oi, order in order_tasks:
        w = np.asarray(wave[oi], dtype=np.float64)
        s = np.asarray(spec[oi], dtype=np.float64)
        e = np.asarray(err[oi] if getattr(err, "ndim", 0) >= 2 else err, dtype=np.float64)
        d_bad = np.asarray(bad[oi] if getattr(bad, "ndim", 0) >= 2 else bad, dtype=bool)

        w_min, w_max = float(np.nanmin(w)), float(np.nanmax(w))
        if not np.isfinite(w_min) or not np.isfinite(w_max) or w_max <= w_min:
            continue

        # Further subdivide an order when its span is larger than window_size_um.
        # Each sub-part is equal width and <= window_size_um.
        n_sub = max(1, int(np.ceil((w_max - w_min) / float(window_size_um))))
        order_ccf_rows: list[np.ndarray] = []
        order_v_grid: np.ndarray | None = None
        n_chunks_order = 0

        for isub in range(n_sub):
            lam_lo = w_min + isub * (w_max - w_min) / n_sub
            lam_hi = w_min + (isub + 1) * (w_max - w_min) / n_sub
            if isub == n_sub - 1:
                in_sub = (w >= lam_lo) & (w <= lam_hi)
            else:
                in_sub = (w >= lam_lo) & (w < lam_hi)
            if int(np.sum(in_sub)) < 20:
                continue

            if not hybrid:
                (
                    rvs_o,
                    errs_o,
                    centers_o,
                    v_grid_o,
                    ccf_2d,
                    _wl_lo_o,
                    _wl_hi_o,
                    ng_o,
                ) = rv_per_chunks_mask_kpf(
                    w[in_sub],
                    s[in_sub],
                    e[in_sub],
                    d_bad[in_sub],
                    mask_lines,
                    window_size_um=window_size_um,
                    v_min_m_s=v_min_m_s,
                    v_max_m_s=v_max_m_s,
                    v_step_m_s=v_step_m_s,
                    zb=zb,
                    max_cores=max_cores,
                    return_ccf=True,
                )
            else:
                (
                    rvs_o,
                    errs_o,
                    centers_o,
                    v_grid_o,
                    ccf_2d,
                    _wl_lo_o,
                    _wl_hi_o,
                    ng_o,
                ) = rv_per_chunks_mask_kpf(
                    w[in_sub],
                    s[in_sub],
                    e[in_sub],
                    d_bad[in_sub],
                    mask_lines,
                    window_size_um=window_size_um,
                    v_min_m_s=float(v_min_m_s),
                    v_max_m_s=float(v_max_m_s),
                    v_step_m_s=float(v_step_m_s),
                    v_grid_m_s=v_grid,
                    zb=zb,
                    max_cores=max_cores,
                    return_ccf=True,
                )
            if rvs_o.size == 0:
                continue

            n_chunks_order += int(rvs_o.size)
            all_rvs.extend([float(x) for x in rvs_o])
            all_rv_errs.extend([float(x) for x in errs_o])
            all_wavelengths.extend([float(x) for x in centers_o])
            all_orders.extend([int(order)] * int(len(centers_o)))

            if ccf_2d is not None and ccf_2d.size > 0:
                for ii in range(int(rvs_o.size)):
                    rvccf_rows_accum.append(np.asarray(ccf_2d[ii], dtype=np.float64))
                    rvccf_n_good_accum.append(int(ng_o[ii]))

            if want_preview_effective and ccf_2d is not None and ccf_2d.size > 0:
                v_now = np.asarray(v_grid_o, dtype=np.float64)
                if order_v_grid is None:
                    order_v_grid = v_now
                    order_ccf_rows.append(ccf_2d)
                elif order_v_grid.shape == v_now.shape and np.allclose(order_v_grid, v_now):
                    order_ccf_rows.append(ccf_2d)

        if want_preview_effective and order_ccf_rows and order_v_grid is not None:
            stacked = np.vstack(order_ccf_rows)
            ccf_mean = np.nanmean(stacked, axis=0)
            order_slots.append(
                {
                    "kind": "order",
                    "order": int(order),
                    "v": order_v_grid,
                    "ccf": np.asarray(ccf_mean, dtype=np.float64),
                    "wl_lo": float(w_min),
                    "wl_hi": float(w_max),
                }
            )
            all_ccf_rows.append(stacked)

        if debug_ccf and n_chunks_order > 0:
            logger.info(
                "CCF-mask_kpf fiber=%s order=%s: n_suborders=%d n_chunks=%d rv_mean~%.1f m/s",
                fiber,
                order,
                n_sub,
                n_chunks_order,
                float(np.nanmedian(all_rvs[-n_chunks_order:])),
            )

    all_rvs_arr = np.asarray(all_rvs, dtype=np.float64)
    all_rv_errs_arr = np.asarray(all_rv_errs, dtype=np.float64)
    all_wavelengths_arr = np.asarray(all_wavelengths, dtype=np.float64)
    all_orders_arr = np.asarray(all_orders, dtype=np.int64)
    if len(all_orders_arr) != len(all_wavelengths_arr):
        all_orders_arr = np.full(len(all_wavelengths_arr), -1, dtype=np.int64)

    valid = np.isfinite(all_rvs_arr) & np.isfinite(all_rv_errs_arr)
    if not np.any(valid):
        inlier = np.zeros(len(all_rvs_arr), dtype=bool)
        rv_mean = np.nan
        rv_std = np.nan
    else:
        inlier, _, _ = combine_rvs_median_mad(all_rvs_arr, all_rv_errs_arr, outlier_sigma)
        rv_mean, rv_std = weighted_mean_rv(all_rvs_arr, all_rv_errs_arr, inlier)

    if hybrid:
        pv_min = float(v_grid[0])
        pv_max = float(v_grid[-1])
        pv_step = float(effective_ccf_v_step_for_fit(v_grid))
    else:
        pv_min = float(v_min_m_s)
        pv_max = float(v_max_m_s)
        pv_step = float(v_step_m_s)

    preview_detail = None
    if want_preview_effective and order_slots and all_ccf_rows:
        logger.info(
            "CCF mask_kpf: fiber=%s building preview n_preview_orders=%d",
            fiber,
            len(order_slots),
        )
        preview_detail = finalize_ccf_preview_dict(
            order_slots,
            all_ccf_rows,
            v_grid,
            pv_min,
            pv_max,
            pv_step,
        )

    logger.info(
        "CCF mask_kpf: fiber=%s finished n_chunks=%d rv_mean=%.2f m/s rv_err=%.2f m/s "
        "n_inlier=%d/%d preview=%s",
        fiber,
        len(all_wavelengths_arr),
        float(rv_mean),
        float(rv_std),
        int(np.sum(inlier)) if len(inlier) else 0,
        len(inlier),
        preview_detail is not None,
    )

    n_rvccf = len(rvccf_rows_accum)
    if n_rvccf > 0:
        rvccf_tbl = build_rvccf_chunk_supplement(
            chunk_id=np.arange(n_rvccf, dtype=np.int32),
            n_good=np.asarray(rvccf_n_good_accum, dtype=np.int32),
            v_min=pv_min,
            v_max=pv_max,
            v_step=pv_step,
            ccf_2d=np.stack(rvccf_rows_accum, axis=0),
        )
    else:
        rvccf_tbl = build_rvccf_chunk_supplement(
            chunk_id=np.array([], dtype=np.int32),
            n_good=np.array([], dtype=np.int32),
            v_min=pv_min,
            v_max=pv_max,
            v_step=pv_step,
            ccf_2d=np.empty((0, 0), dtype=np.float64),
        )

    return (
        all_wavelengths_arr,
        all_rvs_arr,
        all_rv_errs_arr,
        inlier,
        float(rv_mean),
        float(rv_std),
        all_orders_arr,
        preview_detail,
        rvccf_tbl,
    )


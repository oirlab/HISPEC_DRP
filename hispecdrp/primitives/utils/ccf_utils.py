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
.. [2] ``alg_rv_mask_line.py`` (mask geometry / BC bookkeeping in KPF):
       https://github.com/Keck-DataReductionPipelines/KPF-Pipeline/blob/81609b0ac0ab6d12536fb771bfdd0112001340c3/modules/radial_velocity/src/alg_rv_mask_line.py
.. [3] spirou-ccf (SPIRou spectra / CCF ecosystem):
       https://github.com/edermartioli/spirou-ccf/blob/main/spiroulib.py
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass

import numpy as np
from astropy.modeling import fitting
from astropy.modeling.models import Gaussian1D
from scipy.ndimage import median_filter
from astropy.stats import mad_std
from pathlib import Path

from ...datamodels.dqflags import DQ_FLAGS
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

# Typical velocity scale per pixel for L2 (BSPEC/RSPEC) from wavelength sampling:
#   Δv = c * (Δλ/λ).  With λ_center = 147/order µm and pixel size Δλ = (147/order²)/4096 µm,
#   velocity per pixel = c / (order * 4096) m/s.
#   BSPEC (order 110–148): ~494–665 m/s per pixel.  RSPEC (order 60–97): ~755–1220 m/s per pixel.

logger = logging.getLogger(__name__)

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
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """
    High-throughput per-chunk RVs: Numba CCF grid, then SciPy Gaussian per chunk.

    This is the workhorse behind :func:`process_orders_rvs_numba_parallel`.  It is
    not re-exported from ``hispecdrp.primitives.utils``; callers typically use the
    parallel wrapper that sets thread count and packs ``order_tasks``.

    **Pipeline**

    #. Allocate ``all_ccf`` with shape ``(total_chunks, n_vel)`` and flatten chunk
       indices across orders via ``chunk_start`` / ``n_chunks_per_order``.
    #. :func:`_process_all_orders_ccf_numba` — parallel over orders (``prange``):
       for each wavelength chunk, :func:`_ccf_chunk_ccf_numba` fills one row of
       ``all_ccf`` (same weighted covariance as :func:`masked_ccf_rv`).
    #. Python loop: for each row with enough good pixels, :func:`_ccf_rv_from_ccf`
       calls :func:`fit_ccf_rv` + HWHM fallback (serial; cannot run inside Numba).

    Parameters
    ----------
    wave_2d, spec_2d, spec_err_2d, dq_2d
        Per-order 2-D arrays (wavelength micron, flux, error, DQ).
    do_not_use : int
        DQ bitmask for bad pixels.
    template_ln, template_flux
        Full template (log micron, flux).
    window_size_um : float
        Chunk width in wavelength (micron) along each order.
    order_indices : np.ndarray
        Row indices into ``wave_2d`` for orders to process.
    n_chunks_per_order, chunk_start : np.ndarray
        Chunk counts and cumulative start index into the flattened output.
    v_min, v_max, v_step : float
        CCF velocity grid (m/s); usually ``_V_MIN``, ``_V_MAX``, ``_V_STEP``.

    Returns
    -------
    all_rvs : np.ndarray
        RV per chunk (m/s), length ``total_chunks``.
    all_rv_errs : np.ndarray
        Uncertainty per chunk (m/s).
    all_wl : np.ndarray
        Chunk center wavelength (micron); ``nan`` where order skipped.
    all_ccf, v_grid : np.ndarray, optional
        Returned when ``return_ccf`` is True: CCF stack ``(total_chunks, n_vel)`` and
        shared velocity axis (m/s).

    See Also
    --------
    process_orders_rvs_numba_parallel : Public entry with ``order_tasks`` and threads.
    """
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
        return all_rvs, all_rv_errs, all_wl, all_ccf, v_grid
    return all_rvs, all_rv_errs, all_wl


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
        Override the CCF velocity grid (m/s). Defaults are ``_V_MIN``, ``_V_MAX``,
        ``_V_STEP`` (50 m/s steps). **Coarser** ``ccf_v_step`` (e.g. 200–500 m/s)
        cuts Phase A and Phase B cost roughly proportional to ``_V_STEP / ccf_v_step``
        at modest RV resolution cost.

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
    all_ccf, v_grid, order_indices : np.ndarray, optional
        Extra arrays when ``return_ccf`` is True (for preview / diagnostics).

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
    if not np.isfinite(vm) or not np.isfinite(vx) or not np.isfinite(vs) or vs <= 0:
        raise ValueError("ccf_v_min, ccf_v_max must be finite and ccf_v_step must be finite and > 0")
    if vx <= vm:
        raise ValueError("ccf_v_max must be greater than ccf_v_min")
    total_chunks_layout = int(chunk_start[-1] + n_chunks_per_order[-1])
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
        if return_ccf:
            all_rvs, all_rv_errs, all_wl, all_ccf, v_grid = process_all_orders_rvs_numba(
                wave_2d, spec_2d, spec_err_2d, dq_2d, do_not_use,
                template_ln, template_flux, window_size_um,
                order_indices, n_chunks_per_order, chunk_start,
                vm, vx, vs,
                return_ccf=True,
                numba_threads=nthr,
            )
            return (
                all_rvs,
                all_rv_errs,
                all_wl,
                all_ccf,
                v_grid,
                n_chunks_per_order,
                chunk_start,
                order_indices,
            )
        all_rvs, all_rv_errs, all_wl = process_all_orders_rvs_numba(
            wave_2d, spec_2d, spec_err_2d, dq_2d, do_not_use,
            template_ln, template_flux, window_size_um,
            order_indices, n_chunks_per_order, chunk_start,
            vm, vx, vs,
            return_ccf=False,
            numba_threads=nthr,
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

    v_grid = _make_velocity_grid(v_min, v_max, v_step)
    if n_good < 10:
        nan_ccf = np.full_like(v_grid, np.nan)
        if diagnostics:
            return np.nan, np.nan, v_grid, nan_ccf, empty_diag
        return np.nan, np.nan, v_grid, nan_ccf, None

    ln_lam = _ln_lam(wave)
    ccf = np.full(len(v_grid), np.nan, dtype=np.float64)
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

    rv, rv_err = _ccf_rv_from_ccf(v_grid, ccf, v_step, n_good)
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
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Same physics as :func:`masked_ccf_rv`, plus the CCF array for diagnostics/plots.

    Parameters
    ----------
    wave, spec, spec_err, mask, template_ln, template_flux
        Same as :func:`masked_ccf_rv`.
    v_min, v_max, v_step : float, optional
        Velocity grid in m/s.

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
        wave, spec, spec_err, mask, template_ln, template_flux,
        v_min, v_max, v_step, diagnostics=False,
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
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """
    One echelle order: clip template, optional continuum norm, chunk RVs.

    Serial-friendly (picklable) path used when not using the Numba multi-order
    driver.  Returns empty arrays if the template has fewer than 20 points in the
    order wavelength range.

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

    if return_ccf:
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
            return_ccf=True,
        )

    return rv_per_chunks(w, s, e, mask, t_ln, t_flux, window_size_um, return_ccf=False)


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

@njit(cache=True)
def _calc_mask_ccf_numba(
    left_edges: np.ndarray,
    right_edges: np.ndarray,
    pixel_width: np.ndarray,
    s_use: np.ndarray,
    sn: np.ndarray,
    v_grid_m_s: np.ndarray,
    zb: float,
    line_start: np.ndarray,
    line_end: np.ndarray,
    line_w: np.ndarray,
) -> np.ndarray:
    """Mask-shift CCF(v): sum_n flux_n * mask_n(v) * sn_n over pixels."""
    n_pix = left_edges.shape[0]
    n_v = v_grid_m_s.shape[0]
    n_ln = line_start.shape[0]
    ccf = np.empty(n_v, dtype=np.float64)
    zb1 = 1.0 + zb
    
    for i_v in range(n_v):
        sh = (1.0 + v_grid_m_s[i_v] / C_M_S) / zb1
        mask_sp = np.zeros(n_pix, dtype=np.float64)
        for j in range(n_ln):
            ls = line_start[j] * sh
            le = line_end[j] * sh
            if not (np.isfinite(ls) and np.isfinite(le)) or le <= ls:
                continue
            cxp = np.searchsorted(right_edges, ls, side="left")
            cxn = np.searchsorted(left_edges, le, side="right")
            if cxn <= 0 or cxp >= n_pix - 1:
                continue
            lo = cxp if cxp > 0 else 0
            hi = cxn if cxn < n_pix else n_pix
            if hi <= lo:
                continue
            for n in range(lo, hi):
                le_n = left_edges[n]
                ri = right_edges[n]
                if le_n <= le and ri >= ls:
                    ws = le_n if le_n > ls else ls
                    we = ri if ri < le else le
                    ov = we - ws
                    pwi = pixel_width[n]
                    if ov > 0.0 and np.isfinite(pwi) and pwi > 0.0:
                        mask_sp[n] += line_w[j] * (ov / pwi)
        acc = 0.0
        for n in range(n_pix):
            acc += s_use[n] * mask_sp[n] * sn[n]
        ccf[i_v] = acc
    return ccf


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


def calc_ccf_mask_kpf_like(
    wave_um: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray | None,
    mask_lines: MaskLines,
    v_grid_m_s: np.ndarray,
    *,
    zb: float = 0.0,
    sn_floor: float = 0.0,
) -> np.ndarray:
    """
    Compute mask-shifted :math:`\\mathrm{CCF}(v)` for one spectral chunk.

    Builds pixel bin edges from the wavelength grid, then for each trial velocity
    Doppler-shifts mask segments and accumulates overlap-weighted mask density per
    pixel. The CCF value is the sum of flux × mask × signal-to-noise factor over
    pixels (KPF-style structure; see module ``References``).

    Parameters
    ----------
    wave_um : np.ndarray
        Wavelength samples (µm), same length as ``flux``.
    flux : np.ndarray
        Flux per pixel.
    flux_err : np.ndarray or None
        Flux error per pixel; if None, errors are taken as unity internally.
    mask_lines : MaskLines
        Line segments in µm.
    v_grid_m_s : np.ndarray
        Trial velocities (m s⁻¹).
    zb : float, optional
        Barycentric correction factor :math:`(1+z_b)` applied in the shift denominator
        (default 0 → factor 1).
    sn_floor : float, optional
        Floor for SN weighting on bad pixels.

    Returns
    -------
    np.ndarray
        CCF array, same shape as ``v_grid_m_s``.
    """

    w = np.asarray(wave_um, dtype=np.float64).ravel()
    s = np.asarray(flux, dtype=np.float64).ravel()
    if flux_err is None:
        e = np.ones_like(s, dtype=np.float64)
    else:
        e = np.asarray(flux_err, dtype=np.float64).ravel()

    if w.size != s.size or w.size != e.size:
        raise ValueError("wave/flux/flux_err must have matching lengths.")
    n_pixel = w.size
    if n_pixel < 3:
        return np.full_like(v_grid_m_s, np.nan, dtype=np.float64)
    if mask_lines.start_um.size == 0:
        return np.full_like(v_grid_m_s, np.nan, dtype=np.float64)

    left_edges, right_edges = pixel_edges_from_wavelength(w)
    pixel_width = right_edges - left_edges
    pixel_width = np.where(pixel_width > 0, pixel_width, np.nan)

    finite = np.isfinite(s) & np.isfinite(e) & (e > 0) & np.isfinite(pixel_width)
    if not np.any(finite):
        return np.full_like(v_grid_m_s, np.nan, dtype=np.float64)

    s_use = np.where(finite, s, 0.0)
    # KPF's python path uses an SNR scaling array of ones. Using sn=1/err
    # can cancel absorption features for synthetic spectra where err ∝ flux.
    sn = np.where(finite, 1.0, sn_floor)

    return _calc_mask_ccf_numba(
        left_edges,
        right_edges,
        pixel_width,
        s_use,
        sn,
        np.asarray(v_grid_m_s, dtype=np.float64).ravel(),
        float(zb),
        mask_lines.start_um,
        mask_lines.end_um,
        mask_lines.weight,
    )


def rv_from_ccf_mask_kpf(
    v_grid_m_s: np.ndarray,
    ccf_raw: np.ndarray,
    *,
    v_step_m_s: float,
    n_good: int,
) -> tuple[float, float, np.ndarray]:
    """
    Estimate radial velocity and uncertainty from a mask CCF.

    The CCF may appear as a positive peak or a negative trough depending on data and
    mask sign conventions; the array is negated when a trough is deeper than the peak
    is tall, so subsequent peak finding targets a maximum. RV is refined with a
    three-point parabolic interpolation; uncertainty uses curvature and a simple
    :math:`\\sqrt{N}` scaling heuristic (no SciPy Gaussian fit in this path).

    Parameters
    ----------
    v_grid_m_s : np.ndarray
        Velocity grid (m s⁻¹), same length as ``ccf_raw``.
    ccf_raw : np.ndarray
        CCF values on ``v_grid_m_s``.
    v_step_m_s : float
        Uniform grid step (m s⁻¹) for parabolic refinement.
    n_good : int
        Number of good pixels in the chunk (scales reported error).

    Returns
    -------
    rv : float
        Radial velocity (m s⁻¹).
    rv_err : float
        Estimated uncertainty (m s⁻¹).
    ccf : np.ndarray
        CCF array used for peak finding (possibly sign-flipped copy of ``ccf_raw``).
    """

    ccf_raw = np.asarray(ccf_raw, dtype=np.float64).ravel()
    if ccf_raw.size != v_grid_m_s.size:
        raise ValueError("ccf_raw and v_grid must have matching lengths.")

    ccf_min = float(np.nanmin(ccf_raw))
    ccf_max = float(np.nanmax(ccf_raw))

    if not np.isfinite(ccf_min) or not np.isfinite(ccf_max):
        return np.nan, np.nan, np.full_like(v_grid_m_s, np.nan, dtype=np.float64)

    baseline = float(np.nanmedian(ccf_raw))
    # Decide whether the alignment feature is a trough or a peak by comparing
    # its excursion relative to the baseline.
    height = float(ccf_max - baseline)
    depth = float(baseline - ccf_min)
    # If the trough is deeper than the peak is tall, flip sign so the fitter sees
    # a positive maximum at the true RV location.
    if depth > height:
        ccf = -ccf_raw
    else:
        ccf = ccf_raw

    # Stable (non-SciPy) RV estimation:
    # - use quadratic interpolation across the maximum (3-point parabolic fit)
    # - estimate a Gaussian-equivalent sigma from curvature
    # - report rv_err from sigma scaled by sqrt(N) heuristics
    i_peak = int(np.nanargmax(ccf))
    v_peak = float(v_grid_m_s[i_peak])

    # Default fallback: grid peak only.
    if i_peak <= 0 or i_peak >= (len(ccf) - 1):
        rv_err = max(float(abs(v_step_m_s) * 2.0), 1.0)
        return v_peak, rv_err, ccf

    y_m1 = float(ccf[i_peak - 1])
    y_0 = float(ccf[i_peak])
    y_p1 = float(ccf[i_peak + 1])

    baseline = float(np.nanmedian(ccf))
    amp = y_0 - baseline

    # If the peak isn't clearly above baseline, uncertainty becomes very large.
    if not np.isfinite(amp) or amp <= 0:
        rv_err = max(float(abs(v_step_m_s) * 2.0), 1.0)
        return v_peak, rv_err, ccf

    # Parabolic vertex offset for uniform grid:
    # d = (y_- - y_+) / (2*(y_- - 2*y_0 + y_+)) * v_step
    denom = 2.0 * (y_m1 - 2.0 * y_0 + y_p1)
    if not np.isfinite(denom) or denom == 0.0:
        rv_err = max(float(abs(v_step_m_s) * 2.0), 1.0)
        return v_peak, rv_err, ccf

    d_v = (y_m1 - y_p1) / denom * float(v_step_m_s)
    rv = v_peak + d_v

    # Curvature-based sigma estimate:
    # For y = baseline + amp * exp(-0.5*(x/sigma)^2), near the peak:
    # y''(0) = -amp / sigma^2.
    # For quadratic interpolation around a max with spacing v_step:
    # y''(0) ~= (y_- + y_+ - 2*y_0) / v_step^2.
    ypp = (y_m1 + y_p1 - 2.0 * y_0) / (float(v_step_m_s) ** 2)
    if not np.isfinite(ypp) or ypp >= 0:
        sigma_v = float(abs(v_step_m_s) * 2.0)
    else:
        sigma_v2 = max(float(-amp / ypp), (0.1 * float(abs(v_step_m_s))) ** 2)
        sigma_v = float(np.sqrt(sigma_v2))

    # Error scaling heuristic: more good pixels -> smaller error.
    n_scale = max(float(n_good), 1.0) / 50.0
    rv_err = float(max(sigma_v / np.sqrt(n_scale), float(abs(v_step_m_s))))
    return float(rv), float(rv_err), ccf


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
    zb: float = 0.0,
    return_ccf: bool = False,
    min_good_pixels: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Compute per-chunk RVs and optional CCF stacks for one echelle order.

    Splits the order into ``n_chunk`` wavelength bins of width set by
    ``window_size_um`` (derived from full order span), runs
    :func:`calc_ccf_mask_kpf_like` + :func:`rv_from_ccf_mask_kpf` per chunk, and
    returns flattened lists of chunk centers and RVs.

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
        Velocity search grid (m s⁻¹).
    zb : float, optional
        Barycentric factor passed to the CCF shift.
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
    """

    w = np.asarray(wave_um, dtype=np.float64).ravel()
    s = np.asarray(spec, dtype=np.float64).ravel()
    e = np.asarray(spec_err, dtype=np.float64).ravel()
    d = np.asarray(dq_bad_mask, dtype=bool).ravel()

    def _empty(v_grid_local: np.ndarray):
        ccf_out = np.empty((0, len(v_grid_local)), dtype=np.float64) if return_ccf else None
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            v_grid_local,
            ccf_out,
        )

    good_pix = ~d & np.isfinite(w) & np.isfinite(s) & np.isfinite(e)
    v_grid = np.arange(v_min_m_s, v_max_m_s + 0.5 * v_step_m_s, v_step_m_s, dtype=np.float64)
    if np.sum(good_pix) < min_good_pixels:
        return _empty(v_grid)

    w_min, w_max = float(np.nanmin(w)), float(np.nanmax(w))
    if not np.isfinite(w_min) or not np.isfinite(w_max) or w_max <= w_min:
        return _empty(v_grid)

    n_chunk = max(1, int((w_max - w_min) / float(window_size_um)))

    all_rvs: list[float] = []
    all_rv_errs: list[float] = []
    all_centers: list[float] = []
    all_ccf_rows: list[np.ndarray] = []

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
            v_min_m_s=v_min_m_s,
            v_max_m_s=v_max_m_s,
            zb=zb,
        )
        if mask_c.start_um.size == 0:
            continue

        ccf_raw = calc_ccf_mask_kpf_like(
            w_c,
            s_c,
            e_c,
            mask_c,
            v_grid,
            zb=zb,
        )
        rv, rv_err, ccf_fit = rv_from_ccf_mask_kpf(
            v_grid,
            ccf_raw,
            v_step_m_s=v_step_m_s,
            n_good=int(np.sum(in_chunk)),
        )
        if not np.isfinite(rv):
            continue
        all_rvs.append(float(rv))
        all_rv_errs.append(float(rv_err) if np.isfinite(rv_err) and rv_err > 0 else np.nan)
        all_centers.append(float(wl_center))
        if return_ccf:
            all_ccf_rows.append(ccf_fit)

    if not all_rvs:
        return _empty(v_grid)

    rvs_arr = np.asarray(all_rvs, dtype=np.float64)
    rv_errs_arr = np.asarray(all_rv_errs, dtype=np.float64)
    centers_arr = np.asarray(all_centers, dtype=np.float64)
    if return_ccf:
        ccf_2d = np.vstack(all_ccf_rows).astype(np.float64) if all_ccf_rows else np.empty((0, len(v_grid)))
    else:
        ccf_2d = None
    return rvs_arr, rv_errs_arr, centers_arr, v_grid, ccf_2d


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
    zb: float = 0.0,
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
        Velocity grid (m s⁻¹).
    zb : float, optional
        Barycentric factor for mask shift.
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

    v_grid = np.arange(v_min_m_s, v_max_m_s + 0.5 * v_step_m_s, v_step_m_s, dtype=np.float64)

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

    for oi, order in order_tasks:
        w = np.asarray(wave[oi], dtype=np.float64)
        s = np.asarray(spec[oi], dtype=np.float64)
        e = np.asarray(err[oi] if getattr(err, "ndim", 0) >= 2 else err, dtype=np.float64)
        d_bad = np.asarray(bad[oi] if getattr(bad, "ndim", 0) >= 2 else bad, dtype=bool)

        w_min, w_max = float(np.nanmin(w)), float(np.nanmax(w))
        if not np.isfinite(w_min) or not np.isfinite(w_max) or w_max <= w_min:
            continue

        rvs_o, errs_o, centers_o, v_grid_o, ccf_2d = rv_per_chunks_mask_kpf(
            w,
            s,
            e,
            d_bad,
            mask_lines,
            window_size_um=window_size_um,
            v_min_m_s=v_min_m_s,
            v_max_m_s=v_max_m_s,
            v_step_m_s=v_step_m_s,
            zb=zb,
            return_ccf=want_preview_effective,
        )

        if rvs_o.size == 0:
            continue

        all_rvs.extend([float(x) for x in rvs_o])
        all_rv_errs.extend([float(x) for x in errs_o])
        all_wavelengths.extend([float(x) for x in centers_o])
        all_orders.extend([int(order)] * int(len(centers_o)))

        if want_preview_effective and ccf_2d is not None and ccf_2d.size > 0:
            ccf_mean = np.nanmean(ccf_2d, axis=0)
            order_slots.append(
                {
                    "kind": "order",
                    "order": int(order),
                    "v": np.asarray(v_grid_o, dtype=np.float64),
                    "ccf": np.asarray(ccf_mean, dtype=np.float64),
                    "wl_lo": float(w_min),
                    "wl_hi": float(w_max),
                }
            )
            all_ccf_rows.append(ccf_2d)

        if debug_ccf and rvs_o.size > 0:
            logger.info(
                "CCF-mask_kpf fiber=%s order=%s: n_chunks=%d rv_mean~%.1f m/s",
                fiber,
                order,
                len(rvs_o),
                float(np.nanmedian(rvs_o)),
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
            v_min_m_s,
            v_max_m_s,
            v_step_m_s,
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

    return (
        all_wavelengths_arr,
        all_rvs_arr,
        all_rv_errs_arr,
        inlier,
        float(rv_mean),
        float(rv_std),
        all_orders_arr,
        preview_detail,
    )


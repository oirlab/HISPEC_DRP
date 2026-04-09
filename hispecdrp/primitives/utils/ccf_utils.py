"""
Cross-correlation (CCF) and radial-velocity utilities for stellar templates.

Provides weighted-covariance CCF in :math:`\\ln\\lambda`, Gaussian fitting to
:math:`\\mathrm{CCF}(v)` via :func:`fit_ccf_rv` (Astropy ``Gaussian1D`` + ``Const1D``,
``LevMarLSQFitter`` with internal **y** scaling for MINPACK conditioning, two-pass windows,
error heuristics labeled “KPF” in-code), a
Numba multi-order path, continuum normalization, median/MAD outlier rejection, and
inverse-variance chunk combination. **All velocities are in m s\\ :sup:`−1`**.

Per-chunk ``RV_ERR`` combines gradient- and covariance-based estimates (see
:class:`CCFFitKpfConfig.rv_err_cov_kpf_combine`), optional **parametric bootstrap**
on ``ccf_err`` in the final fit window, local velocity sampling at the CCF peak
(finest grid spacing on hybrid ``v`` lists), and simple ``σ/√N`` fallbacks
**without** hard-coded ``/50`` sample-count heuristics. Formal LM errors may still
be **below one velocity pixel** when the CCF is very well sampled and high-SNR;
bootstrap (when enabled) floors ``RV_ERR`` using diagonal ``ccf_err`` perturbations.

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

import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from astropy.table import Table
from astropy.modeling import fitting
from astropy.modeling.models import Const1D, Gaussian1D
from scipy.interpolate import CubicSpline
from scipy.ndimage import correlate1d, gaussian_filter1d, median_filter, uniform_filter1d
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter
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
# Broad step 50 m/s: finer coarse grid than 100 m/s (doubles N_v); override with
# StellarRVPrimitive ``v_step`` or ``resolve_ccf_broad_grid_m_s`` when a different step is required.
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
    Finest velocity sampling (m/s) on ``v_grid`` for Gaussian / fallback CCF fits.

    Uses the **minimum** positive spacing between sorted grid points. Hybrid grids merge a
    coarse broad scan (e.g. 50 m/s) with a dense window (e.g. 0.1 m/s); the **median**
    spacing is still dominated by the coarse leg and incorrectly floors ``RV_ERR`` near
    50 m/s in :func:`_ccf_rv_from_ccf` via ``max(..., v_step)``. The minimum spacing
    matches the refined resolution actually sampling the CCF peak.
    """
    vg = np.asarray(v_grid, dtype=np.float64).ravel()
    if vg.size < 2:
        return 1.0
    d = np.diff(np.sort(vg))
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return 1.0
    return float(np.min(d))


def _ccf_rv_err_numeric_floor(v_step_m_s: float) -> float:
    """
    Minimum positive RV uncertainty (m/s) for numerical stability only.

    Sub-pixel (sub-``v_step``) formal errors from :func:`fit_ccf_rv` or parabolic CCF
    curvature are valid; fallbacks should not clamp them up to a full grid step.
    """
    vs = float(abs(v_step_m_s)) if math.isfinite(v_step_m_s) else 1.0
    return max(1.0e-9 * max(vs, 1.0e-30), 1.0e-12)


def _gaussian_kernel1d_weights(sigma_bins: float, *, truncate: float = 4.0) -> np.ndarray:
    """
    Normalized 1-D Gaussian kernel matching :func:`scipy.ndimage.gaussian_filter1d` (order 0).

    Uses the same radius rule as SciPy (``truncate`` × ``sigma`` in bins, rounded).
    """
    sb = float(sigma_bins)
    if not math.isfinite(sb) or sb <= 0:
        return np.ones(1, dtype=np.float64)
    tr = float(truncate)
    if not math.isfinite(tr) or tr <= 0:
        tr = 4.0
    radius = int(tr * sb + 0.5)
    if radius <= 0:
        return np.ones(1, dtype=np.float64)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    w = np.exp(-0.5 * (x / sb) ** 2)
    w /= float(np.sum(w))
    return w


def _per_bin_sigma_gaussian_smooth1d(ge: np.ndarray, sigma_bins: float) -> np.ndarray:
    """
    Per-sample σ after the same 1-D Gaussian smoothing as :func:`gaussian_filter1d`.

    For i.i.d. noise with variance ``ge[i]²`` at each sample, the variance of the
    smoothed value at :math:`i` is :math:`\\sum_k w_k^2 \\sigma_{i+k}^2` (reflect
    boundaries), implemented via ``correlate1d(ge², w², mode='reflect')``.
    """
    g = np.asarray(ge, dtype=np.float64).ravel()
    w = _gaussian_kernel1d_weights(sigma_bins)
    v2 = correlate1d(g * g, w * w, mode="reflect")
    return np.sqrt(np.maximum(v2, 0.0))


def _rv_err_bouchy_smoothed_ccf_m_s(
    v_m_s: np.ndarray,
    ys: np.ndarray,
    sigma_smooth: np.ndarray,
    *,
    rv_center_m_s: float,
    gaussian_sigma_m_s: float,
    inner_sigma_mult: float = 3.0,
) -> float:
    """
    Bouchy-style formal σ_RV (m/s) from a smoothed CCF and diagonal per-bin σ.

    Under independent Gaussian noise on each velocity sample,
    :math:`\\sigma_{\\mathrm{RV}}^{-2} \\approx \\sum_i \\sigma_i^{-2}\\,(\\mathrm{d\\,CCF}/\\mathrm{d}v)_i^2`
    (Fisher information for a shift in :math:`v`). Sums over
    ``|v - rv_center| \\le inner_sigma_mult \\times \\sigma_\\mathrm{Gaussian}`` so the
    line core dominates; if too few points remain, uses the full passed arrays.

    This matches the scale of the **sub-pixel** parabolic/Gaussian refinement far better
    than parametric bootstrap with diagonal ``ccf_err`` on the **raw** grid, which
    mis-conditions the three-point vertex when :math:`|d^2\\mathrm{CCF}/dv^2|` is tiny in
    CCF amplitude units.
    """
    v0 = np.asarray(v_m_s, dtype=np.float64).ravel()
    y0 = np.asarray(ys, dtype=np.float64).ravel()
    s0 = np.asarray(sigma_smooth, dtype=np.float64).ravel()
    if v0.size != y0.size or v0.size != s0.size or v0.size < 5:
        return float("nan")
    m = np.isfinite(v0) & np.isfinite(y0) & np.isfinite(s0) & (s0 > 0)
    if int(np.count_nonzero(m)) < 5:
        return float("nan")
    vg, yv, se = v0[m], y0[m], s0[m]
    sg = float(gaussian_sigma_m_s)
    if not math.isfinite(sg) or sg <= 0:
        return float("nan")
    rc = float(rv_center_m_s)
    if not math.isfinite(rc):
        return float("nan")
    mult = float(inner_sigma_mult)
    if not math.isfinite(mult) or mult <= 0:
        mult = 3.0
    inner = np.abs(vg - rc) <= mult * sg
    v_u, y_u, s_u = vg[inner], yv[inner], se[inner]
    if v_u.size < 5:
        v_u, y_u, s_u = vg, yv, se
    g1 = np.gradient(y_u, v_u)
    inv_var = 1.0 / (s_u * s_u)
    q = float(np.sum(inv_var * (g1 * g1)))
    if not math.isfinite(q) or q <= 0:
        return float("nan")
    return float(1.0 / math.sqrt(q))


def _nanmean_axis0_ignore_empty_columns(arr: np.ndarray) -> np.ndarray:
    """
    Column-wise mean of finite values (axis 0). Columns with no finite entries are NaN.

    Stacked CCF arrays often have velocity bins where **every** chunk row is NaN;
    ``np.nanmean(..., axis=0)`` then emits ``RuntimeWarning: Mean of empty slice``
    once per such column. This implementation matches ``nanmean`` for mixed
    finite/NaN columns but avoids that warning.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("_nanmean_axis0_ignore_empty_columns: expected a 2-D array")
    n_row, n_col = a.shape
    if n_col == 0:
        return np.array([], dtype=np.float64)
    if n_row == 0:
        return np.full(n_col, np.nan, dtype=np.float64)
    fin = np.isfinite(a)
    cnt = np.sum(fin, axis=0, dtype=np.float64)
    s = np.nansum(a, axis=0)
    out = np.full(n_col, np.nan, dtype=np.float64)
    ok = cnt > 0
    out[ok] = s[ok] / cnt[ok]
    return out


def _nan_stderr_of_mean_axis0(stacked_sigma: np.ndarray, stacked_ccf: np.ndarray) -> np.ndarray:
    """
    Standard error of ``nanmean(stacked_ccf, axis=0)`` when each row is an independent
    estimate with 1σ ``stacked_sigma[i, :]`` at each velocity.

    Uses :math:`\\sigma_{\\mu,j}=\\sqrt{\\sum_i \\sigma_{ij}^2}\\,/\\,n_j` where the sum
    and :math:`n_j` use the same rows as the column-wise mean (finite ``stacked_ccf``;
    only rows with finite nonnegative ``stacked_sigma`` contribute to the sum of
    variances). This matches the scale of the **mean** CCF; using
    ``sqrt(mean(σ²))`` instead would track a **typical single-row** σ and overstate
    uncertainty on the mean by ~ :math:`\\sqrt{n}` when rows are similar.
    """
    sig = np.asarray(stacked_sigma, dtype=np.float64)
    ccf = np.asarray(stacked_ccf, dtype=np.float64)
    if sig.shape != ccf.shape:
        raise ValueError("_nan_stderr_of_mean_axis0: sigma and ccf stacks must have the same shape")
    if ccf.ndim != 2:
        raise ValueError("_nan_stderr_of_mean_axis0: expected 2-D arrays")
    n_row, n_col = ccf.shape
    if n_col == 0:
        return np.array([], dtype=np.float64)
    if n_row == 0:
        return np.full(n_col, np.nan, dtype=np.float64)
    fin_ccf = np.isfinite(ccf)
    cnt = np.sum(fin_ccf, axis=0, dtype=np.float64)
    use = fin_ccf & np.isfinite(sig) & (sig >= 0)
    sum_sq = np.sum(np.where(use, sig**2, 0.0), axis=0)
    out = np.full(n_col, np.nan, dtype=np.float64)
    ok = cnt > 0
    out[ok] = np.sqrt(np.maximum(sum_sq[ok], 0.0)) / cnt[ok]
    return out


def _rv_err_slope_propagation_m_s(
    v_m_s: np.ndarray,
    y_ccf: np.ndarray,
    sigma_ccf: np.ndarray | None,
) -> float:
    """
    Diagonal error propagation σ_RV ≈ σ_y / |d(CCF)/dv| (m/s), v in m/s.

    Uses a robust high-percentile of |dCCF/dv| in the central part of the window so a
    single noisy bin does not dominate. When ``sigma_ccf`` is missing, uses ``mad_std``
    of ``y_ccf`` as a noise scale. The result is **not** clipped to an arbitrary
    upper bound; very large values indicate a near-flat CCF or tiny |dCCF/dv|.
    """
    v0 = np.asarray(v_m_s, np.float64).ravel()
    y0 = np.asarray(y_ccf, np.float64).ravel()
    if v0.size != y0.size or v0.size < 5:
        return float("nan")
    s0 = None
    if sigma_ccf is not None:
        s0 = np.asarray(sigma_ccf, np.float64).ravel()
        if s0.size != v0.size:
            s0 = None

    m = np.isfinite(v0) & np.isfinite(y0)
    if s0 is not None:
        m &= np.isfinite(s0) & (s0 > 0)
    v, y = v0[m], y0[m]
    se = s0[m] if s0 is not None else None
    if v.size < 5:
        m2 = np.isfinite(v0) & np.isfinite(y0)
        v, y = v0[m2], y0[m2]
        se = s0[m2] if s0 is not None else None
    if v.size < 5:
        return float("nan")

    o = np.argsort(v)
    v = v[o]
    y = y[o]
    if se is not None:
        se = se[o]
    if float(np.nanmax(v)) <= float(np.nanmin(v)):
        return float("nan")
    dye = np.gradient(y, v)
    abs_g = np.abs(dye)
    lo, hi = np.nanquantile(v, [0.15, 0.85])
    cen = (v >= lo) & (v <= hi)
    if not np.any(cen):
        cen = np.ones_like(v, dtype=bool)
    g_use = abs_g[cen]
    g95 = float(np.nanpercentile(g_use, 95)) if g_use.size else float("nan")
    if not math.isfinite(g95) or g95 <= 0:
        g95 = float(np.nanmax(abs_g))
    if not math.isfinite(g95) or g95 <= 0:
        return float("nan")
    if se is not None and np.any(np.isfinite(se) & (se > 0)):
        sig_y = float(np.nanmedian(se[np.isfinite(se) & (se > 0)]))
    else:
        sig_y = float(mad_std(y, ignore_nan=True))
    if not math.isfinite(sig_y) or sig_y <= 0:
        return float("nan")
    est = sig_y / g95
    if not math.isfinite(est) or est <= 0:
        return float("nan")
    return float(est)


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


def clip_gaussian_plot_velocity_range_m_s(
    lo_m_s: float,
    hi_m_s: float,
    *,
    v_grid_min_m_s: float | None = None,
    v_grid_max_m_s: float | None = None,
    v_samples_m_s: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Bound ``[lo_m_s, hi_m_s]`` for evaluating a Gaussian CCF model.

    The returned interval is the intersection of ``[lo_m_s, hi_m_s]`` with the span of
    the CCF velocity axis (``v_grid_*`` or ``v_samples_m_s``), further clipped to the
    module absolute limits ``_V_MIN`` … ``_V_MAX`` (m/s), matching the broad CCF grid
    used in :func:`resolve_ccf_broad_grid_m_s` and related paths.
    """
    if v_grid_min_m_s is None or v_grid_max_m_s is None:
        if v_samples_m_s is not None and np.size(v_samples_m_s) > 0:
            va = np.asarray(v_samples_m_s, dtype=np.float64).ravel()
            gmin = float(np.nanmin(va))
            gmax = float(np.nanmax(va))
        else:
            gmin, gmax = float(_V_MIN), float(_V_MAX)
    else:
        gmin = float(v_grid_min_m_s)
        gmax = float(v_grid_max_m_s)
    gmin = max(gmin, float(_V_MIN))
    gmax = min(gmax, float(_V_MAX))
    lo = max(float(lo_m_s), gmin)
    hi = min(float(hi_m_s), gmax)
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo >= hi:
        lo, hi = gmin, gmax
    return lo, hi


def estimate_ccf_uncertainty_highpass_residual(
    ccf: np.ndarray,
    *,
    lowpass_window: int = 51,
    savgol_polyorder: int = 3,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate a **single** CCF uncertainty σ (same units as ``ccf``): Savitzky–Golay
    low-pass smooth, subtract to form a high-frequency residual, then take the RMS of
    successive differences in that residual (pixel-to-pixel scale).

    **Mask-line KPF path** (:func:`run_ccf_mask_kpf_for_fiber`): this is the primary model
    for ``ccf_err`` on the per-order mean CCF (uniform array filled with ``sigma``), and
    for the stacked rows passed into :func:`finalize_ccf_preview_dict` (replacing the
    former per-chunk weighted SD + stderr-of-mean combination).

    Parameters
    ----------
    ccf : ndarray
        CCF samples (same shape as used for Gaussian fitting).
    lowpass_window : int
        Savitzky–Golay window length (forced odd); capped to array length. Larger ⇒ stronger
        low-pass (more large-scale structure removed from the residual).
    savgol_polyorder : int
        Savitzky–Golay polynomial order (must be ``< window_length``).

    Returns
    -------
    sigma : float
        Scalar σ applied uniformly at each velocity bin when building ``ccf_err`` for
        :func:`fit_ccf_rv`.
    ccf_lowpass : ndarray
        Low-pass (smooth) component of ``ccf``.
    residual : ndarray
        ``ccf - ccf_lowpass`` (high-pass remainder).
    """
    y = np.asarray(ccf, dtype=np.float64).ravel()
    if np.any(~np.isfinite(y)):
        m = np.isfinite(y)
        med = float(np.nanmedian(y[m])) if np.any(m) else 0.0
        if not np.isfinite(med):
            med = 0.0
        y = np.where(np.isfinite(y), y, med)
    n = int(y.size)
    if n < 3:
        z = np.zeros_like(y)
        return float("nan"), np.asarray(y, dtype=np.float64), z

    wl = int(lowpass_window)
    if wl % 2 == 0:
        wl += 1
    wl = max(5, wl)
    if wl > n:
        wl = n if n % 2 == 1 else n - 1
        wl = max(5, wl)
    po = int(savgol_polyorder)
    po = max(1, min(po, wl - 1))

    try:
        smooth = savgol_filter(y, wl, po, mode="interp")
    except Exception:
        size = max(3, min(wl, n if n % 2 == 1 else n - 1))
        if size % 2 == 0:
            size -= 1
        size = max(3, size)
        smooth = uniform_filter1d(y, size=size, mode="nearest")

    res = y - smooth
    dd = np.diff(res)
    sigma = float(np.sqrt(np.nanmean(dd * dd)))
    if not math.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.nanstd(res))
    if not math.isfinite(sigma) or sigma <= 0.0:
        sigma = max(float(np.nanstd(y)), 1e-30)
    return sigma, smooth, res


@dataclass
class CCFFitKpfConfig:
    """Two-pass Gaussian CCF fit; half-widths in km/s."""

    # Wider windows help the LM Gaussian recover a sensible baseline away from the peak.
    pass1_half_width_km_s: float = 800.0
    pass2_half_width_km_s: float = 120.0
    initial_stddev_km_s: float = 5.0
    min_points: int = 5
    emission_strength_ratio: float = 1.02
    cal_mask_methods: frozenset = frozenset({"thar", "lfc", "etalon"})
    # Levenberg–Marquardt iterations per Gaussian fit (χ² with optional inverse-variance weights).
    levmar_maxiter: int = 1500
    # Unclipped prefit passes on the **full** velocity window (no point removal): repeated
    # LM fits to drive χ² down before any sigma-clipping refinement.
    sigma_clip_unclipped_prefit_passes: int = 2
    # After prefit, this many iterations: fit → clip outliers → refit (each pass uses LM).
    sigma_clip_enable: bool = True
    sigma_clip_max_iter: int = 10
    # MAD multiplier for residual clipping after prefit (tighter than relaxed).
    sigma_clip_n_sigma: float = 4.0
    # Weaker clipping for the first few post-prefit rounds (keep more points while refining).
    sigma_clip_n_sigma_relaxed: float = 6.0
    sigma_clip_relax_first_clip_iters: int = 3
    sigma_clip_min_keep_frac: float = 0.30
    # How to merge KPF gradient error with LevMar covariance when both pass plausibility.
    # ``"max"`` is more conservative than historic ``"min"`` (often gave tiny σ_RV).
    rv_err_cov_kpf_combine: str = "max"
    # Parametric bootstrap: refit Gaussian on CCF + N(0, σ_ccf) noise (same window/weights as
    # the final pass). 0 disables. Typical 24–64 for science; each sample is one LM fit.
    rv_err_bootstrap_samples: int = 0
    rv_err_bootstrap_seed: int | None = None


# Lighter LevMar / σ-clip budget for :func:`finalize_ccf_preview_dict` only (Plotly metadata).
# Pipeline RV fits on order-mean CCFs keep the default :class:`CCFFitKpfConfig`.
CCF_PREVIEW_SLOT_FIT_CONFIG = CCFFitKpfConfig(
    levmar_maxiter=450,
    sigma_clip_unclipped_prefit_passes=1,
    sigma_clip_max_iter=6,
)


# Sub-pixel RV refinement on coarse velocity grids (e.g. 50 m/s CCF samples): default
# ``smooth_sigma_scale`` when :func:`refine_ccf_rv_coarse_grid_parabolic` runs with
# ``optimize_smooth_sigma_scale=False``, or as a fallback if the v₀-aligned sweep fails.
# (Historically tuned on lte049 mask-KPF debug for sub–10 cm s⁻¹ vs. Gaussian v₀.)
_REFINE_COARSE_GRID_SMOOTH_SIGMA_SCALE_DEFAULT = 0.0645

# Peak search window (m/s) half-width: narrower than legacy max(8000, 6σ, 40·v_step); shrinks
# each iteration when iteratively re-centering on the sub-pixel peak.
_REFINE_PEAK_WINDOW_BASE_MIN_M_S = 3200.0
_REFINE_PEAK_WINDOW_SIGMA_MULT = 3.25
_REFINE_PEAK_WINDOW_VSTEP_MULT = 26.0
_REFINE_PEAK_WINDOW_HALF_WIDTH_FLOOR_M_S = 900.0
_REFINE_PEAK_WINDOW_SIGMA_FLOOR_MULT = 1.25
_REFINE_PEAK_WINDOW_VSTEP_FLOOR_MULT = 8.0
_REFINE_PEAK_ITER_SHRINK = 0.62
_REFINE_PEAK_ITERATIONS_DEFAULT = 3


def _refine_peak_search_half_window_m_s(
    iteration: int,
    gaussian_sigma_m_s: float,
    v_step_m_s: float,
) -> float:
    """Half-width (m/s) of velocity window for spline peak search; shrinks with ``iteration``."""
    sig = float(gaussian_sigma_m_s)
    vs = float(v_step_m_s)
    it = max(0, int(iteration))
    shrink = _REFINE_PEAK_ITER_SHRINK**it
    base = max(
        _REFINE_PEAK_WINDOW_BASE_MIN_M_S,
        _REFINE_PEAK_WINDOW_SIGMA_MULT * sig,
        _REFINE_PEAK_WINDOW_VSTEP_MULT * vs,
    )
    hw = base * shrink
    floor = max(
        _REFINE_PEAK_WINDOW_HALF_WIDTH_FLOOR_M_S,
        _REFINE_PEAK_WINDOW_SIGMA_FLOOR_MULT * sig,
        _REFINE_PEAK_WINDOW_VSTEP_FLOOR_MULT * vs,
    )
    return float(max(hw, floor))


def _parabolic_vertex_from_three_samples_m_s(
    v_lo: float,
    v_mid: float,
    v_hi: float,
    z_lo: float,
    z_mid: float,
    z_hi: float,
) -> tuple[float, str | None]:
    """Unique parabola through three (v, CCF) samples; return vertex v (m/s) or error."""
    A = np.array(
        [[v_lo**2, v_lo, 1.0], [v_mid**2, v_mid, 1.0], [v_hi**2, v_hi, 1.0]],
        dtype=np.float64,
    )
    try:
        abc = np.linalg.solve(A, [z_lo, z_mid, z_hi])
    except np.linalg.LinAlgError:
        return float("nan"), "parabola solve failed"
    a, b = float(abc[0]), float(abc[1])
    if a == 0.0:
        return float("nan"), "degenerate parabola"
    if a >= 0.0:
        return float("nan"), "no concave vertex at peak"
    return float(-b / (2.0 * a)), None


def _collapse_duplicate_velocity_samples_m_s(
    vm: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort by velocity and average CCF values at duplicate ``vm`` knots so downstream
    cubic-spline peak finders see strictly increasing abscissae.
    """
    vm = np.asarray(vm, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if vm.size == 0 or vm.size != y.size:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    o = np.argsort(vm)
    vm = vm[o]
    y = y[o]
    if vm.size < 2:
        return vm, y
    if np.all(np.diff(vm) > 0):
        return vm, y
    vm_u, inv = np.unique(vm, return_inverse=True)
    cnt = np.bincount(inv)
    y_u = np.bincount(inv, weights=y) / np.maximum(cnt.astype(np.float64), 1.0)
    return vm_u, y_u


def _ccf_raw_grid_subpixel_peak_m_s(
    vm: np.ndarray,
    y: np.ndarray,
    *,
    use_maximum: bool = True,
) -> float:
    """
    Sub-pixel location of a **raw** CCF extremum on the velocity grid (no Gaussian model).

    Must follow the same absorption vs. emission convention as :func:`fit_ccf_rv`:
    **absorption** → sub-pixel **maximum** of ``y``; **emission** → location of the **minimum**
    of ``y`` (implemented as the maximum of ``-y``).

    Prefer a **cubic spline** extremum on the sorted samples; fall back to
    :func:`_parabolic_vertex_discrete_grid_m_s` at the discrete extremum.
    """
    vm = np.asarray(vm, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if vm.size < 3 or y.size != vm.size:
        return float("nan")
    ok = np.isfinite(vm) & np.isfinite(y)
    if int(np.count_nonzero(ok)) < 3:
        return float("nan")
    vm = vm[ok]
    y = y[ok]
    vm, y = _collapse_duplicate_velocity_samples_m_s(vm, y)
    if vm.size < 3:
        return float("nan")
    if np.any(np.diff(vm) <= 0):
        return float("nan")
    y_eff = np.asarray(y, dtype=np.float64).ravel()
    if not use_maximum:
        y_eff = -y_eff
    v_c, err = _subpixel_peak_cubic_spline_m_s(vm, y_eff)
    if v_c is not None and math.isfinite(float(v_c)):
        return float(v_c)
    i_max = int(np.nanargmax(y_eff))
    vtx, _msg = _parabolic_vertex_discrete_grid_m_s(vm, y_eff, i_max)
    if math.isfinite(vtx):
        return float(vtx)
    return float(vm[i_max])


def _ccf_raw_grid_subpixel_peak_nearest_disc_max_m_s(vm: np.ndarray, y: np.ndarray) -> float:
    """
    Sub-pixel raw CCF location using the better of the global-**maximum** and **minimum**
    branches, chosen by proximity to the **discrete** velocity at ``argmax(y)``.

    Comparing against the LM Gaussian center would pick the wrong branch when LevMar is
    biased hundreds of km/s away; the discrete maximum bin remains tied to the actual line
    core on the grid.
    """
    a = _ccf_raw_grid_subpixel_peak_m_s(vm, y, use_maximum=True)
    b = _ccf_raw_grid_subpixel_peak_m_s(vm, y, use_maximum=False)
    imx = int(np.nanargmax(y))
    if not (0 <= imx < int(vm.size)):
        return float(a if math.isfinite(a) else b)
    v_disc = float(vm[imx])
    if not math.isfinite(v_disc):
        return float(a if math.isfinite(a) else b)
    if not math.isfinite(a):
        return float(b)
    if not math.isfinite(b):
        return float(a)
    return float(a) if abs(float(a) - v_disc) <= abs(float(b) - v_disc) else float(b)


def _parabolic_vertex_discrete_grid_m_s(
    vm_w: np.ndarray,
    z_w: np.ndarray,
    ix_loc: int,
) -> tuple[float, str | None]:
    """
    Fallback peak location: 5-point quadratic when the discrete max is interior, else 3-point.
    """
    n = int(vm_w.size)
    if n < 3:
        return float("nan"), "insufficient points"
    if 2 <= ix_loc <= n - 3:
        vm5 = np.asarray(vm_w[ix_loc - 2 : ix_loc + 3], dtype=np.float64)
        z5 = np.asarray(z_w[ix_loc - 2 : ix_loc + 3], dtype=np.float64)
        if np.all(np.isfinite(vm5)) and np.all(np.isfinite(z5)):
            coef = np.polyfit(vm5, z5, 2)
            a, b = float(coef[0]), float(coef[1])
            if math.isfinite(a) and a < 0.0 and math.isfinite(b):
                return float(-b / (2.0 * a)), None
    if ix_loc <= 0 or ix_loc >= n - 1:
        return float("nan"), "peak at window edge"
    v_lo, v_mid, v_hi = float(vm_w[ix_loc - 1]), float(vm_w[ix_loc]), float(vm_w[ix_loc + 1])
    z_lo = float(z_w[ix_loc - 1])
    z_mid = float(z_w[ix_loc])
    z_hi = float(z_w[ix_loc + 1])
    return _parabolic_vertex_from_three_samples_m_s(v_lo, v_mid, v_hi, z_lo, z_mid, z_hi)


def _subpixel_peak_cubic_spline_m_s(vm_w: np.ndarray, z_w: np.ndarray) -> tuple[float | None, str | None]:
    """
    Location of the maximum of the smoothed CCF via **cubic spline** interpolation and bounded
    optimization on velocity.

    PCHIP is inappropriate here (it assumes monotone *data*); mask CCFs are unimodal peaks.
    A three-point parabola alone is biased when the peak falls between grid points; the cubic
    spline matches the pipeline ``lte049`` / synthetic Gaussian tests to sub-cm/s bias when
    combined with Gaussian smoothing.
    """
    vw = np.asarray(vm_w, dtype=np.float64).ravel()
    zw = np.asarray(z_w, dtype=np.float64).ravel()
    if vw.size < 3 or zw.size != vw.size:
        return None, "insufficient points"
    if not np.all(np.isfinite(vw)) or not np.all(np.isfinite(zw)):
        return None, "non-finite window data"
    if np.any(np.diff(vw) <= 0):
        return None, "velocity not strictly increasing"
    lo, hi = float(vw[0]), float(vw[-1])
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return None, "invalid velocity bounds"

    try:
        cs = CubicSpline(vw, zw)

        def negc(x: float) -> float:
            v = float(cs(x))
            return -v if math.isfinite(v) else 1e300

        res = minimize_scalar(
            negc,
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-12, "maxiter": 200},
        )
        if not res.success or not math.isfinite(res.x):
            return None, "cubic optimize failed"
        vtx = float(res.x)
        if not math.isfinite(vtx) or vtx < lo - 1.0 or vtx > hi + 1.0:
            return None, "bad vertex"
        return vtx, None
    except Exception:
        return None, "cubic spline failed"


def _gaussian_smooth_parabolic_vertex_m_s(
    vm: np.ndarray,
    y: np.ndarray,
    *,
    sigma_smooth_bins: float,
    rv_anchor_ms: float,
    gaussian_sigma_m_s: float,
    v_step_m_s: float,
    ys_precomputed: np.ndarray | None = None,
    return_peak_diagnostics: bool = False,
    peak_refine_iterations: int = _REFINE_PEAK_ITERATIONS_DEFAULT,
    peak_search_center_m_s: float | None = None,
) -> tuple[float, str | None, dict | None]:
    """
    Gaussian-smooth ``y`` along sorted ``vm``, then **iteratively** narrow a velocity window
    around the running peak estimate and re-fit **cubic spline** + bounded optimization on the
    smoothed curve (sub-grid peak), with 5-/3-point parabolic **fallback** if needed.

    Returns ``(rv_m_s, error_message_or_None, peak_diagnostics_or_None)``. When
    ``return_peak_diagnostics`` is True and the vertex is valid, the third element is a dict
    with ``peak_search_mask``, parabolic sample ``v_*`` / ``ccf_smoothed_*``, and polynomial
    coefficients ``parabola_a``, ``parabola_b``, ``parabola_c`` for :math:`\\mathrm{CCF}=a v^2+b v+c`
    (three grid points straddling the refined RV for plots).

    When ``ys_precomputed`` is given (same shape as ``y``), it is used instead of smoothing
    ``y`` again (must match ``sigma_smooth_bins``).

    ``peak_search_center_m_s`` optionally sets the **first-iteration** search window center
    (e.g. raw CCF sub-pixel peak). When ``None``, ``rv_anchor_ms`` is used (typically Gaussian
    :math:`v_0`).
    """
    sb = float(sigma_smooth_bins)
    ys = np.asarray(ys_precomputed, dtype=np.float64).ravel() if ys_precomputed is not None else gaussian_filter1d(y, sigma=sb)
    if ys.shape != np.asarray(y).ravel().shape:
        return float("nan"), "ys_precomputed shape mismatch", None

    sig_ms = float(gaussian_sigma_m_s)
    vs = float(v_step_m_s)
    n_iter = max(1, int(peak_refine_iterations))
    psc = float(peak_search_center_m_s) if peak_search_center_m_s is not None else float("nan")
    anchor = float(psc) if math.isfinite(psc) else float(rv_anchor_ms)
    vtx = float(anchor)
    last_half_win = _refine_peak_search_half_window_m_s(0, sig_ms, vs)
    vm_w = np.asarray([], dtype=np.float64)
    z_w = np.asarray([], dtype=np.float64)
    win = np.zeros(vm.shape, dtype=bool)
    used_cubic = False
    ix_loc = 0
    vm_w_last: np.ndarray | None = None
    z_w_last: np.ndarray | None = None
    win_last: np.ndarray | None = None
    used_cubic_last = False
    vtx_last = float("nan")

    for it in range(n_iter):
        half_win = _refine_peak_search_half_window_m_s(it, sig_ms, vs)
        last_half_win = half_win
        win = (vm >= anchor - half_win) & (vm <= anchor + half_win)
        if int(np.count_nonzero(win)) < 5:
            if it == 0:
                win = np.isfinite(vm)
            else:
                break
        vm_w = vm[win]
        z_w = ys[win]
        if vm_w.size < 3:
            if it == 0:
                return float("nan"), "insufficient points in refine window", None
            break

        ix_loc = int(np.argmax(z_w))
        vtx_p, _pmsg = _subpixel_peak_cubic_spline_m_s(vm_w, z_w)
        used_cubic = vtx_p is not None and math.isfinite(float(vtx_p))
        if used_cubic:
            vtx = float(vtx_p)
        else:
            if ix_loc <= 0 or ix_loc >= vm_w.size - 1:
                if it == 0:
                    return float("nan"), "peak at window edge", None
                break
            vtx_fb, msg_fb = _parabolic_vertex_discrete_grid_m_s(vm_w, z_w, ix_loc)
            if not math.isfinite(vtx_fb):
                if it == 0:
                    return float("nan"), msg_fb or "peak refinement failed", None
                break
            vtx = float(vtx_fb)

        anchor = vtx
        vm_w_last = np.asarray(vm_w, dtype=np.float64)
        z_w_last = np.asarray(z_w, dtype=np.float64)
        win_last = np.asarray(win, dtype=bool)
        used_cubic_last = used_cubic
        vtx_last = float(vtx)

    if vm_w_last is not None and vm_w_last.size >= 3 and z_w_last is not None:
        vm_w = vm_w_last
        z_w = z_w_last
        win = win_last if win_last is not None else win
        used_cubic = used_cubic_last
        vtx = vtx_last
    elif vm_w.size < 3:
        return float("nan"), "insufficient points in refine window", None

    n_w = int(vm_w.size)
    ix_mid = int(np.argmin(np.abs(np.asarray(vm_w, dtype=np.float64) - vtx)))
    ix_mid = int(np.clip(ix_mid, 1, n_w - 2))
    v_lo = float(vm_w[ix_mid - 1])
    v_mid = float(vm_w[ix_mid])
    v_hi = float(vm_w[ix_mid + 1])
    z_lo = float(z_w[ix_mid - 1])
    z_mid = float(z_w[ix_mid])
    z_hi = float(z_w[ix_mid + 1])
    a, b, c = float("nan"), float("nan"), float("nan")
    try:
        A = np.array(
            [[v_lo**2, v_lo, 1.0], [v_mid**2, v_mid, 1.0], [v_hi**2, v_hi, 1.0]],
            dtype=np.float64,
        )
        abc = np.linalg.solve(A, [z_lo, z_mid, z_hi])
        a, b, c = float(abc[0]), float(abc[1]), float(abc[2])
    except np.linalg.LinAlgError:
        pass

    aux: dict | None = None
    if return_peak_diagnostics:
        aux = {
            "refine_half_window_m_s": float(last_half_win),
            "refine_peak_iterations_used": int(n_iter),
            "peak_search_mask": np.asarray(win, dtype=bool),
            "parabola_v_lo_m_s": v_lo,
            "parabola_v_mid_m_s": v_mid,
            "parabola_v_hi_m_s": v_hi,
            "parabola_ccf_smoothed_lo": z_lo,
            "parabola_ccf_smoothed_mid": z_mid,
            "parabola_ccf_smoothed_hi": z_hi,
            "parabola_a": a,
            "parabola_b": b,
            "parabola_c": c,
            "refine_peak_method": "cubic_subpixel_max" if used_cubic else "discrete_parabolic_fallback",
        }
    return vtx, None, aux


def refine_ccf_rv_coarse_grid_parabolic(
    v_grid_m_s: np.ndarray,
    ccf: np.ndarray,
    v_step_m_s: float,
    *,
    ccf_err: np.ndarray | None = None,
    gaussian_sigma_m_s: float | None = None,
    mask_method: str | None = None,
    config: CCFFitKpfConfig | None = None,
    smooth_sigma_scale: float = _REFINE_COARSE_GRID_SMOOTH_SIGMA_SCALE_DEFAULT,
    optimize_smooth_sigma_scale: bool = True,
    optimize_scale_lo: float = 0.04,
    optimize_scale_hi: float = 0.10,
    optimize_scale_n: int = 31,
    peak_refine_iterations: int = _REFINE_PEAK_ITERATIONS_DEFAULT,
    sigma_smooth_bins_min: float = 0.4,
    sigma_smooth_bins_max: float = 12.0,
    bootstrap_rv_err_samples: int = 0,
    bootstrap_seed: int | None = None,
) -> tuple[float, dict]:
    """
    Refine radial velocity on a **coarse** CCF velocity grid using mild Gaussian smoothing
    along :math:`v` (in units of **samples**) and a **sub-pixel maximum** of the smoothed curve
    (cubic spline optimization, with discrete parabolic fallback).

    When ``ccf`` is sampled every ~50–100 m/s, a plain Levenberg–Marquardt Gaussian can leave
    **tens of m s\\ :sup:`−1`** bias versus the true correlation peak; this routine reduces
    that bias on BT-Settl mask-line diagnostics
    by (1) estimating the line width ``σ`` from :func:`fit_ccf_rv` (Gaussian :math:`v_0` is kept
    for diagnostics only),
    (2) **by default** choosing the smoothing multiplier to minimize
    :math:`|RV_{\\mathrm{refined}} - v_{\\mathrm{raw}}|` over a bounded grid, where
    :math:`v_{\\mathrm{raw}}` comes from :func:`_ccf_raw_grid_subpixel_peak_nearest_disc_max_m_s`
    (see ``rv_raw_grid_peak_m_s`` in ``meta``). When the LM center matches the discrete CCF peak,
    this tracks :math:`v_{\\mathrm{raw}} \\approx v_0`; when LevMar is biased, it still tracks the
    grid line core,
    then smoothing with ``σ_smooth ≈ scale × (σ / v_step)`` bins, (3) taking the **local maximum**
    of the smoothed CCF in a window centered on :math:`v_{\\mathrm{raw}}` when available (else
    :math:`v_0`), and (4) optionally fitting a parabola through three samples straddling
    that peak for plotting (``parabola_*`` in ``meta``).

    **RV uncertainty:** ``rv_err_m_s`` (default) is a Bouchy-style σ from the smoothed
    CCF—summing :math:`\\sigma_i^{-2}\\,(\\mathrm{d\\,CCF}/\\mathrm{d}v)_i^2` in the line
    core (see :func:`_rv_err_bouchy_smoothed_ccf_m_s`), with per-bin ``ccf_err`` propagated
    through the same Gaussian kernel. Optional parametric bootstrap fills
    ``rv_err_bootstrap_m_s`` only and does not replace ``rv_err_m_s``.

    Parameters
    ----------
    v_grid_m_s, ccf
        Same-shape velocity axis and CCF values (m s\\ :sup:`−1`, dimensionless CCF).
    v_step_m_s
        Finest effective spacing (m s\\ :sup:`−1` per bin), e.g. from
        :func:`effective_ccf_v_step_for_fit`.
    ccf_err
        Optional per-sample σ on ``ccf``; used when ``gaussian_sigma_m_s`` must be estimated,
        for Bouchy ``rv_err_m_s`` (and bootstrap when enabled). If missing, a robust
        difference scale substitutes.
    gaussian_sigma_m_s
        Gaussian σ from a prior fit (m s\\ :sup:`−1`). If omitted, estimated via
        :func:`fit_ccf_rv`.
    mask_method, config
        Passed for emission/absorption peak detection (same rules as :func:`fit_ccf_rv`).
    smooth_sigma_scale
        Multiplier relating smoothing σ in bins to ``gaussian_sigma_m_s / v_step_m_s``.
        Used when ``optimize_smooth_sigma_scale`` is False, or as a fallback if the sweep
        fails.
    optimize_smooth_sigma_scale
        If True (default), scan ``smooth_sigma_scale`` over
        ``[optimize_scale_lo, optimize_scale_hi]`` (``optimize_scale_n`` samples) and pick
        the value that minimizes ``|RV_refined − v_raw|``, where
        ``v_raw`` comes from :func:`_ccf_raw_grid_subpixel_peak_nearest_disc_max_m_s`.
        If that peak is unavailable, the sweep uses the Gaussian center ``v_0`` as fallback.
        If False, use a single ``smooth_sigma_scale`` (historic fixed-tuning behaviour).
    optimize_scale_lo, optimize_scale_hi, optimize_scale_n
        Sweep grid when ``optimize_smooth_sigma_scale`` is True.
    peak_refine_iterations
        After Gaussian smoothing, re-center the peak search window this many times, each time
        using a narrower half-width (:func:`_refine_peak_search_half_window_m_s`) around the
        previous sub-pixel peak (cubic or fallback).
    bootstrap_rv_err_samples
        If > 4, additionally run a parametric bootstrap (``CCF + N(0, σ_ccf)`` per bin, same
        heuristics as when ``ccf_err`` is missing) and store the sample standard deviation of
        refined RVs in ``meta["rv_err_bootstrap_m_s"]`` (floored with
        :func:`_ccf_rv_err_numeric_floor`). Does **not** replace the default ``rv_err_m_s``.
    bootstrap_seed
        RNG seed for bootstrap; ``None`` uses non-deterministic entropy.

    Returns
    -------
    rv_refined_m_s : float
        Refined RV in m s\\ :sup:`−1`.
    meta : dict
        Includes ``sigma_smooth_bins``, ``rv_err_m_s`` (Bouchy / Fisher-gradient σ on the
        smoothed CCF; see :func:`_rv_err_bouchy_smoothed_ccf_m_s`), optional
        ``rv_err_bootstrap_m_s``, and **curve arrays** for plotting the refinement basis:
        ``refine_curve_v_m_s``, ``refine_curve_ccf_smoothed`` (Gaussian-smoothed CCF),
        ``refine_curve_ccf_sigma`` (1σ per bin after smoothing ``ccf_err``),
        ``refine_peak_search_mask``, ``refine_parabola_v_dense_m_s``, ``refine_parabola_ccf_dense``
        (dense quadratic through the three samples at the smoothed peak), scalar
        ``parabola_v_*`` / ``parabola_ccf_smoothed_*`` / ``parabola_a`` / ``parabola_b`` / ``parabola_c``,
        and ``refine_peak_method`` (``"cubic_subpixel_max"`` or ``"discrete_parabolic_fallback"``).
        When optimization runs, expect also ``refine_min_bias_optimized``, etc., from
        :func:`sweep_refine_ccf_rv_coarse_grid_min_abs_bias`, and ``refine_optimizer_anchor``
        (``ccf_grid_subpixel_peak`` vs ``gaussian_v0_gated``).
    """
    cfg = config or CCFFitKpfConfig()
    vg = np.asarray(v_grid_m_s, np.float64).ravel()
    y0 = np.asarray(ccf, np.float64).ravel()
    ok = np.isfinite(vg) & np.isfinite(y0)
    if int(np.count_nonzero(ok)) < 5:
        return float("nan"), {"ok": False, "message": "insufficient valid points"}
    vm = vg[ok]
    y = y0[ok]
    o = np.argsort(vm)
    vm, y = vm[o], y[o]
    y_err = None
    if ccf_err is not None:
        er = np.asarray(ccf_err, np.float64).ravel()
        if er.size == vg.size:
            y_err = er[ok][o]

    vstep = float(v_step_m_s)
    if not math.isfinite(vstep) or vstep <= 0:
        return float("nan"), {"ok": False, "message": "invalid v_step_m_s"}

    mm = str(mask_method).lower() if mask_method else ""
    if mm in cfg.cal_mask_methods:
        profile = "absorption"
    else:
        med = float(np.median(y))
        imx, imn = int(np.nanargmax(y)), int(np.nanargmin(y))
        profile = (
            "emission"
            if (med - y[imn]) > cfg.emission_strength_ratio * max(y[imx] - med, 1e-30)
            else "absorption"
        )

    sig_ms = gaussian_sigma_m_s
    rv_anchor_ms: float | None = None
    if sig_ms is None or not math.isfinite(float(sig_ms)) or float(sig_ms) <= 0:
        _rv_aux, _, params, diag = fit_ccf_rv(
            vm,
            y,
            vstep,
            return_params=True,
            ccf_err=y_err,
            mask_method=mask_method,
            config=cfg,
        )
        if not diag.get("fit_ok") or not params:
            return float("nan"), {
                "ok": False,
                "message": "auxiliary fit_ccf_rv for sigma failed",
                "diagnostics": diag,
            }
        sig_ms = float(params.get("sigma", float("nan")))
        rv_anchor_ms = float(params.get("v0", float("nan")))
    sig_ms = float(sig_ms)
    if not math.isfinite(sig_ms) or sig_ms <= 0:
        return float("nan"), {"ok": False, "message": "invalid gaussian_sigma_m_s"}

    if rv_anchor_ms is None or not math.isfinite(rv_anchor_ms):
        _r2, _, p2, d2 = fit_ccf_rv(
            vm,
            y,
            vstep,
            return_params=True,
            ccf_err=y_err,
            mask_method=mask_method,
            config=cfg,
        )
        if d2.get("fit_ok") and p2:
            rv_anchor_ms = float(p2.get("v0", float("nan")))
    if rv_anchor_ms is None or not math.isfinite(rv_anchor_ms):
        imx, imn = int(np.nanargmax(y)), int(np.nanargmin(y))
        rv_anchor_ms = float(vm[imx if profile == "absorption" else imn])

    rv_raw_peak = float(_ccf_raw_grid_subpixel_peak_nearest_disc_max_m_s(vm, y))

    # Prefer the raw sub-pixel extremum for sweep / first spline window when (a) it matches the
    # LM center, or (b) the LM center is **not** near the velocity of the discrete CCF maximum
    # (bad Gaussian on a coarse grid). In (b), anchoring the sweep to :math:`v_0` would lock in a
    # catastrophic LM bias while the raw geometry still tracks the line core.
    _div_gate_m_s = max(8000.0, 4.0 * float(sig_ms))
    imx = int(np.nanargmax(y))
    v_at_disc_max = float(vm[imx]) if 0 <= imx < vm.size else float("nan")
    raw_near_lm = (
        math.isfinite(rv_raw_peak)
        and math.isfinite(rv_anchor_ms)
        and abs(rv_raw_peak - float(rv_anchor_ms)) <= _div_gate_m_s
    )
    lm_near_disc_max = (
        math.isfinite(v_at_disc_max)
        and math.isfinite(rv_anchor_ms)
        and abs(float(rv_anchor_ms) - v_at_disc_max) <= _div_gate_m_s
    )
    if raw_near_lm or not lm_near_disc_max:
        sweep_truth = float(rv_raw_peak)
        peak_search_center_m_s: float | None = (
            float(rv_raw_peak) if math.isfinite(rv_raw_peak) else None
        )
        sweep_anchor_kind = (
            "ccf_grid_subpixel_peak"
            if raw_near_lm
            else "ccf_grid_subpixel_peak_lm_vs_disc_max"
        )
    else:
        sweep_truth = float(rv_anchor_ms)
        peak_search_center_m_s = None
        sweep_anchor_kind = "gaussian_v0_gated"

    if optimize_smooth_sigma_scale:
        _bsc, rv_opt, meta_opt = sweep_refine_ccf_rv_coarse_grid_min_abs_bias(
            vm,
            y,
            vstep,
            truth_rv_m_s=float(sweep_truth),
            ccf_err=y_err,
            gaussian_sigma_m_s=sig_ms,
            mask_method=mask_method,
            config=cfg,
            scale_lo=float(optimize_scale_lo),
            scale_hi=float(optimize_scale_hi),
            n_scale=int(optimize_scale_n),
            peak_refine_iterations=int(peak_refine_iterations),
            sigma_smooth_bins_min=sigma_smooth_bins_min,
            sigma_smooth_bins_max=sigma_smooth_bins_max,
            bootstrap_rv_err_samples=bootstrap_rv_err_samples,
            bootstrap_seed=bootstrap_seed,
        )
        if meta_opt.get("ok"):
            meta_opt = dict(meta_opt)
            meta_opt["refine_optimizer_anchor"] = sweep_anchor_kind
            meta_opt["rv_anchor_gaussian_m_s"] = float(rv_anchor_ms)
            meta_opt["rv_raw_grid_peak_m_s"] = float(rv_raw_peak)
            meta_opt["refine_raw_gaussian_agreement_m_s"] = (
                abs(float(rv_raw_peak) - float(rv_anchor_ms))
                if math.isfinite(rv_raw_peak) and math.isfinite(rv_anchor_ms)
                else float("nan")
            )
            meta_opt["refine_sweep_truth_m_s"] = float(sweep_truth)
            meta_opt["rv_refined_m_s"] = float(rv_opt)
            return float(rv_opt), meta_opt

    sb = float(smooth_sigma_scale) * (sig_ms / vstep)
    sb = float(np.clip(sb, sigma_smooth_bins_min, sigma_smooth_bins_max))

    ys = gaussian_filter1d(y, sigma=sb)
    v_vertex, vmsg, peak_aux = _gaussian_smooth_parabolic_vertex_m_s(
        vm,
        y,
        sigma_smooth_bins=sb,
        rv_anchor_ms=float(rv_anchor_ms),
        gaussian_sigma_m_s=sig_ms,
        v_step_m_s=vstep,
        ys_precomputed=ys,
        return_peak_diagnostics=True,
        peak_refine_iterations=int(peak_refine_iterations),
        peak_search_center_m_s=peak_search_center_m_s,
    )
    if vmsg is not None:
        return float("nan"), {
            "ok": False,
            "message": vmsg,
            "ccf_profile_heuristic": profile,
        }

    pa = peak_aux or {}

    ge = np.full(vm.shape, np.nan, dtype=np.float64)
    if y_err is not None:
        ge = np.asarray(y_err, np.float64).ravel()
        d1 = np.abs(np.diff(y[np.isfinite(y)]))
        rs = float(np.nanmedian(d1)) / 1.253 if d1.size else float(np.nanstd(y))
        rs = max(rs, 1e-30 * (abs(float(np.nanmax(y) - np.nanmin(y))) + 1.0))
        ge = np.where(np.isfinite(ge) & (ge > 0), ge, rs)
    else:
        d1 = np.abs(np.diff(y[np.isfinite(y)]))
        rs = float(np.nanmedian(d1)) / 1.253 if d1.size else float(np.nanstd(y))
        rs = max(rs, 1e-30 * (abs(float(np.nanmax(y) - np.nanmin(y))) + 1.0))
        ge = np.full(vm.shape, rs, dtype=np.float64)

    sig_s = _per_bin_sigma_gaussian_smooth1d(ge, sb)
    half_bouchy = float(pa.get("refine_half_window_m_s", float("nan")))
    if not math.isfinite(half_bouchy):
        half_bouchy = _refine_peak_search_half_window_m_s(0, float(sig_ms), float(vstep))
    win_b = (vm >= float(v_vertex) - half_bouchy) & (vm <= float(v_vertex) + half_bouchy)
    rv_err_m_s = _rv_err_bouchy_smoothed_ccf_m_s(
        vm[win_b],
        ys[win_b],
        sig_s[win_b],
        rv_center_m_s=float(v_vertex),
        gaussian_sigma_m_s=sig_ms,
    )
    if math.isfinite(rv_err_m_s) and rv_err_m_s > 0:
        rv_err_m_s = max(rv_err_m_s, _ccf_rv_err_numeric_floor(vstep))
    else:
        rv_err_m_s = float("nan")

    rv_err_bootstrap_m_s = float("nan")
    n_bs = int(bootstrap_rv_err_samples)
    if n_bs > 4:
        rng = np.random.default_rng(bootstrap_seed)
        trials: list[float] = []
        for _ in range(n_bs):
            yp = y + rng.standard_normal(y.shape) * ge
            vv, _em, _ = _gaussian_smooth_parabolic_vertex_m_s(
                vm,
                yp,
                sigma_smooth_bins=sb,
                rv_anchor_ms=float(rv_anchor_ms),
                gaussian_sigma_m_s=sig_ms,
                v_step_m_s=vstep,
                peak_refine_iterations=int(peak_refine_iterations),
                peak_search_center_m_s=None,
            )
            if np.isfinite(vv):
                trials.append(float(vv))
        if len(trials) >= max(5, n_bs // 4):
            rv_err_bootstrap_m_s = float(np.std(np.asarray(trials, dtype=np.float64), ddof=1))
        if not math.isfinite(rv_err_bootstrap_m_s) or rv_err_bootstrap_m_s <= 0:
            rv_err_bootstrap_m_s = float("nan")
        elif math.isfinite(rv_err_bootstrap_m_s):
            rv_err_bootstrap_m_s = max(rv_err_bootstrap_m_s, _ccf_rv_err_numeric_floor(vstep))

    v_lo_d = float(pa.get("parabola_v_lo_m_s", float("nan")))
    v_hi_d = float(pa.get("parabola_v_hi_m_s", float("nan")))
    a_p = float(pa.get("parabola_a", float("nan")))
    b_p = float(pa.get("parabola_b", float("nan")))
    c_p = float(pa.get("parabola_c", float("nan")))
    vm_min = float(np.min(vm))
    vm_max = float(np.max(vm))
    # Preview plot: three parabola samples span only ~2×v_step; extend the dashed curve over a
    # wider v band around the refined vertex so the HTML preview is readable.
    d_lo = abs(float(v_vertex) - v_lo_d) if math.isfinite(v_lo_d) else 0.0
    d_hi = abs(v_hi_d - float(v_vertex)) if math.isfinite(v_hi_d) else 0.0
    half_span_m = max(
        12_000.0,
        6.0 * float(sig_ms),
        2.5 * max(d_lo, d_hi, 1.0),
    )
    v_lo_plot = max(vm_min, float(v_vertex) - half_span_m)
    v_hi_plot = min(vm_max, float(v_vertex) + half_span_m)
    if (
        math.isfinite(a_p)
        and math.isfinite(b_p)
        and math.isfinite(c_p)
        and v_hi_plot > v_lo_plot
    ):
        n_dense = max(
            96,
            int(
                min(
                    800,
                    max(24, (v_hi_plot - v_lo_plot) / max(0.08 * vstep, 1e-6)),
                )
            ),
        )
        vd = np.linspace(v_lo_plot, v_hi_plot, n_dense, dtype=np.float64)
        zd = a_p * (vd * vd) + b_p * vd + c_p
    else:
        vd = np.array([], dtype=np.float64)
        zd = np.array([], dtype=np.float64)

    meta = {
        "ok": True,
        "ccf_profile_heuristic": profile,
        "refine_optimizer_anchor": sweep_anchor_kind,
        "rv_anchor_gaussian_m_s": float(rv_anchor_ms),
        "rv_raw_grid_peak_m_s": float(rv_raw_peak),
        "refine_raw_gaussian_agreement_m_s": (
            abs(float(rv_raw_peak) - float(rv_anchor_ms))
            if math.isfinite(rv_raw_peak) and math.isfinite(rv_anchor_ms)
            else float("nan")
        ),
        "refine_peak_search_center_m_s": float(peak_search_center_m_s)
        if peak_search_center_m_s is not None
        else float("nan"),
        "refine_half_window_m_s": float(pa.get("refine_half_window_m_s", float("nan"))),
        "peak_refine_iterations": int(pa.get("refine_peak_iterations_used", peak_refine_iterations)),
        "gaussian_sigma_m_s_used": sig_ms,
        "sigma_smooth_bins": sb,
        "smooth_sigma_scale": float(smooth_sigma_scale),
        "refine_method": "gaussian_smooth_1d_parabolic_vertex",
        "rv_err_m_s": float(rv_err_m_s),
        "rv_err_bootstrap_m_s": float(rv_err_bootstrap_m_s),
        "bootstrap_rv_err_samples": int(n_bs),
        "refine_curve_v_m_s": np.asarray(vm, dtype=np.float64).copy(),
        "refine_curve_ccf_smoothed": np.asarray(ys, dtype=np.float64).copy(),
        "refine_curve_ccf_raw": np.asarray(y, dtype=np.float64).copy(),
        "refine_curve_ccf_sigma": np.asarray(sig_s, dtype=np.float64).copy(),
        "refine_peak_search_mask": np.asarray(pa.get("peak_search_mask"), dtype=bool)
        if isinstance(pa.get("peak_search_mask"), np.ndarray)
        else np.zeros(vm.shape, dtype=bool),
        "refine_parabola_v_dense_m_s": vd,
        "refine_parabola_ccf_dense": zd,
        "parabola_v_lo_m_s": float(pa.get("parabola_v_lo_m_s", float("nan"))),
        "parabola_v_mid_m_s": float(pa.get("parabola_v_mid_m_s", float("nan"))),
        "parabola_v_hi_m_s": float(pa.get("parabola_v_hi_m_s", float("nan"))),
        "parabola_ccf_smoothed_lo": float(pa.get("parabola_ccf_smoothed_lo", float("nan"))),
        "parabola_ccf_smoothed_mid": float(pa.get("parabola_ccf_smoothed_mid", float("nan"))),
        "parabola_ccf_smoothed_hi": float(pa.get("parabola_ccf_smoothed_hi", float("nan"))),
        "parabola_a": float(pa.get("parabola_a", float("nan"))),
        "parabola_b": float(pa.get("parabola_b", float("nan"))),
        "parabola_c": float(pa.get("parabola_c", float("nan"))),
        "refine_peak_method": pa.get("refine_peak_method"),
        "rv_refined_m_s": float(v_vertex),
    }
    return float(v_vertex), meta


def sweep_refine_ccf_rv_coarse_grid_min_abs_bias(
    v_grid_m_s: np.ndarray,
    ccf: np.ndarray,
    v_step_m_s: float,
    *,
    truth_rv_m_s: float,
    ccf_err: np.ndarray | None = None,
    gaussian_sigma_m_s: float | None = None,
    mask_method: str | None = None,
    config: CCFFitKpfConfig | None = None,
    scale_lo: float = 0.04,
    scale_hi: float = 0.10,
    n_scale: int = 61,
    sigma_smooth_bins_min: float = 0.4,
    sigma_smooth_bins_max: float = 12.0,
    peak_refine_iterations: int = _REFINE_PEAK_ITERATIONS_DEFAULT,
    bootstrap_rv_err_samples: int = 0,
    bootstrap_seed: int | None = None,
) -> tuple[float, float, dict]:
    """
    Choose ``smooth_sigma_scale`` in ``[scale_lo, scale_hi]`` to minimize
    ``|RV_refined − truth_rv_m_s|`` for :func:`refine_ccf_rv_coarse_grid_parabolic`.

    Callers should pass a **peak-aligned** ``truth_rv_m_s`` (typically the raw CCF sub-pixel
    maximum on the grid). Using the LM Gaussian center as ``truth`` would reintroduce its bias.

    Sweeps with bootstrap disabled for speed, then re-runs once at the best scale with the
    requested bootstrap settings.

    Returns
    -------
    best_scale : float
        ``smooth_sigma_scale`` that minimized absolute bias.
    rv_refined_m_s : float
        Refined RV from the final call (with bootstrap if requested).
    meta : dict
        Metadata from the final refinement, plus
        ``refine_min_bias_optimized`` (True), ``refine_min_bias_truth_rv_m_s``,
        ``refine_min_bias_abs_m_s`` (at optimum), and ``refine_min_bias_sweep_n``.
    """
    truth = float(truth_rv_m_s)
    if not math.isfinite(truth):
        return float("nan"), float("nan"), {"ok": False, "message": "invalid truth_rv_m_s"}

    lo, hi = float(scale_lo), float(scale_hi)
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return float("nan"), float("nan"), {"ok": False, "message": "invalid scale_lo/hi"}
    n_s = max(3, int(n_scale))
    scales = np.linspace(lo, hi, n_s, dtype=np.float64)

    best_sc = float(_REFINE_COARSE_GRID_SMOOTH_SIGMA_SCALE_DEFAULT)
    best_abs = float("inf")

    for sc in scales:
        rv, meta = refine_ccf_rv_coarse_grid_parabolic(
            v_grid_m_s,
            ccf,
            v_step_m_s,
            ccf_err=ccf_err,
            gaussian_sigma_m_s=gaussian_sigma_m_s,
            mask_method=mask_method,
            config=config,
            smooth_sigma_scale=float(sc),
            optimize_smooth_sigma_scale=False,
            sigma_smooth_bins_min=sigma_smooth_bins_min,
            sigma_smooth_bins_max=sigma_smooth_bins_max,
            peak_refine_iterations=int(peak_refine_iterations),
            bootstrap_rv_err_samples=0,
            bootstrap_seed=bootstrap_seed,
        )
        if not meta.get("ok") or not math.isfinite(float(rv)):
            continue
        d = abs(float(rv) - truth)
        if d < best_abs:
            best_abs = d
            best_sc = float(sc)

    if not math.isfinite(best_abs) or math.isinf(best_abs):
        return float("nan"), float("nan"), {
            "ok": False,
            "message": "min-bias sweep: no successful refinement",
        }

    rv_f, meta_f = refine_ccf_rv_coarse_grid_parabolic(
        v_grid_m_s,
        ccf,
        v_step_m_s,
        ccf_err=ccf_err,
        gaussian_sigma_m_s=gaussian_sigma_m_s,
        mask_method=mask_method,
        config=config,
        smooth_sigma_scale=best_sc,
        optimize_smooth_sigma_scale=False,
        sigma_smooth_bins_min=sigma_smooth_bins_min,
        sigma_smooth_bins_max=sigma_smooth_bins_max,
        peak_refine_iterations=int(peak_refine_iterations),
        bootstrap_rv_err_samples=int(bootstrap_rv_err_samples),
        bootstrap_seed=bootstrap_seed,
    )
    if not meta_f.get("ok"):
        return best_sc, float("nan"), meta_f

    meta_out = dict(meta_f)
    if math.isfinite(float(rv_f)):
        meta_out["rv_refined_m_s"] = float(rv_f)
    meta_out["refine_min_bias_optimized"] = True
    meta_out["refine_min_bias_truth_rv_m_s"] = truth
    meta_out["refine_min_bias_abs_m_s"] = abs(float(rv_f) - truth)
    meta_out["refine_min_bias_sweep_n"] = n_s
    meta_out["refine_min_bias_scale_lo"] = lo
    meta_out["refine_min_bias_scale_hi"] = hi
    return best_sc, float(rv_f), meta_out


def _bootstrap_gaussian_rv_std_m_s(
    gx_km: np.ndarray,
    gy0: np.ndarray,
    ge: np.ndarray,
    *,
    seed_mean_km: float,
    seed_std_km: float,
    profile: str,
    n_samples: int,
    rng: np.random.Generator,
    maxiter: int = 400,
) -> float:
    """
    Scatter of the fitted Gaussian mean (m/s) when each CCF sample is perturbed by its
    reported σ (i.i.d.); ignores bin-to-bin correlation. Uses weighted LM like the main fit.
    """
    if n_samples < 4 or gy0.size < 5:
        return float("nan")
    gx = np.asarray(gx_km, np.float64).ravel()
    y0 = np.asarray(gy0, np.float64).ravel()
    gerr = np.asarray(ge, np.float64).ravel()
    if gx.size != y0.size or gerr.size != y0.size:
        return float("nan")
    good = np.isfinite(gx) & np.isfinite(y0) & np.isfinite(gerr) & (gerr > 0)
    if int(np.count_nonzero(good)) < 5:
        return float("nan")
    gx = gx[good]
    y0 = y0[good]
    gerr = gerr[good]
    w = 1.0 / (gerr * gerr)
    ft = fitting.LevMarLSQFitter()
    means = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        yn = y0 + rng.standard_normal(y0.shape) * gerr
        ys = _levmar_ccf_y_scale(yn)
        yn_s = np.asarray(yn, dtype=np.float64) * ys
        bmed = float(np.median(yn_s))
        gy = yn_s - bmed
        span = abs(float(np.nanmax(gy) - np.nanmin(gy))) * 100.0
        amp = (
            max(-1e7, float(np.nanmin(gy)) - span)
            if profile == "emission"
            else min(1e7, float(np.nanmax(gy)) + span)
        )
        try:
            init_b = Gaussian1D(
                amplitude=amp, mean=float(seed_mean_km), stddev=float(seed_std_km)
            ) + Const1D(amplitude=bmed)
            w_eff = np.asarray(w, dtype=np.float64) / (ys * ys)
            gfit = ft(init_b, gx, yn_s, weights=w_eff, maxiter=maxiter)
            means[i] = float(gfit.mean_0.value) * 1000.0
        except Exception:
            means[i] = float("nan")
    fin = np.isfinite(means)
    if int(np.count_nonzero(fin)) < max(4, n_samples // 4):
        return float("nan")
    mv = means[fin]
    return float(np.std(mv, ddof=1))


def dv_per_pixel_m_s(wl_um: float) -> float:
    """Echelle Δv per pixel (m/s): order ≈ 147/λ µm → c/(order·4096)."""
    if not np.isfinite(wl_um) or wl_um <= 0:
        return float("nan")
    return float(C_M_S / (max(147.0 / wl_um, 1.0) * 4096.0))


def _subsample_velocity_ccf_for_gaussian_fit(
    vm: np.ndarray,
    y: np.ndarray,
    y_err_ok: np.ndarray | None,
    v_step_m_s: float,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float, int]:
    """
    Uniform stride subsample along velocity when the grid is very long (mask-KPF broad scans).

    Preserves ordering; scales ``v_step_m_s`` by the stride so error heuristics stay consistent
    with the effective sample spacing.
    """
    n = int(vm.size)
    ms = int(max_samples)
    if n <= ms or ms < 8:
        return vm, y, y_err_ok, float(v_step_m_s), 1
    stride = int(math.ceil(n / float(ms)))
    stride = max(1, stride)
    vm_s = np.asarray(vm[::stride], dtype=np.float64)
    y_s = np.asarray(y[::stride], dtype=np.float64)
    err_s = (
        np.asarray(y_err_ok[::stride], dtype=np.float64)
        if y_err_ok is not None and y_err_ok.size == n
        else None
    )
    vstep_s = float(v_step_m_s) * float(stride)
    return vm_s, y_s, err_s, vstep_s, stride


def _ccf_fit_debug_jsonable(v):
    """Reduce fit_ccf_rv debug context values to JSON-friendly types."""
    if isinstance(v, (bool, str, type(None))):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        xf = float(v)
        return xf if np.isfinite(xf) else None
    if isinstance(v, dict):
        return {str(k): _ccf_fit_debug_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_ccf_fit_debug_jsonable(x) for x in v]
    return str(v)


def _levmar_ccf_y_scale(gy: np.ndarray) -> float:
    """
    Multiplicative scale so ``gy * scale`` has max(|·|) ≈ 1.

    ``scipy.optimize.leastsq`` (used by Astropy :class:`~astropy.modeling.fitting.LevMarLSQFitter`)
    reports ``ierr`` outside {1,…,4} when the problem is poorly scaled (common for mask-CCF
    amplitudes ~1e-18), which triggers ``AstropyUserWarning: The fit may be unsuccessful``.
    Scaling **y** (and inverse-variance weights consistently) preserves Gaussian **mean** and
    **stddev** in velocity while improving conditioning.
    """
    m = float(np.nanmax(np.abs(np.asarray(gy, dtype=np.float64))))
    if not math.isfinite(m) or m <= 0:
        return 1.0
    return 1.0 / m


def fit_ccf_rv(
    v_grid_m_s: np.ndarray,
    ccf: np.ndarray,
    v_step_m_s: float,
    n_good: int | None = None,
    return_params: bool = False,
    *,
    ccf_err: np.ndarray | None = None,
    mask_method: str | None = None,
    config: CCFFitKpfConfig | None = None,
    vel_span_m_s_per_pixel: float | None = None,
    two_pass: bool = True,
    max_velocity_samples: int | None = 8192,
):
    """
    Fit a Gaussian plus constant baseline to a precomputed CCF and return RV ± error in m s⁻¹.

    Uses a robust emission/absorption profile guess, :class:`~astropy.modeling.models.Gaussian1D`
    plus :class:`~astropy.modeling.models.Const1D` on the CCF in a sliding velocity
    window (values are scaled to ~unit magnitude before LM; RV parameters are unchanged), and
    one or two Levenberg–Marquardt passes. Each pass uses a **wide** velocity
    window (see :class:`CCFFitKpfConfig`) so the continuum level is constrained together with
    the line. Within each window the fitter runs **unclipped** χ² passes on the full sample
    (repeated LM with ``levmar_maxiter``), then **sigma-clipping** refinement with relaxed
    thresholds on the first clipping rounds. Formal uncertainty may use a KPF-inspired
    ``ccf_error`` scaling when plausible, else covariance from the fitter, else a
    σ/√n fallback.

    Parameters
    ----------
    v_grid_m_s : np.ndarray
        Velocity samples (m s⁻¹), usually sorted (e.g. broad scan and/or hybrid merge).
    ccf : np.ndarray
        CCF values on the same grid.
    v_step_m_s : float
        Effective **finest** velocity spacing (m s⁻¹) for plausibility checks and
        ``n_sp`` scaling — typically :func:`effective_ccf_v_step_for_fit` on ``v_grid_m_s``.
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
    max_velocity_samples : int or None, optional
        When the input grid has more than this many finite samples, apply uniform stride
        subsampling for the Gaussian fit only (default ``8192``). Set ``None`` to always
        use the full grid. Does not change stored CCF arrays elsewhere — only speeds LM
        fitting on very long mask-KPF broad scans.
    ccf_err : np.ndarray or None, optional
        Per-sample CCF uncertainty (same shape as ``ccf``). When finite and
        predominantly positive, Levenberg–Marquardt fits use ``weights = 1/ccf_err²``
        on the raw CCF samples in each pass window.

    Returns
    -------
    tuple
        If ``return_params`` is False: ``(rv_ms, rv_err_ms)``.
        If True: ``(rv_ms, rv_err_ms, params, diag)`` with fitted Gaussian parameters
        and diagnostic metadata.

    Notes
    -----
    Design parallels public KPF PRV CCF-fitting ideas; see module ``References``.
    When both the KPF-style gradient error and the Levenberg–Marquardt covariance on
    the Gaussian mean pass plausibility checks, :class:`CCFFitKpfConfig` field
    ``rv_err_cov_kpf_combine`` chooses **``min``** (historic) or **``max``**
    (more conservative; default).

    Optional **parametric bootstrap** (``rv_err_bootstrap_samples`` > 0): repeatedly
    add i.i.d. ``N(0, ccf_err)`` noise to the final-window CCF samples, refit the
    weighted Gaussian, and take the sample standard deviation of fitted means (m/s).
    The reported ``rv_err`` is **at least** that bootstrap scatter when bootstrap
    runs (still diagonal / uncorrelated noise; see Notes).

    **Self-consistency of ``ccf_err`` → ``rv_err`` (what is *not* done).**

    - ``ccf_err`` is used only as **diagonal** inverse-variance weights ``1/σ²`` on
      the **raw** CCF samples inside each fit window (Gaussian + constant model). For ``ccf_mask_kpf``,
      those σ come from the weighted SD of per-line :math:`F\\,\\mathrm{SN}` about the
      chunk CCF, then combined across chunks as a standard error of the mean; this is
      not a full propagation from pixel flux errors, and **bin-to-bin correlations** in
      :math:`v` are ignored.
    - The Levenberg–Marquardt **covariance** on the Gaussian mean (when returned by
      Astropy) is a **local** linearization error for the weighted nonlinear fit; it
      does not add systematic terms from model mismatch (true line profile ≠ Gaussian).
    - The **KPF-style** ``_kpf_err_km`` branch uses gradients of the **raw** CCF
      (``√|CCF|`` scaling) and **does not** use ``ccf_err``. It returns **0** if any
      CCF sample in the fit window is **negative**, so for strongly absorption-shaped
      curves the KPF branch may be inactive and ``rv_err`` relies on covariance or
      fallbacks. With ``rv_err_cov_kpf_combine="min"``, **``min(KPF, cov)``** can still
      yield an **optimistically small** ``rv_err``; default ``"max"`` is more conservative.
    - If the LM Gaussian **width** collapses toward zero, the formal covariance on the
      mean can reach ~1e-16 m/s while still passing ``_ok_err``; such fits set
      ``diag["lm_gaussian_width_plausible"]=False``, drop the covariance branch, and
      combine ``rv_err`` with :func:`_rv_err_slope_propagation_m_s` (CCF + ``ccf_err``)
      and a widened ``σ/√n`` fallback using ``v_step``.
    - When the LevMar **covariance** is used for ``rv_err``, slope propagation is still
      computed for ``diag["rv_err_slope_propagation_m_s"]`` but is **not** forced as a
      minimum on ``rv_err`` (no 200 km/s-style floor from ``max(rv_err, σ_slope)``).
    """
    cfg = config or CCFFitKpfConfig()

    def _fail(msg: str, **ctx):
        d = {
            "fit_ok": False,
            "used_covariance_for_error": False,
            "used_sigma_fallback": False,
            "used_kpf_ccf_error": False,
            "used_ccf_pointwise_error": False,
            "sigma_clip_enabled": bool(cfg.sigma_clip_enable),
            "sigma_clip_iterations": 0,
            "sigma_clip_n_points_initial": 0,
            "sigma_clip_n_points_final": 0,
            "rv_err_bootstrap_sigma_m_s": float("nan"),
            "used_rv_err_bootstrap": False,
            "rv_err_cov_kpf_combine": str(cfg.rv_err_cov_kpf_combine),
            "applied_slope_propagation_floor": False,
            "message": msg,
        }
        if ctx:
            dbg = {k: _ccf_fit_debug_jsonable(v) for k, v in ctx.items()}
            d["debug_context"] = dbg
            try:
                extra = json.dumps(dbg, sort_keys=True)
            except (TypeError, ValueError):
                extra = str(dbg)
            logger.warning("CCF fit: %s | %s", msg, extra)
        else:
            logger.warning("CCF fit: %s", msg)
        return (np.nan, np.nan, {}, d) if return_params else (np.nan, np.nan)

    vg = np.asarray(v_grid_m_s, np.float64).ravel()
    y0 = np.asarray(ccf, np.float64).ravel()
    ok = np.isfinite(vg) & np.isfinite(y0)
    n_ok_pre = int(np.sum(ok))
    if n_ok_pre < cfg.min_points + 2:
        return _fail(
            "insufficient valid points",
            fit_stage="input_grid",
            reason="Too few finite (velocity, CCF) pairs after masking non-finite values",
            n_finite_pairs=n_ok_pre,
            min_pairs_required=int(cfg.min_points + 2),
            min_points_fit_config=int(cfg.min_points),
            raw_grid_size=int(vg.size),
        )
    vm, y = vg[ok], y0[ok]
    y_err_ok: np.ndarray | None = None
    if ccf_err is not None:
        er0 = np.asarray(ccf_err, np.float64).ravel()
        if er0.size == vg.size:
            y_err_ok = er0[ok]
    v_step_fit = float(v_step_m_s)
    subsample_stride = 1
    if max_velocity_samples is not None and int(max_velocity_samples) > 0:
        vm, y, y_err_ok, v_step_fit, subsample_stride = _subsample_velocity_ccf_for_gaussian_fit(
            vm, y, y_err_ok, v_step_fit, int(max_velocity_samples)
        )
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
    # Use caller's effective finest step (hybrid grids are mostly coarse in |v|;
    # recomputing mean spacing here biased KPF errors high).
    vstep_k = float(v_step_fit) / 1000.0
    vsp_k = (
        float(vel_span_m_s_per_pixel) / 1000.0
        if vel_span_m_s_per_pixel and np.isfinite(vel_span_m_s_per_pixel) and vel_span_m_s_per_pixel > 0
        else vstep_k
    )

    def _lm(mo, ft):
        if hasattr(mo, "mean_0"):
            mk, sk = float(mo.mean_0.value), abs(float(mo.stddev_0.value))
            mean_pname = "mean_0"
        else:
            mk, sk = float(mo.mean.value), abs(float(mo.stddev.value))
            mean_pname = "mean"
        sm = None
        fi = getattr(ft, "fit_info", {}) or {}
        pc = fi.get("param_cov")
        if pc is None:
            pc = fi.get("cov_x")
        if pc is not None:
            try:
                pc = np.asarray(pc, np.float64)
                j = list(mo.param_names).index(mean_pname)
                if pc.shape[0] > j and pc[j, j] > 0:
                    sm = float(math.sqrt(pc[j, j]))
            except Exception:
                pass
        return mk, sk, sm

    def _pass(center_km: float, hw_km: float, sd0: float, ft):
        lo, hi = center_km - hw_km, center_km + hw_km
        sel = (vk >= lo) & (vk <= hi)
        if not np.any(sel):
            return None, np.array([]), float("nan"), float("nan"), 0.0, False, {}, np.array([]), np.array([])
        gx0 = vk[sel]
        gy00 = y[sel]
        fin = np.isfinite(gx0) & np.isfinite(gy00)
        gx0, gy00 = gx0[fin], gy00[fin]
        if gy00.size < cfg.min_points:
            return None, gx0, float("nan"), float("nan"), 0.0, False, {}, np.array([]), np.array([])
        wts0 = None
        used_pw = False
        if y_err_ok is not None:
            ge = np.asarray(y_err_ok[sel], np.float64)[fin]
            good_w = np.isfinite(ge) & (ge > 0)
            if int(np.count_nonzero(good_w)) >= cfg.min_points:
                med_e = float(np.nanmedian(ge[good_w]))
                floor = max(med_e * 1e-6, 1e-30)
                ge_safe = np.where(good_w, ge, floor)
                wts0 = 1.0 / (ge_safe * ge_safe)
                used_pw = True

        n_start = int(gx0.size)
        min_keep = max(cfg.min_points, int(math.ceil(cfg.sigma_clip_min_keep_frac * n_start)))
        clip_info: dict = {
            "sigma_clip": bool(cfg.sigma_clip_enable),
            "iterations": 0,
            "n_start": n_start,
            "n_final": n_start,
            "prefit_unclipped_passes": int(cfg.sigma_clip_unclipped_prefit_passes),
        }

        def _one_fit(gx: np.ndarray, gy0: np.ndarray, wts: np.ndarray | None):
            ys = _levmar_ccf_y_scale(gy0)
            gy = np.asarray(gy0, dtype=np.float64) * ys
            b0 = float(np.nanmedian(gy))
            gy_det = gy - b0
            span = abs(float(np.nanmax(gy_det) - np.nanmin(gy_det))) * 100.0
            amp = (
                max(-1e7, float(np.nanmin(gy_det)) - span)
                if profile == "emission"
                else min(1e7, float(np.nanmax(gy_det)) + span)
            )
            init = Gaussian1D(amplitude=amp, mean=center_km, stddev=sd0) + Const1D(amplitude=b0)
            mi = int(max(50, min(int(cfg.levmar_maxiter), 50_000)))
            if wts is not None:
                w_eff = np.asarray(wts, dtype=np.float64) / (ys * ys)
                return ft(init, gx, gy, weights=w_eff, maxiter=mi)
            return ft(init, gx, gy, maxiter=mi)

        gx_w = gx0
        gy0_w = gy00
        wts_w = wts0
        best_g = None
        best_bmed = 0.0
        best_gx = gx_w
        snap_gy0 = np.array([], dtype=np.float64)
        snap_ge = np.array([], dtype=np.float64)

        try:
            prefit = max(0, int(cfg.sigma_clip_unclipped_prefit_passes))
            if cfg.sigma_clip_enable:
                clip_max = max(0, int(cfg.sigma_clip_max_iter))
                total_it = prefit + clip_max
            else:
                clip_max = 0
                total_it = max(1, prefit) if prefit > 0 else 1

            for it in range(total_it):
                if gx_w.size < cfg.min_points:
                    break
                g = _one_fit(gx_w, gy0_w, wts_w)
                pred = np.asarray(g(gx_w), dtype=np.float64)
                resid = gy0_w - pred
                best_g = g
                best_bmed = (
                    float(g.amplitude_1.value)
                    if hasattr(g, "amplitude_1")
                    else float(np.nanmedian(gy0_w))
                )
                best_gx = gx_w
                snap_gy0 = np.asarray(gy0_w, dtype=np.float64).copy()
                if wts_w is not None:
                    snap_ge = (1.0 / np.sqrt(np.maximum(wts_w, 1e-300))).astype(np.float64)
                else:
                    snap_ge = np.full(snap_gy0.shape, np.nan, dtype=np.float64)
                clip_info["iterations"] = it + 1
                if not cfg.sigma_clip_enable:
                    continue
                if it >= total_it - 1:
                    break
                # Unclipped χ² passes: refit on full window only (no point removal).
                if it < prefit:
                    continue
                # Sigma-clipping refinement (relaxed threshold for first few clip rounds).
                clip_idx = it - prefit
                sigma_mult = (
                    float(cfg.sigma_clip_n_sigma_relaxed)
                    if clip_idx < int(cfg.sigma_clip_relax_first_clip_iters)
                    else float(cfg.sigma_clip_n_sigma)
                )
                med_r = float(np.median(resid))
                mad = float(np.median(np.abs(resid - med_r)))
                sig_rob = 1.4826 * mad
                span_y = float(np.nanmax(resid) - np.nanmin(resid)) if resid.size else 1.0
                sig_rob = max(sig_rob, 1e-9 * max(abs(span_y), 1.0))
                thr = float(sigma_mult) * sig_rob
                keep = np.abs(resid) <= thr
                n_kept = int(np.count_nonzero(keep))
                if n_kept >= gx_w.size:
                    break
                if n_kept < min_keep:
                    break
                gx_w = gx_w[keep]
                gy0_w = gy0_w[keep]
                if wts_w is not None:
                    wts_w = wts_w[keep]
            clip_info["n_final"] = int(best_gx.size) if best_g is not None else 0
            if best_g is None:
                return (
                    None,
                    gx0,
                    lo * 1000.0,
                    hi * 1000.0,
                    0.0,
                    used_pw,
                    clip_info,
                    np.array([]),
                    np.array([]),
                )
            return (
                best_g,
                best_gx,
                lo * 1000.0,
                hi * 1000.0,
                best_bmed,
                used_pw,
                clip_info,
                snap_gy0,
                snap_ge,
            )
        except Exception:
            return (
                None,
                gx0,
                lo * 1000.0,
                hi * 1000.0,
                0.0,
                used_pw,
                clip_info,
                np.array([]),
                np.array([]),
            )

    def _kpf_err_km(fw_km: float, rv0_km: float) -> float:
        v = np.asarray(vk, np.float64).ravel()
        c = np.asarray(y, np.float64).ravel()
        m = np.isfinite(v) & np.isfinite(c)
        v, c = v[m], c[m]
        if v.size < 3 or fw_km <= 0:
            return 0.0
        dv = np.diff(np.sort(v))
        pos = dv[(dv > 0) & np.isfinite(dv)]
        vs = float(np.min(pos)) if pos.size > 0 else 0.0
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
        """
        Accept finite, positive formal errors.

        Reject only non-finite values, non-positive values, or numerical blow-ups.
        When the fitted Gaussian width ``sig`` (m/s) is known and positive, reject
        covariances vastly larger than ``max(100·σ, 25·vstep)`` (unphysical vs. the
        fitted line width). There is **no** upper cap at ~200 km/s tied to velocity
        grid spacing (that spacing used to reject ``e > 5000·vstep`` and masked valid
        large uncertainties).
        """
        if not (math.isfinite(e) and e > 0):
            return False
        if e > 1.0e15:
            return False
        if sig > 0 and e > max(100.0 * sig, 25.0 * max(vstep, 1e-9)):
            return False
        return True

    ft1 = fitting.LevMarLSQFitter()
    g1, gx1, vlo1, vhi1, med1, pw1, clip1, gy01, ge1 = _pass(
        rv_gk, cfg.pass1_half_width_km_s, cfg.initial_stddev_km_s, ft1
    )
    if g1 is None or gx1.size < cfg.min_points:
        fp_ctx: dict = {
            "fit_stage": "pass1",
            "reason": "LevMar did not return a usable model, or post-window sample count "
            "fell below min_points",
            "g1_is_none": g1 is None,
            "pass1_fit_sample_count": int(gx1.size),
            "min_points": int(cfg.min_points),
            "seed_mean_km_s": float(rv_gk),
            "pass1_half_width_km_s": float(cfg.pass1_half_width_km_s),
            "fit_window_m_s_lo": float(vlo1) if np.isfinite(vlo1) else None,
            "fit_window_m_s_hi": float(vhi1) if np.isfinite(vhi1) else None,
            "ccf_profile": profile,
            "velocity_grid_m_s_min": float(vm.min()),
            "velocity_grid_m_s_max": float(vm.max()),
            "n_velocity_points": int(vm.size),
            "v_step_effective_m_s": float(v_step_fit),
            "velocity_subsample_stride": int(subsample_stride),
        }
        if isinstance(clip1, dict) and clip1:
            fp_ctx["sigma_clip_iterations"] = int(clip1.get("iterations", 0))
            fp_ctx["sigma_clip_n_start"] = int(clip1.get("n_start", 0))
            fp_ctx["sigma_clip_n_final"] = int(clip1.get("n_final", 0))
        return _fail("first-pass fit failed", **fp_ctx)
    m1k, _, _ = _lm(g1, ft1)
    if not (float(gx1.min()) <= m1k <= float(gx1.max())):
        gx_min, gx_max = float(gx1.min()), float(gx1.max())
        ow_ctx = {
            "fit_stage": "pass1_post_lm",
            "reason": "LevMar Gaussian mean (km/s) lies outside the velocity abscissa of "
            "samples retained in the pass-1 window (after any sigma-clipping)",
            "fitted_mean_km_s": float(m1k),
            "fit_samples_v_km_min": gx_min,
            "fit_samples_v_km_max": gx_max,
            "fit_window_m_s_lo": float(vlo1),
            "fit_window_m_s_hi": float(vhi1),
            "pass1_half_width_km_s": float(cfg.pass1_half_width_km_s),
            "n_points_in_pass1_window": int(gx1.size),
            "seed_mean_km_s": float(rv_gk),
            "ccf_profile": profile,
            "velocity_grid_m_s_min": float(vm.min()),
            "velocity_grid_m_s_max": float(vm.max()),
            "n_velocity_points": int(vm.size),
            "v_step_effective_m_s": float(v_step_fit),
            "velocity_subsample_stride": int(subsample_stride),
            "ccf_y_min": float(np.nanmin(y)),
            "ccf_y_max": float(np.nanmax(y)),
        }
        if not math.isfinite(m1k):
            ow_ctx["mean_side"] = "non_finite_mean"
        elif m1k < gx_min:
            ow_ctx["mean_offset_km_s"] = float(gx_min - m1k)
            ow_ctx["mean_side"] = "below_sample_min"
        elif m1k > gx_max:
            ow_ctx["mean_offset_km_s"] = float(m1k - gx_max)
            ow_ctx["mean_side"] = "above_sample_max"
        if isinstance(clip1, dict):
            ow_ctx["sigma_clip_iterations"] = int(clip1.get("iterations", 0))
            ow_ctx["sigma_clip_n_start"] = int(clip1.get("n_start", 0))
            ow_ctx["sigma_clip_n_final"] = int(clip1.get("n_final", 0))
        return _fail("mean outside window", **ow_ctx)
    best_g, best_ft, best_gx, best_med, best_vlo, best_vhi = g1, ft1, gx1, med1, vlo1, vhi1
    used_pw_fit = pw1
    clip_used = clip1
    snap_y0, snap_ge = gy01, ge1
    pw = 1
    if two_pass:
        ft2 = fitting.LevMarLSQFitter()
        g2, gx2, vlo2, vhi2, med2, pw2, clip2, gy02, ge02 = _pass(
            m1k, cfg.pass2_half_width_km_s, float(np.clip(cfg.initial_stddev_km_s, 0.05, 200.0)), ft2
        )
        if g2 is not None and gx2.size >= cfg.min_points:
            m2k, _, _ = _lm(g2, ft2)
            r2 = m2k * 1000.0
            if vlo2 <= r2 <= vhi2 and float(gx2.min()) <= m2k <= float(gx2.max()):
                best_g, best_ft, best_gx, best_med, best_vlo, best_vhi = g2, ft2, gx2, med2, vlo2, vhi2
                used_pw_fit = pw2
                clip_used = clip2
                snap_y0, snap_ge = gy02, ge02
                pw = 2

    mk, sk, sigc = _lm(best_g, best_ft)
    rv_ms, sig_ms = float(mk * 1000.0), float(sk * 1000.0)
    vstep = float(v_step_fit)
    # LM can drive Gaussian σ → ~0; formal covariance on the mean then hits ~1e-16 m/s while
    # _ok_err still accepts it (bound uses max(100·σ, 25·vstep)). Treat that as non-physical.
    lm_width_plausible = math.isfinite(sig_ms) and sig_ms >= max(1e-15, 1e-12 * vstep)
    sig_for_fb_ms = (
        float(sig_ms)
        if lm_width_plausible
        else max(vstep, 1e-12) if math.isfinite(vstep) and vstep > 0 else max(float(sig_ms), 1e-12)
    )
    fw_km = 2.0 * (cfg.pass2_half_width_km_s if pw == 2 else cfg.pass1_half_width_km_s)
    ek = _kpf_err_km(fw_km, mk)
    ek_ms = float(ek * 1000.0)
    ek_ok = ek > 0 and np.isfinite(ek_ms) and _ok_err(ek_ms, sig_ms, vstep)
    cov_ms = (
        float(sigc * 1000.0)
        if (sigc is not None and np.isfinite(sigc) and float(sigc) > 0)
        else float("nan")
    )
    cov_ok = math.isfinite(cov_ms) and cov_ms > 0 and _ok_err(cov_ms, sig_ms, vstep)
    if not lm_width_plausible:
        cov_ok = False

    n_fit = max(int(best_gx.size), 1)
    rv_err_fb = float(sig_for_fb_ms / math.sqrt(float(n_fit)))
    if n_good is not None and int(n_good) > 0:
        rv_err_fb = max(rv_err_fb, float(sig_for_fb_ms / math.sqrt(float(n_good))))

    comb = str(cfg.rv_err_cov_kpf_combine).lower().strip()
    if comb not in ("min", "max"):
        comb = "max"

    used_kpf = used_cov = used_fb = False
    if cov_ok and ek_ok:
        rv_err_ms = min(cov_ms, ek_ms) if comb == "min" else max(cov_ms, ek_ms)
        used_kpf = True
        used_cov = True
        msg = f"{'min' if comb == 'min' else 'max'}(KPF ccf_error, LevMar cov)"
    elif cov_ok:
        rv_err_ms = cov_ms
        used_cov = True
        msg = "LevMar cov"
    elif ek_ok:
        rv_err_ms = ek_ms
        used_kpf = True
        msg = "KPF ccf_error"
    else:
        rv_err_ms = rv_err_fb
        used_fb = True
        if ek > 0 and np.isfinite(ek):
            msg = "σ/√n (KPF ccf_error implausible)"
        elif sigc is not None and np.isfinite(float(sigc)) and float(sigc) > 0:
            msg = "σ/√n (LevMar cov implausible)"
        else:
            msg = "σ/√n fallback"

    used_boot = False
    boot_sig = float("nan")
    n_bs = int(cfg.rv_err_bootstrap_samples)
    if (
        n_bs > 0
        and snap_ge.size == snap_y0.size
        and snap_y0.size > 0
        and np.any(np.isfinite(snap_ge) & (snap_ge > 0))
    ):
        rng = np.random.default_rng(cfg.rv_err_bootstrap_seed)
        _sd_km = (
            abs(float(best_g.stddev_0.value))
            if hasattr(best_g, "stddev_0")
            else abs(float(best_g.stddev.value))
        )
        boot_sig = _bootstrap_gaussian_rv_std_m_s(
            best_gx,
            snap_y0,
            snap_ge,
            seed_mean_km=float(mk),
            seed_std_km=max(_sd_km, 1e-9),
            profile=profile,
            n_samples=n_bs,
            rng=rng,
            maxiter=int(max(50, min(int(cfg.levmar_maxiter), 50_000))),
        )
        if math.isfinite(boot_sig) and boot_sig > 0:
            used_boot = True
            rv_err_ms = max(rv_err_ms, boot_sig)

    slope_prop_ms = float("nan")
    applied_slope_floor = False
    if snap_y0.size >= 5 and best_gx.size == snap_y0.size:
        se_slope = snap_ge if np.any(np.isfinite(snap_ge) & (snap_ge > 0)) else None
        slope_prop_ms = _rv_err_slope_propagation_m_s(best_gx * 1000.0, snap_y0, se_slope)
    if math.isfinite(slope_prop_ms) and slope_prop_ms > 0 and not used_cov:
        rv_err_ms = max(rv_err_ms, slope_prop_ms)
        applied_slope_floor = True

    rv_err_ms = max(rv_err_ms, _ccf_rv_err_numeric_floor(vstep))

    if hasattr(best_g, "amplitude_0"):
        _amp = float(best_g.amplitude_0.value)
        _baseline = float(best_g.amplitude_1.value)
    else:
        _amp = float(best_g.amplitude.value)
        _baseline = float(best_med)
    params = {
        "amplitude": _amp,
        "v0": rv_ms,
        "sigma": sig_ms,
        "baseline": _baseline,
    }
    diag = {
        "fit_ok": True,
        "used_covariance_for_error": used_cov,
        "used_sigma_fallback": used_fb,
        "used_kpf_ccf_error": used_kpf,
        "used_ccf_pointwise_error": bool(used_pw_fit),
        "message": msg,
        "ccf_velocity_grid_min_m_s": float(np.min(vm)),
        "ccf_velocity_grid_max_m_s": float(np.max(vm)),
        "fit_v_lo": float(best_vlo),
        "fit_v_hi": float(best_vhi),
        "profile": profile,
        "kpf_pass_used": pw,
        "ccf_dir_style": "up" if profile == "absorption" else "down",
        "sigma_clip_enabled": bool(clip_used.get("sigma_clip", cfg.sigma_clip_enable)),
        "sigma_clip_iterations": int(clip_used.get("iterations", 0)),
        "sigma_clip_n_points_initial": int(clip_used.get("n_start", 0)),
        "sigma_clip_n_points_final": int(clip_used.get("n_final", best_gx.size)),
        "rv_err_bootstrap_sigma_m_s": boot_sig,
        "used_rv_err_bootstrap": used_boot,
        "rv_err_cov_kpf_combine": comb,
        "lm_gaussian_width_plausible": bool(lm_width_plausible),
        "rv_err_slope_propagation_m_s": slope_prop_ms,
        "applied_slope_propagation_floor": bool(applied_slope_floor),
        "velocity_fit_subsample_stride": int(subsample_stride),
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
        return rv, float(max(v_step * 2.0, _ccf_rv_err_numeric_floor(v_step)))

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
    ng = max(int(n_good), 1)
    return rv, float(max(hwhm / math.sqrt(float(ng)), _ccf_rv_err_numeric_floor(v_step)))


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


def _ccf_preview_finalize_n_workers(n_slots: int) -> int:
    """
    Worker count for parallel Gaussian + refine in :func:`finalize_ccf_preview_dict`.

    Set ``HISPEC_DRP_CCF_PREVIEW_SLOT_THREADS`` to a positive int to cap workers; ``1`` forces
    sequential (legacy behavior). Unset or invalid: ``min(n_slots, cpu_count)``.
    """
    raw = os.environ.get("HISPEC_DRP_CCF_PREVIEW_SLOT_THREADS", "").strip()
    if raw:
        try:
            n = int(raw)
        except ValueError:
            n = 0
        if n < 1:
            return max(1, min(n_slots, os.cpu_count() or 4))
        return max(1, min(n, n_slots))
    return max(1, min(n_slots, os.cpu_count() or 4))


def _fit_ccf_preview_slot_worker(sl: dict, vs: float) -> tuple[dict, dict | None]:
    """
    One slot: :func:`fit_ccf_rv` with :data:`CCF_PREVIEW_SLOT_FIT_CONFIG` and optional
    :func:`_refine_ccf_rv_after_gaussian_fit_ccf_mask_kpf`. Thread-safe (no shared mutable state).
    """
    wl = sl.get("wl_lo", np.nan)
    if not np.isfinite(wl):
        wl = sl.get("wl_hi", np.nan)
    vsp = dv_per_pixel_m_s(float(wl))
    ce = sl.get("ccf_err")
    ccf_err_arg = np.asarray(ce, np.float64) if ce is not None else None
    y_slot = np.asarray(sl["ccf"], np.float64)
    v_slot = np.asarray(sl["v"], np.float64)
    rv, rv_err, params, diag = fit_ccf_rv(
        v_slot,
        y_slot,
        vs,
        return_params=True,
        vel_span_m_s_per_pixel=vsp if np.isfinite(vsp) else None,
        ccf_err=ccf_err_arg,
        config=CCF_PREVIEW_SLOT_FIT_CONFIG,
    )
    refine_meta: dict | None = None
    if bool(diag.get("fit_ok")) and np.isfinite(rv) and np.isfinite(rv_err) and float(rv_err) > 0:
        rv_ref, err_ref, rmeta = _refine_ccf_rv_after_gaussian_fit_ccf_mask_kpf(
            v_slot,
            y_slot,
            vs,
            ccf_err=ccf_err_arg,
            params=params if isinstance(params, dict) else None,
        )
        refine_meta = rmeta if isinstance(rmeta, dict) else None
        if rv_ref is not None and math.isfinite(float(rv_ref)):
            rv = float(rv_ref)
            if err_ref is not None and math.isfinite(float(err_ref)) and float(err_ref) > 0:
                rv_err = float(err_ref)
    kpf_fit = {"rv": rv, "rv_err": rv_err, "params": params, "diagnostics": diag}
    return kpf_fit, refine_meta


def finalize_ccf_preview_dict(
    order_slots: list[dict],
    all_ccf_rows: list[np.ndarray],
    v_grid: np.ndarray,
    v_min: float,
    v_max: float,
    v_step: float,
    all_ccf_err_rows: list[np.ndarray] | None = None,
) -> dict:
    """
    Build the ``_ccf_preview_by_fiber`` structure from per-order CCF 2-D arrays.

    ``order_slots`` entries must already include per-order mean CCF rows; this
    prepends the global mean slot (mean over all chunk rows).

    When ``all_ccf_err_rows`` is provided (one stacked array per order, aligned with
    ``all_ccf_rows``), the **mean** slot gets ``ccf_err`` = standard error of the
    column-wise mean CCF (``sqrt(sum σ²)/n``, not ``sqrt(mean(σ²))``) and
    :func:`fit_ccf_rv` uses them as inverse-variance weights.
    """
    if not all_ccf_rows:
        return {
            "v_step": float(v_step),
            "v_min": float(v_min),
            "v_max": float(v_max),
            "slots": [],
        }
    stacked = np.vstack(all_ccf_rows)
    combined = _nanmean_axis0_ignore_empty_columns(stacked)
    combined_err: np.ndarray | None = None
    if (
        all_ccf_err_rows is not None
        and len(all_ccf_err_rows) == len(order_slots)
        and all(r.size > 0 for r in all_ccf_err_rows)
    ):
        stacked_e = np.vstack(all_ccf_err_rows)
        combined_err = _nan_stderr_of_mean_axis0(stacked_e, stacked)
    mean_slot = {
        "kind": "mean",
        "order": None,
        "v": np.asarray(v_grid, dtype=np.float64),
        "ccf": np.asarray(combined, dtype=np.float64),
        "wl_lo": float("nan"),
        "wl_hi": float("nan"),
    }
    if combined_err is not None:
        mean_slot["ccf_err"] = np.asarray(combined_err, dtype=np.float64)
    slots = [mean_slot] + order_slots
    vs = float(v_step)
    n_workers = _ccf_preview_finalize_n_workers(len(slots))
    logger.info(
        "CCF preview: finalize_ccf_preview_dict fitting Gaussians n_slots=%d n_workers=%d "
        "(mean + orders)",
        len(slots),
        n_workers,
    )

    def _run_slot(s: dict) -> tuple[dict, dict | None]:
        return _fit_ccf_preview_slot_worker(s, vs)

    if n_workers <= 1:
        results = [_run_slot(sl) for sl in slots]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_run_slot, slots))
    for sl, (kpf_fit, refine_meta) in zip(slots, results):
        sl["kpf_fit"] = kpf_fit
        if refine_meta is not None and bool(refine_meta.get("ok")):
            sl["mask_kpf_refine_rmeta"] = refine_meta
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
        ccf_m = _nanmean_axis0_ignore_empty_columns(np.asarray(block, dtype=np.float64))
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
        ccf_m = _nanmean_axis0_ignore_empty_columns(np.asarray(ccf_2d, dtype=np.float64))
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


def combine_rv_estimates_median_mad_weighted(
    rvs: np.ndarray,
    rv_errs: np.ndarray,
    outlier_sigma: float = 4.0,
) -> tuple[float, float, np.ndarray, float, float]:
    """
    Fiber-level RV combination: robust rejection, inverse-variance mean, and a
    conservative uncertainty.

    The reported ``rv_std`` is ``max(rv_std_internal, scatter_se)`` where
    ``rv_std_internal`` is the usual Gaussian standard error of the weighted mean
    (``1/sqrt(sum 1/σ²)``) and ``scatter_se`` is ``MAD/√N`` of inlier RVs. That
    keeps the aggregate error from being smaller than the typical spread of the
    inputs (which would otherwise happen if per-input σ were over-optimistic).
    """
    rvs = np.asarray(rvs, dtype=np.float64).ravel()
    rv_errs = np.asarray(rv_errs, dtype=np.float64).ravel()
    if rvs.size != rv_errs.size:
        raise ValueError("combine_rv_estimates_median_mad_weighted: rvs and rv_errs length mismatch")
    inlier, _, _ = combine_rvs_median_mad(rvs, rv_errs, outlier_sigma)
    rv_mean, rv_std_internal = weighted_mean_rv(rvs, rv_errs, inlier)
    n_in = int(np.sum(inlier))
    scatter_se = float("nan")
    if n_in > 1:
        rv_in = rvs[inlier]
        mad_disp = float(mad_std(rv_in, ignore_nan=True))
        if math.isfinite(mad_disp) and mad_disp > 0:
            scatter_se = mad_disp / math.sqrt(float(n_in))
    if math.isfinite(rv_std_internal) and math.isfinite(scatter_se):
        rv_std = max(rv_std_internal, scatter_se)
    elif math.isfinite(rv_std_internal):
        rv_std = rv_std_internal
    else:
        rv_std = scatter_se
    return float(rv_mean), float(rv_std), inlier, float(rv_std_internal), float(scatter_se)


# =============================================================================
# Mask-shift line CCF (SPIROU .mas / KPF two-column)
# =============================================================================

@njit(cache=True)
def _linear_interp_sorted_1d(xq: float, x: np.ndarray, y: np.ndarray, n: int) -> float:
    """Linear interpolation on sorted ``x[:n]`` (ascending); clamps outside range."""
    if n <= 0:
        return np.nan
    if xq <= x[0]:
        return float(y[0])
    if xq >= x[n - 1]:
        return float(y[n - 1])
    i = int(np.searchsorted(x, xq, side="right"))
    i0 = i - 1
    i1 = i0 + 1
    if i1 >= n:
        return float(y[n - 1])
    x0 = float(x[i0])
    x1 = float(x[i1])
    if x1 <= x0:
        return float(y[i0])
    t = (xq - x0) / (x1 - x0)
    return (1.0 - t) * float(y[i0]) + t * float(y[i1])


@njit(cache=True, parallel=True)
def _calc_mask_ccf_chunks_numba(
    left_edges_2d: np.ndarray,
    right_edges_2d: np.ndarray,
    s_use_2d: np.ndarray,
    e_use_2d: np.ndarray,
    sn_2d: np.ndarray,
    n_pix_arr: np.ndarray,
    v_grid_m_s: np.ndarray,
    zb: float,
    line_lambda_rest_2d: np.ndarray,
    line_w_2d: np.ndarray,
    n_line_arr: np.ndarray,
    compute_line_std: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chunk-batched mask CCF(v) with parallelism over chunks.

    **Delta-line model:** each mask line is a rest wavelength :math:`\\lambda_{\\rm rest}`.
    At trial velocity :math:`v`, the observed wavelength is
    :math:`\\lambda_{\\rm obs} = \\lambda_{\\rm rest}\\,(1+v/c)/(1+z_b)`.

    For each velocity, let :math:`x_j = F_j\\,\\mathrm{SN}_j` at each contributing line
    (interpolated on pixel centers). Accumulate :math:`\\sum_j w_j x_j` and
    :math:`\\sum_j w_j x_j^2` over lines whose shifted :math:`\\lambda_{\\rm obs}` lies in
    the chunk with finite :math:`F` and :math:`\\mathrm{SN}`. The CCF is the weighted
    mean :math:`\\mathrm{CCF}(v)=\\sum_j w_j x_j\\,/\\,\\sum_j w_j`.

    When ``compute_line_std`` is True, the second output is the **weighted standard
    deviation** of the :math:`x_j` about that mean (same frequency weights :math:`w_j`):

    .. math::

        \\sigma_{\\mathrm{CCF}}(v) = \\sqrt{\\max\\left(0,\\;
          \\frac{\\sum_j w_j x_j^2}{\\sum_j w_j} - \\mathrm{CCF}(v)^2\\right)}\\,.

    The third return is kept for array-shape compatibility and is filled with NaN.

    ``e_use_2d`` is unused here but retained in the signature for call/cache stability.

    Input arrays are padded to fixed widths; ``n_pix_arr`` and ``n_line_arr`` give
    valid lengths for each chunk row.
    """
    n_chunk = left_edges_2d.shape[0]
    n_v = v_grid_m_s.shape[0]
    n_line_pad = line_lambda_rest_2d.shape[1]
    out_ccf = np.full((n_chunk, n_v), np.nan, dtype=np.float64)
    if compute_line_std:
        out_std = np.full((n_chunk, n_v), np.nan, dtype=np.float64)
        out_sigma_flux = np.full((n_chunk, n_v), np.nan, dtype=np.float64)
    else:
        out_std = np.empty((0, 0), dtype=np.float64)
        out_sigma_flux = np.empty((0, 0), dtype=np.float64)
    zb1 = 1.0 + zb

    for ic in prange(n_chunk):
        n_pix = int(n_pix_arr[ic])
        n_ln = int(n_line_arr[ic])
        if n_pix <= 0 or n_ln <= 0:
            continue
        chunk_lo = left_edges_2d[ic, 0]
        chunk_hi = right_edges_2d[ic, n_pix - 1]
        centers = np.empty(n_pix, dtype=np.float64)
        for n in range(n_pix):
            centers[n] = 0.5 * (
                left_edges_2d[ic, n] + right_edges_2d[ic, n]
            )
        for i_v in range(n_v):
            sh = (1.0 + v_grid_m_s[i_v] / C_M_S) / zb1
            total = 0.0
            w_sum = 0.0
            sum_wx2 = 0.0
            for j in range(n_ln):
                lam_rest = line_lambda_rest_2d[ic, j]
                if not np.isfinite(lam_rest) or lam_rest <= 0.0:
                    continue
                lam0 = lam_rest * sh
                if lam0 < chunk_lo or lam0 > chunk_hi:
                    continue
                fv = _linear_interp_sorted_1d(
                    lam0, centers, s_use_2d[ic, :], n_pix
                )
                snv = _linear_interp_sorted_1d(
                    lam0, centers, sn_2d[ic, :], n_pix
                )
                if not (np.isfinite(fv) and np.isfinite(snv)):
                    continue
                wj = line_w_2d[ic, j]
                xj = fv * snv
                partial = wj * xj
                total += partial
                w_sum += wj
                sum_wx2 += wj * xj * xj
            if w_sum > 0.0:
                inv_w = 1.0 / w_sum
                ccf = total * inv_w
                out_ccf[ic, i_v] = ccf
                if compute_line_std:
                    mean_x2 = sum_wx2 * inv_w
                    var = mean_x2 - ccf * ccf
                    if var < 0.0:
                        var = 0.0
                    out_std[ic, i_v] = math.sqrt(var)
                    out_sigma_flux[ic, i_v] = np.nan
            else:
                out_ccf[ic, i_v] = np.nan
                if compute_line_std:
                    out_sigma_flux[ic, i_v] = np.nan
                    out_std[ic, i_v] = np.nan
    return out_ccf, out_std, out_sigma_flux


@dataclass(frozen=True)
class MaskLines:
    """
    Mask lines as delta features in wavelength space (microns).

    All arrays are 1-D and equal length after validation in ``__post_init__``.

    **Mask-shift CCF** (``ccf_mask_kpf``) Doppler-shifts each rest wavelength
    :math:`\\lambda_{\\rm rest}` by trial velocity, forms
    :math:`\\sum_j w_j\\,F_j\\,\\mathrm{SN}_j` over in-chunk lines, then divides by
    :math:`\\sum_j w_j` so the statistic stays on scale as lines enter or leave the
    chunk with :math:`v`.

    On-disk SPIROU three-column files still list a wavelength interval; loaders reduce
    that row to :math:`\\lambda_{\\rm rest} = (λ_{\\rm lo}+λ_{\\rm hi})/2`.

    Attributes
    ----------
    line_lambda_um : np.ndarray
        Rest-frame line wavelength (µm), same frame as L2 ``WAVE_*`` (typically vacuum).
    weight : np.ndarray
        Per-line weight; contributes to both the CCF numerator and the per-velocity
        normalization denominator.
    """

    line_lambda_um: np.ndarray
    weight: np.ndarray

    def __post_init__(self) -> None:
        line_lambda_um = np.asarray(self.line_lambda_um, dtype=np.float64)
        weight = np.asarray(self.weight, dtype=np.float64)
        if line_lambda_um.shape != weight.shape:
            raise ValueError("MaskLines arrays must have matching shapes.")
        if line_lambda_um.ndim != 1:
            raise ValueError("MaskLines arrays must be 1D.")

        object.__setattr__(self, "line_lambda_um", line_lambda_um)
        object.__setattr__(self, "weight", weight)


def _mask_line_order_edge_taper_multipliers(
    order_wave_um: np.ndarray,
    line_lambda_um: np.ndarray,
    *,
    edge_taper_pixels: float,
    exp_shape: float = 5.0,
) -> np.ndarray:
    """
    Per-line weights multipliers for :func:`rv_per_chunks_mask_kpf` deweighting mask lines
    near echelle order edges.

    For each rest wavelength, the nearest spectral pixel index :math:`j` along the order
    (0 … :math:`n-1`) is found. With :math:`d = \\min(j, n-1-j)` the distance in pixels to
    the nearer physical edge, full weight applies for :math:`d \\ge N`; for :math:`d < N`
    the multiplier rises from 0 at the edge to 1 at :math:`d = N` as

    .. math::

        m = \\frac{e^{s \\cdot d/N} - 1}{e^{s} - 1},

    with shape parameter ``exp_shape`` :math:`= s` (default 5).

    Vectorized for large mask lists (``searchsorted`` on monotonic ``order_wave_um``); falls
    back to a per-line loop if the wavelength vector is not monotone.

    Parameters
    ----------
    order_wave_um
        1-D wavelength samples (µm) for the **full** echelle order (or the same span used
        for CCF); length :math:`n` defines pixel indices ``0 … n-1``.
    line_lambda_um
        Rest wavelengths (µm) of mask lines, same as in :class:`MaskLines`.
    edge_taper_pixels
        :math:`N` — taper applies for :math:`0 \\le d < N`; use ``0`` to disable (all ones).
    exp_shape
        :math:`s > 0` controlling how sharply the multiplier rises with :math:`d` inside
        the edge band.

    Returns
    -------
    np.ndarray
        Shape matching ``line_lambda_um``, values in :math:`[0, 1]`.
    """
    w = np.asarray(order_wave_um, dtype=np.float64).ravel()
    lam = np.asarray(line_lambda_um, dtype=np.float64).ravel()
    if lam.size == 0:
        return np.array([], dtype=np.float64)
    out = np.ones(lam.size, dtype=np.float64)
    n = int(w.size)
    if n < 2 or float(edge_taper_pixels) <= 0.0:
        return out
    N = float(edge_taper_pixels)
    s = float(exp_shape)
    if not np.isfinite(s) or s <= 0.0:
        s = 5.0
    denom = float(np.exp(s) - 1.0)
    if denom <= 1e-30:
        denom = 1.0
    inv_denom = 1.0 / denom

    diff = np.diff(w)
    mono_inc = diff.size > 0 and bool(np.all(diff >= 0))
    mono_dec = diff.size > 0 and bool(np.all(diff <= 0))

    def _nearest_index_searchsorted(lx: np.ndarray, increasing: bool) -> np.ndarray:
        """Nearest sample index along 1-D grid ``w`` (monotone)."""
        if increasing:
            w_grid = w
            jj = np.searchsorted(w_grid, lx, side="left")
            jj = np.clip(jj, 1, n - 1)
            jm = jj - 1
        else:
            w_grid = w[::-1]
            jj = np.searchsorted(w_grid, lx, side="left")
            jj = np.clip(jj, 1, n - 1)
            jm = jj - 1
            pick = np.abs(lx - w_grid[jm]) <= np.abs(lx - w_grid[jj])
            j_rev = np.where(pick, jm, jj).astype(np.intp, copy=False)
            return (n - 1 - j_rev).astype(np.intp, copy=False)
        pick = np.abs(lx - w_grid[jm]) <= np.abs(lx - w_grid[jj])
        return np.where(pick, jm, jj).astype(np.intp, copy=False)

    fin = np.isfinite(lam)
    j_idx = np.zeros(lam.shape[0], dtype=np.intp)
    if mono_inc or mono_dec:
        if np.any(fin):
            lx = lam[fin]
            j_idx[fin] = _nearest_index_searchsorted(lx, mono_inc)
    else:
        for i in range(lam.size):
            lx = float(lam[i])
            if not np.isfinite(lx):
                out[i] = 0.0
                continue
            j_idx[i] = int(np.argmin(np.abs(w - lx)))
        d = np.minimum(j_idx, n - 1 - j_idx).astype(np.float64)
        m = d >= N
        out[~fin] = 0.0
        out[fin & m] = 1.0
        inner = fin & ~m
        if np.any(inner):
            t = d[inner] / N
            out[inner] = np.clip((np.exp(t * s) - 1.0) * inv_denom, 0.0, 1.0)
        return out

    d = np.minimum(j_idx, n - 1 - j_idx).astype(np.float64)
    out[~fin] = 0.0
    m = d >= N
    out[fin & m] = 1.0
    inner = fin & ~m
    if np.any(inner):
        t = d[inner] / N
        out[inner] = np.clip((np.exp(t * s) - 1.0) * inv_denom, 0.0, 1.0)
    return out


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
    Load a SPIROU-style ``.mas`` mask file.

    **Three columns:** wavelength start, wavelength end, weight (legacy SPIROU). Each
    row becomes :math:`\\lambda_{\\rm rest} = (λ_{\\rm start}+λ_{\\rm end})/2` in µm
    after nm→µm scaling.
    **Two columns:** vacuum wavelength (nm), weight — template masks from
    :func:`hispecdrp.utils.line_mask_utils.build_line_mask_from_template`; each line is
    :math:`\\lambda_{\\rm rest} = λ_{\\rm nm}/1000`.

    Wavelength columns are **divided by 1000** to convert **nanometers → microns**. No
    additional air→vacuum correction is applied in code. Bundled HISPEC masks document
    **vacuum** nm in file headers; some upstream SPIROU/APERO masks may use **air**
    nm—in that case convert externally or pre-process the file.

    Comment lines beginning with ``#``, ``;``, or ``//`` are skipped.

    Parameters
    ----------
    mask_file : str
        Path to the ``.mas`` file.

    Returns
    -------
    MaskLines
        Line centers and weights in microns (same spectral frame as the numbers in the
        file after the nm→µm scale change).

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
    ncols = int(data.shape[1])
    if ncols == 2:
        w_um = data[:, 0].astype(np.float64) / 1e3
        weight = data[:, 1].astype(np.float64)
        return MaskLines(line_lambda_um=w_um, weight=weight)
    if ncols < 3:
        raise ValueError(
            f"Need 2 columns (wavelength_nm, weight) or ≥3 (start_nm, end_nm, weight): {mask_file}"
        )
    start = data[:, 0].astype(np.float64) / 1e3
    end = data[:, 1].astype(np.float64) / 1e3
    weight = data[:, 2].astype(np.float64)
    lo = np.minimum(start, end)
    hi = np.maximum(start, end)
    line_um = 0.5 * (lo + hi)
    return MaskLines(line_lambda_um=line_um, weight=weight)


# Legacy default for ``StellarRVPrimitive(mask_width_um=...)``; two-column masks no
# longer use a synthetic segment width (see :func:`load_mask_kpf_two_column`).
DEFAULT_KPF_MASK_SEGMENT_WIDTH_UM: float = 1.0e-6


def load_mask_kpf_two_column(mask_file: str, *, mask_width_um: float | None) -> MaskLines:
    """
    Load a KPF-style two-column mask file.

    The file is interpreted as:
    - wavelength (column 0), in Å, nm, or µm (see :func:`_infer_and_convert_to_micron`)
    - weight (column 1)

    Each row is a **delta** mask line at that wavelength; ``mask_width_um`` is accepted
    for API compatibility with older callers but is **ignored**.
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
    _ = mask_width_um  # unused; delta-line model
    return MaskLines(line_lambda_um=w, weight=weight)


def load_mask(mask_path: str, *, mask_width_um: float | None = None) -> MaskLines:
    """
    Load a line mask from disk, dispatching on file extension.

    Parameters
    ----------
    mask_path : str
        Path to ``.mas`` (SPIROU-style) or ``.txt`` / ``.dat`` / ``.csv`` (KPF-style).
    mask_width_um : float or None, optional
        Ignored (retained for backward-compatible call signatures). Two-column files are
        loaded as delta lines at the listed wavelengths.

    Returns
    -------
    MaskLines
        Line centers and weights in microns.

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
    Restrict mask lines to those whose Doppler-shifted wavelength can fall inside the
    chunk span for some trial velocity in ``[v_min_m_s, v_max_m_s]``.
    """

    zb1 = 1.0 + zb
    sh_a = (1.0 + v_min_m_s / C_M_S) / zb1
    sh_b = (1.0 + v_max_m_s / C_M_S) / zb1
    sh_lo = min(sh_a, sh_b)
    sh_hi = max(sh_a, sh_b)
    lam = mask_lines.line_lambda_um
    keep = (lam * sh_hi >= w_min_um) & (lam * sh_lo <= w_max_um)
    if not np.any(keep):
        return MaskLines(
            line_lambda_um=np.array([], dtype=np.float64),
            weight=np.array([], dtype=np.float64),
        )
    return MaskLines(
        line_lambda_um=mask_lines.line_lambda_um[keep],
        weight=mask_lines.weight[keep],
    )


@njit(cache=True)
def _ccf_rv_numba_err_floor(vs: float) -> float:
    """Match :func:`_ccf_rv_err_numeric_floor` (m/s); sub-grid σ_RV allowed."""
    ax = abs(vs)
    if ax < 1.0e-30:
        ax = 1.0
    return max(1.0e-9 * ax, 1.0e-12)


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
def _smooth_uniform5_numba(y: np.ndarray, n: int) -> np.ndarray:
    """5-point moving average; non-finite samples excluded from each local mean."""
    ys = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        c = 0.0
        for k in range(i - 2, i + 3):
            if 0 <= k < n:
                yk = float(y[k])
                if np.isfinite(yk):
                    s += yk
                    c += 1.0
        if c > 0.0:
            ys[i] = s / c
        else:
            ys[i] = np.nan
    return ys


@njit(cache=True)
def _mask_kpf_peak_index_integral_and_prominence(
    ys: np.ndarray,
    n: int,
    prom_half_width: int,
    integral_half_width: int,
) -> int:
    """
    Choose among strict local maxima of the smoothed CCF using **integrated excess**
    (sum of positive deviation from the global median in a wide window) times
    ``sqrt(prominence + 1)``.

    Pure prominence favors **narrow** spikes (large drop to a nearby valley). The physical
    stellar-RV peak from thousands of mask lines is **broad**; its prominence can be modest
    while the integrated contrast in ±~20–40 km/s is large. Combining integral × √prominence
    prefers that coherent peak over spurious narrow maxima (e.g. grid artifacts near
    unrelated velocities).
    """
    W = prom_half_width
    if W < 3:
        W = 3
    Wi = integral_half_width
    if Wi < 10:
        Wi = 10

    y_med = _nanmedian_numba_1d(ys)
    if not np.isfinite(y_med):
        y_med = 0.0

    best_i = -1
    best_score = -np.inf
    best_y = -np.inf
    for i in range(1, n - 1):
        yi = ys[i]
        ym = ys[i - 1]
        yp = ys[i + 1]
        if not (np.isfinite(yi) and np.isfinite(ym) and np.isfinite(yp)):
            continue
        if not (yi > ym and yi > yp):
            continue
        lo = i - W
        if lo < 0:
            lo = 0
        hi = i + W + 1
        if hi > n:
            hi = n
        left_min = np.inf
        for j in range(lo, i):
            yj = ys[j]
            if np.isfinite(yj) and yj < left_min:
                left_min = yj
        right_min = np.inf
        for j in range(i + 1, hi):
            yj = ys[j]
            if np.isfinite(yj) and yj < right_min:
                right_min = yj
        if not np.isfinite(left_min) or not np.isfinite(right_min):
            continue
        base = left_min
        if right_min > base:
            base = right_min
        prom = yi - base
        if prom <= 0.0:
            continue

        lo2 = i - Wi
        if lo2 < 0:
            lo2 = 0
        hi2 = i + Wi + 1
        if hi2 > n:
            hi2 = n
        integral = 0.0
        for j in range(lo2, hi2):
            yj = ys[j]
            if np.isfinite(yj) and yj > y_med:
                integral += yj - y_med

        score = integral * np.sqrt(prom + 1.0)
        if score > best_score or (score == best_score and yi > best_y):
            best_score = score
            best_i = i
            best_y = yi

    if best_i >= 0:
        return best_i
    # Fallback: global maximum of smoothed profile
    ig = -1
    yg = -np.inf
    for i in range(n):
        yi = ys[i]
        if np.isfinite(yi) and yi > yg:
            yg = yi
            ig = i
    return ig


@njit(cache=True)
def _mask_kpf_parabolic_vertex_lstsq_5pt_numba(
    v_grid_m_s: np.ndarray,
    y_signed: np.ndarray,
    i_peak: int,
) -> tuple[float, float, bool]:
    """
    Fit :math:`y \\approx a v^2 + b v + c` to five consecutive samples straddling ``i_peak``
    and return the vertex (maximum), ``a``, and success. Rejects non-concave fits (``a >= 0``)
    or vertices outside the spanned velocity interval.
    """
    n = int(v_grid_m_s.size)
    if n < 5 or i_peak < 2 or i_peak > n - 3:
        return np.nan, np.nan, False
    A = np.empty((5, 3), dtype=np.float64)
    rhs = np.empty(5, dtype=np.float64)
    for k in range(5):
        ii = i_peak - 2 + k
        vi = float(v_grid_m_s[ii])
        yi = float(y_signed[ii])
        if not (np.isfinite(vi) and np.isfinite(yi)):
            return np.nan, np.nan, False
        A[k, 0] = vi * vi
        A[k, 1] = vi
        A[k, 2] = 1.0
        rhs[k] = yi
    coeff = np.linalg.lstsq(A, rhs, rcond=-1)[0]
    a = float(coeff[0])
    b = float(coeff[1])
    if not np.isfinite(a) or not np.isfinite(b) or abs(a) < 1e-40:
        return np.nan, np.nan, False
    if a >= 0.0:
        return np.nan, np.nan, False
    rv = -b / (2.0 * a)
    if not np.isfinite(rv):
        return np.nan, np.nan, False
    v_lo = float(v_grid_m_s[i_peak - 2])
    v_hi = float(v_grid_m_s[i_peak + 2])
    vmin = v_lo if v_lo < v_hi else v_hi
    vmax = v_hi if v_hi > v_lo else v_lo
    if rv < vmin - 1e-9 or rv > vmax + 1e-9:
        return np.nan, np.nan, False
    return rv, a, True


def _merge_mask_ccf_segment_ranges(
    segments: list[tuple[int, int]], min_len: int
) -> list[tuple[int, int]]:
    """Merge adjacent half-open [lo, hi) segments until each has length >= min_len (last may merge back)."""
    if not segments:
        return []
    cur_lo, cur_hi = segments[0]
    out: list[tuple[int, int]] = []
    for lo, hi in segments[1:]:
        if cur_hi - cur_lo < min_len:
            cur_hi = hi
        else:
            out.append((cur_lo, cur_hi))
            cur_lo, cur_hi = lo, hi
    out.append((cur_lo, cur_hi))
    if len(out) >= 2 and out[-1][1] - out[-1][0] < min_len:
        lo0, _ = out[-2]
        _, hi1 = out[-1]
        out[-2] = (lo0, hi1)
        out.pop()
    return out


def mask_ccf_detrend_break_velocities_m_s(
    chunk_wl_lo_um: float,
    chunk_wl_hi_um: float,
    mask_line_lambda_um: np.ndarray,
    zb: float,
) -> np.ndarray:
    """
    Velocities where each mask line crosses the chunk wavelength bounds.

    Uses the same Doppler scaling as :func:`_calc_mask_ccf_chunks_numba`:
    ``λ_obs = λ_rest * (1+v/c) / (1+zb)``. For each rest wavelength ``λ_rest``, for
    each chunk edge ``λ_edge`` in ``{chunk_lo, chunk_hi}``, solve ``λ_rest·sh = λ_edge``
    for ``v``.
    """
    zb1 = 1.0 + float(zb)
    lo = float(min(chunk_wl_lo_um, chunk_wl_hi_um))
    hi = float(max(chunk_wl_lo_um, chunk_wl_hi_um))
    lam_rest = np.asarray(mask_line_lambda_um, dtype=np.float64).ravel()
    out: list[float] = []
    for i in range(lam_rest.size):
        lam0 = float(lam_rest[i])
        if not np.isfinite(lam0) or lam0 <= 0.0:
            continue
        for lam_edge in (lo, hi):
            sh = lam_edge / lam0
            if not np.isfinite(sh) or sh <= 0.0:
                continue
            v = C_M_S * (zb1 * sh - 1.0)
            if np.isfinite(v):
                out.append(float(v))
    return np.asarray(out, dtype=np.float64)


def _index_segments_from_velocity_breaks(
    v_grid_m_s: np.ndarray,
    break_vel_m_s: np.ndarray,
) -> list[tuple[int, int]]:
    """
    Partition ``v_grid_m_s`` into contiguous index ranges using velocity breakpoints
    (half-open in velocity between consecutive sorted breaks). Supports monotonic
    **increasing** or **decreasing** ``v_grid_m_s``.
    """
    vg = np.asarray(v_grid_m_s, dtype=np.float64).ravel()
    n = int(vg.size)
    if n <= 0:
        return []
    flip = n > 1 and float(vg[0]) > float(vg[-1])
    if flip:
        vg = vg[::-1].copy()

    v_lo = float(vg[0])
    v_hi = float(vg[-1])
    b = np.asarray(break_vel_m_s, dtype=np.float64).ravel()
    b = b[np.isfinite(b)]
    b = np.unique(np.concatenate([b, [v_lo, v_hi]]))
    b.sort()
    b = b[(b >= v_lo - 1e-6) & (b <= v_hi + 1e-6)]
    if b.size < 2:
        segs: list[tuple[int, int]] = [(0, n)]
    else:
        segs = []
        for k in range(b.size - 1):
            va, vb = float(b[k]), float(b[k + 1])
            if vb <= va + 1e-18:
                continue
            i0 = int(np.searchsorted(vg, va, side="left"))
            i1 = int(np.searchsorted(vg, vb, side="left"))
            if i1 > i0:
                segs.append((i0, i1))
        if not segs:
            segs = [(0, n)]

    if flip:
        segs = [(n - hi, n - lo) for lo, hi in segs]
    return segs


def _apply_piecewise_baseline_detrend(
    ccf: np.ndarray,
    segments: list[tuple[int, int]],
    base_percentile: float,
) -> np.ndarray:
    """Subtract per-segment lower-percentile baseline, then global median; same length as ``ccf``."""
    n = int(ccf.size)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    bp = float(base_percentile)
    if bp < 0.0:
        bp = 0.0
    if bp > 100.0:
        bp = 100.0
    for lo, hi in segments:
        seg = ccf[lo:hi]
        ok = np.isfinite(seg)
        if not np.any(ok):
            continue
        sub = seg[ok]
        base = float(np.percentile(sub, bp))
        out[lo:hi] = np.where(np.isfinite(ccf[lo:hi]), ccf[lo:hi] - base, np.nan)
    m = float(np.nanmedian(out))
    if np.isfinite(m):
        out -= m
    return out


def _detrend_mask_ccf_piecewise(
    ccf_raw: np.ndarray,
    v_grid_m_s: np.ndarray,
    *,
    chunk_wl_lo_um: float,
    chunk_wl_hi_um: float,
    mask_line_lambda_um: np.ndarray,
    zb: float,
    base_percentile: float = 35.0,
    min_segment_pixels: int = 5,
    breaks: str = "mask",
    jump_mad_sigma: float = 8.0,
) -> np.ndarray:
    """
    Prepare mask-line CCF(v) for peak finding and display.

    **mask** (default): segment boundaries are **RV values where each mask line**
    crosses the chunk wavelength limits (same geometry as :func:`_calc_mask_ccf_chunks_numba`:
    first/last pixel edges as ``chunk_wl_*_um``). Within each velocity segment, a
    **baseline** is estimated with a **lower** percentile, subtracted, then a global
    median subtraction centers the curve **near zero**.

    **empirical**: legacy segmentation from large |ΔCCF| between adjacent velocity
    samples (MAD-based), for diagnostics only.

    Parameters
    ----------
    ccf_raw
        One CCF row (same length as ``v_grid_m_s``).
    v_grid_m_s
        Velocity samples (m s⁻¹), monotonic increasing.
    chunk_wl_lo_um, chunk_wl_hi_um
        Chunk wavelength bounds (µm), matching the CCF kernel (pixel edge span).
    mask_line_lambda_um
        Rest wavelengths for this chunk (restricted mask), same frame as the kernel.
    zb
        Barycentric factor passed to the mask shift (see Numba CCF).
    breaks
        ``"mask"`` or ``"empirical"``.
    jump_mad_sigma
        Used only when ``breaks == "empirical"``.
    """
    ccf = np.asarray(ccf_raw, dtype=np.float64).ravel()
    vg = np.asarray(v_grid_m_s, dtype=np.float64).ravel()
    n = int(ccf.size)
    if n == 0:
        return ccf.copy()
    if vg.size != n:
        raise ValueError("ccf_raw and v_grid_m_s must have the same length")
    if not np.any(np.isfinite(ccf)):
        out = np.empty(n, dtype=np.float64)
        out[:] = np.nan
        return out

    if breaks == "mask":
        bv = mask_ccf_detrend_break_velocities_m_s(
            chunk_wl_lo_um,
            chunk_wl_hi_um,
            mask_line_lambda_um,
            zb,
        )
        segments = _index_segments_from_velocity_breaks(vg, bv)
        segments = _merge_mask_ccf_segment_ranges(segments, int(min_segment_pixels))
    elif breaks == "empirical":
        d = np.abs(np.diff(ccf))
        d_good = d[np.isfinite(d)]
        if d_good.size == 0:
            m = float(np.nanmedian(ccf))
            out = np.where(np.isfinite(ccf), ccf - m, np.nan)
            return np.asarray(out, dtype=np.float64)
        med_d = float(np.median(d_good))
        mad_d = float(np.median(np.abs(d_good - med_d))) + 1e-30
        thresh = med_d + float(jump_mad_sigma) * mad_d
        abs_c = np.abs(np.nan_to_num(ccf, nan=0.0, posinf=0.0, neginf=0.0))
        finite_mask = np.isfinite(ccf)
        if np.any(finite_mask):
            scale = float(np.percentile(abs_c[finite_mask], 95))
        else:
            scale = 1.0
        scale = max(scale, 1e-30)
        thresh = max(thresh, 0.015 * scale)

        br: list[int] = [0]
        for i in range(n - 1):
            a, b = ccf[i], ccf[i + 1]
            if not (np.isfinite(a) and np.isfinite(b)):
                br.append(i + 1)
                continue
            if abs(b - a) > thresh:
                br.append(i + 1)
        if br[-1] != n:
            br.append(n)
        segments = [(br[i], br[i + 1]) for i in range(len(br) - 1)]
        segments = _merge_mask_ccf_segment_ranges(segments, int(min_segment_pixels))
    else:
        raise ValueError('breaks must be "mask" or "empirical"')

    return _apply_piecewise_baseline_detrend(ccf, segments, base_percentile)


@njit(cache=True)
def _rv_from_ccf_mask_kpf_numba_single(
    v_grid_m_s: np.ndarray,
    ccf_raw: np.ndarray,
    n_good: int,
) -> tuple[float, float, int]:
    """
    RV from mask-line (KPF / SPIROU-style) CCF.

    Convention is **fixed by algorithm**, not inferred from each CCF shape: the mask weights
    target **absorption** lines, so the weight-normalized statistic (``Σ w f·SN / Σ w``) is
    minimized near the
    correct Doppler shift. We therefore always work in ``-CCF`` (search maxima / parabolic
    vertex on ``-CCF``), and ``flipped`` is always 1 so returned/preview CCF rows are negated
    consistently across echelle orders.

    ``ccf_raw`` is expected to be **piecewise detrended** (see :func:`_detrend_mask_ccf_piecewise`)
    by the caller so the profile is near zero before peak finding.

    Sub-pixel RV uses a **five-sample quadratic least-squares** vertex when the grid has at
    least five contiguous valid samples around the smoothed peak; otherwise a three-point
    analytic parabola (historic behaviour).
    """
    n = v_grid_m_s.size
    if ccf_raw.size != n:
        return np.nan, np.nan, 0

    baseline_raw = _nanmedian_numba_1d(ccf_raw)
    if not np.isfinite(baseline_raw):
        return np.nan, np.nan, 0

    sign = -1.0
    flipped = 1

    y_work = np.empty(n, dtype=np.float64)
    for i in range(n):
        yi = ccf_raw[i]
        if np.isfinite(yi):
            y_work[i] = sign * yi
        else:
            y_work[i] = np.nan

    # Low-pass filter, then choose among local maxima by integrated contrast × √prominence.
    ys = _smooth_uniform5_numba(y_work, n)
    Wprom = n // 10
    if Wprom < 80:
        Wprom = 80
    if Wprom > 2000:
        Wprom = 2000
    # Wide window for integrated line power (~20–40 km/s at 50 m/s sampling typical).
    Wint = n // 12
    if Wint < 100:
        Wint = 100
    if Wint > 800:
        Wint = 800
    i_peak = _mask_kpf_peak_index_integral_and_prominence(ys, n, Wprom, Wint)
    if i_peak < 0 or i_peak >= n:
        return np.nan, np.nan, flipped

    v_peak = float(v_grid_m_s[i_peak])
    v_step_edge = float(abs(v_grid_m_s[1] - v_grid_m_s[0])) if n > 1 else 1.0
    if i_peak <= 0 or i_peak >= (n - 1):
        fl = _ccf_rv_numba_err_floor(v_step_edge)
        rv_err_edge = max(float(v_step_edge * 2.0), fl)
        return v_peak, rv_err_edge, flipped

    v0 = float(v_grid_m_s[i_peak - 1])
    v1 = float(v_grid_m_s[i_peak])
    v2 = float(v_grid_m_s[i_peak + 1])
    h_m = v1 - v0
    h_p = v2 - v1
    v_step_eff = 0.5 * (h_m + h_p)
    if h_m <= 0.0 or h_p <= 0.0:
        fl = _ccf_rv_numba_err_floor(v_step_eff)
        return v_peak, max(float(abs(v_step_eff) * 2.0), fl), flipped

    y_m1 = float(sign * ccf_raw[i_peak - 1])
    y_0 = float(sign * ccf_raw[i_peak])
    y_p1 = float(sign * ccf_raw[i_peak + 1])
    baseline = float(sign * baseline_raw)
    amp = y_0 - baseline
    if not np.isfinite(amp) or amp <= 0.0:
        fl = _ccf_rv_numba_err_floor(v_step_eff)
        return v_peak, max(float(abs(v_step_eff) * 2.0), fl), flipped

    fl = _ccf_rv_numba_err_floor(v_step_eff)
    rv_ls, a_ls, ok_ls = _mask_kpf_parabolic_vertex_lstsq_5pt_numba(v_grid_m_s, y_work, i_peak)
    if ok_ls:
        rv = float(rv_ls)
        ypp = 2.0 * float(a_ls)
        if not np.isfinite(ypp) or ypp >= 0.0:
            sigma_v = float(abs(v_step_eff) * 2.0)
        else:
            sigma_v2 = max(float(-amp / ypp), fl * fl)
            sigma_v = float(np.sqrt(sigma_v2))
        n_scale = max(float(n_good), 1.0)
        rv_err = float(max(sigma_v / np.sqrt(n_scale), fl))
        return rv, rv_err, flipped

    # Parabola y = a*v^2 + b*v + c through (v0,y_m1), (v1,y_0), (v2,y_p1); vertex at -b/(2a).
    A = v1 * v1 - v0 * v0
    B = v1 - v0
    C = y_0 - y_m1
    D = v2 * v2 - v1 * v1
    E = v2 - v1
    F = y_p1 - y_0
    det = A * E - B * D
    if not np.isfinite(det) or abs(det) < 1e-30:
        fl = _ccf_rv_numba_err_floor(v_step_eff)
        return v_peak, max(float(abs(v_step_eff) * 2.0), fl), flipped
    a = (C * E - B * F) / det
    b = (A * F - C * D) / det
    if not np.isfinite(a) or abs(a) < 1e-30:
        fl = _ccf_rv_numba_err_floor(v_step_eff)
        return v_peak, max(float(abs(v_step_eff) * 2.0), fl), flipped
    rv = -b / (2.0 * a)
    if rv < min(v0, v2) or rv > max(v0, v2):
        rv = v_peak

    ypp = 2.0 * a
    if not np.isfinite(ypp) or ypp >= 0.0:
        sigma_v = float(abs(v_step_eff) * 2.0)
    else:
        sigma_v2 = max(float(-amp / ypp), fl * fl)
        sigma_v = float(np.sqrt(sigma_v2))
    n_scale = max(float(n_good), 1.0)
    rv_err = float(max(sigma_v / np.sqrt(n_scale), fl))
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
    mask_ccf_detrend: bool = False,
    mask_ccf_detrend_breaks: str = "mask",
    mask_ccf_detrend_jump_mad_sigma: float = 8.0,
    mask_ccf_detrend_base_percentile: float = 35.0,
    mask_ccf_detrend_min_segment_pixels: int = 5,
    mask_order_edge_taper_pixels: float = 100.0,
    mask_order_edge_taper_exp_shape: float = 5.0,
    order_wave_um_for_edge_taper: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
]:
    """
    Compute per-chunk RVs and optional CCF stacks for one echelle order.

    Splits the order into ``n_chunk`` wavelength bins of width set by
    ``window_size_um`` (derived from full order span), then computes chunk CCF rows
    in one Numba batch via :func:`_calc_mask_ccf_chunks_numba` (parallel over chunks;
    delta-line interpolation at each Doppler-shifted rest wavelength), followed by a Numba chunk-parallel
    parabolic RV extraction. RV extraction assumes a
    **fixed absorption-line convention** for all chunks and orders (fit on ``-CCF``, not
    data-driven emission/absorption guessing). Peak selection uses a 5-point smoothed CCF
    and a **combined score** (wide-window integrated excess above the global median times
    ``√prominence``) over strict local maxima—favoring the broad, physically coherent peak
    at the true RV over narrow spurious maxima. Returns flattened lists of chunk centers
    and RVs.

    Before RV extraction, each CCF row is optionally **piecewise detrended** (see
    :func:`_detrend_mask_ccf_piecewise`) when ``mask_ccf_detrend`` is True. Detrend
    **segments along RV** are bounded by velocities where **mask lines cross the chunk
    wavelength limits** (first/last pixel edges, matching the mask CCF kernel), not by
    empirical CCF jumps. Within each segment a lower-percentile baseline is removed, then
    a global median shift centers the curve **near zero** for peak finding on ``ccf_2d``.
    By default (``mask_ccf_detrend=False``), no detrend is applied: ``ccf_2d`` is the
    weight-normalized CCF used for RV extraction and stacking; ``ccf_preview_2d`` uses the
    same rows before the sign flip used for ``ccf_2d`` (when detrend is off, both match the
    physical weighted CCF up to that convention).

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
    mask_ccf_detrend : bool, optional
        If True, apply :func:`_detrend_mask_ccf_piecewise` to each CCF row before peak
        finding and before stacking ``ccf_2d``. Default ``False`` (undetrended CCF for
        analysis and preview).
    mask_ccf_detrend_breaks : str, optional
        ``"mask"`` (default): segment boundaries from mask-line / chunk-edge crossing
        velocities. ``"empirical"``: legacy |ΔCCF| jump detection (diagnostics).
    mask_ccf_detrend_jump_mad_sigma, mask_ccf_detrend_base_percentile, mask_ccf_detrend_min_segment_pixels
        Passed to :func:`_detrend_mask_ccf_piecewise` when ``mask_ccf_detrend`` is True
        (``jump_mad_sigma`` only for ``mask_ccf_detrend_breaks="empirical"``).
    mask_order_edge_taper_pixels : float, optional
        If ``> 0``, multiply mask line weights by an exponential taper when the line’s
        nearest pixel lies within this many pixels of either echelle edge (see
        :func:`_mask_line_order_edge_taper_multipliers`). Default ``100`` — unity weight
        from 100 pixels inward from each edge; weight tends to ``0`` at the edges.
        Use ``0`` to disable.
    mask_order_edge_taper_exp_shape : float, optional
        Exponential shape parameter inside the edge band (default ``5``).
    order_wave_um_for_edge_taper : np.ndarray or None, optional
        Full-order wavelength vector for edge-distance mapping. If ``None``, uses
        ``wave_um``. When ``wave_um`` is only a slice of an order, pass the full row so
        tapering references the true order extent.

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
        If ``return_ccf``, shape ``(n_chunks, n_v)`` of CCFs: optionally piecewise-detrended
        when ``mask_ccf_detrend`` is True, then **sign-flipped** per chunk to match the
        parabolic RV extraction convention; used for RVCCF supplements and chunk-level
        diagnostics. Else ``None``.
    wl_lo, wl_hi : np.ndarray
        Per-chunk wavelength bounds (µm), same length as ``rvs``.
    n_good : np.ndarray
        Per-chunk count of good pixels used in the mask CCF.
    ccf_line_std_2d : np.ndarray or None
        If ``return_ccf``, shape ``(n_chunks, n_v)``: weighted standard deviation of
        :math:`x_j = F_j\\,\\mathrm{SN}_j` about :math:`\\mathrm{CCF}(v)` at each sample
        (see :func:`_calc_mask_ccf_chunks_numba`). Not flipped with the CCF row; use as a
        symmetric band around ``ccf_preview_2d``. If ``return_ccf`` is False, ``None``.
    ccf_preview_2d : np.ndarray or None
        If ``return_ccf``, same shape as ``ccf_2d``: **weight-normalized** mask CCF
        (matching :func:`_calc_mask_ccf_chunks_numba`) **before** optional piecewise detrend,
        then the **same** per-chunk sign flip as ``ccf_2d``. When ``mask_ccf_detrend`` is
        False, matches the undetrended CCF used for analysis. Else ``None``.
    """

    w = np.asarray(wave_um, dtype=np.float64).ravel()
    s = np.asarray(spec, dtype=np.float64).ravel()
    e = np.asarray(spec_err, dtype=np.float64).ravel()
    d = np.asarray(dq_bad_mask, dtype=bool).ravel()

    def _empty(v_grid_local: np.ndarray):
        z = np.array([], dtype=np.float64)
        zi = np.array([], dtype=np.int32)
        if not return_ccf:
            return (z, z, z, v_grid_local, None, z, z, zi, None, None)
        ccf_out = np.empty((0, len(v_grid_local)), dtype=np.float64)
        std_out = np.empty((0, len(v_grid_local)), dtype=np.float64)
        return (z, z, z, v_grid_local, ccf_out, z, z, zi, std_out, ccf_out)

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
    all_ccf_preview_rows: list[np.ndarray] = []
    all_ccf_std_rows: list[np.ndarray] = []
    chunk_payloads: list[
        tuple[
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

        # Restrict mask lines so we don't waste time on lines that cannot land in this
        # chunk for any trial velocity in the search range.
        mask_c = restrict_mask_to_chunk(
            mask_lines,
            float(np.nanmin(w_c)),
            float(np.nanmax(w_c)),
            v_min_m_s=vr_lo,
            v_max_m_s=vr_hi,
            zb=zb,
        )
        if mask_c.line_lambda_um.size == 0:
            continue

        if float(mask_order_edge_taper_pixels) > 0.0:
            w_taper = (
                order_wave_um_for_edge_taper
                if order_wave_um_for_edge_taper is not None
                else w
            )
            tap = _mask_line_order_edge_taper_multipliers(
                np.asarray(w_taper, dtype=np.float64).ravel(),
                mask_c.line_lambda_um,
                edge_taper_pixels=float(mask_order_edge_taper_pixels),
                exp_shape=float(mask_order_edge_taper_exp_shape),
            )
            mask_c = MaskLines(
                line_lambda_um=mask_c.line_lambda_um,
                weight=mask_c.weight * tap,
            )

        left_edges, right_edges = pixel_edges_from_wavelength(w_c)
        pixel_width = right_edges - left_edges
        pixel_width = np.where(pixel_width > 0, pixel_width, np.nan)
        finite = np.isfinite(s_c) & np.isfinite(e_c) & (e_c > 0) & np.isfinite(pixel_width)
        if int(np.sum(finite)) < min_good_pixels:
            continue
        s_use = np.where(finite, s_c, 0.0).astype(np.float64)
        sn = np.where(finite, 1.0, 0.0).astype(np.float64)
        e_use = np.where(finite, e_c, 0.0).astype(np.float64)
        chunk_payloads.append(
            (
                left_edges.astype(np.float64),
                right_edges.astype(np.float64),
                s_use,
                sn,
                e_use,
                mask_c.line_lambda_um.astype(np.float64),
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
    s_use_2d = np.zeros((n_chunk_valid, n_pix_max), dtype=np.float64)
    sn_2d = np.zeros((n_chunk_valid, n_pix_max), dtype=np.float64)
    e_use_2d = np.zeros((n_chunk_valid, n_pix_max), dtype=np.float64)
    n_pix_arr = np.zeros(n_chunk_valid, dtype=np.int64)
    line_lambda_rest_2d = np.full((n_chunk_valid, n_line_max), np.nan, dtype=np.float64)
    line_w_2d = np.zeros((n_chunk_valid, n_line_max), dtype=np.float64)
    n_line_arr = np.zeros(n_chunk_valid, dtype=np.int64)
    centers = np.zeros(n_chunk_valid, dtype=np.float64)
    n_good_arr = np.zeros(n_chunk_valid, dtype=np.int64)
    wl_lo_arr = np.zeros(n_chunk_valid, dtype=np.float64)
    wl_hi_arr = np.zeros(n_chunk_valid, dtype=np.float64)

    for i, payload in enumerate(chunk_payloads):
        left_i, right_i, s_i, sn_i, e_i, ll_i, lw_i, lam_lo_i, lam_hi_i, wl_i, ng_i = payload
        npix = len(left_i)
        nln = len(ll_i)
        left_2d[i, :npix] = left_i
        right_2d[i, :npix] = right_i
        s_use_2d[i, :npix] = s_i
        sn_2d[i, :npix] = sn_i
        e_use_2d[i, :npix] = e_i
        n_pix_arr[i] = npix
        line_lambda_rest_2d[i, :nln] = ll_i
        line_w_2d[i, :nln] = lw_i
        n_line_arr[i] = nln
        centers[i] = wl_i
        n_good_arr[i] = ng_i
        wl_lo_arr[i] = lam_lo_i
        wl_hi_arr[i] = lam_hi_i

    with numba_thread_scope(max_cores):
        ccf_rows, ccf_std_rows, _ = _calc_mask_ccf_chunks_numba(
            left_2d,
            right_2d,
            s_use_2d,
            e_use_2d,
            sn_2d,
            n_pix_arr,
            v_grid,
            float(zb),
            line_lambda_rest_2d,
            line_w_2d,
            n_line_arr,
            bool(return_ccf),
        )
        # σ_CCF: weighted SD of x_j = F_j·SN_j about CCF(v) = Σ w_j x_j / Σ w_j (from Numba).
        if bool(return_ccf) and ccf_std_rows.size > 0:
            sf = np.asarray(ccf_std_rows, dtype=np.float64)
            ccf_std_rows = np.where(np.isfinite(sf), np.maximum(0.0, sf), np.nan)
        ccf_preview_unflipped: np.ndarray | None = (
            np.copy(ccf_rows) if bool(return_ccf) else None
        )
        if mask_ccf_detrend:
            brk = str(mask_ccf_detrend_breaks).lower().strip()
            if brk not in ("mask", "empirical"):
                raise ValueError('mask_ccf_detrend_breaks must be "mask" or "empirical"')
            zb_f = float(zb)
            for ic in range(n_chunk_valid):
                npix = int(n_pix_arr[ic])
                chunk_lo = float(left_2d[ic, 0])
                chunk_hi = float(right_2d[ic, npix - 1])
                nln = int(n_line_arr[ic])
                ll = line_lambda_rest_2d[ic, :nln]
                ccf_rows[ic, :] = _detrend_mask_ccf_piecewise(
                    ccf_rows[ic, :],
                    v_grid,
                    chunk_wl_lo_um=chunk_lo,
                    chunk_wl_hi_um=chunk_hi,
                    mask_line_lambda_um=ll,
                    zb=zb_f,
                    base_percentile=float(mask_ccf_detrend_base_percentile),
                    min_segment_pixels=int(mask_ccf_detrend_min_segment_pixels),
                    breaks=brk,
                    jump_mad_sigma=float(mask_ccf_detrend_jump_mad_sigma),
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
            pu = ccf_preview_unflipped
            all_ccf_preview_rows.append(
                -pu[i] if int(flip_rows[i]) == 1 else pu[i].copy()
            )
            all_ccf_std_rows.append(ccf_std_rows[i].copy())
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
        ccf_preview_2d = (
            np.vstack(all_ccf_preview_rows).astype(np.float64)
            if all_ccf_preview_rows
            else np.empty((0, len(v_grid)))
        )
        ccf_std_2d = (
            np.vstack(all_ccf_std_rows).astype(np.float64) if all_ccf_std_rows else np.empty((0, len(v_grid)))
        )
    else:
        ccf_2d = None
        ccf_preview_2d = None
        ccf_std_2d = None
    return (
        rvs_arr,
        rv_errs_arr,
        centers_arr,
        v_grid,
        ccf_2d,
        wl_lo_out,
        wl_hi_out,
        n_good_out,
        ccf_std_2d,
        ccf_preview_2d,
    )


def _refine_ccf_rv_after_gaussian_fit_ccf_mask_kpf(
    v_grid_m_s: np.ndarray,
    ccf_mean: np.ndarray,
    vstep: float,
    *,
    ccf_err: np.ndarray | None,
    params: dict | None,
) -> tuple[float | None, float | None, dict | None]:
    """
    Coarse-grid parabolic / spline refinement (:func:`refine_ccf_rv_coarse_grid_parabolic`)
    after a successful Gaussian fit to the order-mean mask CCF — same path as Plotly
    ``ccf_mask_kpf`` refine overlays.
    """
    sig: float | None = None
    if params:
        s = params.get("sigma")
        if s is not None:
            sf = float(s)
            if math.isfinite(sf) and sf > 0:
                sig = sf
    rv_r, meta = refine_ccf_rv_coarse_grid_parabolic(
        np.asarray(v_grid_m_s, dtype=np.float64),
        np.asarray(ccf_mean, dtype=np.float64),
        float(vstep),
        ccf_err=ccf_err,
        gaussian_sigma_m_s=sig,
        mask_method="stellar",
        config=CCFFitKpfConfig(),
    )
    if not meta.get("ok") or not math.isfinite(float(rv_r)):
        return None, None, None
    err_m = float(meta.get("rv_err_m_s", float("nan")))
    err_out: float | None = err_m if math.isfinite(err_m) and err_m > 0 else None
    return float(rv_r), err_out, meta


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
    mask_order_edge_taper_pixels: float = 100.0,
    mask_order_edge_taper_exp_shape: float = 5.0,
    ccf_mask_kpf_auto_hybrid_scout: bool = True,
    mask_ccf_detrend: bool = False,
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
    each requested echelle order with :func:`rv_per_chunks_mask_kpf`. For every
    echelle order, all chunk rows in ``RVCCF`` are assigned the **same** RV and
    ``RV_ERR``, taken from :func:`fit_ccf_rv` on the **order-mean** mask CCF
    (nanmean over chunk CCF rows on the velocity grid), then refined with
    :func:`refine_ccf_rv_coarse_grid_parabolic` when that step succeeds (same
    sub-pixel refinement as the default ``ccf_mask_kpf`` Plotly overlays). Chunk
    wavelengths stay as chunk centers. Those
    Fiber-level RV and uncertainty use :func:`combine_rv_estimates_median_mad_weighted`
    on **one aggregate sample per echelle order** when the order-mean Gaussian succeeds
    (so duplicate chunk rows are not treated as independent measurements). That
    returns an inverse-variance mean and ``max(internal stderr, MAD/√N)`` on the
    aggregates. Per-chunk ``INLIER`` flags map the aggregate mask back to each row.
    If the Gaussian fit fails, Numba per-chunk RVs are kept and each chunk remains its
    own aggregate for combination.

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
    mask_order_edge_taper_pixels : float, optional
        Deweight mask lines near echelle order edges (default ``100``); ``0`` disables.
        See :func:`rv_per_chunks_mask_kpf`.
    mask_order_edge_taper_exp_shape : float, optional
        Exponential shape parameter for the edge taper (default ``5``).
    ccf_mask_kpf_auto_hybrid_scout : bool, optional
        When ``ccf_hybrid_center_m_s`` is unset and **exactly one** echelle order is processed,
        run a **full-order** coarse mask CCF (one chunk spanning the echelle row), take the
        sub-pixel peak on the order-mean CCF (same basis as the pipeline fit), build
        :func:`build_hybrid_merged_velocity_grid` at that center, and evaluate all wavelength
        sub-bins on the merged grid. Multi-order runs skip this so preview stacking keeps one
        velocity axis.
    mask_ccf_detrend : bool, optional
        If True, apply piecewise baseline removal to each mask CCF row before RV extraction
        (see :func:`rv_per_chunks_mask_kpf`). Default ``False``: RV table, Gaussian fit, RVCCF
        HDU rows, and Plotly preview use the **undetrended** weight-normalized CCF.

    Returns
    -------
    all_wavelengths : np.ndarray
        Chunk center wavelengths (µm).
    all_rvs : np.ndarray
        Per-chunk RVs (m s⁻¹); within one echelle order, values are identical when the
        order-mean CCF Gaussian fit succeeds.
    all_rv_errs : np.ndarray
        Per-chunk uncertainties (same within each order when the Gaussian fit succeeds).
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
    if mask_lines.line_lambda_um.size == 0:
        raise ValueError(f"Mask loaded no lines: {resolved_mask}")
    logger.info(
        "CCF mask_kpf: loaded mask path=%s n_line_segments=%d",
        resolved_mask,
        int(mask_lines.line_lambda_um.size),
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
    single_order_mask_kpf = len(order_tasks) == 1
    fiber_merged_v_grid: np.ndarray | None = None

    all_rvs: list[float] = []
    all_rv_errs: list[float] = []
    all_wavelengths: list[float] = []
    all_orders: list[int] = []

    # Per-chunk RVs and RVCCF table CCF rows use ``ccf_2d`` from :func:`rv_per_chunks_mask_kpf`
    # (default: undetrended). The order-mean Gaussian + parabolic refine stacks the same chunk
    # rows as chunk RV extraction.
    want_preview_effective = bool(want_preview)

    order_slots: list[dict] = []
    all_ccf_rows: list[np.ndarray] = []
    all_ccf_err_rows: list[np.ndarray] = []

    rvccf_rows_accum: list[np.ndarray] = []
    rvccf_n_good_accum: list[int] = []

    # One inverse-variance weight per statistically independent RV estimate (see loop body).
    fiber_agg_rvs: list[float] = []
    fiber_agg_errs: list[float] = []
    chunk_fiber_agg_index: list[int] = []

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
        order_chunk_rvs: list[float] = []
        order_chunk_errs: list[float] = []
        order_chunk_wl: list[float] = []
        order_chunk_ng: list[int] = []
        v_auto_grid: np.ndarray | None = None
        order_auto_hybrid_merged = False

        kw_edge = dict(
            mask_order_edge_taper_pixels=float(mask_order_edge_taper_pixels),
            mask_order_edge_taper_exp_shape=float(mask_order_edge_taper_exp_shape),
            order_wave_um_for_edge_taper=w,
        )
        kw_rv = {**kw_edge, "mask_ccf_detrend": bool(mask_ccf_detrend)}
        # Single-order auto-hybrid: full-order CCF (one chunk spanning the echelle row) for scout.
        span_um = float(w_max - w_min)
        window_um_scout = max(float(window_size_um), span_um * (1.0 + 1e-12))
        if (
            bool(ccf_mask_kpf_auto_hybrid_scout)
            and single_order_mask_kpf
            and not hybrid
        ):
            scout = rv_per_chunks_mask_kpf(
                w,
                s,
                e,
                d_bad,
                mask_lines,
                window_size_um=window_um_scout,
                v_min_m_s=v_min_m_s,
                v_max_m_s=v_max_m_s,
                v_step_m_s=v_step_m_s,
                zb=zb,
                max_cores=max_cores,
                return_ccf=True,
                **kw_rv,
            )
            (
                _rvs_sc,
                _errs_sc,
                _centers_sc,
                v_grid_sc,
                ccf_2d_sc,
                _wl_lo_sc,
                _wl_hi_sc,
                _ng_sc,
                _std_sc,
                _ccf_preview_sc,
            ) = scout
            if int(_rvs_sc.size) > 0:
                vg_sc = np.asarray(v_grid_sc, dtype=np.float64)
                m_dt = _nanmean_axis0_ignore_empty_columns(
                    np.asarray(ccf_2d_sc, dtype=np.float64)
                )
                v_c = _ccf_raw_grid_subpixel_peak_nearest_disc_max_m_s(vg_sc, m_dt)

                if math.isfinite(v_c):
                    v_auto_grid = build_hybrid_merged_velocity_grid(
                        float(v_min_m_s),
                        float(v_max_m_s),
                        float(v_step_m_s),
                        float(v_c),
                        float(ccf_hybrid_half_width_m_s),
                        float(ccf_hybrid_step_m_s),
                    )
                    order_auto_hybrid_merged = True
                    fiber_merged_v_grid = v_auto_grid
                    logger.info(
                        "CCF mask_kpf: auto-hybrid scout (full-order mean CCF) center=%.2f m/s "
                        "n_vel=%d fiber=%s order=%s window_um_scout=%.6f",
                        float(v_c),
                        int(v_auto_grid.size),
                        fiber,
                        order,
                        float(window_um_scout),
                    )
                else:
                    v_auto_grid = np.asarray(v_grid_sc, dtype=np.float64)

        for isub in range(n_sub):
            lam_lo = w_min + isub * (w_max - w_min) / n_sub
            lam_hi = w_min + (isub + 1) * (w_max - w_min) / n_sub
            if isub == n_sub - 1:
                in_sub = (w >= lam_lo) & (w <= lam_hi)
            else:
                in_sub = (w >= lam_lo) & (w < lam_hi)
            if int(np.sum(in_sub)) < 20:
                continue

            w_sub = w[in_sub]
            s_sub = s[in_sub]
            e_sub = e[in_sub]
            d_sub = d_bad[in_sub]

            if hybrid:
                (
                    rvs_o,
                    errs_o,
                    centers_o,
                    v_grid_o,
                    ccf_2d,
                    _wl_lo_o,
                    _wl_hi_o,
                    ng_o,
                    _ccf_std_2d,
                    ccf_preview_2d,
                ) = rv_per_chunks_mask_kpf(
                    w_sub,
                    s_sub,
                    e_sub,
                    d_sub,
                    mask_lines,
                    window_size_um=window_size_um,
                    v_min_m_s=float(v_min_m_s),
                    v_max_m_s=float(v_max_m_s),
                    v_step_m_s=float(v_step_m_s),
                    v_grid_m_s=v_grid,
                    zb=zb,
                    max_cores=max_cores,
                    return_ccf=True,
                    **kw_rv,
                )
            elif v_auto_grid is not None:
                (
                    rvs_o,
                    errs_o,
                    centers_o,
                    v_grid_o,
                    ccf_2d,
                    _wl_lo_o,
                    _wl_hi_o,
                    ng_o,
                    _ccf_std_2d,
                    ccf_preview_2d,
                ) = rv_per_chunks_mask_kpf(
                    w_sub,
                    s_sub,
                    e_sub,
                    d_sub,
                    mask_lines,
                    window_size_um=window_size_um,
                    v_min_m_s=float(v_min_m_s),
                    v_max_m_s=float(v_max_m_s),
                    v_step_m_s=float(v_step_m_s),
                    v_grid_m_s=v_auto_grid,
                    zb=zb,
                    max_cores=max_cores,
                    return_ccf=True,
                    **kw_rv,
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
                    _ccf_std_2d,
                    ccf_preview_2d,
                ) = rv_per_chunks_mask_kpf(
                    w_sub,
                    s_sub,
                    e_sub,
                    d_sub,
                    mask_lines,
                    window_size_um=window_size_um,
                    v_min_m_s=v_min_m_s,
                    v_max_m_s=v_max_m_s,
                    v_step_m_s=v_step_m_s,
                    zb=zb,
                    max_cores=max_cores,
                    return_ccf=True,
                    **kw_rv,
                )
            if rvs_o.size == 0:
                continue

            for ii in range(int(rvs_o.size)):
                order_chunk_rvs.append(float(rvs_o[ii]))
                ev = float(errs_o[ii])
                order_chunk_errs.append(ev if np.isfinite(ev) and ev > 0 else float("nan"))
                order_chunk_wl.append(float(centers_o[ii]))
                order_chunk_ng.append(int(ng_o[ii]))

            if ccf_2d is not None and ccf_2d.size > 0:
                for ii in range(int(rvs_o.size)):
                    rvccf_rows_accum.append(np.asarray(ccf_2d[ii], dtype=np.float64))
                    rvccf_n_good_accum.append(int(ng_o[ii]))

                v_now = np.asarray(v_grid_o, dtype=np.float64)
                # Stack chunk CCF rows for order-mean RV (same basis as chunk RV / RVCCF HDU).
                stack_for_order_mean = ccf_2d
                if order_v_grid is None:
                    order_v_grid = v_now
                    order_ccf_rows.append(stack_for_order_mean)
                elif order_v_grid.shape == v_now.shape and np.allclose(order_v_grid, v_now):
                    order_ccf_rows.append(stack_for_order_mean)

        n_chunks_order = len(order_chunk_rvs)
        if n_chunks_order == 0:
            continue

        order_used_gaussian_mean = False
        stacked_order: np.ndarray | None = None
        ccf_mean_order: np.ndarray | None = None
        ccf_err_mean_order: np.ndarray | None = None
        sigma_hp_order = float("nan")
        if order_ccf_rows and order_v_grid is not None:
            stacked_order = np.vstack(order_ccf_rows)
            ccf_mean_order = _nanmean_axis0_ignore_empty_columns(stacked_order)
            sigma_hp_order, _, _ = estimate_ccf_uncertainty_highpass_residual(ccf_mean_order)
            if np.isfinite(sigma_hp_order) and sigma_hp_order > 0.0:
                ccf_err_mean_order = np.full_like(
                    np.asarray(ccf_mean_order, dtype=np.float64),
                    float(sigma_hp_order),
                    dtype=np.float64,
                )
            else:
                ccf_err_mean_order = None

            v_fit = np.asarray(order_v_grid, dtype=np.float64)
            vstep_fit = (
                float(effective_ccf_v_step_for_fit(v_fit))
                if (hybrid or order_auto_hybrid_merged)
                else float(v_step_m_s)
            )
            wl_c = 0.5 * (w_min + w_max)
            vsp = dv_per_pixel_m_s(float(wl_c))
            n_good_fit = max(int(np.nansum(order_chunk_ng)), 1)
            rv_g, err_g, _par_g, diag_g = fit_ccf_rv(
                v_fit,
                ccf_mean_order,
                vstep_fit,
                n_good=n_good_fit,
                return_params=True,
                vel_span_m_s_per_pixel=vsp if np.isfinite(vsp) else None,
                ccf_err=ccf_err_mean_order,
            )
            if (
                bool(diag_g.get("fit_ok"))
                and np.isfinite(rv_g)
                and np.isfinite(err_g)
                and float(err_g) > 0.0
            ):
                rv_gf = float(rv_g)
                err_gf = float(err_g)
                rv_ref, err_ref, _rmeta = _refine_ccf_rv_after_gaussian_fit_ccf_mask_kpf(
                    v_fit,
                    ccf_mean_order,
                    vstep_fit,
                    ccf_err=ccf_err_mean_order,
                    params=_par_g if isinstance(_par_g, dict) else None,
                )
                if rv_ref is not None and math.isfinite(float(rv_ref)):
                    rv_gf = float(rv_ref)
                    if err_ref is not None and math.isfinite(float(err_ref)) and float(err_ref) > 0:
                        err_gf = float(err_ref)
                order_chunk_rvs = [rv_gf] * n_chunks_order
                order_chunk_errs = [err_gf] * n_chunks_order
                order_used_gaussian_mean = True

        if order_used_gaussian_mean:
            fiber_agg_rvs.append(float(order_chunk_rvs[0]))
            fiber_agg_errs.append(float(order_chunk_errs[0]))
            ai = len(fiber_agg_rvs) - 1
            chunk_fiber_agg_index.extend([ai] * n_chunks_order)
        else:
            for rv_c, er_c in zip(order_chunk_rvs, order_chunk_errs):
                fiber_agg_rvs.append(float(rv_c))
                fiber_agg_errs.append(float(er_c))
                chunk_fiber_agg_index.append(len(fiber_agg_rvs) - 1)

        all_rvs.extend(order_chunk_rvs)
        all_rv_errs.extend(order_chunk_errs)
        all_wavelengths.extend(order_chunk_wl)
        all_orders.extend([int(order)] * n_chunks_order)

        if want_preview_effective and stacked_order is not None and order_v_grid is not None and ccf_mean_order is not None:
            slot_payload: dict = {
                "kind": "order",
                "order": int(order),
                "v": order_v_grid,
                "ccf": np.asarray(ccf_mean_order, dtype=np.float64),
                "wl_lo": float(w_min),
                "wl_hi": float(w_max),
            }
            if ccf_err_mean_order is not None and np.any(np.isfinite(ccf_err_mean_order)):
                slot_payload["ccf_err"] = np.asarray(ccf_err_mean_order, dtype=np.float64)
            if np.isfinite(sigma_hp_order) and sigma_hp_order > 0.0:
                err_stack = np.full_like(stacked_order, float(sigma_hp_order), dtype=np.float64)
            else:
                err_stack = np.full_like(stacked_order, np.nan, dtype=np.float64)
            all_ccf_err_rows.append(err_stack)
            order_slots.append(slot_payload)
            all_ccf_rows.append(stacked_order)

        if debug_ccf and n_chunks_order > 0:
            logger.info(
                "CCF-mask_kpf fiber=%s order=%s: n_suborders=%d n_chunks=%d rv_mean~%.1f m/s",
                fiber,
                order,
                n_sub,
                n_chunks_order,
                float(np.nanmedian(order_chunk_rvs)),
            )

    all_rvs_arr = np.asarray(all_rvs, dtype=np.float64)
    all_rv_errs_arr = np.asarray(all_rv_errs, dtype=np.float64)
    all_wavelengths_arr = np.asarray(all_wavelengths, dtype=np.float64)
    all_orders_arr = np.asarray(all_orders, dtype=np.int64)
    if len(all_orders_arr) != len(all_wavelengths_arr):
        all_orders_arr = np.full(len(all_wavelengths_arr), -1, dtype=np.int64)

    agg_rvs_arr = np.asarray(fiber_agg_rvs, dtype=np.float64)
    agg_errs_arr = np.asarray(fiber_agg_errs, dtype=np.float64)
    chunk_map = np.asarray(chunk_fiber_agg_index, dtype=np.intp)
    valid_agg = np.isfinite(agg_rvs_arr) & np.isfinite(agg_errs_arr) & (agg_errs_arr > 0)
    if not np.any(valid_agg):
        inlier = np.zeros(len(all_rvs_arr), dtype=bool)
        rv_mean = np.nan
        rv_std = np.nan
    else:
        rv_mean, rv_std, inlier_agg, _rv_std_int, _scatter_se = combine_rv_estimates_median_mad_weighted(
            agg_rvs_arr, agg_errs_arr, outlier_sigma
        )
        if chunk_map.size == len(all_rvs_arr) and inlier_agg.size == agg_rvs_arr.size:
            inlier = inlier_agg[chunk_map]
        else:
            inlier, _, _ = combine_rvs_median_mad(all_rvs_arr, all_rv_errs_arr, outlier_sigma)
            rv_mean, rv_std = weighted_mean_rv(all_rvs_arr, all_rv_errs_arr, inlier)

    if hybrid:
        pv_min = float(v_grid[0])
        pv_max = float(v_grid[-1])
        pv_step = float(effective_ccf_v_step_for_fit(v_grid))
        preview_v_grid = v_grid
    elif fiber_merged_v_grid is not None:
        pv_min = float(fiber_merged_v_grid[0])
        pv_max = float(fiber_merged_v_grid[-1])
        pv_step = float(effective_ccf_v_step_for_fit(fiber_merged_v_grid))
        preview_v_grid = fiber_merged_v_grid
    else:
        pv_min = float(v_min_m_s)
        pv_max = float(v_max_m_s)
        pv_step = float(v_step_m_s)
        preview_v_grid = v_grid

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
            preview_v_grid,
            pv_min,
            pv_max,
            pv_step,
            all_ccf_err_rows if len(all_ccf_err_rows) == len(order_slots) else None,
        )

    logger.info(
        "CCF mask_kpf: fiber=%s finished n_chunks=%d rv_mean=%.2f m/s "
        "rv_stderr_agg=%.2f m/s (IV mean err max MAD/sqrt(N); n_inlier_chunks=%d/%d "
        "n_agg=%d) preview=%s",
        fiber,
        len(all_wavelengths_arr),
        float(rv_mean),
        float(rv_std),
        int(np.sum(inlier)) if len(inlier) else 0,
        len(inlier),
        int(agg_rvs_arr.size),
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


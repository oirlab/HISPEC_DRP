"""Shared CCF constants and small helpers used across mask / fit / grid modules."""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "V_MAX_DEFAULT_M_S",
    "V_MIN_DEFAULT_M_S",
    "V_STEP_DEFAULT_M_S",
    "_ccf_background_sigma_clipped_median",
    "_ccf_background_sigma_clipped_median_dv_weighted",
    "_ccf_rv_err_numeric_floor",
    "_make_velocity_grid",
    "_velocity_grid_from_window",
    "ccf_oversample_nsp",
    "effective_ccf_v_step_for_fit",
    "resolve_ccf_broad_grid_m_s",
    "science_centroid_spec_err",
]

# Default coarse CCF velocity grid (m/s) for broad scans and StellarRV fallbacks.
V_MIN_DEFAULT_M_S = -500_000.0
V_MAX_DEFAULT_M_S = 500_000.0
V_STEP_DEFAULT_M_S = 50.0


def resolve_ccf_broad_grid_m_s(
    v_min: float | None = None,
    v_max: float | None = None,
    v_step: float | None = None,
) -> tuple[float, float, float]:
    """Broad uniform CCF velocity grid (m/s); ``None`` uses module defaults."""
    vm = float(V_MIN_DEFAULT_M_S if v_min is None else v_min)
    vx = float(V_MAX_DEFAULT_M_S if v_max is None else v_max)
    vs = float(V_STEP_DEFAULT_M_S if v_step is None else v_step)
    if not np.isfinite(vm) or not np.isfinite(vx) or not np.isfinite(vs):
        raise ValueError("v_min, v_max, v_step must be finite when set (m/s)")
    if vs <= 0:
        raise ValueError("v_step must be finite and > 0 (m/s)")
    if vx <= vm:
        raise ValueError("v_max must be greater than v_min (m/s)")
    return vm, vx, vs


def effective_ccf_v_step_for_fit(v_grid: np.ndarray) -> float:
    """Finest positive velocity spacing (m/s) on ``v_grid`` for CCF RV fits."""
    vg = np.asarray(v_grid, dtype=np.float64).ravel()
    if vg.size < 2:
        return 1.0
    d = np.diff(np.sort(vg))
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return 1.0
    return float(np.min(d))


def ccf_oversample_nsp(pix_size_m_s: float | None, v_step_m_s: float) -> float:
    """
    Number of CCF velocity samples per independent detector pixel.

    Interpolated ``CCF(v)`` on a grid finer than ``Δv_pix`` reuses the same
    pixels, so Fisher / LM covariances that treat bins as i.i.d. must inflate
    ``σ_μ`` by ``√nsp`` with ``nsp = max(Δv_pix / Δv_grid, 1)``.
    """
    if pix_size_m_s is None or not math.isfinite(float(pix_size_m_s)) or float(pix_size_m_s) <= 0.0:
        return 1.0
    vs = float(v_step_m_s) if math.isfinite(float(v_step_m_s)) and float(v_step_m_s) > 0.0 else 1.0
    return max(float(pix_size_m_s) / vs, 1.0)


def science_centroid_spec_err(
    spec: np.ndarray,
    spec_err: np.ndarray,
    spec_uncal: np.ndarray | None = None,
    spec_uncal_err: np.ndarray | None = None,
) -> np.ndarray:
    """
    Per-pixel σ for centroiding blaze-corrected (or native) flux.

    A multiplicative fiberflat / blaze error is almost entirely amplitude, not
    line centroid. When L1 ``spec_uncal`` / ``spec_uncal_err`` are present,
    map the **science-extract** relative error onto ``spec``:

    ``σ = |spec| · |spec_uncal_err / spec_uncal|``

    (equivalent to ``σ_uncal / blaze``). Fall back to ``spec_err`` where those
    arrays are missing or invalid.
    """
    flux = np.asarray(spec, dtype=np.float64)
    out = np.array(spec_err, dtype=np.float64, copy=True)
    if out.shape != flux.shape:
        out = np.broadcast_to(out, flux.shape).astype(np.float64, copy=True)
    if spec_uncal is None or spec_uncal_err is None:
        return out
    uncal = np.asarray(spec_uncal, dtype=np.float64)
    uncal_err = np.asarray(spec_uncal_err, dtype=np.float64)
    if uncal.shape != flux.shape or uncal_err.shape != flux.shape:
        return out
    good = (
        np.isfinite(flux)
        & np.isfinite(uncal)
        & np.isfinite(uncal_err)
        & (np.abs(uncal) > 0.0)
        & (uncal_err >= 0.0)
    )
    mapped = np.abs(flux) * np.abs(uncal_err / uncal)
    use = good & np.isfinite(mapped) & (mapped > 0.0)
    return np.where(use, mapped, out)


def _ccf_rv_err_numeric_floor(v_step_m_s: float) -> float:
    """Minimum positive RV uncertainty (m/s) for numerical stability only."""
    vs = float(abs(v_step_m_s)) if math.isfinite(v_step_m_s) else 1.0
    return max(1.0e-9 * max(vs, 1.0e-30), 1.0e-12)


def _velocity_grid_from_window(
    vel_window: float,
    vel_step: float,
    vsys: float = 0.0,
) -> np.ndarray:
    """Uniform RV trial grid spanning ``vel_window`` (m/s) centered on ``vsys``."""
    v_min = -float(vel_window) / 2.0 + float(vsys)
    v_max = float(vel_window) / 2.0 + float(vsys)
    return np.arange(v_min, v_max + float(vel_step) * 0.5, float(vel_step), dtype=np.float64)


def _make_velocity_grid(v_min: float, v_max: float, v_step: float) -> np.ndarray:
    """Uniform RV trial grid in m/s (matches Numba chunk CCF grid spacing)."""
    return np.arange(v_min, v_max + v_step * 0.5, v_step, dtype=np.float64)


def _ccf_background_sigma_clipped_median(
    y: np.ndarray,
    *,
    n_sigma: float = 3.0,
    max_iter: int = 8,
    min_keep: int = 5,
) -> float:
    """Iterative sigma-clipped median — robust CCF continuum / background level."""
    yy = np.asarray(y, dtype=np.float64).ravel()
    m = np.isfinite(yy)
    yy = yy[m]
    n = int(yy.size)
    if n == 0:
        return float("nan")
    if n < 3:
        return float(np.median(yy))
    med = float(np.median(yy))
    for _ in range(max_iter):
        resid = yy - med
        mad = float(np.median(np.abs(resid - float(np.median(resid)))))
        sig = 1.4826 * mad
        if not np.isfinite(sig) or sig <= 0:
            sig = float(np.std(yy))
        if not np.isfinite(sig) or sig <= 0:
            break
        thr = float(n_sigma) * sig
        keep = np.abs(resid) <= thr
        n_kept = int(np.count_nonzero(keep))
        if n_kept < max(min_keep, max(3, n // 4)):
            break
        if n_kept == n:
            break
        yy = yy[keep]
        med = float(np.median(yy))
    return med


def _ccf_trapezoid_dv_weights_m_s(v_m_s: np.ndarray) -> np.ndarray:
    """Trapezoid Δv weights (m/s) for a sorted velocity grid."""
    v = np.asarray(v_m_s, dtype=np.float64).ravel()
    n = int(v.size)
    w = np.zeros(n, dtype=np.float64)
    if n == 0:
        return w
    if n == 1:
        w[0] = 1.0
        return w
    w[0] = max(float(v[1] - v[0]), 0.0)
    w[-1] = max(float(v[-1] - v[-2]), 0.0)
    if n > 2:
        w[1:-1] = 0.5 * np.maximum(v[2:] - v[:-2], 0.0)
    return w


def _ccf_weighted_median(y: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median of ``y`` with non-negative ``weights``."""
    yy = np.asarray(y, dtype=np.float64).ravel()
    ww = np.asarray(weights, dtype=np.float64).ravel()
    ok = np.isfinite(yy) & np.isfinite(ww) & (ww > 0.0)
    if not np.any(ok):
        return float("nan")
    yy = yy[ok]
    ww = ww[ok]
    order = np.argsort(yy)
    ys = yy[order]
    cs = np.cumsum(ww[order])
    mid = 0.5 * float(cs[-1])
    return float(ys[int(np.searchsorted(cs, mid))])


def _ccf_background_sigma_clipped_median_dv_weighted(
    v_m_s: np.ndarray,
    y: np.ndarray,
    *,
    n_sigma: float = 3.0,
    max_iter: int = 8,
    min_keep: int = 5,
) -> float:
    """
    Continuum level with trapezoid-Δv weights.

    Weighting by Δv restores a wing-dominated baseline on non-uniform grids and
    matches uniform-grid behaviour when spacing is constant.
    """
    v = np.asarray(v_m_s, dtype=np.float64).ravel()
    yy = np.asarray(y, dtype=np.float64).ravel()
    if v.size != yy.size or v.size == 0:
        return _ccf_background_sigma_clipped_median(yy)
    order = np.argsort(v)
    v = v[order]
    yy = yy[order]
    ok = np.isfinite(v) & np.isfinite(yy)
    v = v[ok]
    yy = yy[ok]
    n = int(yy.size)
    if n == 0:
        return float("nan")
    if n < 3:
        return float(np.median(yy))
    w = _ccf_trapezoid_dv_weights_m_s(v)
    med = _ccf_weighted_median(yy, w)
    if not math.isfinite(med):
        return _ccf_background_sigma_clipped_median(yy)
    for _ in range(max_iter):
        resid = yy - med
        # Weight-aware MAD: median of |resid| with same weights
        mad = _ccf_weighted_median(np.abs(resid), w)
        sig = 1.4826 * float(mad) if math.isfinite(mad) else float("nan")
        if not math.isfinite(sig) or sig <= 0.0:
            sig = float(np.std(yy))
        if not math.isfinite(sig) or sig <= 0.0:
            break
        thr = float(n_sigma) * sig
        keep = np.abs(resid) <= thr
        n_kept = int(np.count_nonzero(keep))
        if n_kept < max(min_keep, max(3, n // 4)):
            break
        if n_kept == n:
            break
        yy = yy[keep]
        w = w[keep]
        n = n_kept
        med = _ccf_weighted_median(yy, w)
        if not math.isfinite(med):
            break
    return float(med)

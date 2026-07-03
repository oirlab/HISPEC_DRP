"""Shared CCF constants and small helpers used across mask / fit / grid modules."""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "V_MAX_DEFAULT_M_S",
    "V_MIN_DEFAULT_M_S",
    "V_STEP_DEFAULT_M_S",
    "_ccf_background_sigma_clipped_median",
    "_ccf_rv_err_numeric_floor",
    "_make_velocity_grid",
    "_velocity_grid_from_window",
    "effective_ccf_v_step_for_fit",
    "resolve_ccf_broad_grid_m_s",
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

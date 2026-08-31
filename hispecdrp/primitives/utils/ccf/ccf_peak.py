"""Sub-pixel CCF peak location on a coarse velocity grid (seeding / scout only).

These helpers locate a raw CCF extremum for Gaussian-fit *seeds* and auto-center
scouts. They are **not** used as the pipeline final RV estimator (that is the
Gaussian centroid from :func:`~hispecdrp.primitives.utils.ccf.ccf_fitting.calc_ccf_metrics`
/ :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.fit_ccf_rv`).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

__all__ = [
    "_ccf_raw_grid_subpixel_peak_nearest_disc_max_m_s",
]


def _collapse_duplicate_velocity_samples_m_s(
    vm: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort by velocity and average CCF values at duplicate ``vm`` knots."""
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


def _subpixel_peak_cubic_spline_m_s(vm_w: np.ndarray, z_w: np.ndarray) -> tuple[float | None, str | None]:
    """Location of the maximum via cubic spline + bounded optimization on velocity."""
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


def _ccf_raw_grid_subpixel_peak_m_s(
    vm: np.ndarray,
    y: np.ndarray,
    *,
    use_maximum: bool = True,
) -> float:
    """
    Sub-pixel location of a raw CCF extremum on the velocity grid.

    Prefer a cubic-spline extremum; fall back to the discrete argmax/argmin bin.
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
    v_c, _err = _subpixel_peak_cubic_spline_m_s(vm, y_eff)
    if v_c is not None and math.isfinite(float(v_c)):
        return float(v_c)
    return float(vm[int(np.nanargmax(y_eff))])


def _ccf_raw_grid_subpixel_peak_nearest_disc_max_m_s(vm: np.ndarray, y: np.ndarray) -> float:
    """
    Sub-pixel raw CCF location using the better of the global-maximum and minimum
    branches, chosen by proximity to the discrete velocity at ``argmax(y)``.
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

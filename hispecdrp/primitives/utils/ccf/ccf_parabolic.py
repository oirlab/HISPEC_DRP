"""CCF parabolic-vertex RV refinement (coarse-grid smooth + line-mask LSTSQ).

Mirrors :mod:`hispecdrp.primitives.utils.ccf.ccf_bisector`: :func:`fit_ccf_rv_parabolic`
refines a precomputed CCF on a velocity grid; :func:`_refine_ccf_rv_parabolic_after_gaussian_fit`
is the post-:func:`~hispecdrp.primitives.utils.ccf.ccf_utils.fit_ccf_rv` hook used by the pipeline.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .ccf_utils import CCFFitConfig

import numpy as np
from astropy.modeling import fitting
from astropy.modeling.models import Polynomial1D
from scipy.interpolate import CubicSpline
from scipy.ndimage import correlate1d, gaussian_filter1d
from scipy.optimize import minimize_scalar

try:
    from astropy.utils.exceptions import AstropyUserWarning
except ImportError:  # pragma: no cover
    AstropyUserWarning = UserWarning

from ....utils.parallelization_utils import njit
from .ccf_common import (
    _ccf_rv_err_numeric_floor,
    effective_ccf_v_step_for_fit,
)

__all__ = [
    "CCFParabolicConfig",
    "fit_ccf_rv_parabolic",
    "_refine_ccf_rv_parabolic_after_gaussian_fit",
    "refine_ccf_rv_coarse_grid_parabolic",
    "sweep_refine_ccf_rv_coarse_grid_min_abs_bias",
    "refine_ccf_rv_line_mask_lstsq_from_stored",
    "ensure_line_mask_lstsq_preview_plot_metadata",
    "line_mask_parabolic_vertex_rv_err_levmar_m_s",
    "line_mask_parabolic_vertex_rv_err_lstsq_homoscedastic_m_s",
    "require_ccf_parabolic_vertex_n_samples",
    "_line_mask_parabolic_vertex_lstsq_npt_numba",
    "_refine_ccf_rv_after_gaussian_fit_ccf_line_mask",
    "_refine_ccf_rv_gaussian_smooth_after_gaussian_fit_ccf_line_mask",
]

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


# Sub-pixel RV refinement on coarse velocity grids (e.g. 50 m/s CCF samples): default
# ``smooth_sigma_scale`` when :func:`refine_ccf_rv_coarse_grid_parabolic` runs with
# ``optimize_smooth_sigma_scale=False``, or as a fallback if the v₀-aligned sweep fails.
# (Historically tuned on lte049 line-mask debug for sub–10 cm s⁻¹ vs. Gaussian v₀.)
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
    config: CCFFitConfig | None = None,
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
    from .ccf_utils import CCFFitConfig, fit_ccf_rv

    cfg = config or CCFFitConfig()
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
    config: CCFFitConfig | None = None,
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


def _require_ccf_parabolic_vertex_n_samples(n: int) -> int:
    """
    Odd integer :math:`\\ge 3`: number of coarse velocity samples in the LSTSQ parabolic vertex
    window (symmetric about the peak index).

    Raises
    ------
    ValueError
        If ``n`` is below 3 or not odd.
    """
    n = int(n)
    if n < 3:
        raise ValueError(
            f"ccf_parabolic_vertex_n_samples must be >= 3, got {n}"
        )
    if n % 2 == 0:
        raise ValueError(
            f"ccf_parabolic_vertex_n_samples must be odd, got {n}"
        )
    return n


def _nearest_velocity_grid_index(v_grid_m_s: np.ndarray, v_target_m_s: float) -> int:
    """Index of the velocity sample closest to ``v_target_m_s`` (any monotonic grid)."""
    vg = np.asarray(v_grid_m_s, dtype=np.float64).ravel()
    if vg.size == 0 or not np.isfinite(v_target_m_s):
        return -1
    j = int(np.nanargmin(np.abs(vg - float(v_target_m_s))))
    if j < 0 or j >= vg.size:
        return -1
    return j


def _line_mask_parabolic_vertex_window_indices(
    n: int, i_peak: int, n_pt_want: int
) -> tuple[int, int] | None:
    """
    Odd-length velocity window centered on ``i_peak``, matching
    :func:`_line_mask_parabolic_vertex_lstsq_npt_numba` / LSTSQ extract.

    Returns ``(i0, n_use)`` with ``v_grid[i0:i0+n_use]`` the window, or ``None`` if invalid.
    """
    if n < 3:
        return None
    ip = int(i_peak)
    if ip < 0 or ip >= n:
        return None
    nw = int(n_pt_want)
    if nw < 3:
        nw = 3
    if (nw % 2) == 0:
        nw += 1
    half_w = (nw - 1) // 2
    half_lim = ip
    if (n - 1 - ip) < half_lim:
        half_lim = n - 1 - ip
    if half_lim < 0:
        half_lim = 0
    half_use = half_w
    if half_use > half_lim:
        half_use = half_lim
    if half_use < 1:
        return None
    n_use = 2 * half_use + 1
    i0 = ip - half_use
    return int(i0), int(n_use)


def _extract_line_mask_parabolic_vertex_lstsq_window(
    v_grid_m_s: np.ndarray,
    y_work: np.ndarray,
    i_peak: int,
    n_pt_want: int,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Velocity / signed-CCF samples in the LSTSQ window, matching
    :func:`_line_mask_parabolic_vertex_lstsq_npt_numba`.
    """
    vg = np.asarray(v_grid_m_s, dtype=np.float64).ravel()
    yw = np.asarray(y_work, dtype=np.float64).ravel()
    n = int(vg.size)
    if int(yw.size) != n or n < 3:
        return None, None
    wi = _line_mask_parabolic_vertex_window_indices(n, int(i_peak), int(n_pt_want))
    if wi is None:
        return None, None
    i0, n_use = wi
    v_s = np.ascontiguousarray(vg[i0 : i0 + n_use], dtype=np.float64)
    y_s = np.ascontiguousarray(yw[i0 : i0 + n_use], dtype=np.float64)
    return v_s, y_s


def line_mask_parabolic_vertex_rv_err_levmar_m_s(
    v_grid_m_s: np.ndarray,
    ccf_raw: np.ndarray,
    *,
    i_peak: int,
    n_pt_want: int,
    v_step_m_s_for_floor: float | None = None,
    ccf_err: np.ndarray | None = None,
    min_abs_curvature: float = 1e-30,
    levmar_maxiter: int = 512,
) -> tuple[float | None, dict[str, float]]:
    """
    Vertex σ (m/s) from a local quadratic fit in the same window as
    :func:`line_mask_parabolic_vertex_rv_err_lstsq_homoscedastic_m_s`, using
    :class:`~astropy.modeling.models.Polynomial1D` (degree 2) +
    :class:`~astropy.modeling.fitting.LevMarLSQFitter`, **y**-scaling via
    :func:`_levmar_ccf_y_scale`, and optional inverse-variance weights
    ``∝ 1 / (σ_CCF² · ys²)`` when ``ccf_err`` is aligned with the velocity grid (same
    convention as ``examples/plot_lte049_line_mask_ccf_segment.py``).

    Fits :math:`y \\approx c_0 + c_1 v + c_2 v^2` to :math:`y = -\\mathrm{CCF}_{\\rm raw}`.
    Vertex :math:`v = -c_1/(2c_2)`; uncertainty from LM ``param_cov`` and the same
    Jacobian as the plot script. Rejects non-concave fits (:math:`c_2 \\ge 0`) or vertices
    outside the window span (same acceptance as the Numba parabolic path).

    Returns
    -------
    sigma_m_s : float or None
        :math:`\\sigma_v` from the covariance propagation, floored with
        :func:`_ccf_rv_err_numeric_floor`.
    meta : dict
        Includes ``rv_levmar_m_s``, ``n_samples``, ``rss`` (scaled-y residuals), and
        polynomial coefficients when the fit succeeds.
    """
    vg = np.asarray(v_grid_m_s, dtype=np.float64).ravel()
    cr = np.asarray(ccf_raw, dtype=np.float64).ravel()
    if vg.size != cr.size or vg.size < 3:
        return None, {}
    y_work = np.where(np.isfinite(cr), -cr, np.nan)
    wi = _line_mask_parabolic_vertex_window_indices(
        int(vg.size), int(i_peak), int(n_pt_want)
    )
    if wi is None:
        return None, {}
    i0, n_use = wi
    vw = vg[i0 : i0 + n_use].astype(np.float64, copy=False)
    yw = y_work[i0 : i0 + n_use].astype(np.float64, copy=False)
    ew = (
        np.asarray(ccf_err[i0 : i0 + n_use], dtype=np.float64).ravel()
        if ccf_err is not None and int(np.asarray(ccf_err).size) == int(vg.size)
        else None
    )
    m = np.isfinite(vw) & np.isfinite(yw)
    if ew is not None:
        m = m & np.isfinite(ew)
    vw = vw[m]
    yw = yw[m]
    if ew is not None:
        ew = ew[m]
    n_fit = int(vw.size)
    if n_fit < 3:
        return None, {"n_samples": float(n_fit)}
    from .ccf_utils import _levmar_ccf_y_scale

    ys = float(_levmar_ccf_y_scale(yw))
    if not math.isfinite(ys) or ys <= 0.0:
        ys = 1.0
    y_scaled = yw * ys

    weights: np.ndarray | None = None
    if ew is not None and ew.size == vw.size:
        good_w = np.isfinite(ew) & (ew > 0)
        if int(np.count_nonzero(good_w)) >= max(3, min(8, vw.size // 2)):
            med_e = float(np.nanmedian(ew[good_w]))
            floor = max(med_e * 1e-6, 1e-30)
            ge_safe = np.where(good_w, ew, floor)
            weights = np.asarray(1.0 / (ge_safe * ge_safe), dtype=np.float64) / (
                ys * ys
            )

    try:
        p0 = np.polyfit(vw, y_scaled, 2)
    except (np.linalg.LinAlgError, ValueError):
        return None, {"n_samples": float(n_fit)}

    init = Polynomial1D(
        degree=2,
        c0=float(p0[2]),
        c1=float(p0[1]),
        c2=float(p0[0]),
    )
    ft = fitting.LevMarLSQFitter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (AstropyUserWarning, RuntimeWarning))
            mf = ft(
                init,
                vw,
                y_scaled,
                weights=weights,
                maxiter=int(levmar_maxiter),
            )
    except Exception:
        return None, {"n_samples": float(n_fit)}

    c1 = float(mf.c1.value)
    c2 = float(mf.c2.value)
    if (
        not math.isfinite(c1)
        or not math.isfinite(c2)
        or abs(c2) < float(min_abs_curvature)
        or c2 >= 0.0
    ):
        return None, {"n_samples": float(n_fit)}

    v_vertex = -0.5 * c1 / c2
    if not math.isfinite(v_vertex):
        return None, {"n_samples": float(n_fit)}
    v_lo = float(np.nanmin(vw))
    v_hi = float(np.nanmax(vw))
    vmin = v_lo if v_lo < v_hi else v_hi
    vmax = v_hi if v_hi > v_lo else v_lo
    if v_vertex < vmin - 1e-9 or v_vertex > vmax + 1e-9:
        return None, {"n_samples": float(n_fit)}

    sigma_v: float | None = None
    fi = getattr(ft, "fit_info", {}) or {}
    pc = fi.get("param_cov")
    if pc is None:
        pc = fi.get("cov_x")
    if pc is not None:
        try:
            pc = np.asarray(pc, dtype=np.float64)
            names = list(mf.param_names)
            j = np.zeros(len(names), dtype=np.float64)
            j[names.index("c1")] = -0.5 / c2
            j[names.index("c2")] = 0.5 * c1 / (c2 * c2)
            if pc.shape == (j.size, j.size) and np.all(np.isfinite(pc)):
                var_v = float(j @ pc @ j)
                if math.isfinite(var_v) and var_v >= 0.0:
                    sigma_v = float(math.sqrt(var_v))
        except (ValueError, IndexError, TypeError):
            pass

    yfit = np.asarray(mf(vw), dtype=np.float64)
    rss = float(np.sum((y_scaled - yfit) ** 2))

    if sigma_v is None or not math.isfinite(sigma_v) or sigma_v <= 0.0:
        return None, {
            "n_samples": float(n_fit),
            "rv_levmar_m_s": float(v_vertex),
            "rss": rss,
            "c0": float(mf.c0.value),
            "c1": c1,
            "c2": c2,
        }

    v_step_eff = (
        float(np.mean(np.abs(np.diff(vw)))) if vw.size > 1 else 1.0
    )
    fl = _ccf_rv_err_numeric_floor(
        float(v_step_m_s_for_floor)
        if v_step_m_s_for_floor is not None and math.isfinite(float(v_step_m_s_for_floor))
        else v_step_eff
    )
    sigma = max(float(sigma_v), fl)
    return float(sigma), {
        "n_samples": float(n_fit),
        "rv_levmar_m_s": float(v_vertex),
        "rss": rss,
        "c0": float(mf.c0.value),
        "c1": c1,
        "c2": c2,
        "v_lo_m_s": vmin,
        "v_hi_m_s": vmax,
    }


def line_mask_parabolic_vertex_rv_err_lstsq_homoscedastic_m_s(
    v_grid_m_s: np.ndarray,
    ccf_raw: np.ndarray,
    *,
    i_peak: int,
    n_pt_want: int,
    v_step_m_s_for_floor: float | None = None,
) -> tuple[float | None, dict[str, float]]:
    """
    Formal σ (m/s) on the LSTSQ parabolic vertex from **homoscedastic** residuals.

    Fits :math:`y \\approx a \\delta v^2 + b \\delta v + c` on the same velocity window as
    :func:`_line_mask_parabolic_vertex_lstsq_npt_numba`, with
    :math:`\\delta v = v - v_{\\rm center}` (window center, for numerical conditioning;
    equivalent polynomial in absolute :math:`v` as in Numba) and
    :math:`y = -\\mathrm{CCF}_{\\rm raw}`. Uses

    * residual variance :math:`s^2 = \\mathrm{RSS}/(N-3)`;
    * :math:`\\mathrm{Cov}(\\hat\\beta) = s^2 (X^{\\mathsf T}X)^{-1}` for
      :math:`X=[\\delta v^2,\\delta v,1]`; and
    * the delta method for :math:`\\mathrm{RV} = v_{\\rm center} - \\hat b/(2\\hat a)`:

      .. math::

         \\mathrm{Var}(\\mathrm{RV}) \\approx
             J^{\\mathsf T}\\,\\mathrm{Cov}(\\hat a,\\hat b,\\hat c)\\,J,
             \\quad J_i = \\partial \\mathrm{RV}/\\partial \\beta_i .

    This targets **refinement-limited** uncertainty from the :math:`N` coarse velocity
    samples (typically ``ccf_parabolic_vertex_n_samples``), not the per-pixel count
    :math:`n_{\\rm good}` used inside :func:`_rv_from_ccf_line_mask_numba_single`.

    **Caveats:** ignores correlated noise between CCF bins (from finite pixel sampling);
    for per-bin :math:`\\sigma_{\\mathrm{CCF}}(v_i)`, extend to weighted LS / GLS. See also
    :func:`_rv_err_bouchy_smoothed_ccf_m_s` for a Fisher-gradient σ on a smoothed CCF.

    Returns
    -------
    sigma_m_s : float or None
        Square root of the delta-method variance, floored with
        :func:`_ccf_rv_err_numeric_floor` using ``v_step_m_s_for_floor`` or the mean
        spacing in the LSTSQ window.
    meta : dict
        ``rss``, ``dof``, ``n_samples``, ``rv_lstsq_m_s``, and coefficients ``a``, ``b``
        when the fit succeeds; empty or partial on failure.
    """
    vg = np.asarray(v_grid_m_s, dtype=np.float64).ravel()
    cr = np.asarray(ccf_raw, dtype=np.float64).ravel()
    if vg.size != cr.size or vg.size < 3:
        return None, {}
    y_work = np.where(np.isfinite(cr), -cr, np.nan)
    v_s, y_s = _extract_line_mask_parabolic_vertex_lstsq_window(
        vg, y_work, int(i_peak), int(n_pt_want)
    )
    if v_s is None or y_s is None:
        return None, {}
    if not np.all(np.isfinite(v_s) & np.isfinite(y_s)):
        return None, {}
    n_use = int(v_s.size)
    if n_use < 3:
        return None, {}
    v_center = float(v_s[n_use // 2])
    dv = v_s - v_center
    X = np.column_stack((dv * dv, dv, np.ones(n_use, dtype=np.float64)))
    y = y_s
    coef, _res, rank, _s = np.linalg.lstsq(X, y, rcond=None)
    rank = int(rank)
    if rank < 3:
        return None, {"n_samples": float(n_use), "rank": float(rank)}
    yfit = X @ coef
    rss = float(np.sum((y - yfit) ** 2))
    dof = float(n_use - 3)
    if dof <= 0.0:
        return None, {"rss": rss, "n_samples": float(n_use)}
    s2 = rss / dof
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X.T @ X)
    cov = s2 * XtX_inv
    a = float(coef[0])
    b = float(coef[1])
    if not np.isfinite(a) or not np.isfinite(b) or abs(a) < 1e-40 or a >= 0.0:
        return None, {"rss": rss, "n_samples": float(n_use)}
    rv_v = v_center - b / (2.0 * a)
    if not np.isfinite(rv_v):
        return None, {"rss": rss, "n_samples": float(n_use)}
    j = np.array(
        [b / (2.0 * a * a), -1.0 / (2.0 * a), 0.0],
        dtype=np.float64,
    )
    var_rv = float(j @ cov @ j)
    if not np.isfinite(var_rv) or var_rv < 0.0:
        var_rv = 0.0
    sigma = math.sqrt(var_rv)
    v_step_eff = float(np.mean(np.abs(np.diff(v_s)))) if v_s.size > 1 else 1.0
    fl = _ccf_rv_err_numeric_floor(
        float(v_step_m_s_for_floor)
        if v_step_m_s_for_floor is not None and math.isfinite(float(v_step_m_s_for_floor))
        else v_step_eff
    )
    sigma = max(sigma, fl)
    return float(sigma), {
        "rss": rss,
        "dof": dof,
        "n_samples": float(n_use),
        "rv_lstsq_m_s": float(rv_v),
        "a": a,
        "b": b,
        "v_center_m_s": v_center,
    }


@njit(cache=True)
def _line_mask_parabolic_vertex_lstsq_npt_numba(
    v_grid_m_s: np.ndarray,
    y_signed: np.ndarray,
    i_peak: int,
    n_pt_want: int,
) -> tuple[float, float, bool]:
    """
    Fit :math:`y \\approx a v^2 + b v + c` to an odd number of consecutive samples centered on
    ``i_peak`` (at most ``n_pt_want``, fewer near velocity-grid edges) and return the vertex
    (maximum), ``a``, and success. Rejects non-concave fits (``a >= 0``) or vertices outside
    the spanned velocity interval.
    """
    n = int(v_grid_m_s.size)
    nw = int(n_pt_want)
    if nw < 3:
        nw = 3
    if (nw % 2) == 0:
        nw = nw + 1
    half_w = (nw - 1) // 2
    half_lim = i_peak
    if (n - 1 - i_peak) < half_lim:
        half_lim = n - 1 - i_peak
    if half_lim < 0:
        half_lim = 0
    half_use = half_w
    if half_use > half_lim:
        half_use = half_lim
    if half_use < 1:
        return np.nan, np.nan, False
    n_use = 2 * half_use + 1
    A = np.empty((n_use, 3), dtype=np.float64)
    rhs = np.empty(n_use, dtype=np.float64)
    for k in range(n_use):
        ii = i_peak - half_use + k
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
    v_lo = float(v_grid_m_s[i_peak - half_use])
    v_hi = float(v_grid_m_s[i_peak + half_use])
    vmin = v_lo if v_lo < v_hi else v_hi
    vmax = v_hi if v_hi > v_lo else v_lo
    if rv < vmin - 1e-9 or rv > vmax + 1e-9:
        return np.nan, np.nan, False
    return rv, a, True


def _line_mask_lstsq_parabola_plot_metadata(
    v_grid_m_s: np.ndarray,
    ccf_stored: np.ndarray,
    *,
    i_peak: int,
    n_par: int,
    stored_ccf_negated_from_raw: bool,
) -> dict[str, np.ndarray] | None:
    """
    Dense parabola curve and coarse-grid sample coordinates for Plotly (CCF vs velocity).

    ``y`` values are in the same units as the **stored** slot ``ccf`` (``stored_ccf_negated_from_raw``
    matches :data:`LINE_MASK_CCF_STORED_NEGATED_FROM_RAW`). The LSTSQ fit uses signed ``-CCF`` as in
    :func:`_line_mask_parabolic_vertex_lstsq_npt_numba`.
    """
    vg = np.asarray(v_grid_m_s, dtype=np.float64).ravel()
    ys = np.asarray(ccf_stored, dtype=np.float64).ravel()
    if vg.size != ys.size or vg.size < 3:
        return None
    if i_peak < 0 or i_peak >= vg.size:
        return None
    if bool(stored_ccf_negated_from_raw):
        ccf_raw = -ys
    else:
        ccf_raw = ys
    n = vg.size
    y_work = np.where(np.isfinite(ccf_raw), -ccf_raw, np.nan)
    nw = int(n_par)
    if nw < 3 or (nw % 2) != 0:
        return None
    half_w = (nw - 1) // 2
    half_lim = min(i_peak, n - 1 - i_peak)
    if half_lim < 0:
        half_lim = 0
    half_use = min(half_w, half_lim)
    if half_use < 1:
        return None
    n_use = 2 * half_use + 1
    i0 = i_peak - half_use
    i1 = i_peak + half_use + 1
    v_samp = vg[i0:i1].astype(np.float64, copy=False)
    y_samp = y_work[i0:i1].astype(np.float64, copy=False)
    if not np.all(np.isfinite(v_samp) & np.isfinite(y_samp)):
        return None
    a_mat = np.column_stack((v_samp * v_samp, v_samp, np.ones(n_use)))
    coeff, _, _, _ = np.linalg.lstsq(a_mat, y_samp, rcond=None)
    a, b, c0 = float(coeff[0]), float(coeff[1]), float(coeff[2])
    if not (np.isfinite(a) and np.isfinite(b)) or abs(a) < 1e-40 or a >= 0.0:
        return None
    rv = -b / (2.0 * a)
    v_lo = float(vg[i0])
    v_hi = float(vg[i1 - 1])
    vmin = v_lo if v_lo < v_hi else v_hi
    vmax = v_hi if v_hi > v_lo else v_lo
    if not np.isfinite(rv) or rv < vmin - 1e-9 or rv > vmax + 1e-9:
        return None
    n_dense = int(max(50, min(200, 10 * n_use)))
    v_dense = np.linspace(vmin, vmax, n_dense, dtype=np.float64)
    y_dense_w = a * v_dense * v_dense + b * v_dense + c0
    if bool(stored_ccf_negated_from_raw):
        y_dense_plot = y_dense_w
        y_samp_plot = y_samp
    else:
        y_dense_plot = -y_dense_w
        y_samp_plot = -y_samp
    return {
        "refine_parabola_v_dense_m_s": v_dense,
        "refine_parabola_ccf_dense": y_dense_plot.astype(np.float64, copy=False),
        "lstsq_parabola_v_samples_m_s": v_samp.astype(np.float64, copy=False),
        "lstsq_parabola_ccf_plot_samples": y_samp_plot.astype(np.float64, copy=False),
    }


def refine_ccf_rv_line_mask_lstsq_from_stored(
    v_grid_m_s: np.ndarray,
    ccf_stored: np.ndarray,
    *,
    n_good: int,
    ccf_parabolic_vertex_n_samples: int,
    v_gaussian_mean_m_s: float | None = None,
    ccf_err_stored: np.ndarray | None = None,
) -> tuple[float | None, float | None, dict | None]:
    """
    Sub-pixel RV from the order-mean (or preview-slot) mask CCF using the same Numba
    least-squares parabolic vertex as per-chunk :func:`rv_per_chunks_line_mask`.

    ``ccf_stored`` must match the RVCCF / preview convention (negated from the raw weighted
    statistic when :data:`LINE_MASK_CCF_STORED_NEGATED_FROM_RAW` is True).

    When the Numba parabolic window is accepted, the pipeline prefers
    :func:`line_mask_parabolic_vertex_rv_err_levmar_m_s` (Astropy LevMar on ``Polynomial1D``,
    scaled **y**, optional ``ccf_err`` weights, ``param_cov`` vertex σ) and uses its
    ``rv_levmar_m_s`` / σ when available; otherwise it uses **homoscedastic** LSTSQ σ as
    before. If neither succeeds, ``rv_err_m_s`` falls back to the Numba heuristic.

    When ``v_gaussian_mean_m_s`` is finite (typically the Gaussian **mean** :math:`v_0` from
    :func:`fit_ccf_rv` on the same CCF), the parabolic LSTSQ window is centered on the velocity
    grid index nearest that value — a prior on the line center before sub-grid refinement.
    When ``None``, peak placement matches the per-chunk path (smoothed-CCF prominence peak).

    ``ccf_err_stored`` must match ``ccf_stored`` in length (per-velocity-bin σ on the stored CCF).
    """
    from .ccf_utils import (
        LINE_MASK_CCF_STORED_NEGATED_FROM_RAW,
        _rv_from_ccf_line_mask_numba_single,
        _rv_from_ccf_line_mask_peak_index_numba,
    )

    vg = np.ascontiguousarray(
        np.asarray(v_grid_m_s, dtype=np.float64).ravel(), dtype=np.float64
    )
    ys = np.ascontiguousarray(
        np.asarray(ccf_stored, dtype=np.float64).ravel(), dtype=np.float64
    )
    if vg.size != ys.size or vg.size < 3:
        return None, None, None
    if bool(LINE_MASK_CCF_STORED_NEGATED_FROM_RAW):
        ccf_raw = np.ascontiguousarray(-ys, dtype=np.float64)
    else:
        ccf_raw = np.ascontiguousarray(ys, dtype=np.float64)
    n_par = _require_ccf_parabolic_vertex_n_samples(
        int(ccf_parabolic_vertex_n_samples)
    )
    i_prior = -1
    if v_gaussian_mean_m_s is not None and math.isfinite(float(v_gaussian_mean_m_s)):
        i_prior = _nearest_velocity_grid_index(vg, float(v_gaussian_mean_m_s))
    rv, rv_err, _flipped = _rv_from_ccf_line_mask_numba_single(
        vg,
        ccf_raw,
        max(1, int(n_good)),
        int(n_par),
        int(i_prior),
    )
    if not math.isfinite(float(rv)):
        return None, None, None
    err_m = float(rv_err) if math.isfinite(rv_err) else float("nan")
    meta: dict = {
        "ok": True,
        "refine_method": "line_mask_lstsq_parabolic_vertex",
        "ccf_parabolic_vertex_n_samples": int(n_par),
        "rv_err_m_s": err_m,
        "rv_refined_m_s": float(rv),
        "parabolic_peak_index_prior": int(i_prior),
        "v_gaussian_mean_prior_m_s": float(v_gaussian_mean_m_s)
        if v_gaussian_mean_m_s is not None and math.isfinite(float(v_gaussian_mean_m_s))
        else None,
    }
    i_peak = int(_rv_from_ccf_line_mask_peak_index_numba(vg, ccf_raw, int(i_prior)))
    y_work_np = np.ascontiguousarray(
        np.where(np.isfinite(ccf_raw), -ccf_raw, np.nan),
        dtype=np.float64,
    )
    _rv_ls, _a_ls, ok_lstsq = _line_mask_parabolic_vertex_lstsq_npt_numba(
        vg, y_work_np, i_peak, int(n_par)
    )
    err_out: float | None = err_m if math.isfinite(err_m) and err_m > 0 else None
    rv_out = float(rv)
    if ok_lstsq:
        pm = _line_mask_lstsq_parabola_plot_metadata(
            vg,
            ys,
            i_peak=i_peak,
            n_par=int(n_par),
            stored_ccf_negated_from_raw=bool(LINE_MASK_CCF_STORED_NEGATED_FROM_RAW),
        )
        if pm:
            meta.update(pm)
        vfloor = effective_ccf_v_step_for_fit(vg)
        err_a = (
            np.ascontiguousarray(
                np.asarray(ccf_err_stored, dtype=np.float64).ravel(),
                dtype=np.float64,
            )
            if ccf_err_stored is not None
            and int(np.asarray(ccf_err_stored).size) == int(vg.size)
            else None
        )
        sig_lm, diag_lm = line_mask_parabolic_vertex_rv_err_levmar_m_s(
            vg,
            ccf_raw,
            i_peak=i_peak,
            n_pt_want=int(n_par),
            v_step_m_s_for_floor=vfloor,
            ccf_err=err_a,
        )
        rvl = diag_lm.get("rv_levmar_m_s")
        use_lm = (
            sig_lm is not None
            and math.isfinite(float(sig_lm))
            and float(sig_lm) > 0.0
            and rvl is not None
            and math.isfinite(float(rvl))
        )
        if use_lm:
            meta["rv_levmar_m_s"] = float(rvl)
            meta["rv_levmar_diag"] = diag_lm
            meta["rv_err_levmar_m_s"] = float(sig_lm)
            meta["rv_refined_m_s"] = float(rvl)
            meta["rv_err_m_s"] = float(sig_lm)
            meta["rv_err_numba_heuristic_m_s"] = float(err_m)
            rv_out = float(rvl)
            err_out = float(sig_lm)
        else:
            sig_h, diag_h = line_mask_parabolic_vertex_rv_err_lstsq_homoscedastic_m_s(
                vg,
                ccf_raw,
                i_peak=i_peak,
                n_pt_want=int(n_par),
                v_step_m_s_for_floor=vfloor,
            )
            if (
                sig_h is not None
                and math.isfinite(float(sig_h))
                and float(sig_h) > 0.0
            ):
                meta["rv_err_lstsq_homoscedastic_m_s"] = float(sig_h)
                meta["rv_err_lstsq_homoscedastic_diag"] = diag_h
                meta["rv_err_numba_heuristic_m_s"] = float(err_m)
                meta["rv_err_m_s"] = float(sig_h)
                err_out = float(sig_h)
    return rv_out, err_out, meta


def ensure_line_mask_lstsq_preview_plot_metadata(
    rmeta: dict | None,
    v_grid_m_s: np.ndarray,
    ccf_stored: np.ndarray,
) -> dict | None:
    """
    Ensure ``line_mask_refine_rmeta`` from LSTSQ refinement carries Plotly arrays
    (dense parabola + sample markers). Recomputes them from the velocity grid and stored CCF
    when missing so the unified preview can always draw the fit inspection overlays.
    """
    if rmeta is None or not rmeta.get("ok"):
        return rmeta
    if rmeta.get("refine_method") != "line_mask_lstsq_parabolic_vertex":
        return rmeta
    rs = rmeta.get("lstsq_parabola_v_samples_m_s")
    rd = rmeta.get("refine_parabola_v_dense_m_s")
    if rs is not None and rd is not None:
        rsa = np.asarray(rs, dtype=np.float64).ravel()
        rda = np.asarray(rd, dtype=np.float64).ravel()
        if rsa.size > 0 and rda.size > 1:
            return rmeta
    vg = np.ascontiguousarray(
        np.asarray(v_grid_m_s, dtype=np.float64).ravel(), dtype=np.float64
    )
    ys = np.ascontiguousarray(
        np.asarray(ccf_stored, dtype=np.float64).ravel(), dtype=np.float64
    )
    if vg.size != ys.size or vg.size < 3:
        return rmeta
    from .ccf_utils import (
        LINE_MASK_CCF_STORED_NEGATED_FROM_RAW,
        _rv_from_ccf_line_mask_peak_index_numba,
    )

    try:
        n_par = _require_ccf_parabolic_vertex_n_samples(
            int(
                rmeta.get(
                    "ccf_parabolic_vertex_n_samples",
                    rmeta.get("line_mask_parabolic_vertex_n_samples", 11),
                )
            )
        )
    except ValueError:
        return rmeta
    i_prior = int(rmeta.get("parabolic_peak_index_prior", -1))
    if bool(LINE_MASK_CCF_STORED_NEGATED_FROM_RAW):
        ccf_raw = np.ascontiguousarray(-ys, dtype=np.float64)
    else:
        ccf_raw = np.ascontiguousarray(ys, dtype=np.float64)
    i_peak = int(_rv_from_ccf_line_mask_peak_index_numba(vg, ccf_raw, i_prior))
    y_work_np = np.ascontiguousarray(
        np.where(np.isfinite(ccf_raw), -ccf_raw, np.nan),
        dtype=np.float64,
    )
    _rv_ls, _a_ls, ok_lstsq = _line_mask_parabolic_vertex_lstsq_npt_numba(
        vg, y_work_np, i_peak, int(n_par)
    )
    if not ok_lstsq:
        return rmeta
    pm = _line_mask_lstsq_parabola_plot_metadata(
        vg,
        ys,
        i_peak=i_peak,
        n_par=int(n_par),
        stored_ccf_negated_from_raw=bool(LINE_MASK_CCF_STORED_NEGATED_FROM_RAW),
    )
    if not pm:
        return rmeta
    out = dict(rmeta)
    out.update(pm)
    return out


def _refine_ccf_rv_gaussian_smooth_after_gaussian_fit_ccf_line_mask(
    v_grid_m_s: np.ndarray,
    ccf_mean: np.ndarray,
    vstep: float,
    *,
    ccf_err: np.ndarray | None,
    params: dict | None,
) -> tuple[float | None, float | None, dict | None]:
    """
    Gaussian-smoothed coarse-grid refinement (:func:`refine_ccf_rv_coarse_grid_parabolic`).

    Used as fallback when :func:`refine_ccf_rv_line_mask_lstsq_from_stored` does not return a
    finite vertex, and for non-mask-line preview slots (template CCF).
    """
    from .ccf_utils import CCFFitConfig

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
        config=CCFFitConfig(),
    )
    if not meta.get("ok") or not math.isfinite(float(rv_r)):
        return None, None, None
    err_m = float(meta.get("rv_err_m_s", float("nan")))
    err_out: float | None = err_m if math.isfinite(err_m) and err_m > 0 else None
    return float(rv_r), err_out, meta


def _refine_ccf_rv_after_gaussian_fit_ccf_line_mask(
    v_grid_m_s: np.ndarray,
    ccf_mean: np.ndarray,
    vstep: float,
    *,
    ccf_err: np.ndarray | None,
    params: dict | None,
    ccf_parabolic_vertex_n_samples: int = 11,
    n_good: int | None = None,
) -> tuple[float | None, float | None, dict | None]:
    """
    Mask-line CCF refinement after :func:`fit_ccf_rv`: parabolic vertex via
    :func:`refine_ccf_rv_line_mask_lstsq_from_stored` (LevMar + optional ``ccf_err``, then
    homoscedastic LSTSQ fallback) first, then
    :func:`_refine_ccf_rv_gaussian_smooth_after_gaussian_fit_ccf_line_mask` if needed.

    The parabolic window uses the Gaussian fit mean ``params["v0"]`` (m s\\ :sup:`−1`) as a
    prior: centered on the velocity sample nearest ``v0``, with
    ``ccf_parabolic_vertex_n_samples`` points along the grid (odd, :math:`\\ge 3`).
    """
    ng = (
        int(n_good)
        if n_good is not None
        else max(1, int(np.count_nonzero(np.isfinite(np.asarray(ccf_mean)))))
    )
    v_prior: float | None = None
    if params and isinstance(params, dict):
        v0 = params.get("v0")
        if v0 is not None and np.isfinite(float(v0)):
            v_prior = float(v0)
    rv_ls, err_ls, meta_ls = refine_ccf_rv_line_mask_lstsq_from_stored(
        v_grid_m_s,
        ccf_mean,
        n_good=max(1, ng),
        ccf_parabolic_vertex_n_samples=ccf_parabolic_vertex_n_samples,
        v_gaussian_mean_m_s=v_prior,
        ccf_err_stored=ccf_err,
    )
    if rv_ls is not None and math.isfinite(float(rv_ls)):
        return rv_ls, err_ls, meta_ls
    return _refine_ccf_rv_gaussian_smooth_after_gaussian_fit_ccf_line_mask(
        v_grid_m_s, ccf_mean, vstep, ccf_err=ccf_err, params=params
    )


# ---------------------------------------------------------------------------
# Pipeline config + public RV API (parallel to :mod:`ccf_bisector`)
# ---------------------------------------------------------------------------

@dataclass
class CCFParabolicConfig:
    """
    Tunables for :func:`fit_ccf_rv_parabolic`.

    ``mode`` selects the refinement kernel:

    * ``"coarse_grid"`` — Gaussian-smooth the CCF along :math:`v`, then sub-pixel peak
      (:func:`refine_ccf_rv_coarse_grid_parabolic`).
    * ``"line_mask_lstsq"`` — odd-window quadratic LSTSQ on stored mask-line CCF rows
      (:func:`refine_ccf_rv_line_mask_lstsq_from_stored`).
    """

    mode: Literal["coarse_grid", "line_mask_lstsq"] = "coarse_grid"
    ccf_parabolic_vertex_n_samples: int = 11
    line_mask_stored_negated: bool = True
    smooth_sigma_scale: float = _REFINE_COARSE_GRID_SMOOTH_SIGMA_SCALE_DEFAULT
    optimize_smooth_sigma_scale: bool = True
    optimize_scale_lo: float = 0.04
    optimize_scale_hi: float = 0.10
    optimize_scale_n: int = 31
    peak_refine_iterations: int = _REFINE_PEAK_ITERATIONS_DEFAULT
    sigma_smooth_bins_min: float = 0.4
    sigma_smooth_bins_max: float = 12.0
    bootstrap_rv_err_samples: int = 0
    bootstrap_seed: int | None = None
    mask_method: str | None = None


def require_ccf_parabolic_vertex_n_samples(n: int) -> int:
    """Public alias of :func:`_require_ccf_parabolic_vertex_n_samples`."""
    return _require_ccf_parabolic_vertex_n_samples(n)


def fit_ccf_rv_parabolic(
    v_grid_m_s: np.ndarray,
    ccf: np.ndarray,
    v_step_m_s: float,
    *,
    ccf_err: np.ndarray | None = None,
    config: CCFParabolicConfig | None = None,
    return_diagnostics: bool = False,
    n_good: int | None = None,
    v_gaussian_mean_m_s: float | None = None,
    gaussian_sigma_m_s: float | None = None,
) -> tuple[float, float] | tuple[float, float, dict]:
    """
    Radial velocity (m/s) from parabolic-vertex refinement of a CCF.

    Dispatches on :class:`CCFParabolicConfig.mode` to either coarse-grid Gaussian smoothing
    or mask-line LSTSQ parabolic vertex fitting. API mirrors
    :func:`~hispecdrp.primitives.utils.ccf.ccf_bisector.fit_ccf_rv_bisector`.
    """
    cfg = config or CCFParabolicConfig()
    v_step_eff = (
        float(v_step_m_s)
        if math.isfinite(v_step_m_s) and v_step_m_s > 0
        else effective_ccf_v_step_for_fit(v_grid_m_s)
    )
    if cfg.mode == "line_mask_lstsq":
        ng = (
            int(n_good)
            if n_good is not None
            else max(1, int(np.count_nonzero(np.isfinite(np.asarray(ccf)))))
        )
        rv, err, diag = refine_ccf_rv_line_mask_lstsq_from_stored(
            v_grid_m_s,
            ccf,
            n_good=max(1, ng),
            ccf_parabolic_vertex_n_samples=int(cfg.ccf_parabolic_vertex_n_samples),
            v_gaussian_mean_m_s=v_gaussian_mean_m_s,
            ccf_err_stored=ccf_err,
        )
        if rv is None or not math.isfinite(float(rv)):
            out = (float("nan"), float("nan"), {"ok": False, "message": "line_mask_lstsq failed"})
            return out if return_diagnostics else (out[0], out[1])
        err_out = float(err) if err is not None and math.isfinite(float(err)) and float(err) > 0 else float("nan")
        meta = diag if isinstance(diag, dict) else {"ok": True, "rv_refined_m_s": float(rv)}
        if return_diagnostics:
            return float(rv), err_out, meta
        return float(rv), err_out

    from .ccf_utils import CCFFitConfig

    rv_r, meta = refine_ccf_rv_coarse_grid_parabolic(
        v_grid_m_s,
        ccf,
        v_step_eff,
        ccf_err=ccf_err,
        gaussian_sigma_m_s=gaussian_sigma_m_s,
        mask_method=cfg.mask_method,
        config=CCFFitConfig(),
        smooth_sigma_scale=float(cfg.smooth_sigma_scale),
        optimize_smooth_sigma_scale=bool(cfg.optimize_smooth_sigma_scale),
        optimize_scale_lo=float(cfg.optimize_scale_lo),
        optimize_scale_hi=float(cfg.optimize_scale_hi),
        optimize_scale_n=int(cfg.optimize_scale_n),
        peak_refine_iterations=int(cfg.peak_refine_iterations),
        sigma_smooth_bins_min=float(cfg.sigma_smooth_bins_min),
        sigma_smooth_bins_max=float(cfg.sigma_smooth_bins_max),
        bootstrap_rv_err_samples=int(cfg.bootstrap_rv_err_samples),
        bootstrap_seed=cfg.bootstrap_seed,
    )
    if not meta.get("ok") or not math.isfinite(float(rv_r)):
        out = (float("nan"), float("nan"), meta if isinstance(meta, dict) else {"ok": False})
        return out if return_diagnostics else (out[0], out[1])
    err_m = float(meta.get("rv_err_m_s", float("nan")))
    err_out = err_m if math.isfinite(err_m) and err_m > 0 else float("nan")
    if return_diagnostics:
        return float(rv_r), err_out, meta
    return float(rv_r), err_out


def _refine_ccf_rv_parabolic_after_gaussian_fit(
    v_grid_m_s: np.ndarray,
    ccf: np.ndarray,
    vstep: float,
    *,
    ccf_err: np.ndarray | None = None,
    config: CCFParabolicConfig | None = None,
    params: dict | None = None,
    n_good: int | None = None,
    ccf_parabolic_vertex_n_samples: int | None = None,
) -> tuple[float | None, float | None, dict | None]:
    """
    Parabolic RV after :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.fit_ccf_rv`.

    Mask-line CCF: LSTSQ vertex first, then coarse-grid fallback. Template / preview slots
    without mask-line storage use coarse-grid refinement only.
    """
    cfg = config or CCFParabolicConfig()
    n_par = (
        int(ccf_parabolic_vertex_n_samples)
        if ccf_parabolic_vertex_n_samples is not None
        else int(cfg.ccf_parabolic_vertex_n_samples)
    )
    return _refine_ccf_rv_after_gaussian_fit_ccf_line_mask(
        v_grid_m_s,
        ccf,
        float(vstep),
        ccf_err=ccf_err,
        params=params,
        ccf_parabolic_vertex_n_samples=n_par,
        n_good=n_good,
    )

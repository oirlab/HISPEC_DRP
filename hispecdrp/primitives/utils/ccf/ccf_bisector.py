"""CCF bisector utilities: Queloz BIS (Numba) and NEID-style weighted percentile RV."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from astropy.stats import mad_std
from numba import njit
from scipy.interpolate import CubicSpline

from .ccf_common import (
    _ccf_background_sigma_clipped_median,
    _ccf_background_sigma_clipped_median_dv_weighted,
    _ccf_rv_err_numeric_floor,
    effective_ccf_v_step_for_fit,
)

__all__ = [
    "CCFBisectorConfig",
    "calc_bisector_inverse_slope",
    "fit_ccf_rv_bisector",
    "_refine_ccf_rv_bisector_after_gaussian_fit",
]

@njit(cache=True, nogil=True)
def _interp_x_at_y(x0: float, y0: float, x1: float, y1: float, y: float) -> float:
    """Linear interpolation: return x where a segment (x0,y0)-(x1,y1) hits y."""
    dy = y1 - y0
    if dy == 0.0:
        return 0.5 * (x0 + x1)
    t = (y - y0) / dy
    return x0 + t * (x1 - x0)


@njit(cache=True, nogil=True)
def _find_ccf_level(
    vel_grid: np.ndarray,
    ccf: np.ndarray,
    ccf_err: np.ndarray,
    i_ext: int,
    level: float,
    direction: int,
) -> tuple[float, float, float, bool]:
    """
    Find where one flank crosses `level`, walking outward from the extremum.

    direction = -1 walks toward index 0 (blue flank); +1 toward n-1 (red flank).
    Returns (v_cross, flux_err_at_cross, dv_dflux, found). A single pass yields
    everything the caller needs, so no bracket is searched more than once.

    Parameters
    ----------
    vel_grid : np.ndarray
        Velocity grid (m/s).
    ccf : np.ndarray
        CCF values on vel_grid.
    ccf_err : np.ndarray
        1-sigma uncertainties on ccf (same shape).
    i_ext : int
        Index of the extremum (minimum for absorption dip, maximum for emission peak).
    level : float
        CCF value at which to find the flank crossing.
    direction : int
        -1 to walk left, +1 to walk right.
    """
    n = len(vel_grid)
    if direction == -1:
        i = i_ext
        while i > 0:
            a = ccf[i]
            b = ccf[i - 1]
            if (a - level) * (b - level) <= 0.0 and a != b:
                v_cross = _interp_x_at_y(vel_grid[i], a, vel_grid[i - 1], b, level)
                tt = (level - a) / (b - a)
                e_cross = ccf_err[i] * (1.0 - tt) + ccf_err[i - 1] * tt
                slope = (vel_grid[i - 1] - vel_grid[i]) / (b - a)
                return v_cross, e_cross, slope, True
            i -= 1
    elif direction == 1:
        i = i_ext
        while i < n - 1:
            a = ccf[i]
            b = ccf[i + 1]
            if (a - level) * (b - level) <= 0.0 and a != b:
                v_cross = _interp_x_at_y(vel_grid[i], a, vel_grid[i + 1], b, level)
                tt = (level - a) / (b - a)
                e_cross = ccf_err[i] * (1.0 - tt) + ccf_err[i + 1] * tt
                slope = (vel_grid[i + 1] - vel_grid[i]) / (b - a)
                return v_cross, e_cross, slope, True
            i += 1
    return np.nan, 0.0, 0.0, False


@njit(cache=True, nogil=True)
def _calc_bis(
    vel_grid: np.ndarray,
    ccf: np.ndarray,
    ccf_err: np.ndarray,
    n_levels: int = 100,
    shallow_lo: float = 0.10,
    shallow_hi: float = 0.40,
    core_lo: float = 0.55,
    core_hi: float = 0.90,
    pix_size: float | None = None
):
    """Bisector Inverse Slope (BIS) of a CCF, after Queloz et al. (2001).

    The bisector is the locus of midpoints of horizontal (constant-CCF-value)
    slices through the profile: at each flux level the velocities of the blue
    and red flanks are found and averaged. The BIS is then

        BIS = v_shallow - v_core

    where ``v_shallow`` is the mean bisector velocity over the shallow
    (near-continuum) depth region and ``v_core`` the mean over the core
    (near-core) region.

    Depth fractions are measured from the continuum toward the extremum, so the
    region definitions are orientation-invariant: a fraction of 0 is the
    continuum and a fraction of 1 is the core, whether the feature is an
    absorption dip or an emission peak. The Queloz et al. (2001) definition uses
    shallow = 10%-40% and core = 55%-90%, which are the defaults here.

    The CCF may be either an absorption-like dip or an emission-like peak; the
    orientation is detected automatically from whether the extremum lies below
    or above the wing (continuum) level.

    Parameters
    ----------
    vel_grid : ndarray
        Velocities (m/s), strictly ascending.
    ccf : ndarray
        CCF values on ``vel_grid``.
    ccf_err : ndarray
        1-sigma uncertainties on ``ccf`` (same shape). Used only to propagate
        the velocity uncertainties; pass an array of ones if you only want the
        velocity values.
    n_levels : int
        Number of evenly spaced depth levels at which the bisector is sampled
        across the full 0-1 depth range. Levels falling inside the shallow/core
        windows are the ones actually averaged into ``v_shallow``/``v_core``.
    shallow_lo, shallow_hi : float
        Lower/upper depth fractions (0-1, measured from the continuum) bounding
        the shallow (near-continuum) region used for ``v_shallow``. Defaults
        0.10, 0.40 (Queloz 2001).
    core_lo, core_hi : float
        Lower/upper depth fractions bounding the core (near-core) region used
        for ``v_core``. Defaults 0.55, 0.90 (Queloz 2001).

    Returns
    -------
    bis : float
        The bisector inverse slope (``v_shallow - v_core``), in the same
        velocity units as ``vel_grid``. NaN if the bisector cannot be measured
        (e.g. flat/degenerate profile, a non-interior extremum, or no levels
        bracketed inside the shallow/core windows).
    bis_err : float
        Propagated 1-sigma uncertainty on ``bis``: the quadrature sum of
        ``shallow_err`` and ``core_err``. NaN if ``bis`` is NaN.
    v_shallow : float
        Mean bisector velocity over the shallow (near-continuum) region. NaN if
        the fit failed before this could be computed.
    shallow_err : float
        1-sigma uncertainty on ``v_shallow``: the mean per-level bisector
        uncertainty in the shallow window divided by sqrt(N_shallow). NaN on
        failure.
    v_core : float
        Mean bisector velocity over the core (near-core) region. NaN on failure.
    core_err : float
        1-sigma uncertainty on ``v_core``: the mean per-level bisector
        uncertainty in the core window divided by sqrt(N_core). NaN on failure.
    levels : ndarray
        Depth fractions (0-1, measured from the continuum toward the extremum)
        at which the bisector was successfully measured. Levels where either
        flank could not be bracketed are omitted (no NaN padding), so this is
        generally shorter than ``n_levels - 1``. Empty if the profile was
        degenerate before sampling began.
    bis_curve : ndarray
        Bisector velocity at each level in ``levels``; paired element-for-
        element with it. Same length as ``levels``.

    Notes
    -----
    All returns have fixed arity (8 values) regardless of success or failure;
    failure paths fill the scalars with NaN and return whatever portion of
    ``levels``/``bis_curve`` had been collected (possibly empty).
    """
    n = vel_grid.shape[0]
    if n < 4:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    i_ext = 0
    cmin = ccf[0]
    cmax = ccf[0]
    i_min = 0
    i_max = 0
    for i in range(n):
        v = ccf[i]
        if v < cmin:
            cmin = v
            i_min = i
        if v > cmax:
            cmax = v
            i_max = i

    cont = 0.5 * (ccf[0] + ccf[n - 1])

    if (cont - cmin) >= (cmax - cont):
        is_dip = True
        i_ext = i_min
        extreme = cmin
    else:
        is_dip = False
        i_ext = i_max
        extreme = cmax

    full_depth = cont - extreme if is_dip else extreme - cont
    if not (full_depth > 0.0):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    if i_ext <= 0 or i_ext >= n - 1:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    shallow_sum = 0.0
    shallow_var = 0.0
    shallow_cnt = 0
    core_sum = 0.0
    core_var = 0.0
    core_cnt = 0

    curve_buf = np.empty(n_levels, dtype=np.float64)
    level_buf = np.empty(n_levels, dtype=np.float64)
    n_curve = 0

    for j in range(1, n_levels):
        f = j / n_levels

        if is_dip:
            level = cont - f * full_depth
        else:
            level = cont + f * full_depth

        vb, eb, sb, found_b = _find_ccf_level(
            vel_grid, ccf, ccf_err, i_ext, level, -1,
        )
        vr, er, sr, found_r = _find_ccf_level(
            vel_grid, ccf, ccf_err, i_ext, level, 1,
        )

        if not (found_b and found_r):
            continue

        v_bis = 0.5 * (vb + vr)

        curve_buf[n_curve] = v_bis
        level_buf[n_curve] = f
        n_curve += 1

        ev_b = abs(sb) * eb
        ev_r = abs(sr) * er
        v_bis_err = 0.5 * np.sqrt(ev_b * ev_b + ev_r * ev_r)

        if (f >= shallow_lo) and (f <= shallow_hi):
            shallow_sum += v_bis
            shallow_var += v_bis_err * v_bis_err
            shallow_cnt += 1
        if (f >= core_lo) and (f <= core_hi):
            core_sum += v_bis
            core_var += v_bis_err * v_bis_err
            core_cnt += 1

    bis_curve = curve_buf[:n_curve].copy()
    levels = level_buf[:n_curve].copy()

    if shallow_cnt == 0 or core_cnt == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, levels, bis_curve

    v_shallow = shallow_sum / shallow_cnt
    v_core = core_sum / core_cnt
    bis = v_shallow - v_core

    shallow_err = np.sqrt(shallow_var / shallow_cnt) / np.sqrt(shallow_cnt)
    core_err = np.sqrt(core_var / core_cnt) / np.sqrt(core_cnt)
    bis_err = np.sqrt(shallow_err * shallow_err + core_err * core_err)

    if pix_size is not None:
        shallow_err = max(shallow_err, pix_size / np.sqrt(shallow_cnt))
        core_err = max(core_err, pix_size / np.sqrt(core_cnt))
    else:
        shallow_err = max(shallow_err, np.sqrt(shallow_var) / shallow_cnt)
        core_err = max(core_err, np.sqrt(core_var) / core_cnt)
    bis_err = max(bis_err, np.sqrt(shallow_err * shallow_err + core_err * core_err))

    return bis, bis_err, v_shallow, shallow_err, v_core, core_err, levels, bis_curve


def calc_bisector_inverse_slope(
    vel_grid: np.ndarray,
    ccf: np.ndarray,
    ccf_err: np.ndarray | None = None,
    n_levels: int = 100,
    quant_shallow : tuple[float, float] = (0.10, 0.40),
    quant_core : tuple[float, float] = (0.6, 0.90)
):
    """
    Bisector Inverse Slope (BIS) of a CCF, after Queloz et al. (2001).

    Thin wrapper over the njit core (``_calc_bis``). ``ccf_err`` is optional: it
    is used only to propagate the velocity uncertainties, never the velocity
    values themselves. When ``ccf_err`` is None the bisector velocities are
    computed normally and all three error outputs (``bis_err``, ``shallow_err``,
    ``core_err``) are returned as NaN, since there is no uncertainty information
    to propagate.

    See ``_calc_bis`` for the full parameter and algorithm description. In
    brief, the bisector midpoints are averaged over two depth-fraction windows
    measured from the continuum toward the extremum (orientation-invariant for
    both absorption dips and emission peaks):

        BIS = v_shallow - v_core

    with ``shallow`` near the continuum and ``core`` near the core.

    Parameters
    ----------
    vel_grid, ccf : ndarray
        Velocity grid (ascending) and CCF values.
    ccf_err : ndarray | None
        1-sigma uncertainties on ``ccf``. Optional; if None, the error outputs
        are NaN.
    n_levels : int
        Number of evenly spaced depth levels sampled across the 0-1 range.
    quant_shallow : tuple[float, float] | None
        Lower/upper depth fractions (0-1, measured from the continuum) bounding
        the shallow (near-continuum) region used for ``v_shallow``.
        Defaults to (0.10, 0.40).
    quant_core : tuple[float, float]
        Lower/upper depth fractions (0-1, measured from the continuum) bounding
        the core (near-core) region used for ``v_core``.
        Defaults to (0.6, 0.90).

    Returns
    -------
    bis : float
        The bisector inverse slope (``v_shallow - v_core``). NaN if not
        measurable.
    bis_err : float
        Propagated 1-sigma uncertainty on ``bis`` (quadrature sum of
        ``shallow_err`` and ``core_err``). NaN if not measurable or if
        ``ccf_err`` was None.
    v_shallow : float
        Mean bisector velocity over the shallow (near-continuum) region.
    shallow_err : float
        1-sigma uncertainty on ``v_shallow``. NaN if not measurable or if
        ``ccf_err`` was None.
    v_core : float
        Mean bisector velocity over the core (near-core) region.
    core_err : float
        1-sigma uncertainty on ``v_core``. NaN if not measurable or if
        ``ccf_err`` was None.
    levels, bis_curve : ndarray
        Depth fractions and bisector velocities at each successfully bracketed
        level, sampled across the full 0-1 depth range. Paired element-for-
        element.
    """
    if ccf_err is None:
        (bis, _, v_shallow, _, v_core, _,
         levels, bis_curve) = _calc_bis(
            vel_grid, ccf, np.ones_like(ccf),
            n_levels, quant_shallow[0], quant_shallow[1],
            quant_core[0], quant_core[1]
        )
        return (bis, np.nan, v_shallow, np.nan, v_core, np.nan,
                levels, bis_curve)
    return _calc_bis(
        vel_grid, ccf, ccf_err,
        n_levels, quant_shallow[0], quant_shallow[1],
        quant_core[0], quant_core[1]
    )

# ---------------------------------------------------------------------------
# NEID-style weighted-percentile bisector RV (pipeline / fit_ccf_rv_bisector)
# ---------------------------------------------------------------------------

@dataclass
class CCFBisectorConfig:
    """
    Tunables for :func:`fit_ccf_rv_bisector` (traditional CCF bisector RV).

    Follows the NEID-DRP bisector construction (percentile cuts through the CCF in
    velocity space; see NEID algorithms § CCF Bisector Calculation). Radial velocity
    **and** its uncertainty are a **weighted mean** of bisector midpoints
    :math:`(v_\\mathrm{left}+v_\\mathrm{right})/2` at multiple depth percentiles:
    with per-cut midpoint uncertainties :math:`e_i` and weights :math:`w_i`,

    .. math::

        \\hat{v} = \\frac{\\sum_i w_i v_i}{\\sum_i w_i},\\qquad
        \\sigma_{\\hat{v}} = \\frac{\\sum_i w_i e_i}{\\sum_i w_i}.

    (The uncertainty is the weighted average of the cut uncertainties, not an
    independent-measurement reduction of those errors.) Cut weights are the core
    performance weight only; per-cut :math:`e_i` comes from CCF-error propagation
    when informative, otherwise from the smooth-profile discretization estimate.

    Attributes
    ----------
    rv_percentile_lo, rv_percentile_hi : float
        Percentile range (0–100, relative to peak depth above baseline) over which
        bisector midpoints contribute to the RV estimate.
    rv_percentile_step : float
        Step between percentile cuts (percent).
    bis_top_lo, bis_top_hi : float
        Top section for bisector inverse slope (BIS), NEID default 60–90%.
    bis_bottom_lo, bis_bottom_hi : float
        Bottom section for BIS, NEID default 10–40%.
    core_weight_sigma_percentile : float
        Gaussian performance weight on percentile cuts, centered at 50% depth
        (emphasizes the CCF core). Set very large to approximate uniform weighting.
    rv_err_core_percentile : float
        Depth percentile recorded in diagnostics as the representative “core” cut
        (does **not** set ``rv_err``; retained for previews / BIS section labeling).
    min_cuts : int
        Minimum successful bisector cuts required for a valid RV.
    min_peak_contrast_sigma : float
        Minimum peak amplitude (above baseline) in units of robust wing noise (MAD of
        ``y - baseline``). Weak peaks scale ``rv_err`` up by
        ``min_peak_contrast_sigma / (peak/MAD)`` so alias fits are down-weighted.
    rv_err_min_vstep_fraction : float
        Optional absolute floor as this fraction of the minimum positive velocity
        spacing. Default ``0`` (disabled): sub-pixel bisector precision should not
        be forced to a large fraction of a coarse grid step.
    """

    rv_percentile_lo: float = 10.0
    rv_percentile_hi: float = 90.0
    rv_percentile_step: float = 5.0
    bis_top_lo: float = 60.0
    bis_top_hi: float = 90.0
    bis_bottom_lo: float = 10.0
    bis_bottom_hi: float = 40.0
    core_weight_sigma_percentile: float = 25.0
    rv_err_core_percentile: float = 50.0
    rv_err_bootstrap_samples: int = 0
    rv_err_bootstrap_seed: int | None = None
    min_cuts: int = 5
    min_peak_contrast_sigma: float = 5.0
    rv_err_min_vstep_fraction: float = 0.0
def _ccf_bisector_wing_crossing_m_s(
    v_m_s: np.ndarray,
    y_peak_up: np.ndarray,
    i_peak: int,
    target: float,
    side: str,
) -> tuple[float, int]:
    """
    Velocity (m/s) and bracket index where ``y_peak_up`` crosses ``target``.

    Returns ``(v_cross, i_lo)`` with the crossing lying between ``v_m_s[i_lo]`` and
    ``v_m_s[i_lo + 1]``.
    """
    n = len(v_m_s)
    if side == "left":
        for i in range(i_peak - 1, -1, -1):
            y0, y1 = float(y_peak_up[i]), float(y_peak_up[i + 1])
            if (y0 - target) * (y1 - target) <= 0.0 and y1 != y0:
                frac = (target - y0) / (y1 - y0)
                return float(v_m_s[i] + frac * (v_m_s[i + 1] - v_m_s[i])), int(i)
        return float("nan"), -1
    if side == "right":
        for i in range(i_peak, n - 1):
            y0, y1 = float(y_peak_up[i]), float(y_peak_up[i + 1])
            if (y0 - target) * (y1 - target) <= 0.0 and y1 != y0:
                frac = (target - y0) / (y1 - y0)
                return float(v_m_s[i] + frac * (v_m_s[i + 1] - v_m_s[i])), int(i)
        return float("nan"), -1
    raise ValueError(f"side must be 'left' or 'right', got {side!r}")


def _ccf_bisector_wing_velocity_m_s(
    v_m_s: np.ndarray,
    y_peak_up: np.ndarray,
    i_peak: int,
    target: float,
    side: str,
) -> float:
    """
    Velocity (m/s) where ``y_peak_up`` crosses ``target`` on the left or right wing of ``i_peak``.

    Uses linear interpolation between bracketing samples. ``y_peak_up`` must be peak-up
    (non-negative at the extremum).
    """
    v_cross, _ = _ccf_bisector_wing_crossing_m_s(v_m_s, y_peak_up, i_peak, target, side)
    return v_cross


def _ccf_bisector_discretization_sigma_m_s(
    v_m_s: np.ndarray,
    y_peak_up: np.ndarray,
    i_lo: int,
) -> float:
    """
    Sub-grid 1σ uncertainty (m/s) for a linearly interpolated wing crossing.

    For a smooth CCF sampled on a uniform grid, interpolation error scales as
    :math:`\\mathcal{O}(\\Delta v^2)` (not :math:`\\mathcal{O}(\\Delta v)`).
    """
    n = len(v_m_s)
    i_lo = int(i_lo)
    if i_lo < 0 or i_lo >= n - 1:
        return float("nan")
    dv = float(v_m_s[i_lo + 1] - v_m_s[i_lo])
    if dv <= 0.0:
        return float("nan")
    dy_dv = float(y_peak_up[i_lo + 1] - y_peak_up[i_lo]) / dv
    if abs(dy_dv) <= 1e-30:
        return float("nan")
    if 0 < i_lo < n - 2:
        d2y_dv2 = float(y_peak_up[i_lo + 1] - 2.0 * y_peak_up[i_lo] + y_peak_up[i_lo - 1]) / (dv * dv)
    elif i_lo == 0:
        d2y_dv2 = float(y_peak_up[i_lo + 2] - 2.0 * y_peak_up[i_lo + 1] + y_peak_up[i_lo]) / (dv * dv)
    else:
        d2y_dv2 = float(y_peak_up[i_lo + 1] - 2.0 * y_peak_up[i_lo] + y_peak_up[i_lo - 1]) / (dv * dv)
    return float(abs(dv * dv * d2y_dv2 / (2.0 * dy_dv)))


def _ccf_bisector_wing_velocity_uncertainty_m_s(
    v_m_s: np.ndarray,
    y_peak_up: np.ndarray,
    i_lo: int,
    target: float,
    *,
    y_err: np.ndarray | None = None,
    v_step_m_s: float,
) -> float:
    """
    Approximate 1σ uncertainty (m/s) on a bisector wing crossing.

    With per-bin ``y_err``, propagates noise through the linear interpolation that
    locates the crossing when that formal σ is below the uniform-in-bracket prior
    (:math:`\\Delta v/\\sqrt{12}`). When formal noise saturates that prior, returns
    the :math:`\\mathcal{O}(\\Delta v^2)` discretization estimate for a smooth CCF
    instead of locking σ to ~0.3 grid steps. Without meaningful ``y_err``, uses
    discretization only.
    """
    n = len(v_m_s)
    i_lo = int(i_lo)
    if i_lo < 0 or i_lo >= n - 1:
        return float(max(v_step_m_s, _ccf_rv_err_numeric_floor(v_step_m_s)))
    y0 = float(y_peak_up[i_lo])
    y1 = float(y_peak_up[i_lo + 1])
    v0 = float(v_m_s[i_lo])
    v1 = float(v_m_s[i_lo + 1])
    dy = y1 - y0
    dv = v1 - v0
    if abs(dy) <= 0.0:
        return float(max(v_step_m_s, _ccf_rv_err_numeric_floor(v_step_m_s)))

    use_noise = False
    s0 = s1 = 0.0
    if y_err is not None and len(y_err) == n:
        s0 = float(y_err[i_lo]) if np.isfinite(y_err[i_lo]) else 0.0
        s1 = float(y_err[i_lo + 1]) if np.isfinite(y_err[i_lo + 1]) else 0.0
        use_noise = (s0 > 0.0) or (s1 > 0.0)

    # Guard against tiny bracket gradients on noisy samples (would inflate σ_v).
    local_span = max(abs(dv), float(v_step_m_s))
    grad_floor = max(abs(y1 - y0) / local_span, 1e-30)
    if use_noise and abs(dy) < 0.02 * grad_floor * local_span:
        dy = math.copysign(0.02 * grad_floor * local_span, dy if dy != 0.0 else 1.0)

    sig_disc = _ccf_bisector_discretization_sigma_m_s(v_m_s, y_peak_up, i_lo)
    if use_noise:
        # v* = v0 + (target - y0) * (v1 - v0) / (y1 - y0)
        dv_dy0 = -(v1 - v0) / dy
        dv_dy1 = (target - y0) * (v1 - v0) / (dy * dy)
        sig_noise = math.sqrt((dv_dy0 * s0) ** 2 + (dv_dy1 * s1) ** 2)
        # Uniform-in-bracket prior σ = Δv/√12. When formal noise sits at or above
        # that ceiling, bin spacing — not CCF S/N — dominates the linear
        # interpolator; use the smooth-profile O(Δv²) discretization estimate so a
        # coarse velocity grid is not stuck near ~0.3 Δv per wing.
        bracket_prior = abs(dv) / math.sqrt(12.0)
        if (
            math.isfinite(bracket_prior)
            and bracket_prior > 0.0
            and math.isfinite(sig_noise)
            and sig_noise >= 0.95 * bracket_prior
            and math.isfinite(sig_disc)
            and sig_disc > 0.0
        ):
            sig_v = float(sig_disc)
        else:
            sig_v = float(sig_noise)
    else:
        sig_v = float(sig_disc) if math.isfinite(sig_disc) and sig_disc > 0.0 else 0.0

    if not math.isfinite(sig_v) or sig_v <= 0.0:
        sig_v = float(max(v_step_m_s, _ccf_rv_err_numeric_floor(v_step_m_s)))
    return float(sig_v)


def _ccf_bisector_cut_m_s(
    v_m_s: np.ndarray,
    y_peak_up: np.ndarray,
    i_peak: int,
    depth_frac: float,
    peak_amp: float,
    *,
    y_err: np.ndarray | None = None,
    v_step_m_s: float,
) -> tuple[float, float, float, float, float] | None:
    """
    Bisector midpoint at one depth percentile.

    Returns ``(v_mid, v_mid_err, v_left, v_right, percentile)`` or ``None`` if the
    cut does not bracket both wings.
    """
    if not (0.0 < depth_frac < 1.0) or peak_amp <= 0.0:
        return None
    target = depth_frac * peak_amp
    v_left, i_left = _ccf_bisector_wing_crossing_m_s(v_m_s, y_peak_up, i_peak, target, "left")
    v_right, i_right = _ccf_bisector_wing_crossing_m_s(v_m_s, y_peak_up, i_peak, target, "right")
    if not (math.isfinite(v_left) and math.isfinite(v_right)):
        return None
    sig_l = _ccf_bisector_wing_velocity_uncertainty_m_s(
        v_m_s, y_peak_up, i_left, target, y_err=y_err, v_step_m_s=v_step_m_s
    )
    sig_r = _ccf_bisector_wing_velocity_uncertainty_m_s(
        v_m_s, y_peak_up, i_right, target, y_err=y_err, v_step_m_s=v_step_m_s
    )
    v_mid = 0.5 * (v_left + v_right)
    v_mid_err = math.sqrt(0.25 * (sig_l * sig_l + sig_r * sig_r))
    return v_mid, v_mid_err, v_left, v_right, depth_frac * 100.0


def fit_ccf_rv_bisector(
    v_grid_m_s: np.ndarray,
    ccf: np.ndarray,
    v_step_m_s: float,
    *,
    ccf_err: np.ndarray | None = None,
    config: CCFBisectorConfig | None = None,
    return_diagnostics: bool = False,
    gaussian_seed_mean_m_s: float | None = None,
) -> (
    tuple[float, float]
    | tuple[float, float, dict]
):
    """
    Radial velocity (m/s) from a traditional CCF bisector analysis.

    At each depth percentile between baseline and peak, finds the left and right
    wing velocities where the CCF crosses that flux level and takes their midpoint.
    The RV estimate **and** its uncertainty are a performance-weighted mean of
    midpoints over ``config.rv_percentile_lo`` … ``config.rv_percentile_hi``, with
    Gaussian weight emphasizing the CCF core (50% depth). The reported σ is the
    weighted average of per-cut midpoint uncertainties
    (:math:`\\sum_i w_i e_i\\,/\\,\\sum_i w_i`), not an independent-cut
    :math:`1/\\sqrt{N}` reduction.

    Also computes bisector inverse slope (BIS) following NEID-DRP: average midpoints
    in the top (60–90%) minus bottom (10–40%) sections, with the same weighted-mean
    uncertainty on each section.

    Parameters
    ----------
    v_grid_m_s : np.ndarray
        Velocity samples (m/s), need not be uniform.
    ccf : np.ndarray
        CCF values on the same grid.
    v_step_m_s : float
        Effective finest velocity spacing (m/s) for uncertainty floors; use
        :func:`effective_ccf_v_step_for_fit` on ``v_grid_m_s``.
    ccf_err : np.ndarray or None, optional
        Per-sample CCF uncertainty for wing-crossing error propagation.
    config : CCFBisectorConfig or None, optional
        Bisector percentile ranges and weighting.
    return_diagnostics : bool, optional
        If True, return ``(rv, rv_err, diag)`` with per-cut midpoints, BIS, etc.
    gaussian_seed_mean_m_s : float or None, optional
        When set (typically the Gaussian CCF fit center), the bisector peak index is
        the sample nearest this velocity (with a small local refine). Do **not**
        re-search ``argmax(|CCF − baseline|)`` over a multi-km/s window: a global
        contrast search can lock onto a window edge when the core has near-zero
        contrast relative to an unweighted continuum.

    Returns
    -------
    rv_m_s, rv_err_m_s : float
        Weighted bisector RV and combined uncertainty (m/s).
    diag : dict, optional
        Present when ``return_diagnostics=True``.
    """
    cfg = config or CCFBisectorConfig()
    vg = np.asarray(v_grid_m_s, dtype=np.float64).ravel()
    y0 = np.asarray(ccf, dtype=np.float64).ravel()
    ok = np.isfinite(vg) & np.isfinite(y0)
    if int(np.sum(ok)) < 5:
        out = (float("nan"), float("nan"), {"ok": False, "message": "insufficient finite samples"})
        return out if return_diagnostics else (out[0], out[1])

    order = np.argsort(vg[ok])
    v_m_s = vg[ok][order]
    y = y0[ok][order]
    y_err: np.ndarray | None = None
    has_ccf_err = False
    if ccf_err is not None:
        ye = np.asarray(ccf_err, dtype=np.float64).ravel()
        if ye.size == y0.size:
            y_err = ye[ok][order]
            has_ccf_err = bool(np.any(np.isfinite(y_err) & (y_err > 0.0)))

    baseline = _ccf_background_sigma_clipped_median_dv_weighted(v_m_s, y)
    if not math.isfinite(baseline):
        baseline = _ccf_background_sigma_clipped_median(y)
    if not math.isfinite(baseline):
        baseline = float(np.nanmedian(y))
    y_rel = y - baseline
    n = int(v_m_s.size)
    v_step = float(v_step_m_s) if math.isfinite(v_step_m_s) and v_step_m_s > 0 else effective_ccf_v_step_for_fit(v_m_s)

    if (
        gaussian_seed_mean_m_s is not None
        and math.isfinite(float(gaussian_seed_mean_m_s))
        and n >= 3
    ):
        # Lock the bisector peak to the Gaussian / scout seed. A wide unweighted
        # |y_rel| search can lock onto the window edge when the core has near-zero
        # contrast.
        v_seed = float(gaussian_seed_mean_m_s)
        i_peak = int(np.argmin(np.abs(v_m_s - v_seed)))
        i_peak = int(min(max(i_peak, 1), n - 2))
        # Optional micro-refine within ±2 steps of the seed (not a wide half-window
        # that reintroduces edge locking).
        refine_half = max(2.0 * float(v_step), 100.0)
        idx = np.arange(n, dtype=np.int64)
        local = (
            (v_m_s >= v_seed - refine_half)
            & (v_m_s <= v_seed + refine_half)
            & (idx > 0)
            & (idx < n - 1)
        )
        if np.any(local):
            loc = np.where(local)[0]
            i_local = int(loc[int(np.nanargmax(np.abs(y_rel[local])))])
            # Accept refine only if it stays near the seed (reject wing-edge spikes).
            if abs(float(v_m_s[i_local]) - v_seed) <= refine_half:
                i_peak = i_local
    else:
        i_peak = int(np.nanargmax(np.abs(y_rel)))
        # Wing walks need an interior peak; fall back to nearest interior sample.
        if i_peak <= 0 or i_peak >= n - 1:
            if n >= 3:
                interior = np.abs(y_rel[1 : n - 1])
                if np.any(np.isfinite(interior)):
                    i_peak = int(np.nanargmax(interior)) + 1
                else:
                    i_peak = n // 2
            else:
                out = (float("nan"), float("nan"), {"ok": False, "message": "peak at array edge"})
                return out if return_diagnostics else (out[0], out[1])

    peak_amp = float(abs(y_rel[i_peak]))
    # Prefer a wing continuum when the window is wide enough so a global
    # (even Δv-weighted) median does not sit near the peak.
    wing_min = max(20.0 * float(v_step), 2000.0)
    wing = np.abs(v_m_s - float(v_m_s[i_peak])) >= wing_min
    if int(np.count_nonzero(wing)) >= 5:
        baseline_w = _ccf_background_sigma_clipped_median_dv_weighted(v_m_s[wing], y[wing])
        if math.isfinite(baseline_w):
            amp_w = float(abs(float(y[i_peak]) - float(baseline_w)))
            if amp_w > peak_amp or peak_amp <= 0.0 or not math.isfinite(peak_amp):
                baseline = float(baseline_w)
                y_rel = y - baseline
                peak_amp = amp_w
    if peak_amp <= 0.0 or not math.isfinite(peak_amp):
        out = (float("nan"), float("nan"), {"ok": False, "message": "zero peak amplitude"})
        return out if return_diagnostics else (out[0], out[1])

    sign = 1.0 if y_rel[i_peak] >= 0.0 else -1.0
    y_peak_up = sign * y_rel

    p_lo = float(cfg.rv_percentile_lo)
    p_hi = float(cfg.rv_percentile_hi)
    p_step = float(cfg.rv_percentile_step)
    if p_step <= 0.0 or p_hi <= p_lo:
        out = (float("nan"), float("nan"), {"ok": False, "message": "invalid percentile grid"})
        return out if return_diagnostics else (out[0], out[1])

    percentiles = np.arange(p_lo, p_hi + 0.5 * p_step, p_step, dtype=np.float64)
    cuts: list[dict] = []
    for pct in percentiles:
        depth = float(pct) / 100.0
        row = _ccf_bisector_cut_m_s(
            v_m_s,
            y_peak_up,
            i_peak,
            depth,
            peak_amp,
            y_err=y_err,
            v_step_m_s=v_step,
        )
        if row is None:
            continue
        v_mid, v_mid_err, v_left, v_right, pct_out = row
        core_sig = float(cfg.core_weight_sigma_percentile)
        if core_sig <= 0.0:
            w_perf = 1.0
        else:
            w_perf = math.exp(-0.5 * ((pct_out - 50.0) / core_sig) ** 2)
        # Performance weight only. Inverse-variance cut weights are inappropriate
        # for strongly correlated percentile cuts and become unstable when a cut's
        # formal e_i collapses to the discretization floor.
        w_cut = w_perf
        cuts.append(
            {
                "percentile": float(pct_out),
                "v_mid_m_s": float(v_mid),
                "v_mid_err_m_s": float(v_mid_err),
                "v_left_m_s": float(v_left),
                "v_right_m_s": float(v_right),
                "weight": float(w_cut),
                "target_flux": float(depth * peak_amp),
            }
        )

    if len(cuts) < int(cfg.min_cuts):
        out = (
            float("nan"),
            float("nan"),
            {"ok": False, "message": f"too few bisector cuts ({len(cuts)})", "cuts": cuts},
        )
        return out if return_diagnostics else (out[0], out[1])

    w_arr = np.array([c["weight"] for c in cuts], dtype=np.float64)
    v_arr = np.array([c["v_mid_m_s"] for c in cuts], dtype=np.float64)
    w_sum = float(np.sum(w_arr))
    if w_sum <= 0.0 or not math.isfinite(w_sum):
        out = (float("nan"), float("nan"), {"ok": False, "message": "invalid weights", "cuts": cuts})
        return out if return_diagnostics else (out[0], out[1])

    rv = float(np.sum(w_arr * v_arr) / w_sum)
    e_arr = np.array([c["v_mid_err_m_s"] for c in cuts], dtype=np.float64)
    # Weighted mean of per-cut midpoint uncertainties. Cuts share one CCF, so do
    # **not** apply an independent-measurement :math:`1/\\sqrt{N}` reduction
    # (:math:`\\sqrt{\\sum (w e)^2}/\\sum w`); that understates σ by ~√N_eff.
    rv_err = float(np.sum(w_arr * e_arr) / w_sum)

    core_pct = float(cfg.rv_err_core_percentile)
    core_cut = min(cuts, key=lambda c: abs(c["percentile"] - core_pct))

    cut_spread = float(
        math.sqrt(max(float(np.average((v_arr - rv) ** 2, weights=w_arr / w_sum)), 0.0))
    )

    d_pos = np.diff(v_m_s)
    d_pos = d_pos[np.isfinite(d_pos) & (d_pos > 0)]
    d_min = float(np.min(d_pos)) if d_pos.size > 0 else float(v_step)
    vstep_floor_frac = float(cfg.rv_err_min_vstep_fraction)
    if vstep_floor_frac > 0.0 and math.isfinite(d_min) and d_min > 0.0:
        rv_err = max(rv_err, vstep_floor_frac * d_min)

    wing_noise = float(mad_std(y_rel, ignore_nan=True))
    if not math.isfinite(wing_noise) or wing_noise <= 0.0:
        wing_noise = float(np.nanstd(y_rel)) if y.size > 1 else peak_amp
    peak_contrast_sigma = float(peak_amp / max(wing_noise, 1e-30))
    min_contrast = float(cfg.min_peak_contrast_sigma)
    if min_contrast > 0.0 and peak_contrast_sigma < min_contrast:
        # Scale formal σ (do not floor to a multiple of the velocity grid step).
        rv_err *= float(min_contrast / max(peak_contrast_sigma, 1e-30))

    rv_err = max(rv_err, _ccf_rv_err_numeric_floor(v_step))
    boot_sig = float("nan")
    if has_ccf_err and int(cfg.rv_err_bootstrap_samples) > 0:
        y_model = CubicSpline(v_m_s, y, extrapolate=False)(v_m_s)
        rng = np.random.default_rng(cfg.rv_err_bootstrap_seed)
        boot_cfg = replace(cfg, rv_err_bootstrap_samples=0)
        ye_draw = np.where(np.isfinite(y_err), y_err, 0.0)
        boot_rvs: list[float] = []
        for _ in range(int(cfg.rv_err_bootstrap_samples)):
            y_boot = y_model + rng.normal(0.0, ye_draw)
            rv_b, _ = fit_ccf_rv_bisector(
                v_m_s,
                y_boot,
                v_step,
                ccf_err=y_err,
                config=boot_cfg,
            )
            if math.isfinite(rv_b):
                boot_rvs.append(float(rv_b))
        if len(boot_rvs) >= 2:
            boot_sig = float(np.std(boot_rvs, ddof=1))
            rv_err = max(rv_err, boot_sig)

    def _section_mean(lo: float, hi: float) -> tuple[float, float]:
        sel = [c for c in cuts if lo <= c["percentile"] <= hi]
        if not sel:
            return float("nan"), float("nan")
        ww = np.array([c["weight"] for c in sel], dtype=np.float64)
        vv = np.array([c["v_mid_m_s"] for c in sel], dtype=np.float64)
        ee = np.array([c["v_mid_err_m_s"] for c in sel], dtype=np.float64)
        wsum = float(np.sum(ww))
        if wsum <= 0.0:
            return float("nan"), float("nan")
        mean_v = float(np.sum(ww * vv) / wsum)
        mean_e = float(np.sum(ww * ee) / wsum)
        return mean_v, mean_e

    v_top, e_top = _section_mean(cfg.bis_top_lo, cfg.bis_top_hi)
    v_bot, e_bot = _section_mean(cfg.bis_bottom_lo, cfg.bis_bottom_hi)
    bis = float(v_top - v_bot) if math.isfinite(v_top) and math.isfinite(v_bot) else float("nan")
    bis_err = float(
        math.sqrt(e_top * e_top + e_bot * e_bot)
    ) if math.isfinite(e_top) and math.isfinite(e_bot) else float("nan")

    diag = {
        "ok": True,
        "rv_m_s": rv,
        "rv_err_m_s": rv_err,
        "rv_err_core_percentile": core_pct,
        "rv_err_core_cut_percentile": float(core_cut["percentile"]),
        "rv_err_core_cut_m_s": float(core_cut["v_mid_err_m_s"]),
        "rv_err_bootstrap_sigma_m_s": boot_sig,
        "used_rv_err_bootstrap": math.isfinite(boot_sig) and boot_sig > 0.0,
        "baseline": float(baseline),
        "peak_amplitude": float(peak_amp),
        "v_peak_m_s": float(v_m_s[i_peak]),
        "n_cuts": len(cuts),
        "cuts": cuts,
        "bis_m_s": bis,
        "bis_err_m_s": bis_err,
        "bis_top_mean_m_s": v_top,
        "bis_bottom_mean_m_s": v_bot,
        "method": "ccf_bisector_weighted_percentile",
        "cut_midpoint_spread_m_s": cut_spread,
        "wing_noise_mad": wing_noise,
        "peak_contrast_sigma": peak_contrast_sigma,
    }
    if return_diagnostics:
        return rv, rv_err, diag
    return rv, rv_err
def _refine_ccf_rv_bisector_after_gaussian_fit(
    v_grid_m_s: np.ndarray,
    ccf: np.ndarray,
    vstep: float,
    *,
    ccf_err: np.ndarray | None = None,
    config: CCFBisectorConfig | None = None,
    gaussian_seed_mean_m_s: float | None = None,
) -> tuple[float | None, float | None, dict | None]:
    """
    Bisector RV after :func:`fit_ccf_rv` (NEID-style weighted percentile cuts).

    Pass ``gaussian_seed_mean_m_s`` (Gaussian CCF center) so the bisector peak is
    taken near that velocity rather than a global ``argmax`` that can lock onto a
    window edge.

    Returns ``(None, None, diag)`` when bisector fitting fails; ``diag`` may still
    carry ``ok=False`` and cut metadata for previews.
    """
    v_step_eff = (
        float(vstep)
        if math.isfinite(vstep) and vstep > 0
        else effective_ccf_v_step_for_fit(v_grid_m_s)
    )
    rv_b, err_b, diag = fit_ccf_rv_bisector(
        v_grid_m_s,
        ccf,
        v_step_eff,
        ccf_err=ccf_err,
        config=config,
        return_diagnostics=True,
        gaussian_seed_mean_m_s=gaussian_seed_mean_m_s,
    )
    if not isinstance(diag, dict) or not diag.get("ok") or not math.isfinite(float(rv_b)):
        return None, None, diag if isinstance(diag, dict) else None
    err_out: float | None = None
    if err_b is not None and math.isfinite(float(err_b)) and float(err_b) > 0.0:
        err_out = float(err_b)
    return float(rv_b), err_out, diag


from __future__ import annotations

import numpy as np
from numba import njit

__all__ = [
    "calc_bisector_inverse_slope",
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
import numpy as np
from numba import njit

from hispecdrp.utils.physical_constants import C_M_S

__all__ = [
    "cubic_spline_build",
    "cubic_spline_eval",
    "cubic_spline_eval_scalar",
    "cubic_spline_eval_scalar_arrays",
    "cubic_spline2d_build",
    "cubic_spline2d_eval",
    "cubic_spline2d_eval_cell",
    "cubic_spline2d_eval_grid",
    "cubic_spline2d_eval_scalar",
    "cubic_spline2d_eval_scalar_arrays",
    "doppler_paint_cubic_spline",
]


@njit(cache=True)
def _solve_slopes_not_a_knot(x : np.ndarray, y : np.ndarray):
    """
    Return the first derivatives (slopes) k[i] at each knot for a not-a-knot
    cubic spline.

    Uses the slope (Hermite) formulation. Interior knots give the standard
    tridiagonal equations; the not-a-knot end conditions translate into clean
    two-term first/last rows, so the whole system is tridiagonal and (for
    sensible data) diagonally dominant — solvable by plain Thomas with no
    pivoting on arbitrary (irregular) spacing.

    Requires n >= 3.

    Parameters
    ----------
    x : np.ndarray
        1D array of strictly increasing knot locations.
    y : np.ndarray
        1D array of knot values.

    Returns
    -------
    slopes : np.ndarray
        1D array of slopes at each knot.
    """
    n = x.shape[0]
    h = np.empty(n - 1)
    d = np.empty(n - 1)   # secant slopes
    for i in range(n - 1):
        h[i] = x[i + 1] - x[i]
        d[i] = (y[i + 1] - y[i]) / h[i]

    lo = np.zeros(n)
    di = np.zeros(n)
    up = np.zeros(n)
    rh = np.zeros(n)

    # Interior rows (i = 1..n-2):
    #   h[i]*k[i-1] + 2*(h[i-1]+h[i])*k[i] + h[i-1]*k[i+1]
    #     = 3*(h[i]*d[i-1] + h[i-1]*d[i])
    for i in range(1, n - 1):
        lo[i] = h[i]
        di[i] = 2.0 * (h[i - 1] + h[i])
        up[i] = h[i - 1]
        rh[i] = 3.0 * (h[i] * d[i - 1] + h[i - 1] * d[i])

    if n == 3:
        # Not-a-knot with 3 points => single parabola; its slope system is
        # well-defined with the simple natural-like closure that reproduces the
        # quadratic. Use end conditions k0 and k2 from the parabola through the
        # three points.
        # Parabola p(x); slopes at ends:
        h0 = h[0]; h1 = h[1]
        # second derivative (constant) of the interpolating parabola:
        a2 = (d[1] - d[0]) / (h0 + h1)
        k0 = d[0] - a2 * h0
        k2 = d[1] + a2 * h1
        di[0] = 1.0; rh[0] = k0
        di[2] = 1.0; rh[2] = k2
    else:
        # Not-a-knot left: third derivative continuous across x[1].
        # Closed form (de Boor): 
        #   k[0]*h[1]*(h[0]+h[1]) ... -> the standard reduced row is
        #   k0*h1 + k1*(h0+h1) = ((h0+2*(h0+h1))*h1*d0 + h0^2*d1)/(h0+h1)
        h0 = h[0]; h1 = h[1]
        di[0] = h1
        up[0] = h0 + h1
        rh[0] = ((h0 + 2.0 * (h0 + h1)) * h1 * d[0] + h0 * h0 * d[1]) / (h0 + h1)

        # Not-a-knot right: symmetric across x[n-2].
        hm1 = h[n - 2]; hm2 = h[n - 3]
        lo[n - 1] = hm2 + hm1
        di[n - 1] = hm2
        rh[n - 1] = ((hm1 + 2.0 * (hm2 + hm1)) * hm2 * d[n - 2]
                     + hm1 * hm1 * d[n - 3]) / (hm2 + hm1)

    # Thomas algorithm
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = up[0] / di[0]
    dp[0] = rh[0] / di[0]
    for i in range(1, n):
        m = di[i] - lo[i] * cp[i - 1]
        if i < n - 1:
            cp[i] = up[i] / m
        dp[i] = (rh[i] - lo[i] * dp[i - 1]) / m
    slopes = np.empty(n)
    slopes[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        slopes[i] = dp[i] - cp[i] * slopes[i + 1]
    return slopes


@njit(cache=True)
def _coeffs_from_slopes(x : np.ndarray, y : np.ndarray, slopes : np.ndarray):
    """
    Convert knot slopes into per-segment (a,b,c,d) Horner coefficients
    about the left knot of each segment. n-1 segments.

    On segment i with width h and secant slope dd=(y[i+1]-y[i])/h:
        a = y[i]
        b = slopes[i]
        c = (3*dd - 2*slopes[i] - slopes[i+1]) / h
        d = (slopes[i] + slopes[i+1] - 2*dd) / h**2

    Parameters
    ----------
    x : np.ndarray
        1D array of strictly increasing knot locations.
    y : np.ndarray
        1D array of knot values.
    slopes : np.ndarray
        1D array of slopes at each knot.
    """
    n = x.shape[0]
    a = np.empty(n - 1)
    b = np.empty(n - 1)
    cc = np.empty(n - 1)
    d = np.empty(n - 1)
    for i in range(n - 1):
        h = x[i + 1] - x[i]
        dd = (y[i + 1] - y[i]) / h
        a[i] = y[i]
        b[i] = slopes[i]
        cc[i] = (3.0 * dd - 2.0 * slopes[i] - slopes[i + 1]) / h
        d[i] = (slopes[i] + slopes[i + 1] - 2.0 * dd) / (h * h)
    return a, b, cc, d


@njit(cache=True)
def cubic_spline_build(x : np.ndarray, y : np.ndarray):
    """
    Build & cache the not-a-knot cubic spline.

    Returns a tuple (x, a, b, c, d) — pass this same tuple to the eval funcs.
    Call once; reuse for many evaluations.

    Parameters
    ----------
    x : np.ndarray
        1D array of strictly increasing knot locations.
    y : np.ndarray
        1D array of knot values.
    """
    k = _solve_slopes_not_a_knot(x, y)
    a, b, c, d = _coeffs_from_slopes(x, y, k)
    return x, a, b, c, d


@njit(cache=True, inline="always")
def _find_segment(xk : np.ndarray, xq : float):
    """Binary search: largest i with xk[i] <= xq, clamped to [0, n-2]."""
    n = xk.shape[0]
    if xq <= xk[0]:
        return 0
    if xq >= xk[n - 1]:
        return n - 2
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if xk[mid] <= xq:
            lo = mid
        else:
            hi = mid
    return lo


@njit(cache=True, nogil=True, inline="always")
def cubic_spline_eval_scalar_arrays(
    xk: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    xq: float,
) -> float:
    """Evaluate a spline from unpacked coefficient arrays (Numba-friendly)."""
    i = _find_segment(xk, xq)
    t = xq - xk[i]
    return a[i] + t * (b[i] + t * (c[i] + t * d[i]))


@njit(cache=True)
def cubic_spline_eval_scalar(cache : tuple, xq : float):
    """
    Evaluate the cached spline at a single scalar point.
    Points outside the knot range are extrapolated with the end cubics.

    Parameters
    ----------
    cache : tuple
        The tuple returned by `cubic_spline_build`.
    xq : float
        The query point at which to evaluate the spline.

    Returns
    -------
    float
        The spline value at the query point.
    """
    xk, a, b, c, d = cache
    i = _find_segment(xk, xq)
    t = xq - xk[i]
    return a[i] + t * (b[i] + t * (c[i] + t * d[i]))


@njit(cache=True)
def cubic_spline_eval(cache : tuple, xq : np.ndarray):
    """
    Evaluate the cached spline at an array of sorted query points.
    Points outside the knot range are extrapolated with the end cubics.

    Parameters
    ----------
    cache : tuple
        The tuple returned by `cubic_spline_build`.
    xq : np.ndarray
        The query points at which to evaluate the spline.

    Returns
    -------
    np.ndarray
        The spline value at the query point.
    """
    xk, a, b, c, d = cache
    n = xk.shape[0]
    m = xq.shape[0]
    out = np.empty(m)
    i = 0
    for k in range(m):
        xx = xq[k]
        # advance segment pointer while the next knot is still <= xx
        while i < n - 2 and xk[i + 1] <= xx:
            i += 1
        # (no need to walk backwards: xq is sorted ascending)
        t = xx - xk[i]
        out[k] = a[i] + t * (b[i] + t * (c[i] + t * d[i]))
    return out


@njit(cache=True)
def cubic_spline2d_build(x : np.ndarray, y : np.ndarray, z : np.ndarray):
    """
    Build & cache the tensor-product not-a-knot cubic spline of a 2D grid.

    The knot slopes come from the 1D not-a-knot solver: along each row
    (d/dx), along each column (d/dy), and along each column of the x-slopes
    (d2/dxdy). Together with the values these define a bicubic patch on
    every cell, which is exactly the tensor-product spline.

    Returns a tuple (x, y, z, zx, zy, zxy) — pass this same tuple to the
    2D eval funcs. Call once; reuse for many evaluations.

    Requires at least 3 knots along each axis.

    Parameters
    ----------
    x : np.ndarray
        1D array of strictly increasing knot locations along the columns (length Ncols).
    y : np.ndarray
        1D array of strictly increasing knot locations along the rows (length Nrows).
    z : np.ndarray
        2D array of knot values with shape (Nrows, Ncols), z[j, i] at (x[i], y[j]).
    """
    Nrows, Ncols = z.shape
    zx = np.empty((Nrows, Ncols))
    zy = np.empty((Nrows, Ncols))
    zxy = np.empty((Nrows, Ncols))
    for j in range(Nrows):
        zx[j, :] = _solve_slopes_not_a_knot(x, z[j, :].copy())
    for i in range(Ncols):
        zy[:, i] = _solve_slopes_not_a_knot(y, z[:, i].copy())
        zxy[:, i] = _solve_slopes_not_a_knot(y, zx[:, i].copy())
    return x, y, z, zx, zy, zxy


@njit(cache=True, nogil=True, inline="always")
def _cubic_from_ends(f0 : float, s0 : float, f1 : float, s1 : float, h : float, t : float):
    """
    Cubic through (0, f0) and (h, f1) with end slopes s0 and s1, evaluated at offset t
    from the left end. Same Horner coefficients as `_coeffs_from_slopes`, so extrapolation
    far outside the segment does not lose precision.
    """
    dd = (f1 - f0) / h
    c = (3.0 * dd - 2.0 * s0 - s1) / h
    d = (s0 + s1 - 2.0 * dd) / (h * h)
    return f0 + t * (s0 + t * (c + t * d))


@njit(cache=True, nogil=True, inline="always")
def cubic_spline2d_eval_scalar_arrays(
    xk: np.ndarray,
    yk: np.ndarray,
    z: np.ndarray,
    zx: np.ndarray,
    zy: np.ndarray,
    zxy: np.ndarray,
    xq: float,
    yq: float,
) -> float:
    """Evaluate a 2D spline from unpacked arrays (Numba-friendly)."""
    i = _find_segment(xk, xq)
    j = _find_segment(yk, yq)
    return cubic_spline2d_eval_cell(
        z, zx, zy, zxy, i, j, xk[i + 1] - xk[i], yk[j + 1] - yk[j], xq - xk[i], yq - yk[j]
    )


@njit(cache=True, nogil=True, inline="always")
def cubic_spline2d_eval_cell(
    z: np.ndarray,
    zx: np.ndarray,
    zy: np.ndarray,
    zxy: np.ndarray,
    i: int,
    j: int,
    hx: float,
    hy: float,
    tx: float,
    ty: float,
) -> float:
    """
    Evaluate a 2D spline (unpacked from `cubic_spline2d_build`) on a known cell (Numba-friendly).

    The cell spans knots i, i+1 along x (width hx) and j, j+1 along y (width hy); tx and ty are the
    offsets of the query point from knot (i, j). Use this when the cell is known without a search,
    e.g. on a uniform grid; offsets outside [0, hx] x [0, hy] extrapolate that cell's bicubic.
    """
    # Interpolate values and y-slopes along x on rows j and j+1, then along y
    v0 = _cubic_from_ends(z[j, i], zx[j, i], z[j, i + 1], zx[j, i + 1], hx, tx)
    v1 = _cubic_from_ends(z[j + 1, i], zx[j + 1, i], z[j + 1, i + 1], zx[j + 1, i + 1], hx, tx)
    s0 = _cubic_from_ends(zy[j, i], zxy[j, i], zy[j, i + 1], zxy[j, i + 1], hx, tx)
    s1 = _cubic_from_ends(zy[j + 1, i], zxy[j + 1, i], zy[j + 1, i + 1], zxy[j + 1, i + 1], hx, tx)
    return _cubic_from_ends(v0, s0, v1, s1, hy, ty)


@njit(cache=True)
def cubic_spline2d_eval_scalar(cache : tuple, xq : float, yq : float):
    """
    Evaluate the cached 2D spline at a single point.
    Points outside the knot range are extrapolated with the end cubics.

    Parameters
    ----------
    cache : tuple
        The tuple returned by `cubic_spline2d_build`.
    xq : float
        The query x location.
    yq : float
        The query y location.

    Returns
    -------
    float
        The spline value at the query point.
    """
    xk, yk, z, zx, zy, zxy = cache
    return cubic_spline2d_eval_scalar_arrays(xk, yk, z, zx, zy, zxy, xq, yq)


@njit(cache=True)
def cubic_spline2d_eval(cache : tuple, xq : np.ndarray, yq : np.ndarray):
    """
    Evaluate the cached 2D spline at an array of points (xq[k], yq[k]), in any order.
    Points outside the knot range are extrapolated with the end cubics.

    Parameters
    ----------
    cache : tuple
        The tuple returned by `cubic_spline2d_build`.
    xq : np.ndarray
        1D array of query x locations.
    yq : np.ndarray
        1D array of query y locations, same length as xq.

    Returns
    -------
    np.ndarray
        The spline values at the query points.
    """
    xk, yk, z, zx, zy, zxy = cache
    m = xq.shape[0]
    out = np.empty(m)
    for k in range(m):
        out[k] = cubic_spline2d_eval_scalar_arrays(xk, yk, z, zx, zy, zxy, xq[k], yq[k])
    return out


@njit(cache=True)
def cubic_spline2d_eval_grid(cache : tuple, xq : np.ndarray, yq : np.ndarray):
    """
    Evaluate the cached 2D spline on the grid of all (xq[i], yq[j]).
    Points outside the knot range are extrapolated with the end cubics.

    Parameters
    ----------
    cache : tuple
        The tuple returned by `cubic_spline2d_build`.
    xq : np.ndarray
        1D array of query x locations (length Ncols).
    yq : np.ndarray
        1D array of query y locations (length Nrows).

    Returns
    -------
    np.ndarray
        The spline values with shape (Nrows, Ncols), out[j, i] at (xq[i], yq[j]).
    """
    xk, yk, z, zx, zy, zxy = cache
    out = np.empty((yq.shape[0], xq.shape[0]))
    for j in range(yq.shape[0]):
        for i in range(xq.shape[0]):
            out[j, i] = cubic_spline2d_eval_scalar_arrays(xk, yk, z, zx, zy, zxy, xq[i], yq[j])
    return out


def doppler_paint_cubic_spline(
    wave_obs_um: np.ndarray,
    wave_rest_um: np.ndarray,
    flux_rest: np.ndarray,
    rv_m_s: float,
    *,
    zb: float = 0.0,
) -> np.ndarray:
    """
    Resample a rest-frame spectrum onto an observed wavelength grid via cubic spline.

    Uses the same not-a-knot cubic spline as
    :func:`~hispecdrp.primitives.utils.ccf.ccf_spline.compute_ccf_spline`
    (alternate / research path). For paint that matches the **science**
    ``ccf_line_mask`` sampler, use
    :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.doppler_paint_lagrange`
    instead (Numba Lanczos / ``mask_ccf_pixel_interp_order``).

    At injected radial velocity ``rv_m_s`` and barycentric factor ``zb``,

    .. math::

        F_{\\rm obs}(\\lambda) = F_{\\rm rest}\\!\\left(
            \\lambda\\,\\frac{1+z_b}{1+v/c}
        \\right)

    which is the inverse of the mask Doppler map
    :math:`\\lambda_{\\rm obs}=\\lambda_{\\rm rest}(1+v/c)/(1+z_b)`.

    Parameters
    ----------
    wave_obs_um
        Observed wavelengths (µm), **strictly ascending** (as for one echelle order).
    wave_rest_um, flux_rest
        Rest-frame knots (µm) and fluxes; ``wave_rest_um`` must be strictly ascending.
    rv_m_s
        Injected stellar RV (m s⁻¹).
    zb
        Barycentric redshift factor (same convention as the mask CCF).

    Returns
    -------
    np.ndarray
        Flux on ``wave_obs_um`` (float64).
    """
    w_obs = np.asarray(wave_obs_um, dtype=np.float64).ravel()
    w_rest = np.asarray(wave_rest_um, dtype=np.float64).ravel()
    f_rest = np.asarray(flux_rest, dtype=np.float64).ravel()
    if w_rest.size != f_rest.size:
        raise ValueError("wave_rest_um and flux_rest must have the same length")
    if w_rest.size < 3:
        raise ValueError("need at least 3 rest-frame knots for cubic spline")
    if w_obs.size and not np.all(np.diff(w_obs) > 0.0):
        raise ValueError("wave_obs_um must be strictly ascending")
    if not np.all(np.diff(w_rest) > 0.0):
        raise ValueError("wave_rest_um must be strictly ascending")

    cache = cubic_spline_build(w_rest, f_rest)
    beta = (1.0 + float(zb)) / (1.0 + float(rv_m_s) / float(C_M_S))
    return cubic_spline_eval(cache, w_obs * beta)

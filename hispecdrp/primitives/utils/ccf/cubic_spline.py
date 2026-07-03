import numpy as np
from numba import njit

__all__ = [
    "cubic_spline_build",
    "cubic_spline_eval",
    "cubic_spline_eval_scalar",
    "cubic_spline_eval_scalar_arrays",
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

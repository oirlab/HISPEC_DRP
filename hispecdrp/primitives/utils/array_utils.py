"""Generic Numba / array / polynomial helpers for HISPEC DRP primitives.

Polynomial evaluation, Chebyshev OLS fits, short-circuit any/all, median filters,
and related array utilities formerly housed in ``math.py``.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from numpy.polynomial.chebyshev import chebvander2d

__all__ = [
    "all_sc",
    "any_sc",
    "bitwise_or_reduce_uint32",
    "median_filter_1d",
    "median_filter_2d",
    "nanmedian_axis0_ignore_empty_columns",
    "polyval1d",
    "solve_chebyshev2d_ols",
]

@njit(nogil=True)
def polyval1d(coeffs : np.ndarray, x : np.ndarray) -> np.ndarray:
    """
    Evaluate a polynomial with given coefficients at specified points using the Horner method.

    Args:
        coeffs (np.ndarray): Coefficients of the polynomial, ordered from lowest degree to highest (NOTE: numpy is reversed).
        x (np.ndarray): Points at which to evaluate the polynomial.
    """
    v = np.zeros(x.shape, dtype=coeffs.dtype)
    for coeff in coeffs[::-1]:
        v = v * x + coeff
    return v

def solve_chebyshev2d_ols(
    x : np.ndarray, y : np.ndarray, z : np.ndarray,
    deg_x : int, deg_y : int,
    weights : np.ndarray | None = None,
) -> np.ndarray:
    """
    Fit a 2D Chebyshev polynomial using ordinary (uncorrelated) weighted least squares.

    Args:
        x: 1D array of x coordinates (length N).
        y: 1D array of y coordinates (length N).
        z (np.ndarray): 1D array of values at coordinates (x,y) (length N).
        deg_x (int): X Degree of the Chebyshev polynomial.
        deg_y (int): Y Degree of the Chebyshev polynomial.
        weights (np.ndarray | None): 1D array of weights (length N). If None, uniform weights are used.

    Returns:
        coeffs: 2D array of Chebyshev coefficients of shape (deg_x+1, deg_y+1)
    """

    # Construct the 2D Vandermonde matrix for Chebyshev basis
    V = chebvander2d(x, y, [deg_x, deg_y])  # shape: (N, (deg_x+1)*(deg_y+1))

    # Apply weights, if provided
    if weights is not None:
        W = np.sqrt(weights)
        V *= W[:, None]
        z = z * W

    # Solve the least squares problem
    coeffs_flat, *_ = np.linalg.lstsq(V, z, rcond=None)
    coeffs = coeffs_flat.reshape((deg_x + 1, deg_y + 1))
    return coeffs

@njit(nogil=True)
def all_sc(arr : np.ndarray) -> bool:
    """
    Optimized version of np.all which short circuits, unlike numpy.all.
    """
    for item in arr.ravel():
        if item:
            continue
        else:
            return False
    return True

@njit(nogil=True)
def any_sc(arr : np.ndarray) -> bool:
    """
    Optimized version of np.any which short circuits, unlike numpy.any.
    """
    for item in arr.ravel():
        if item:
            return True
    return False

@njit(nogil=True, cache=True, parallel=False)
def bitwise_or_reduce_uint32(arr: np.ndarray) -> np.uint32:
    result = np.uint32(0)
    for k in range(arr.shape[0]):
        result |= arr[k]
    return result

@njit(nogil=True)
def _calc_median_insertion_sort(vals, m):
    # insertion sort first m entries, then return median
    for i in range(1, m):
        key = vals[i]
        j = i - 1
        while j >= 0 and vals[j] > key:
            vals[j + 1] = vals[j]
            j -= 1
        vals[j + 1] = key
    if m == 0:
        return np.nan
    if m % 2 == 1:
        return vals[m // 2]
    else:
        a = vals[(m // 2) - 1]
        b = vals[m // 2]
        return 0.5 * (a + b)

@njit(nogil=True)
def median_filter_1d(data : np.ndarray, dq : np.ndarray, window_size : int):
    n = data.shape[0]
    out = np.empty(n, dtype=data.dtype)
    half = window_size // 2
    # temporary array sized to window for collecting valid values
    temp = np.empty(window_size, dtype=data.dtype)
    for i in range(n):
        m = 0
        start = i - half
        end = i + half
        for j in range(start, end + 1):
            if 0 <= j < n and dq[j] == 0:
                temp[m] = data[j]
                m += 1
        if m == 0:
            out[i] = np.nan
        else:
            out[i] = _calc_median_insertion_sort(temp, m)
    return out

@njit(nogil=True)
def median_filter_2d(data, dq, window_size : tuple[int, int]):
    ny, nx = data.shape
    out = np.empty((ny, nx), dtype=data.dtype)
    hy = window_size[0] // 2
    hx = window_size[1] // 2
    temp = np.empty(window_size[0] * window_size[1], dtype=data.dtype)

    for iy in range(ny):
        for ix in range(nx):
            m = 0
            for jy in range(iy - hy, iy + hy + 1):
                if jy < 0 or jy >= ny:
                    continue
                for jx in range(ix - hx, ix + hx + 1):
                    if jx < 0 or jx >= nx:
                        continue
                    if dq[jy, jx] == 0:
                        temp[m] = data[jy, jx]
                        m += 1
            if m == 0:
                out[iy, ix] = np.nan
            else:
                out[iy, ix] = _calc_median_insertion_sort(temp, m)
    return out

def nanmedian_axis0_ignore_empty_columns(arr: np.ndarray) -> np.ndarray:
    """
    Column-wise median of finite values (axis 0). Columns with no finite entries are NaN.

    ``np.nanmedian(..., axis=0)`` emits ``RuntimeWarning: All-NaN slice`` when entire
    columns are NaN; this matches ``nanmedian`` for mixed finite/NaN columns but avoids
    that warning.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("nanmedian_axis0_ignore_empty_columns: expected a 2-D array")
    n_row, n_col = a.shape
    if n_col == 0:
        return np.array([], dtype=np.float64)
    if n_row == 0:
        return np.full(n_col, np.nan, dtype=np.float64)
    fin = np.isfinite(a)
    cnt = np.sum(fin, axis=0)
    out = np.full(n_col, np.nan, dtype=np.float64)
    for j in range(n_col):
        if cnt[j] > 0:
            out[j] = float(np.median(a[fin[:, j], j]))
    return out


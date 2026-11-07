import numpy as np
from numba import njit

from hispecdrp.primitives.utils.cubic_spline import _find_segment

__all__ = [
    "linear_interp2d_eval",
    "linear_interp2d_eval_cell",
    "linear_interp2d_eval_grid",
    "linear_interp2d_eval_scalar",
    "linear_interp2d_from_corners",
]


@njit(cache=True, nogil=True, inline="always")
def linear_interp2d_from_corners(f00 : float, f01 : float, f10 : float, f11 : float, tx : float, ty : float):
    """
    Bilinear blend of the four corner values of a cell (Numba-friendly).

    f00 is at (x0, y0), f01 at (x1, y0), f10 at (x0, y1) and f11 at (x1, y1); tx and ty are the
    fractional offsets of the query point along x and y, in units of the cell width. Values of
    tx or ty outside [0, 1] extrapolate linearly. Use this when the corner values are computed on
    the fly rather than read from a grid.
    """
    return (f00 * (1.0 - tx) + f01 * tx) * (1.0 - ty) + (f10 * (1.0 - tx) + f11 * tx) * ty


@njit(cache=True, nogil=True, inline="always")
def linear_interp2d_eval_cell(z : np.ndarray, i : int, j : int, tx : float, ty : float):
    """
    Bilinear interpolation of a 2D grid z (Nrows, Ncols) on a known cell (Numba-friendly).

    The cell spans knots i, i+1 along x and j, j+1 along y; tx and ty are the fractional offsets of
    the query point from knot (i, j), in units of the cell width. Use this when the cell is known
    without a search, e.g. on a uniform grid; offsets outside [0, 1] extrapolate that cell linearly.
    """
    return linear_interp2d_from_corners(z[j, i], z[j, i + 1], z[j + 1, i], z[j + 1, i + 1], tx, ty)


@njit(cache=True)
def linear_interp2d_eval_scalar(x : np.ndarray, y : np.ndarray, z : np.ndarray, xq : float, yq : float):
    """
    Bilinear interpolation of a 2D grid at a single point.
    Points outside the knot range are extrapolated linearly from the edge cells.

    Parameters
    ----------
    x : np.ndarray
        1D array of strictly increasing knot locations along the columns (length Ncols >= 2).
    y : np.ndarray
        1D array of strictly increasing knot locations along the rows (length Nrows >= 2).
    z : np.ndarray
        2D array of knot values with shape (Nrows, Ncols), z[j, i] at (x[i], y[j]).
    xq : float
        The query x location.
    yq : float
        The query y location.

    Returns
    -------
    float
        The interpolated value at the query point.
    """
    i = _find_segment(x, xq)
    j = _find_segment(y, yq)
    tx = (xq - x[i]) / (x[i + 1] - x[i])
    ty = (yq - y[j]) / (y[j + 1] - y[j])
    return linear_interp2d_eval_cell(z, i, j, tx, ty)


@njit(cache=True)
def linear_interp2d_eval(x : np.ndarray, y : np.ndarray, z : np.ndarray, xq : np.ndarray, yq : np.ndarray):
    """
    Bilinear interpolation of a 2D grid at an array of points (xq[k], yq[k]), in any order.
    Points outside the knot range are extrapolated linearly from the edge cells.

    Parameters
    ----------
    x : np.ndarray
        1D array of strictly increasing knot locations along the columns (length Ncols >= 2).
    y : np.ndarray
        1D array of strictly increasing knot locations along the rows (length Nrows >= 2).
    z : np.ndarray
        2D array of knot values with shape (Nrows, Ncols), z[j, i] at (x[i], y[j]).
    xq : np.ndarray
        1D array of query x locations.
    yq : np.ndarray
        1D array of query y locations, same length as xq.

    Returns
    -------
    np.ndarray
        The interpolated values at the query points.
    """
    m = xq.shape[0]
    out = np.empty(m)
    for k in range(m):
        out[k] = linear_interp2d_eval_scalar(x, y, z, xq[k], yq[k])
    return out


@njit(cache=True)
def linear_interp2d_eval_grid(x : np.ndarray, y : np.ndarray, z : np.ndarray, xq : np.ndarray, yq : np.ndarray):
    """
    Bilinear interpolation of a 2D grid on the grid of all (xq[i], yq[j]).
    Points outside the knot range are extrapolated linearly from the edge cells.

    Parameters
    ----------
    x : np.ndarray
        1D array of strictly increasing knot locations along the columns (length Ncols >= 2).
    y : np.ndarray
        1D array of strictly increasing knot locations along the rows (length Nrows >= 2).
    z : np.ndarray
        2D array of knot values with shape (Nrows, Ncols), z[j, i] at (x[i], y[j]).
    xq : np.ndarray
        1D array of query x locations (length Ncols_out).
    yq : np.ndarray
        1D array of query y locations (length Nrows_out).

    Returns
    -------
    np.ndarray
        The interpolated values with shape (Nrows_out, Ncols_out), out[j, i] at (xq[i], yq[j]).
    """
    out = np.empty((yq.shape[0], xq.shape[0]))
    for j in range(yq.shape[0]):
        for i in range(xq.shape[0]):
            out[j, i] = linear_interp2d_eval_scalar(x, y, z, xq[i], yq[j])
    return out

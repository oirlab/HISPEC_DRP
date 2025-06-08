import numpy as np
from numpy.polynomial.chebyshev import chebvander2d

__all__ = ['solve_chebyshev2d']


def solve_chebyshev2d(x : np.ndarray, y : np.ndarray, z : np.ndarray, weights : np.ndarray, deg_x : int, deg_y : int):
    """
    Fit a 2D Chebyshev polynomial using weighted least squares.

    Args:
        x, y: 1D arrays of coordinates (length N)
        z: 1D array of values at those coordinates (length N)
        weights: 1D array of weights (length N).
        deg_x, deg_y: degrees of the Chebyshev polynomial in x and y

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

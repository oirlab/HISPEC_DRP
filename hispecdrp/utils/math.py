import numpy as np
import scipy.constants
from numpy.polynomial.chebyshev import chebvander2d
from numba import njit

__all__ = [
    'delta_wave_to_delta_v', 'delta_v_to_delta_wave', 'delta_pix_to_delta_wave',
    'polyval1d', 'solve_chebyshev2d',
]

def delta_wave_to_delta_v(dw, w):
    """
    Converts a wavelength difference to a velocity difference.

    Args:
        dw (float): Wavelength difference.
        w (float): Wavelength.

    Returns:
        float: Velocity difference.
    """
    return dw / w * scipy.constants.c

def delta_v_to_delta_wave(dv, w):
    """
    Converts a velocity difference to a wavelength difference.

    Args:
        dv (float): Velocity difference.
        w (float): Wavelength.

    Returns:
        float: Wavelength difference.
    """
    return dv / scipy.constants.c * w

def delta_pix_to_delta_wave(dx, x, wavesol=None, dispersion=None):
    """
    Converts a pixel difference to a wavelength difference.

    Args:
        dx (float): Pixel difference.
        x (float): Pixel value.
        wavesol (np.ndarray, optional): Wavelength solution array. Defaults to None.
        dispersion (np.ndarray, optional): Dispersion array. Defaults to None.

    Returns:
        float: Wavelength difference.
    """
    if dispersion is None:
        dispersion = np.gradient(wavesol)
    disp = np.interp(x, np.arange(len(dispersion)), dispersion)
    dw = disp * dx
    return dw

@njit(nogil=True)
def polyval1d(coeffs, x):
    v = np.zeros_like(x)
    for coeff in coeffs[::-1]:
        v = v * x + coeff
    return v

def solve_chebyshev2d_ols(
    x : np.ndarray, y : np.ndarray, z : np.ndarray,
    deg_x : int, deg_y : int,
    weights : np.ndarray | None = None,
) -> np.ndarray:
    """
    Fit a 2D Chebyshev polynomial using ordinary (uncorrelated) least squares.

    Args:
        x, y (np.ndarray): 1D arrays of coordinates (length N).
        z (np.ndarray): 1D array of values at those coordinates (length N).
        deg_x, deg_y (int): degrees of the Chebyshev polynomial in x and y.
        weights (np.ndarray, optional): 1D array of weights (length N). Defaults to uniform weights.

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
from numba import njit, prange
import numpy as np

__all__ = ['correct_nonlinearity']


@njit
def correct_nonlinearity(ramps, coeffs):
    ny, nx, _ = ramps.shape
    for i in range(ny):
        for j in range(nx):
            ramps[i, j, :] = polyval(coeffs[i, j, :], ramps[i, j, :])
    return ramps


@njit
def polyval(coeffs, x):
    v = np.zeros_like(x)
    for coeff in coeffs[::-1]:
        v = v * x + coeff
    return v
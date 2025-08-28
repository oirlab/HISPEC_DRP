from numba import njit
from ...utils.math import polyval1d

__all__ = ['correct_nonlinearity']

@njit(nogil=True)
def correct_nonlinearity(ramps, coeffs):
    _, ny, nx = ramps.shape
    for i in range(ny):
        for j in range(nx):
            ramps[:, i, j] = polyval1d(coeffs[:, i, j], ramps[:, i, j])
    return ramps

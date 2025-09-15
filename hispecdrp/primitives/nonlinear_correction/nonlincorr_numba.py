from numba import njit
from ...utils.math import polyval1d

__all__ = ['correct_nonlinearity']

@njit(nogil=True)
def correct_nonlinearity(ramps, coeffs):
    """
    Apply non-linearity correction to up-the-ramps.
    Note: First read is subtracted before applying the polynomial correction, but it is added back after such that the
    output ramp still includes the bias.

    Args:
        ramps (np.ndarray): 3D array of shape (nints, ny, nx) containing the ramps.
        coeffs (np.ndarray): 3D array of shape (ncoeffs, ny, nx) containing the polynomial coefficients of P().
            such that, "corrected ramp (DN)" = P("measured ramp (DN)").

    Returns:
        np.ndarray: The corrected ramps (same shape as input).

    """
    _, ny, nx = ramps.shape
    for i in range(ny):
        for j in range(nx):
            ramps[:, i, j] = ramps[0, i, j]+polyval1d(coeffs[:, i, j], ramps[:, i, j]-ramps[0, i, j])
    return ramps

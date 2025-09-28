from numba import njit, prange
from ...utils.math import polyval1d
import numpy as np
from ...datamodels.dqflags import DQ_FLAGS

__all__ = ['correct_nonlinearity']

NON_LINEAR_FLAG =  DQ_FLAGS["NON_LINEAR"]
NON_LINEAR_FLAG_uint8 =  np.uint8(DQ_FLAGS["NON_LINEAR"])

@njit(nogil=True,parallel=True,cache=True)
def correct_nonlinearity(ramps, dq_raw, dq, coeffs, nonlin_max, nonlin_dq):
    """
    Apply non-linearity correction to up-the-ramps.
    Note: First read is subtracted before applying the polynomial correction, but it is added back after such that the
    output ramp still includes the bias.

    Args:
        ramps (np.ndarray): 3D array of shape (n_reads, ny, nx) containing the ramps.
        dq_raw (np.ndarray): 3D array of shape (n_reads, ny, nx) containing the raw DQ flags.
            Any reads beyong `nonlin_max` will be flagged as "non linear" and not trustworthy.
        dq (np.ndarray): 2D array of shape (ny, nx) containing the DQ flags to be updated with non-linearity flags.
        coeffs (np.ndarray): 3D array of shape (ncoeffs, ny, nx) containing the polynomial coefficients of P().
            such that, "corrected ramp (DN)" = P("measured ramp (DN)").
        nonlin_max (float): The maximum DN value for which the non-linearity correction is good to within 1%.
        nonlin_dq (np.ndarray): 2D array of shape (ny, nx) containing the DQ flags from the non-linearity model
            to be applied to all pixels.

    Returns:

    """
    n_reads, ny, nx = ramps.shape
    for i in prange(ny):
        for j in range(nx):
            nl_corr_ramp = ramps[0, i, j] + polyval1d(coeffs[:, i, j],
                                       (ramps[:, i, j] - ramps[0, i, j]).astype(coeffs.dtype))
            ramps[:, i, j] = nl_corr_ramp.astype(ramps.dtype)
            for k in range(n_reads):
                if ramps[k, i, j] > nonlin_max[i,j]:
                    dq_raw[k, i, j] |= NON_LINEAR_FLAG_uint8
            dq[i,j] |= nonlin_dq[i,j]
    return

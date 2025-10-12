from numba import njit, prange
from ...utils.math import polyval1d
import numpy as np
from ...datamodels.dqflags import DQ_FLAGS

__all__ = ['correct_nonlinearity']

NON_LINEAR_FLAG_uint8 = np.uint8(DQ_FLAGS["NON_LINEAR"])

@njit(nogil=True, parallel=True, cache=True)
def _correct_nonlinearity_numba(
    ramps : np.ndarray,
    dq_raw : np.ndarray,
    dq : np.ndarray,
    coeffs : np.ndarray,
    nonlin_max : np.ndarray,
    nonlin_dq : np.ndarray
):
    n_reads, ny, nx = ramps.shape
    for i in prange(ny):
        for j in range(nx):
            first_read = ramps[0, i, j]
            nl_corr_ramp = first_read + polyval1d(coeffs[:, i, j], ramps[:, i, j] - first_read)
            ramps[:, i, j] = nl_corr_ramp
            for k in range(n_reads):
                if ramps[k, i, j] > nonlin_max[i, j]:
                    dq_raw[k, i, j] |= NON_LINEAR_FLAG_uint8
            dq[i, j] |= nonlin_dq[i, j]
    return


def correct_nonlinearity(
    ramps : np.ndarray,
    dq_raw : np.ndarray,
    dq : np.ndarray,
    coeffs : np.ndarray,
    nonlin_max : np.ndarray,
    nonlin_dq : np.ndarray
):
    """
    Apply non-linearity correction to up-the-ramps. Correction is done in place.
    The first read is subtracted before applying the polynomial correction, but it is added back after such that the output ramp still includes the bias.
    The dq arrays are updated in place to flag any reads that exceed the non-linearity limit.

    Parameters
    ----------
    ramps
        3D array of shape (Nreads, Ny, Nx) containing the ramps. Must be floating point.
    dq_raw
        3D array of shape (Nreads, Ny, Nx) containing the raw DQ flags.
        Any reads beyong `nonlin_max` will be flagged as "non linear" and not trustworthy.
    dq
        2D array of shape (Ny, Nx) containing the DQ flags to be updated with non-linearity flags.
    coeffs
        3D array of shape (Ncoeffs, Ny, Nx) containing the polynomial coefficients of P().
        such that, "corrected ramp (DN)" = P("measured ramp (DN)").
    nonlin_max
        The maximum DN value for which the non-linearity correction is good to within 1%.
    nonlin_dq
        2D array of shape (Ny, Nx) containing the DQ flags from the non-linearity model
        to be applied to all pixels.
    """
    assert not np.issubdtype(ramps.dtype, np.integer), f"Ramps must be floating point, got {ramps.dtype}"
    return _correct_nonlinearity_numba(ramps, dq_raw, dq, coeffs, nonlin_max, nonlin_dq)

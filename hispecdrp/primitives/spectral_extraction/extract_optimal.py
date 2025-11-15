from numba import njit
import numpy as np
from ...datamodels.dqflags import DQ_FLAGS
from .extraction_utils import get_extraction_window
import matplotlib.pyplot as plt
DO_NOT_USE = DQ_FLAGS['DO_NOT_USE']


def extract_spectrum_optimal(
    input_data : np.ndarray,
    input_err : np.ndarray,
    input_dq : np.ndarray,
    input_var_rnoise : np.ndarray,
    input_var_poisson : np.ndarray,
    trace_positions : np.ndarray,
    profile : np.ndarray,
    extraction_aperture : float,
    xrange : tuple | None = None,
    spec_output : np.ndarray | None = None,
    err_output : np.ndarray | None = None,
    var_rnoise_output : np.ndarray | None = None,
    var_poisson_output : np.ndarray | None = None,
    dq_output : np.ndarray | None = None,
) -> dict:
    """
    Optimal spectral extraction.

    Parameters
    ----------
    input_data : np.ndarray
        2D array of input data.
    input_err : np.ndarray
        2D array of input errors.
    input_dq : np.ndarray
        2D array of input data quality flags.
    input_var_rnoise : np.ndarray
        2D array of input read noise variance.
    input_var_poisson : np.ndarray
        2D array of input Poisson noise variance.
    trace_positions : np.ndarray
        2D array of y-coordinates for the extraction (order, ypos).
    profile : np.ndarray
        2D array profiles to use for optimal extraction wiht the same shape as input_data.
    extraction_aperture : float
        Width of the box to use for extraction.
    xrange : tuple
        Range of x-coordinates to extract (xmin, xmax).
    spec_output : np.ndarray
        1D array to store extracted spectrum.
    err_output : np.ndarray
        1D array to store extracted errors.
    var_rnoise_output : np.ndarray
        1D array to store extracted read noise variance.
    var_poisson_output : np.ndarray
        1D array to store extracted Poisson noise variance.
    dq_output : np.ndarray
        1D array to store extracted data quality flags.

    Returns
    -------
    dict
        Dictionary containing extracted flux, error, read noise variance for the box extracted spectrum.
        If output arrays are provided, they will be filled in place.
    """

    # Default X range
    ny, nx = input_data.shape
    if xrange is None:
        xrange = (0, nx - 1)

    # Default outputs
    if spec_output is None:
        spec_output = np.full(nx, np.nan, dtype=input_data.dtype)
    if err_output is None:
        err_output = np.full(nx, np.nan, dtype=input_err.dtype)
    if var_rnoise_output is None:
        var_rnoise_output = np.full(nx, np.nan, dtype=input_var_rnoise.dtype)
    if var_poisson_output is None:
        var_poisson_output = np.full(nx, np.nan, dtype=input_var_poisson.dtype)
    if dq_output is None:
        dq_output = np.full(nx, 0, dtype=input_dq.dtype)

    # Call numba extraction routine
    _extract_optimal_numba(
        input_data=input_data,
        input_err=input_err,
        input_dq=input_dq,
        input_var_rnoise=input_var_rnoise,
        input_var_poisson=input_var_poisson,
        trace_positions=trace_positions,
        profile=profile,
        extraction_aperture=extraction_aperture,
        xrange=xrange,
        spec_output=spec_output,
        err_output=err_output,
        var_rnoise_output=var_rnoise_output,
        var_poisson_output=var_poisson_output,
        dq_output=dq_output,
    )

    # Outputs
    return dict(
        spec=spec_output,
        err=err_output,
        var_rnoise=var_rnoise_output,
        var_poisson=var_poisson_output,
        dq=dq_output
    )


@njit(nogil=True, parallel=False, cache=True)
def _extract_optimal_numba(
    input_data : np.ndarray,
    input_err : np.ndarray,
    input_dq : np.ndarray,
    input_var_rnoise : np.ndarray,
    input_var_poisson : np.ndarray,
    trace_positions : np.ndarray,
    profile : np.ndarray,
    extraction_aperture : float,
    xrange : tuple,
    spec_output : np.ndarray,
    err_output : np.ndarray,
    var_rnoise_output : np.ndarray,
    var_poisson_output : np.ndarray,
    dq_output : np.ndarray,
):

    # Loop over spectral bins
    for x in range(xrange[0], xrange[1] + 1):

        window_inds, W = get_extraction_window(trace_positions[x], extraction_aperture)

        # Slices for this spectral bin
        D = input_data[window_inds, x].copy()
        D_var = input_err[window_inds, x]**2
        D_var_poisson = input_var_poisson[window_inds, x].copy()
        D_var_rnoise = input_var_rnoise[window_inds, x].copy()
        D_dq = input_dq[window_inds, x].copy()
        P = profile[window_inds, x].copy()

        # Mask bad pixels
        # NOTE: Ignore column if any profile value is bad, keep eye on this behavior
        M = (D_dq == 0) & (D > 0) & (D_var > 0) & (D_var_poisson > 0) & (D_var_rnoise > 0)
        if np.any(~np.isfinite(P)):
            M[:] = False
        bad = np.where(~M)[0]
        n_good = len(M) - len(bad)
        if n_good == 0:
            spec_output[x] = np.nan
            err_output[x] = np.nan
            var_rnoise_output[x] = np.nan
            var_poisson_output[x] = np.nan
            dq_output[x] = DO_NOT_USE
            continue

        # Set bad pixels to zero / inf
        D[bad] = 0
        D_var[bad] = np.inf
        D_var_poisson[bad] = np.inf
        D_var_rnoise[bad] = np.inf
        
        # Normalize profile
        PP = P / np.sum(P*W)

        #if np.any(D_var == 0) or np.any(D_var_poisson == 0) or np.any(D_var_rnoise == 0) or np.sum(M) == 0 or np.sum(M * PP**2 * W / D_var_poisson) == 0 or np.sum(M * PP**2 * W / D_var_rnoise) == 0:
        #    breakpoint()

        # Calculate optimal spectrum and errors
        spec_output[x] = np.sum(D * M * PP * W / D_var) / np.sum(M * PP**2 * W / D_var)
        err_output[x] = np.sqrt(np.sum(PP * W) / np.sum(M * PP**2 * W / D_var))
        var_rnoise_output[x] = np.sum(PP * W) / np.sum(M * PP**2 * W / D_var_rnoise)
        var_poisson_output[x] = np.sum(PP * W) / np.sum(M * PP**2 * W / D_var_poisson)

        # Always good if we get here
        dq_output[x] = 0
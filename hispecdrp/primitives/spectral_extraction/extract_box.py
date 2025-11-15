from numba import njit
import numpy as np
from ...utils.math import any_sc
from ...datamodels.dqflags import DQ_FLAGS
import matplotlib.pyplot as plt
from .extraction_utils import get_extraction_window
DO_NOT_USE = DQ_FLAGS['DO_NOT_USE']


def extract_spectrum_box(
    input_data : np.ndarray,
    input_err : np.ndarray,
    input_dq : np.ndarray,
    input_var_rnoise : np.ndarray,
    input_var_poisson : np.ndarray,
    trace_positions : np.ndarray,
    extraction_aperture : float,
    xrange : tuple | None = None,
    spec_output : np.ndarray = None,
    err_output : np.ndarray = None,
    var_rnoise_output : np.ndarray = None,
    var_poisson_output : np.ndarray = None,
    dq_output : np.ndarray = None,
) -> dict:
    """
    Box spectral extraction.

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
    extraction_aperture : float
        Width of the aperture to use for extraction.
    xrange : tuple, optional.
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
    """
    ny, nx = input_data.shape
    if xrange is None:
        xrange = (0, nx - 1)

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
    
    _extract_box_numba(
        input_data=input_data,
        input_err=input_err,
        input_dq=input_dq,
        input_var_rnoise=input_var_rnoise,
        input_var_poisson=input_var_poisson,
        trace_positions=trace_positions,
        extraction_aperture=extraction_aperture,
        xrange=xrange,
        spec_output=spec_output,
        err_output=err_output,
        var_rnoise_output=var_rnoise_output,
        var_poisson_output=var_poisson_output,
        dq_output=dq_output,
    )

    return dict(
        spec=spec_output,
        err=err_output,
        var_rnoise=var_rnoise_output,
        var_poisson=var_poisson_output,
        dq=dq_output,
    )

@njit(nogil=True, parallel=False, cache=True)
def _extract_box_numba(
    input_data : np.ndarray,
    input_err : np.ndarray,
    input_dq : np.ndarray,
    input_var_rnoise : np.ndarray,
    input_var_poisson : np.ndarray,
    trace_positions : np.ndarray,
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

        # Get extraction window
        window_inds, W = get_extraction_window(trace_positions[x], extraction_aperture)

        # Slices for this spectral bin
        D = input_data[window_inds, x].copy()
        D_var = input_err[window_inds, x]**2
        D_var_poisson = input_var_poisson[window_inds, x].copy()
        D_var_rnoise = input_var_rnoise[window_inds, x].copy()
        D_dq = input_dq[window_inds, x]

        # Mask bad pixels
        M = (D_dq == 0) & (D > 0) & (D_var > 0) & (D_var_poisson > 0) & (D_var_rnoise > 0)
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

        # Debugging
        # if x == 1024:
        #     breakpoint()

        # Normalization factor
        N = np.sum(M * W) / np.sum(W)

        if N == 0:
            spec_output[x] = np.nan
            err_output[x] = np.nan
            var_rnoise_output[x] = np.nan
            var_poisson_output[x] = np.nan
            dq_output[x] = DO_NOT_USE
            continue

        # Compute raw box extraction sums using full pixels
        spec_output[x] = np.sum(M * D * W) / N
        err_output[x] = np.sqrt(np.sum(M * W**2 * D_var)) / N
        var_rnoise_output[x] = np.sum(M * W**2 * D_var_rnoise) / N**2
        var_poisson_output[x] = np.sum(M * W**2 * D_var_poisson) / N**2
        dq_output[x] = 0
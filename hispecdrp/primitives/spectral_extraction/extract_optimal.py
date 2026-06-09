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
    input_mid_time : np.ndarray,
    trace_positions : np.ndarray,
    profile : np.ndarray,
    extraction_aperture : float,
    pix_range : tuple | None = None,
    spec_output : np.ndarray | None = None,
    err_output : np.ndarray | None = None,
    var_rnoise_output : np.ndarray | None = None,
    var_poisson_output : np.ndarray | None = None,
    dq_output : np.ndarray | None = None,
    mid_time_output : np.ndarray | None = None,
    orders_are_vertical : bool = True,
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
    input_mid_time : np.ndarray
        2D array of input mid times for each pixel.
    trace_positions : np.ndarray
        2D array of y-coordinates for the extraction (order, ypos).
    profile : np.ndarray
        2D array profiles to use for optimal extraction wiht the same shape as input_data.
    extraction_aperture : float
        Width of the box to use for extraction.
    pix_range : tuple
        Range of pixel-coordinates to extract (min, max).
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
    mid_time_output : np.ndarray
        1D array to store extracted mid times.
    orders_are_vertical : bool
        Whether the orders are oriented vertically.

    Returns
    -------
    dict
        Dictionary containing extracted flux, error, read noise variance for the box extracted spectrum.
        If output arrays are provided, they will be filled in place.
    """

    # Default X range
    ny, nx = input_data.shape
    if pix_range is None:
        if orders_are_vertical:
            pix_range = (0, ny - 1)
            npix = ny
        else:
            pix_range = (0, nx - 1)
            npix = nx

    # Default outputs
    if spec_output is None:
        spec_output = np.full(npix, np.nan, dtype=input_data.dtype)
    if err_output is None:
        err_output = np.full(npix, np.nan, dtype=input_err.dtype)
    if var_rnoise_output is None:
        var_rnoise_output = np.full(npix, np.nan, dtype=input_var_rnoise.dtype)
    if var_poisson_output is None:
        var_poisson_output = np.full(npix, np.nan, dtype=input_var_poisson.dtype)
    if dq_output is None:
        dq_output = np.full(npix, 0, dtype=input_dq.dtype)
    if mid_time_output is None:
        mid_time_output = np.full(npix, np.nan, dtype=input_mid_time.dtype)

    # Call numba extraction routine
    _extract_optimal_numba(
        input_data=input_data,
        input_err=input_err,
        input_dq=input_dq,
        input_var_rnoise=input_var_rnoise,
        input_var_poisson=input_var_poisson,
        input_mid_time=input_mid_time,
        trace_positions=trace_positions,
        profile=profile,
        extraction_aperture=extraction_aperture,
        pix_range=pix_range,
        spec_output=spec_output,
        err_output=err_output,
        var_rnoise_output=var_rnoise_output,
        var_poisson_output=var_poisson_output,
        dq_output=dq_output,
        mid_time_output=mid_time_output,
        orders_are_vertical=orders_are_vertical,
    )

    # Outputs
    return dict(
        spec=spec_output,
        err=err_output,
        var_rnoise=var_rnoise_output,
        var_poisson=var_poisson_output,
        dq=dq_output,
        mid_time=mid_time_output,
    )


@njit(nogil=True, parallel=False, cache=True)
def _extract_optimal_numba(
    input_data : np.ndarray,
    input_err : np.ndarray,
    input_dq : np.ndarray,
    input_var_rnoise : np.ndarray,
    input_var_poisson : np.ndarray,
    input_mid_time : np.ndarray,
    trace_positions : np.ndarray,
    profile : np.ndarray,
    extraction_aperture : float,
    pix_range : tuple,
    spec_output : np.ndarray,
    err_output : np.ndarray,
    var_rnoise_output : np.ndarray,
    var_poisson_output : np.ndarray,
    dq_output : np.ndarray,
    mid_time_output : np.ndarray,
    orders_are_vertical : bool = True,
):

    # Loop over spectral bins
    for x in range(pix_range[0], pix_range[1] + 1):

        #if x == 2000:
        #    breakpoint()

        window_inds, W = get_extraction_window(trace_positions[x], extraction_aperture)

        # Slices for this spectral bin
        if orders_are_vertical:
            D = input_data[x, window_inds].copy()
            D_var = input_err[x, window_inds]**2
            D_var_poisson = input_var_poisson[x, window_inds].copy()
            D_var_rnoise = input_var_rnoise[x, window_inds].copy()
            D_dq = input_dq[x, window_inds]
            P = profile[x, window_inds].copy()
            T = input_mid_time[x, window_inds].copy()
        else:
            D = input_data[window_inds, x].copy()
            D_var = input_err[window_inds, x]**2
            D_var_poisson = input_var_poisson[window_inds, x].copy()
            D_var_rnoise = input_var_rnoise[window_inds, x].copy()
            D_dq = input_dq[window_inds, x].copy()
            P = profile[window_inds, x].copy()
            T = input_mid_time[window_inds, x].copy()

        # Mask bad pixels
        M = (D_dq & DO_NOT_USE) == 0

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
            mid_time_output[x] = np.nan
            continue

                # Set bad pixels to zero / inf
        D[bad] = 0.0
        D_var[bad] = np.inf
        D_var_poisson[bad] = np.inf
        D_var_rnoise[bad] = np.inf

        # Normalize profile
        PP = P / np.sum(P * W)

        # Build safe weights
        w_tot = np.zeros_like(D_var)
        w_rn = np.zeros_like(D_var_rnoise)
        w_p = np.zeros_like(D_var_poisson)

        good_tot = D_var > 0
        good_rn = D_var_rnoise > 0
        good_p = D_var_poisson > 0

        w_tot[good_tot] = 1.0 / D_var[good_tot]
        w_rn[good_rn] = 1.0 / D_var_rnoise[good_rn]
        w_p[good_p] = 1.0 / D_var_poisson[good_p]

        # Zero-variance pixels → infinite weight
        zero_tot = D_var == 0
        zero_rn = D_var_rnoise == 0
        zero_p = D_var_poisson == 0

        if np.any(zero_tot):
            w_tot[:] = 0.0
            w_tot[zero_tot] = 1.0

        if np.any(zero_rn):
            w_rn[:] = 0.0
            w_rn[zero_rn] = 1.0

        if np.any(zero_p):
            w_p[:] = 0.0
            w_p[zero_p] = 1.0

        denom_tot = np.sum(M * PP**2 * W * w_tot)
        denom_rn = np.sum(M * PP**2 * W * w_rn)
        denom_p = np.sum(M * PP**2 * W * w_p)

        if denom_tot == 0:
            spec_output[x] = np.nan
            err_output[x] = np.nan
            var_rnoise_output[x] = np.nan
            var_poisson_output[x] = np.nan
            dq_output[x] = DO_NOT_USE
            continue

        spec_output[x] = np.sum(D * M * PP * W * w_tot) / denom_tot
        err_output[x] = np.sqrt(np.sum(PP * W) / denom_tot)

        var_rnoise_output[x] = (
            np.sum(PP * W) / denom_rn if denom_rn > 0 else 0.0
        )
        var_poisson_output[x] = (
            np.sum(PP * W) / denom_p if denom_p > 0 else 0.0
        )

        dq_output[x] = 0

        # Mid time
        mid_time_output[x] = np.sum(T * M * w_tot) / np.sum(w_tot * M)
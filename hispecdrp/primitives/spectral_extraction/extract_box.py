from numba import njit
import numpy as np
from ..utils.array_utils import any_sc
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
    input_mid_time : np.ndarray,
    trace_positions : np.ndarray,
    extraction_aperture : float,
    pix_range : tuple | None = None,
    spec_output : np.ndarray = None,
    err_output : np.ndarray = None,
    var_rnoise_output : np.ndarray = None,
    var_poisson_output : np.ndarray = None,
    dq_output : np.ndarray = None,
    mid_time_output : np.ndarray = None,
    orders_are_vertical : bool = True,
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
    input_mid_time : np.ndarray
        2D array of input mid times for each pixel.
    trace_positions : np.ndarray
        2D array of y-coordinates for the extraction (order, ypos).
    extraction_aperture : float
        Width of the aperture to use for extraction.
    pix_range : tuple, optional.
        Range of pixel-coordinates to extract (min, max) in the spectral direction.
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
    """
    ny, nx = input_data.shape
    if pix_range is None:
        if orders_are_vertical:
            pix_range = (0, ny - 1)
            npix = ny
        else:
            pix_range = (0, nx - 1)
            npix = nx

    # Mock / sim L0s may omit optional companion arrays — fill zeros so dtype
    # access and the numba kernel never see None.
    data_dtype = np.asarray(input_data).dtype
    if input_err is None:
        input_err = np.zeros((ny, nx), dtype=data_dtype)
    if input_var_rnoise is None:
        input_var_rnoise = np.zeros((ny, nx), dtype=data_dtype)
    if input_var_poisson is None:
        input_var_poisson = np.zeros((ny, nx), dtype=data_dtype)
    if input_dq is None:
        input_dq = np.zeros((ny, nx), dtype=np.uint32)
    if input_mid_time is None:
        input_mid_time = np.zeros((ny, nx), dtype=np.float64)

    if spec_output is None:
        spec_output = np.full(npix, np.nan, dtype=data_dtype)
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

    _extract_box_numba(
        input_data=input_data,
        input_err=input_err,
        input_dq=input_dq,
        input_var_rnoise=input_var_rnoise,
        input_var_poisson=input_var_poisson,
        input_mid_time=input_mid_time,
        trace_positions=trace_positions,
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

    return dict(
        spec=spec_output,
        err=err_output,
        var_rnoise=var_rnoise_output,
        var_poisson=var_poisson_output,
        dq=dq_output,
        mid_time=mid_time_output,
    )

@njit(nogil=True, parallel=False, cache=True)
def _extract_box_numba(
    input_data : np.ndarray,
    input_err : np.ndarray,
    input_dq : np.ndarray,
    input_var_rnoise : np.ndarray,
    input_var_poisson : np.ndarray,
    input_mid_time : np.ndarray,
    trace_positions : np.ndarray,
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

    n_cross = input_data.shape[1] if orders_are_vertical else input_data.shape[0]

    # Loop over spectral bins
    for x in range(pix_range[0], pix_range[1] + 1):

        # Get extraction window (clip to detector; non-finite trace → empty window)
        window_inds, W = get_extraction_window(
            trace_positions[x], extraction_aperture, n_cross
        )
        if len(window_inds) == 0 or np.sum(W) <= 0.0:
            spec_output[x] = np.nan
            err_output[x] = np.nan
            var_rnoise_output[x] = np.nan
            var_poisson_output[x] = np.nan
            dq_output[x] = DO_NOT_USE
            continue

        # Slices for this spectral bin
        if orders_are_vertical:
            D = input_data[x, window_inds].copy()
            D_var = input_err[x, window_inds]**2
            D_var_poisson = input_var_poisson[x, window_inds].copy()
            D_var_rnoise = input_var_rnoise[x, window_inds].copy()
            D_dq = input_dq[x, window_inds]
        else:
            D = input_data[window_inds, x].copy()
            D_var = input_err[window_inds, x]**2
            D_var_poisson = input_var_poisson[window_inds, x].copy()
            D_var_rnoise = input_var_rnoise[window_inds, x].copy()
            D_dq = input_dq[window_inds, x]

        # Finite rate with non-finite err: treat variance as 0 and keep the pixel.
        # Dropping empty aperture-edge pixels (NaN rate_err) and renormalizing
        # caused large spectral spikes for undersampled PSFs.
        for i in range(D.size):
            if np.isfinite(D[i]) and not np.isfinite(D_var[i]):
                D_var[i] = 0.0
            if np.isfinite(D[i]) and not np.isfinite(D_var_poisson[i]):
                D_var_poisson[i] = 0.0
            if np.isfinite(D[i]) and not np.isfinite(D_var_rnoise[i]):
                D_var_rnoise[i] = 0.0

        # Mask bad pixels: DQ DO_NOT_USE and non-finite rate only
        M = ((D_dq & DO_NOT_USE) == 0) & np.isfinite(D)
        bad = np.where(~M)[0]
        n_good = len(M) - len(bad)
        if n_good == 0:
            spec_output[x] = np.nan
            err_output[x] = np.nan
            var_rnoise_output[x] = np.nan
            var_poisson_output[x] = np.nan
            dq_output[x] = DO_NOT_USE
            continue

        # Flux that would have been contributed by masked pixels (before zeroing).
        # If that flux is negligible, skip aperture renormalization so empty /
        # DQ-only edge pixels do not inflate the spectrum.
        flux_keep = 0.0
        flux_lost = 0.0
        w_good = 0.0
        w_tot = 0.0
        for i in range(D.size):
            wi = W[i]
            w_tot += wi
            di = D[i] if np.isfinite(D[i]) else 0.0
            if M[i]:
                w_good += wi
                flux_keep += di * wi
            else:
                flux_lost += di * wi

        # Zero bad pixels (M already excludes them). Use 0 not inf for variances:
        # ``M * inf`` is NaN and would poison the error sum.
        D[bad] = 0
        D_var[bad] = 0
        D_var_poisson[bad] = 0
        D_var_rnoise[bad] = 0

        if w_good <= 0.0:
            spec_output[x] = np.nan
            err_output[x] = np.nan
            var_rnoise_output[x] = np.nan
            var_poisson_output[x] = np.nan
            dq_output[x] = DO_NOT_USE
            continue

        # Renormalize only when masked pixels carried real flux.
        scale = max(abs(flux_keep), 1e-30)
        if abs(flux_lost) > 1e-6 * scale and w_tot > 0.0:
            N = w_good / w_tot
        else:
            N = 1.0

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

        # Mid times
        mid_time_output[x] = np.sum(M * W * input_mid_time[x, window_inds]) / np.sum(M * W)
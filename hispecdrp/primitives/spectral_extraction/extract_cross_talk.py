import numpy as np
from ...datamodels.dqflags import DQ_FLAGS
from .extraction_utils import get_extraction_window
import matplotlib.pyplot as plt
from numba import njit

DO_NOT_USE = DQ_FLAGS['DO_NOT_USE']

def extract_spectra_cross_talk(
    input_data : np.ndarray,
    input_err : np.ndarray,
    input_dq : np.ndarray,
    input_var_rnoise : np.ndarray,
    input_var_poisson : np.ndarray,
    input_mid_time : np.ndarray,
    trace_positions : list[np.ndarray],
    profile : list[np.ndarray],
    extraction_aperture : float,
    pix_range : tuple | None = None,
    spec_output : dict[str, np.ndarray] = None,
    err_output : dict[str, np.ndarray] = None,
    var_rnoise_output : dict[str, np.ndarray] = None,
    var_poisson_output : dict[str, np.ndarray] = None,
    dq_output : dict[str, np.ndarray] = None,
    mid_time_output : dict[str, np.ndarray] = None,
    orders_are_vertical : bool = True,
) -> dict:
    """
    Extract spectra for all fibers for a given order accounting for cross-talk between fibers.

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
        2D array of trace positions for each fiber. Shape = (n_fibers, nx).
    profile : np.ndarray
        2D array of profiles for each fiber. Shape = (n_fibers, ny, nx).
    extraction_aperture : float
        Width of the aperture to use for extraction.
    pix_range : tuple
        Range of pixel-coordinates to extract (min, max) in the spectral direction. Defaults to the full range.
    spec_output : np.ndarray
        2D array to store extracted spectrum. Shape = (n_fibers, nx).
    err_output : np.ndarray
        2D array to store extracted errors. Shape = (n_fibers, nx).
    var_rnoise_output : np.ndarray
        2D array to store extracted read noise variance. Shape = (n_fibers, nx).
    var_poisson_output : np.ndarray
        2D array to store extracted Poisson noise variance. Shape = (n_fibers, nx).
    dq_output : np.ndarray
        2D array to store extracted data quality flags. Shape = (n_fibers, nx).
    mid_time_output : np.ndarray
        2D array to store extracted mid times. Shape = (n_fibers, nx).
    orders_are_vertical : bool
        Whether the orders are oriented vertically.

    Returns
    -------
    dict
        Dictionary containing extracted flux, error, read noise variance for the box extracted spectrum.
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

    # Sanity check
    n_fibers = trace_positions.shape[0]
    assert profile.shape == (n_fibers, ny, nx), f"profiles must have shape ({n_fibers}, {ny}, {nx}), got {profile.shape}"

    # Mock / sim L0s may omit optional companion arrays.
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

    # Default outputs
    if spec_output is None:
        spec_output = np.full((n_fibers, npix), np.nan, dtype=data_dtype)
    if err_output is None:
        err_output = np.full((n_fibers, npix), np.nan, dtype=input_err.dtype)
    if var_rnoise_output is None:
        var_rnoise_output = np.full((n_fibers, npix), np.nan, dtype=input_var_rnoise.dtype)
    if var_poisson_output is None:
        var_poisson_output = np.full((n_fibers, npix), np.nan, dtype=input_var_poisson.dtype)
    if dq_output is None:
        dq_output = np.full((n_fibers, npix), 0, dtype=input_dq.dtype)
    if mid_time_output is None:
        mid_time_output = np.full((n_fibers, npix), np.nan, dtype=input_mid_time.dtype)

    # Call numba extraction routine
    _extract_cross_talk_numba(
        input_data=input_data,
        input_err=input_err,
        input_dq=input_dq,
        input_var_rnoise=input_var_rnoise,
        input_var_poisson=input_var_poisson,
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

    # Return
    return dict(
        spec=spec_output,
        err=err_output,
        var_rnoise=var_rnoise_output,
        var_poisson=var_poisson_output,
        dq=dq_output,
        mid_time=mid_time_output,
    )

@njit(nogil=True, parallel=False, cache=True)
def _extract_cross_talk_numba(
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
    n_fibers = trace_positions.shape[0]
    for x in range(pix_range[0], pix_range[1] + 1):

        trace_positions_x = trace_positions[:, x]
        profiles_x = profile[:, :, x]

        # Build the profile matrix P for this column or row
        P, W, window_full = get_design_matrix(
            trace_positions_x=trace_positions_x,
            profiles_x=profiles_x,
            extraction_aperture=extraction_aperture,
            orders_are_vertical=orders_are_vertical,
        )
        if orders_are_vertical:
            D = input_data[x, window_full].copy()
            D_var = input_err[x, window_full]**2
            D_var_poisson = input_var_poisson[x, window_full].copy()
            D_var_rnoise = input_var_rnoise[x, window_full].copy()
            D_dq = input_dq[x, window_full].copy()
        else:
            D = input_data[window_full, x].copy()
            D_var = input_err[window_full, x]**2
            D_var_poisson = input_var_poisson[window_full, x].copy()
            D_var_rnoise = input_var_rnoise[window_full, x].copy()
            D_dq = input_dq[window_full, x].copy()

        # Mask bad pixels
        # NOTE: Ignore column if any profile value is bad, keep eye on this behavior
        M = (
            ((D_dq & DO_NOT_USE) == 0)
            & np.isfinite(D)
            & np.isfinite(D_var)
            & (D > 0)
            & (D_var > 0)
            & (D_var_poisson > 0)
            & (D_var_rnoise > 0)
        )
        for i in range(n_fibers):
            if np.any(~np.isfinite(P[:, i])):
                M[:] = False
                break
        bad = np.where(~M)[0]
        n_good = len(M) - len(bad)
        if n_good == 0:
            spec_output[:, x] = np.nan
            err_output[:, x] = np.nan
            var_rnoise_output[:, x] = np.nan
            var_poisson_output[:, x] = np.nan
            dq_output[:, x] = DO_NOT_USE
            continue

        # Zero bad pixels (M already excludes them; avoid M*inf → NaN).
        D[bad] = 0
        D_var[bad] = 0
        D_var_poisson[bad] = 0
        D_var_rnoise[bad] = 0

        # Extract the spectrum at this column
        spec_fibers, err_fibers, var_rnoise_fibers, var_poisson_fibers = solve_cross_talk(
            D=D,
            D_err=D_var,
            D_var_rnoise=D_var_rnoise,
            D_var_poisson=D_var_poisson,
            P=P,
            W=W,
            M=M,
        )

        # Store the results
        spec_output[:, x] = spec_fibers
        err_output[:, x] = err_fibers
        var_rnoise_output[:, x] = var_rnoise_fibers
        var_poisson_output[:, x] = var_poisson_fibers
        dq_output[:, x] = 0

        # TODO: Compute mid times for cross talk extraction.
        # mid_time_output

        #if x == 1018:
        #    breakpoint()


@njit(nogil=True, parallel=False, cache=True)
def get_design_matrix(
    trace_positions_x : np.ndarray,
    profiles_x : np.ndarray,
    extraction_aperture : float,
    orders_are_vertical : bool = True,
) -> np.ndarray:
    """
    Build the profile matrix P for cross-talk extraction.
    """
    n_fibers = len(trace_positions_x)
    window_first, W_first = get_extraction_window(
        yc=trace_positions_x[0],
        window_size=extraction_aperture
    )
    window_last, W_last = get_extraction_window(
        yc=trace_positions_x[-1],
        window_size=extraction_aperture
    )
    if trace_positions_x[0] < trace_positions_x[-1]:
        window_full = np.arange(window_first[0], window_last[-1] + 1, dtype=np.int64)
    else:
        window_full = np.arange(window_last[0], window_first[-1] + 1, dtype=np.int64)
    ny_full = len(window_full)
    P = np.zeros((ny_full, n_fibers), dtype=profiles_x.dtype)
    W = np.zeros((ny_full, n_fibers), dtype=profiles_x.dtype)
    for i in range(n_fibers):
        # NOTE: If i == 0 or i == n_fibers - 1, use first or last window above eventually.
        window_i, W_i = get_extraction_window(
            yc=trace_positions_x[i],
            window_size=extraction_aperture
        )
        window_sub = window_i - window_full[0]
        if orders_are_vertical: # NOTE: Confirm this is correct
            P[window_sub, i] = profiles_x[window_i, i]
            W[window_sub, i] = W_i
        else:
            P[window_sub, i] = profiles_x[i, window_i]
            W[window_sub, i] = W_i
    return P, W, window_full


# NOTE: Look closer at numba C vs. Fortran ordering issues and performance implications.
@njit(nogil=True, parallel=False, cache=False)
def solve_cross_talk(
    D: np.ndarray,
    D_err: np.ndarray,
    D_var_rnoise: np.ndarray,
    D_var_poisson: np.ndarray,
    P: np.ndarray,
    W: np.ndarray,
    M: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    denom = np.sum(P * W, axis=0)
    P_norm = P / denom[None, :]

    D_var = D_err**2

    # Inverse variance for each element, INCLUDING W (W is already 2D!)
    Vinv = M[:, None] * P_norm**2 * W / D_var[:, None]
    Vinv = np.where(np.isfinite(Vinv), Vinv, np.float32(0))

    # Weighted least squares matrix A and vector b
    A = np.ascontiguousarray(P_norm.T) @ np.ascontiguousarray(Vinv * P_norm)
    b = np.ascontiguousarray((P_norm * Vinv).T) @ D

    # Solve for spectrum
    spec = np.linalg.pinv(A) @ b

    # Covariance and errors
    cov = np.linalg.pinv(A)
    spec_err = np.sqrt(np.diag(cov))
    #spec_err = np.sqrt(np.sum(P_norm * W, axis=0) / np.sum(M[:, None] * P_norm**2 * W / D_var[:, None], axis=0))

    # Separate variance contributions
    spec_var_rnoise = np.sum(P_norm * W, axis=0) / np.sum(M[:, None] * P_norm**2 * W / D_var_rnoise[:, None], axis=0)
    spec_var_poisson = np.sum(P_norm * W, axis=0) / np.sum(M[:, None] * P_norm**2 * W / D_var_poisson[:, None], axis=0)

    return spec, spec_err, spec_var_rnoise, spec_var_poisson

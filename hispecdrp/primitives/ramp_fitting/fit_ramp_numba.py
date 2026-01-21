from numba import njit, prange
import numpy as np
from ...datamodels.dqflags import DQ_FLAGS
from .utils import get_mid_delta_times_ramp, get_ramp_chunks, bitwise_or_reduce
from ...utils.parallelization_utils import numba_thread_scope

SATURATED_FLAG = DQ_FLAGS["SATURATED"]
DO_NOT_USE_FLAG = DQ_FLAGS["DO_NOT_USE"]
JUMP_DET_FLAG = DQ_FLAGS["JUMP_DET"]
SATURATED_FLAG_uint8 =  np.uint8(DQ_FLAGS["SATURATED"])
DO_NOT_USE_FLAG_uint8 = np.uint8(DQ_FLAGS["DO_NOT_USE"])
JUMP_DET_FLAG_uint8 = np.uint8(DQ_FLAGS["JUMP_DET"])

# TODO: Implement jump detection, look into more sophisticated methods
# https://iopscience.iop.org/article/10.1088/1538-3873/ad38d9/pdf
# https://www.stsci.edu/files/live/sites/www/files/home/roman/_documents/Roman-STScI-000394_DeterminingTheBestFittingSlope.pdf
# NOTE: If we want to use numba, we may want to cache some arrays
# NOTE: like np.sum(times**2) since times are same for all pixels, etc.

__all__ = ['fit_ramps_ols', 'fit_ramps_mcds']


@njit(nogil=True, cache=True)
def _fit_ramp_slope_pedestal(
    times : np.ndarray,
    ramp : np.ndarray,
    dq_raw : np.ndarray,
    rdnoise : float,
    gain : float,
    dark_current : float,
) -> tuple[float, float, float, float, float, float]:
    """
    Fits a line to the given data (times, ramp) with a free y-intercept.

    Parameters
    ----------
    times : np.ndarray
        Read times.
    ramp : np.ndarray
        1D array of up the ramp data.
    dq_raw : np.ndarray
        1D array of data quality flags for each read.
    rdnoise : float
        Read noise for the pixel.
    gain : float
        Gain for the pixel.
    dark_current : np.ndarray or float, optional
        Dark current (DN/s) for the pixel.
    """
    first_indices, last_indices = get_ramp_chunks(dq_raw)

    slope_num_sum = 0
    slope_denom_sum = 0
    rnoise_denom_sum = 0
    poisson_denom_sum = 0
    b_num_sum = 0
    b_denom_sum = 0

    n_chunk = 0
    for k in range(len(first_indices)):
        if first_indices[k] == last_indices[k] or rdnoise == 0:
            continue
        _times = times[first_indices[k]: last_indices[k]]
        _ramp = ramp[first_indices[k]: last_indices[k]]

        n = len(_times) / rdnoise**2
        times_tot = np.sum(_times) / rdnoise**2
        times2_tot = np.sum(_times**2) / rdnoise**2
        ramp_tot = np.sum(_ramp) / rdnoise**2
        timesramp_tot = np.sum(_times * _ramp) / rdnoise**2
        deno = (n * times2_tot - times_tot**2)

        slope = (n * timesramp_tot - times_tot * ramp_tot) / deno
        b = (times2_tot * ramp_tot - times_tot * timesramp_tot) / deno

        var_rnoise = n / deno
        var_poisson = np.abs((slope + dark_current)) / ((_times[-1] - _times[0])* gain)
        b_var = times2_tot / deno

        if var_rnoise + var_poisson > 0:
            slope_num_sum += slope / (var_rnoise + var_poisson)
            slope_denom_sum += 1 / (var_rnoise + var_poisson)
            n_chunk += 1
        if var_rnoise > 0:
            rnoise_denom_sum += 1 / var_rnoise
        if var_poisson > 0:
            poisson_denom_sum += 1 / var_poisson
        if b_var > 0 and first_indices[k] == 0: # only record b for the first chunk starting at the 0th read
            b_num_sum += b / b_var
            b_denom_sum += 1 / b_var

    if n_chunk == 0 or slope_denom_sum == 0:
        slope, slope_err, var_rnoise, var_poisson, b, b_err = 0.0, -1.0, -1.0, -1.0, 0.0, -1.0
    else:
        if slope_denom_sum > 0:
            slope = slope_num_sum / slope_denom_sum
            slope_err = np.sqrt(1 / slope_denom_sum)
        if rnoise_denom_sum > 0:
            var_rnoise = 1 / rnoise_denom_sum
        if poisson_denom_sum > 0:
            var_poisson = 1 / poisson_denom_sum
        if b_denom_sum > 0:
            b = b_num_sum / b_denom_sum
            b_err = np.sqrt(1 / b_denom_sum)
    return slope, slope_err, var_rnoise, var_poisson, b, b_err

@njit(nogil=True, parallel=True, cache=True)
def _fit_ramps_slope_pedestal(
    times : np.ndarray,
    ramps : np.ndarray,
    dq_raw : np.ndarray,
    dq : np.ndarray,
    rdnoise : np.ndarray,
    gain : np.ndarray,
    dark_current : np.ndarray,
    start_times : np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    _, ny, nx = ramps.shape

    slope_image = np.zeros(shape=(ny, nx), dtype=np.float32)
    slope_err = np.zeros(shape=(ny, nx), dtype=np.float32)
    var_rnoise = np.zeros(shape=(ny, nx), dtype=np.float32)
    var_poisson = np.zeros(shape=(ny, nx), dtype=np.float32)
    new_dq = np.zeros(shape=(ny, nx), dtype=np.uint32)
    mid_time = np.zeros(shape=(ny, nx), dtype=np.float64)
    b = np.zeros(shape=(ny, nx), dtype=np.float32)
    b_err = np.zeros(shape=(ny, nx), dtype=np.float32)

    for i in prange(ny):
        for j in range(nx):
            out = _fit_ramp_slope_pedestal(
                times, ramps[:, i, j], dq_raw[:, i, j],
                rdnoise[i, j], gain[i, j], dark_current[i, j]
            )

            slope_image[i, j], slope_err[i, j], var_rnoise[i, j], var_poisson[i, j], b[i, j], b_err[i, j] = out
            mid_time[i, j] = start_times[i, j] + get_mid_delta_times_ramp(times, dq_raw[:, i, j])

            # if all reads are flagged as bad, set output DQ of rate to DO_NOT_USE
            # _if_all_reads_bad = np.all(dq[:,i, j] & DO_NOT_USE_FLAG_uint8 == DO_NOT_USE_FLAG_uint8)
            _dq_raw = dq_raw[:, i, j] - (dq_raw[:, i, j] & DO_NOT_USE_FLAG_uint8)
            new_dq[i, j] = new_dq[i, j] | bitwise_or_reduce(_dq_raw) #+ _if_all_reads_bad
            if slope_err[i, j] < 0 or var_rnoise[i, j] < 0 or var_poisson[i, j] < 0:
                new_dq[i, j] = new_dq[i, j] | DO_NOT_USE_FLAG
            new_dq[i, j] |= dq[i, j] # propagate image DQ flags

    return slope_image, slope_err, var_rnoise, var_poisson, new_dq, mid_time, b, b_err

def fit_ramps_ols(
    times : np.ndarray,
    ramps : np.ndarray,
    dq_raw : np.ndarray | None = None,
    dq : np.ndarray | None = None,
    rdnoise : np.ndarray | None = None,
    gain : np.ndarray | None = None,
    dark_current : np.ndarray | None = None,
    start_times : np.ndarray | None = None,
    max_cores : int = 1
) -> dict[str, np.ndarray]:
    """
    Fits ramps using the ordinary least squares (OLS) method.

    Parameters
    ----------
    times : np.ndarray
        Read times.
    ramps : np.ndarray
        3D array of ramp data with shape (n_reads, ny, nx) in units of DN.
    dq_raw : np.ndarray, optional
        3D array of data quality flags across reads and detector (n_reads, ny, nx).
    dq : np.ndarray, optional
        2D array of data quality flags across detector (ny, nx).
    rdnoise : np.ndarray, optional
        2D array of read noise across detector (ny, nx) in units of e-.
    gain : np.ndarray, optional
        2D array of gain across detector (ny, nx) in units of e-/DN.
    dark_current : np.ndarray, optional
        2D array of dark current across detector (ny, nx) in units of DN/s.
        Caution: Only use when the ramps have been already dark subtracted.
    start_times : np.ndarray, optional
        2D array of MJD times of the first read across detector (ny, nx).
    num_coadd : int, optional
        Number of coadds. Default is 1.
    max_cores : int, optional
        Maximum number of CPU cores to use. Default is 1.

    Returns
    -------
    output : dict[str, np.ndarray]
        Dictionary with keys matching L0 rate objects.
    """
    n_reads, ny, nx = ramps.shape

    if dq_raw is None:
        dq_raw = np.zeros((n_reads, ny, nx), dtype=np.uint8)
    if dq is None:
        dq = np.zeros((ny, nx), dtype=np.uint32)
    if rdnoise is None:
        rdnoise = np.zeros((ny, nx), dtype=np.float32)
    if gain is None:
        gain = np.ones((ny, nx), dtype=np.float32)
    if dark_current is None:
        dark_current = np.zeros((ny, nx), dtype=np.float32)
    if start_times is None:
        start_times = np.zeros((ny, nx), dtype=np.float64)

    with numba_thread_scope(max_cores):
        result = _fit_ramps_slope_pedestal(
            times, ramps, dq_raw, dq,
            rdnoise, gain, dark_current, start_times
        )

    output = {}
    output['rate'] = result[0]
    output['rate_err'] = result[1]
    output['rate_var_rnoise'] = result[2]
    output['rate_var_poisson'] = result[3]
    output['dq'] = result[4]
    output['mid_time'] = result[5]
    if len(result) > 6:
        output['bias'] = result[6]
        output['bias_err'] = result[7]

    return output


##############################
############ MCDS ############
##############################

@njit(nogil=True, cache=True)
def _fit_ramp_mcds(
    times : np.ndarray,
    ramp : np.ndarray,
    dq_raw : np.ndarray,
    rdnoise : float,
    gain : float,
    dark_current : float,
    num_coadd : int = 1
) -> tuple[float, float, float, float]:
    """
    Fits a ramp using the MCDS method.

    Parameters
    ----------
    times : np.ndarray
        Read times.
    ramp : np.ndarray
        1D array of up the ramp data.
    dq_raw : np.ndarray
        1D array of data quality flags for each read.
    rdnoise : float
        Read noise for the pixel.
    gain : float
        Gain for the pixel.
    dark_current : np.ndarray or float, optional
        Dark current (DN/s) for the pixel.
    num_coadd : int, optional
        Number of coadds on each end. Default is 1.
    
    Returns
    -------
    slope : float
        Fitted slope of the ramp.
    slope_err : float
        Uncertainty in the fitted slope.
    var_rnoise : float
        Variance due to read noise.
    var_poisson : float
        Variance due to Poisson noise.
    """

    first_indices, last_indices = get_ramp_chunks(dq_raw)

    slope_num_sum = 0
    slope_denom_sum = 0
    rnoise_denom_sum = 0
    poisson_denom_sum = 0

    n_chunk = 0
    for k in range(len(first_indices)):
        if first_indices[k] == last_indices[k]+1:
            continue
        _times = times[first_indices[k]: last_indices[k]]
        _ramp = ramp[first_indices[k]: last_indices[k]]

        n1 = np.sum(_ramp[:num_coadd])
        n2 = np.sum(_ramp[-num_coadd:])
        d1 = np.sum(_times[:num_coadd])
        d2 = np.sum(_times[-num_coadd:])
        if num_coadd > 1:
            n1 = n1 / num_coadd
            n2 = n2 / num_coadd
            d1 = d1 / num_coadd
            d2 = d2 / num_coadd
        d2md1 = (d2 - d1)
        slope = (n2 - n1) / d2md1

        var_rnoise = (2 * num_coadd) * (rdnoise/num_coadd)**2 / d2md1**2
        if dark_current is not None:
            var_poisson = np.abs((slope + dark_current)) / (d2md1*gain)
        else:
            var_poisson = np.abs(slope) / (d2md1*gain)

        var_sum = var_rnoise + var_poisson
        if var_sum > 0:
            slope_num_sum += slope / var_sum
            slope_denom_sum += 1 / var_sum
            n_chunk += 1
        if var_rnoise > 0:
            rnoise_denom_sum += 1 / var_rnoise
        if var_poisson > 0:
            poisson_denom_sum += 1 / var_poisson

    if n_chunk == 0 or slope_denom_sum == 0:
        slope, slope_err, var_rnoise, var_poisson = 0.0, -1.0, -1.0, -1.0
    else:
        slope = slope_num_sum / slope_denom_sum
        slope_err = np.sqrt(1 / slope_denom_sum)
        if rnoise_denom_sum > 0:
            var_rnoise = 1 / rnoise_denom_sum
        if poisson_denom_sum > 0:
            var_poisson = 1 / poisson_denom_sum
    return slope, slope_err, var_rnoise, var_poisson

@njit(nogil=True, parallel=True, cache=True)
def _fit_ramps_mcds(
    times : np.ndarray,
    ramps : np.ndarray,
    dq_raw : np.ndarray,
    dq : np.ndarray,
    rdnoise : np.ndarray,
    gain : np.ndarray,
    dark_current : np.ndarray,
    start_times : np.ndarray,
    num_coadd : int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    n_reads, ny, nx = ramps.shape
    slope_image = np.zeros(shape=(ny, nx), dtype=np.float32)
    slope_err = np.zeros(shape=(ny, nx), dtype=np.float32)
    var_rnoise = np.zeros(shape=(ny, nx), dtype=np.float32)
    var_poisson = np.zeros(shape=(ny, nx), dtype=np.float32)
    new_dq = np.zeros(shape=(ny, nx), dtype=np.uint32)
    mid_time = np.zeros(shape=(ny, nx), dtype=np.float64)
    for i in prange(ny):
        for j in range(nx):
            out  = _fit_ramp_mcds(
                times, ramps[:, i, j], dq_raw[:,i,j],
                rdnoise[i,j], gain[i,j], dark_current[i,j],
                num_coadd=num_coadd
            )

            slope_image[i, j], slope_err[i, j], var_rnoise[i, j], var_poisson[i, j] = out
            mid_time[i, j] = start_times[i, j] + get_mid_delta_times_ramp(times, dq_raw[:, i, j])

            # if all reads are flagged as bad, set output DQ of rate to DO_NOT_USE
            # _if_all_reads_bad = np.all(dq[:,i, j] & DO_NOT_USE_FLAG_uint8 == DO_NOT_USE_FLAG_uint8)
            _dq_raw = dq_raw[:,i, j] - (dq_raw[:,i, j] & DO_NOT_USE_FLAG_uint8)
            new_dq[i, j] = new_dq[i, j] | bitwise_or_reduce(_dq_raw) #+ _if_all_reads_bad
            if slope_err[i,j] < 0 or var_rnoise[i,j] < 0 or var_poisson[i,j] < 0:
                new_dq[i, j] = new_dq[i, j] | DO_NOT_USE_FLAG
            new_dq[i,j] |= dq[i,j] # propagate image DQ flags

    return slope_image, slope_err, var_rnoise, var_poisson,new_dq,mid_time

def fit_ramps_mcds(
    times : np.ndarray,
    ramps : np.ndarray,
    dq_raw : np.ndarray | None = None,
    dq : np.ndarray | None = None,
    rdnoise : np.ndarray | None = None,
    gain : np.ndarray | None = None,
    dark_current : np.ndarray | None = None,
    start_times : np.ndarray | None = None,
    num_coadd : int = 1,
    max_cores : int = 1
) -> dict[str, np.ndarray]:
    """
    Fits ramps using the MCDS method.

    Parameters
    ----------
    times : np.ndarray
        Read times.
    ramps : np.ndarray
        3D array of ramp data with shape (n_reads, ny, nx) in units of DN.
    dq_raw : np.ndarray, optional
        3D array of data quality flags across reads and detector (n_reads, ny, nx).
    dq : np.ndarray, optional
        2D array of data quality flags across detector (ny, nx).
    rdnoise : np.ndarray, optional
        2D array of read noise across detector (ny, nx) in units of e-.
    gain : np.ndarray, optional
        2D array of gain across detector (ny, nx) in units of e-/DN.
    dark_current : np.ndarray, optional
        2D array of dark current across detector (ny, nx) in units of DN/s.
        Caution: Only use when the ramps have been already dark subtracted.
    start_times : np.ndarray, optional
        2D array of MJD times of the first read across detector (ny, nx).
    num_coadd : int, optional
        Number of coadds. Default is 1.
    max_cores : int, optional
        Maximum number of CPU cores to use. Default is 1.

    Returns
    -------
    output : dict[str, np.ndarray]
        Dictionary with keys matching L0 rate objects.
    """
    n_reads, ny, nx = ramps.shape

    # Create remaining dummy variables if input arrays are not provided
    if dq_raw is None:
        dq_raw = np.zeros((n_reads, ny, nx), dtype=np.uint8)
    if dq is None:
        dq = np.zeros((ny, nx), dtype=np.uint32)
    if rdnoise is None:
        rdnoise = np.zeros((ny, nx), dtype=np.float32)
    if gain is None:
        gain = np.ones((ny, nx), dtype=np.float32)
    if dark_current is None:
        dark_current = np.zeros((ny, nx), dtype=np.float32)
    if start_times is None:
        start_times = np.zeros((ny, nx), dtype=np.float64)

    with numba_thread_scope(max_cores):
        result = _fit_ramps_mcds(
            times, ramps, dq_raw, dq,
            rdnoise, gain, dark_current, start_times, num_coadd
        )

    slope_image = result[0]
    slope_err = result[1]
    var_rnoise = result[2]
    var_poisson = result[3]
    new_dq = result[4]
    mid_times = result[5]

    output = {}
    output['rate'] = slope_image
    output['rate_err'] = slope_err
    output['rate_var_rnoise'] = var_rnoise
    output['rate_var_poisson'] = var_poisson
    output['dq'] = new_dq
    output['mid_time'] = mid_times
    return output

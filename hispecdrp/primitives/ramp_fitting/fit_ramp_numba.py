from numba import njit, prange
import numpy as np
from ...datamodels.dqflags import DQ_FLAGS

saturation_flag = DQ_FLAGS["SATURATED"]
do_not_use_flag = DQ_FLAGS["DO_NOT_USE"]

# TODO: Implement jump detection, look into more sophisticated methods
# https://iopscience.iop.org/article/10.1088/1538-3873/ad38d9/pdf
# https://www.stsci.edu/files/live/sites/www/files/home/roman/_documents/Roman-STScI-000394_DeterminingTheBestFittingSlope.pdf
# NOTE: If we want to use numba, we may want to cache some arrays
# NOTE: like np.sum(times**2) since times are same for all pixels, etc.


from numba import njit
import numpy as np

from numba import njit
import numpy as np
from ...datamodels.dqflags import DQ_FLAGS

__all__ = ['fit_ramps_ols', 'fit_ramps_mcds']

@njit(nogil=True)
def first_with_flag(dq: np.ndarray, flag: np.uint32) -> int:
    """
    Return the index of the first element whose DQ has `flag` set.
    Returns dq.size if none found.
    """
    n = dq.size
    for i in range(n):
        if (dq[i] & flag) != 0:
            return i
    return n

# @njit(nogil=True)
def _fit_ramp_slope_pedestal(times, ramp, dq, rdnoise, gain, dark_current,start_time):
    """
    Fits a line to the given data (times, ramp) with a free y-intercept.
    """

    first_do_not_use = first_with_flag(dq,do_not_use_flag)
    first_saturated = first_with_flag(dq,saturation_flag)
    if first_saturated <= first_do_not_use:
        first_do_not_use = first_saturated

    # if not enough reads for mcds, return 0 slope and 0 error
    if first_do_not_use < 2:
        return 0.0, -1.0, -1.0, 0.0, 0.0, -1.0

    times = times[:first_do_not_use]
    ramp = ramp[:first_do_not_use]

    n = len(times)/rdnoise**2
    times_tot = np.sum(times)/rdnoise**2
    times2_tot = np.sum(times**2)/rdnoise**2
    ramp_tot = np.sum(ramp)/rdnoise**2
    timesramp_tot = np.sum(times * ramp)/rdnoise**2
    deno = (n * times2_tot - times_tot**2)

    slope = (n * timesramp_tot - times_tot * ramp_tot) / deno
    b = (times2_tot * ramp_tot - times_tot * timesramp_tot)/ deno

    var_rnoise = n / deno
    var_poisson = np.abs((slope + dark_current) * gain) / (times[-1] - times[0])
    b_err = np.sqrt(times2_tot/deno)

    mid_time = start_time + np.sum(times) / n / 86400.0  # Convert seconds to days for MJD

    return slope, var_rnoise, var_poisson, mid_time, b, b_err


# @njit(nogil=True)
def _fit_ramp_slope_fixed_pedestal(times, ramp, dq, rdnoise, gain, dark_current,start_time,  b):
    """
    Fits a line to the given data (times, ramp) with a fixed y-intercept.
    """

    first_do_not_use = first_with_flag(dq,do_not_use_flag)
    first_saturated = first_with_flag(dq,saturation_flag)
    if first_saturated <= first_do_not_use:
        first_do_not_use = first_saturated

    # if not enough reads for mcds, return 0 slope and 0 error
    if first_do_not_use < 2:
        return 0.0, -1.0, -1.0, 0.0

    times = times[:first_do_not_use]
    ramp = ramp[:first_do_not_use]

    n = len(times)
    times2_tot = np.sum((times/rdnoise)**2)
    slope = np.sum(times * (ramp - b)/rdnoise**2) / times2_tot

    var_rnoise = 1/times2_tot
    var_poisson = np.abs((slope+dark_current)*gain)/(times[-1] - times[0])

    mid_time = start_time + np.sum(times)/n / 86400.0  # Convert seconds to days for MJD

    return slope, var_rnoise,var_poisson,mid_time


# @njit(nogil=True)
def _fit_ramps_slope_pedestal(times, ramps, dq, rdnoise, gain, dark_current,start_times):
    _, ny, nx = ramps.shape
    slope_image = np.zeros(shape=(ny, nx), dtype=np.float32)
    var_rnoise = np.zeros(shape=(ny, nx), dtype=np.float32)
    var_poisson = np.zeros(shape=(ny, nx), dtype=np.float32)
    b = np.zeros(shape=(ny, nx), dtype=np.float32)
    b_err = np.zeros(shape=(ny, nx), dtype=np.float32)
    mid_times = np.zeros(shape=(ny, nx), dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            slope_image[i, j], var_rnoise[i, j], var_poisson[i, j], mid_times[i, j],b[i,j],b_err[i,j] = _fit_ramp_slope_pedestal(
                                                times, ramps[:, i, j], dq[:, i, j],
                                                rdnoise[i, j], gain[i, j], dark_current[i, j],start_times[i, j])
    return slope_image, var_rnoise, var_poisson, mid_times,b,b_err


# @njit(nogil=True)
def _fit_ramps_slope_fixed_pedestal_array(times, ramps, dq, rdnoise, gain, dark_current,start_times, b):
    _, ny, nx = ramps.shape
    slope_image = np.zeros(shape=(ny, nx), dtype=np.float32)
    var_rnoise = np.zeros(shape=(ny, nx), dtype=np.float32)
    var_poisson = np.zeros(shape=(ny, nx), dtype=np.float32)
    mid_times = np.zeros(shape=(ny, nx), dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            slope_image[i, j], var_rnoise[i, j], var_poisson[i, j], mid_times[i, j] = _fit_ramp_slope_fixed_pedestal(
                                                times, ramps[:, i, j], dq[:, i, j],
                                                rdnoise[i, j], gain[i, j], dark_current[i, j],start_times[i, j],
                                                b[i, j])
    return slope_image, var_rnoise, var_poisson, mid_times


# @njit(nogil=True)
def _fit_ramps_slope_fixed_pedestal_float(times, ramps, dq, rdnoise, gain, dark_current,start_times, b):
    _, ny, nx = ramps.shape
    slope_image = np.zeros(shape=(ny, nx), dtype=np.float32)
    var_rnoise = np.zeros(shape=(ny, nx), dtype=np.float32)
    var_poisson = np.zeros(shape=(ny, nx), dtype=np.float32)
    mid_times = np.zeros(shape=(ny, nx), dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            slope_image[i, j], var_rnoise[i, j], var_poisson[i, j], mid_times[i, j]  = _fit_ramp_slope_fixed_pedestal(
                                                times, ramps[:, i, j], dq[:, i, j],
                                                rdnoise[i, j], gain[i, j], dark_current[i, j],start_times[i, j],
                                                b)
    return slope_image, var_rnoise, var_poisson, mid_times


def fit_ramps_ols(times, ramps, dq=None, rdnoise=None, gain=None, dark_current=None, start_times=None, b=None):
    """
    Top-level function for fitting ramps with or without a fixed y-intercept with no correlated noise.
    Dispatches to the appropriate specialized function based on whether b is provided.

    Args:
        times (np.ndarray): Read times.
        ramps (np.ndarray): 3D array of ramp data with shape (n_reads, ny, nx).
        dq (np.ndarray, optional): 2D array of data quality flags across detector (ny, nx).
        rdnoise (np.ndarray, optional): 2D array of read noise across detector (ny, nx).
        gain (np.ndarray, optional): 2D array of gain across detector (ny, nx).
        dark_current (np.ndarray, optional): 2D array of dark current (DN/s) across detector (ny, nx).
            Caution: Only use when the ramps have been already dark subtracted.
        start_times (np.ndarray, optional): 2D array of MJD times of the first read across detector (ny, nx).
        b (float | np.ndarray, optional): Fixed y-intercept value. If None, y-intercept will be fit.

    Returns:
        dict[str, np.ndarray]: Dictionary with keys matching L0 rate objects.
    """
    n_reads,ny, nx = ramps.shape

    if dq is None:
        dq = np.zeros((ny, nx)).astype(np.uint32)
    if rdnoise is None:
        rdnoise = np.zeros((ny, nx)).astype(np.float32)
    if gain is None:
        gain = np.ones((ny, nx)).astype(np.float32)
    if dark_current is None:
        dark_current = np.zeros((ny, nx)).astype(np.float32)
    if start_times is None:
        start_times = np.zeros((ny, nx)).astype(np.float32)

    if b is None:
        result = _fit_ramps_slope_pedestal(times, ramps,dq, rdnoise, gain, dark_current,start_times)
    elif isinstance(b, np.ndarray):
        result = _fit_ramps_slope_fixed_pedestal_array(times, ramps, dq, rdnoise, gain, dark_current,start_times, b)
    else:
        result = _fit_ramps_slope_fixed_pedestal_float(times, ramps, dq, rdnoise, gain, dark_current,start_times, b)


    output = {}
    output['rate'] = result[0]
    output['rate_err'] = np.sqrt(result[1] + result[2])
    output['rate_var_rnoise'] = result[1]
    output['rate_var_poisson'] = result[2]
    output['mid_time'] = result[3]
    if len(result) >4:
        output['bias'] = result[4]
        output['bias_err'] = result[5]
    # if all reads are flagged as bad, set output DQ of rate to DO_NOT_USE
    _if_all_reads_bad = np.bitwise_and.reduce(dq & DQ_FLAGS["DO_NOT_USE"], axis=0)
    _dq = dq-dq & DQ_FLAGS["DO_NOT_USE"]
    output['DQ'] = np.bitwise_or.reduce(_dq, axis=0)+_if_all_reads_bad

    where_bad = np.where(result[1] == -1)
    output['DQ'][where_bad] |= DQ_FLAGS["DO_NOT_USE"]
    output['rate'][where_bad] = np.nan
    output['rate_err'][where_bad] = np.nan
    output['rate_var_rnoise'][where_bad] = np.nan
    output['rate_var_poisson'][where_bad] = np.nan

    return output


##############################
############ MCDS ############
##############################



@njit(nogil=True)
def _fit_ramp_mcds(times, ramp, dq, rdnoise, gain, dark_current,start_time, num_coadd=1):

    first_do_not_use = first_with_flag(dq,do_not_use_flag)
    first_saturated = first_with_flag(dq,saturation_flag)
    if first_saturated <= first_do_not_use:
        first_do_not_use = first_saturated

    # if not enough reads for mcds, return 0 slope and 0 error
    if first_do_not_use < 2*num_coadd:
        return 0.,-1.,-1.,0.

    times = times[:first_do_not_use]
    ramp = ramp[:first_do_not_use]

    n1 = np.sum(ramp[:num_coadd])
    n2 = np.sum(ramp[-num_coadd:])
    d1 = np.sum(times[:num_coadd])
    d2 = np.sum(times[-num_coadd:])
    if num_coadd > 1:
        n1 = n1 / num_coadd
        n2 = n2 / num_coadd
        d1 = d1 / num_coadd
        d2 = d2 / num_coadd
    slope = (n2 - n1) / (d2 - d1)

    var_rnoise = (2*num_coadd)*rdnoise**2/ (d2 - d1)**2
    var_poisson = np.abs((slope+dark_current)*gain)/(d2 - d1)

    mid_time = start_time + (d2 + d1) / 2.0 / 86400.0  # Convert seconds to days for MJD

    return slope, var_rnoise,var_poisson,mid_time


@njit(nogil=True)
def _fit_ramps_mcds(times, ramps, dq, rdnoise, gain, dark_current, start_times,num_coadd=1):
    _, ny, nx = ramps.shape
    slope_image = np.zeros(shape=(ny, nx), dtype=np.float32)
    var_rnoise = np.zeros(shape=(ny, nx), dtype=np.float32)
    var_poisson = np.zeros(shape=(ny, nx), dtype=np.float32)
    mid_times = np.zeros(shape=(ny, nx), dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            out  = _fit_ramp_mcds(times,
                                  ramps[:, i, j], dq[:,i,j], rdnoise[i,j], gain[i,j], dark_current[i,j], start_times[i,j],
                                  num_coadd=num_coadd)
            slope_image[i, j], var_rnoise[i, j], var_poisson[i, j], mid_times[i, j] = out
    return slope_image, var_rnoise, var_poisson, mid_times


def fit_ramps_mcds(times, ramps, dq=None, rdnoise=None, gain=None, dark_current=None, start_times=None,num_coadd=1):
    """
    Fits ramps using the MCDS method.

    Args:
        times (np.ndarray): Read times.
        ramps (np.ndarray): 3D array of ramp data with shape (n_reads, ny, nx).
        dq (np.ndarray): 2D array of data quality flags across detector (ny, nx)
        rdnoise (np.ndarray): 2D array of read noise across detector (ny, nx)
        gain (np.ndarray): 2D array of gain across detector (ny, nx)
        dark_current (np.ndarray): 2D array of dark current (DN/s) across detector (ny, nx).
            Caution: Only use when the ramps have been already dark subtracted.
        start_times (np.ndarray): 2D array of MJD times of the first read across detector (ny, nx).
        num_coadd (int): Number of coadds.

    Returns:
        dict[str,
         np.ndarray]: Dictionary with keys matching L0 rate objects.
    """
    n_reads,ny, nx = ramps.shape

    # Create remaining dummy variables if input arrays are not provided
    if dq is None:
        dq = np.zeros((ny, nx)).astype(np.uint32)
    if rdnoise is None:
        rdnoise = np.zeros((ny, nx)).astype(np.float32)
    if gain is None:
        gain = np.ones((ny, nx)).astype(np.float32)
    if dark_current is None:
        dark_current = np.zeros((ny, nx)).astype(np.float32)
    if start_times is None:
        start_times = np.zeros((ny, nx)).astype(np.float32)

    result = _fit_ramps_mcds(times, ramps,dq, rdnoise, gain, dark_current,start_times, num_coadd=num_coadd)

    output = {}
    output['rate'] = result[0]
    output['rate_err'] = np.sqrt(result[1] + result[2])
    output['rate_var_rnoise'] = result[1]
    output['rate_var_poisson'] = result[2]
    output['mid_time'] = result[3]
    # if all reads are flagged as bad, set output DQ of rate to DO_NOT_USE
    _if_all_reads_bad = np.bitwise_and.reduce(dq & DQ_FLAGS["DO_NOT_USE"], axis=0)
    _dq = dq-dq & DQ_FLAGS["DO_NOT_USE"]
    output['DQ'] = np.bitwise_or.reduce(_dq, axis=0)+_if_all_reads_bad

    where_bad = np.where(result[1] == -1)
    output['DQ'][where_bad] |= DQ_FLAGS["DO_NOT_USE"]
    output['rate'][where_bad] = np.nan
    output['rate_err'][where_bad] = np.nan
    output['rate_var_rnoise'][where_bad] = np.nan
    output['rate_var_poisson'][where_bad] = np.nan
    return output

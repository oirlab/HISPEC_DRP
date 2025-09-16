from numba import njit, prange
import numpy as np

# TODO: Implement jump detection, look into more sophisticated methods
# https://iopscience.iop.org/article/10.1088/1538-3873/ad38d9/pdf
# https://www.stsci.edu/files/live/sites/www/files/home/roman/_documents/Roman-STScI-000394_DeterminingTheBestFittingSlope.pdf
# NOTE: If we want to use numba, we may want to cache some arrays
# NOTE: like np.sum(times**2) since times are same for all pixels, etc.


from numba import njit
import numpy as np

from numba import njit
import numpy as np

__all__ = ['fit_ramps_ols', 'fit_ramps_mcds','fit_ramps_likely']


@njit(nogil=True)
def _fit_ramp_slope_pedestal(times, ramp):
    """
    Fits a line to the given data (times, ramp) with a free y-intercept.

    Args:
        times (np.ndarray): The times at which the data was taken.
        ramp (np.ndarray): The ramp data.

    Returns:
        float32: slope
        float32: slope error
        float32: y-intercept
        float32: y-intercept error
    """
    n = len(times)
    times_tot = np.sum(times)
    times2_tot = np.sum(times**2)
    ramp_tot = np.sum(ramp)
    
    slope = (n * np.sum(times * ramp) - times_tot * ramp_tot) / (n * times2_tot - times_tot**2)
    b = (ramp_tot - slope * times_tot) / n
    
    # Compute residual variance
    residual_var = np.sum((ramp - slope * times - b)**2) / (n - 2)
    
    # Compute errors
    slope_err = np.sqrt(residual_var / (times2_tot - times_tot**2 / n))
    b_err = np.sqrt(residual_var * times2_tot / (n * (times2_tot - times_tot**2 / n)))
    
    return slope, slope_err, b, b_err


@njit(nogil=True)
def _fit_ramp_slope_fixed_pedestal(times, ramp, b):
    """
    Fits a line to the given data (times, ramp) with a fixed y-intercept.

    Args:
        times (np.ndarray): The read times (1D).
        ramp (np.ndarray): The ramp data (1D).
        fixed_pedestal (float32): The fixed y-intercept value.

    Returns:
        float32: slope
        float32: slope error
    """
    n = len(times)
    times2_tot = np.sum(times**2)
    slope = np.sum(times * (ramp - b)) / times2_tot
    residual_var = np.sum((ramp - slope * times - b)**2) / (n - 1)
    slope_err = np.sqrt(residual_var / times2_tot)
    return slope, slope_err


@njit(nogil=True)
def _fit_ramps_slope_pedestal(times, ramps):
    _, ny, nx = ramps.shape
    slope_image = np.zeros(shape=(ny, nx), dtype=np.float32)
    slope_errors = np.zeros(shape=(ny, nx), dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            slope_image[i, j], slope_errors[i, j] = _fit_ramp_slope_pedestal(times, ramps[:, i, j])
    return slope_image, slope_errors


@njit(nogil=True)
def _fit_ramps_slope_fixed_pedestal_array(times, ramps, b):
    _, ny, nx = ramps.shape
    slope_image = np.zeros(shape=(ny, nx), dtype=np.float32)
    slope_errors = np.zeros(shape=(ny, nx), dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            slope_image[i, j], slope_errors[i, j] = _fit_ramp_slope_fixed_pedestal(times, ramps[:, i, j], b[i, j])
    return slope_image, slope_errors


@njit(nogil=True)
def _fit_ramps_slope_fixed_pedestal_float(times, ramps, b):
    _, ny, nx = ramps.shape
    slope_image = np.zeros(shape=(ny, nx), dtype=np.float32)
    slope_errors = np.zeros(shape=(ny, nx), dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            slope_image[i, j], slope_errors[i, j] = _fit_ramp_slope_fixed_pedestal(times, ramps[:, i, j], b)
    return slope_image, slope_errors


def fit_ramps_ols(times, ramps, b=None):
    """
    Top-level function for fitting ramps with or without a fixed y-intercept with no correlated noise.
    Dispatches to the appropriate specialized function based on whether b is provided.

    Args:
        times (np.ndarray): Read times.
        ramps (np.ndarray): 3D array of ramp data with shape (n_reads, ny, nx).
        b (float | np.ndarray, optional): Fixed y-intercept value. If None, y-intercept will be fit.

    Returns:
        dict[str, np.ndarray]: Dictionary with keys 'slope' and 'slope_err' containing the fitted slopes and errors. If b is not provided, the result also contains 'b' and 'b_err'.
    """
    if b is None:
        result = _fit_ramps_slope_pedestal(times, ramps)
        return dict(slope=result[0], slope_err=result[1], b=result[2], b_err=result[3])
    elif isinstance(b, np.ndarray):
        result = _fit_ramps_slope_fixed_pedestal_array(times, ramps, b)
        return dict(slope=result[0], slope_err=result[1])
    else:
        result = _fit_ramps_slope_fixed_pedestal_float(times, ramps, b)
        return dict(slope=result[0], slope_err=result[1])



##############################
############ MCDS ############
##############################


@njit(nogil=True)
def _fit_ramp_mcds(times, ramp, num_coadd=1):
    n = len(times)
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
    residuals = ramp - slope * times
    slope_err = np.sqrt(np.sum(residuals**2) / (n - 1) / np.sum(times**2))
    return slope, slope_err


@njit(nogil=True)
def _fit_ramps_mcds(times, ramps, num_coadd=1):
    _, ny, nx = ramps.shape
    slope_image = np.zeros(shape=(ny, nx), dtype=np.float32)
    slope_errors = np.zeros(shape=(ny, nx), dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            slope_image[i, j], slope_errors[i, j] = _fit_ramp_mcds(times, ramps[:, i, j], num_coadd=num_coadd)
    return slope_image, slope_errors


def fit_ramps_mcds(times, ramps, num_coadd=1):
    """
    Fits ramps using the MCDS method.

    Args:
        times (np.ndarray): Read times.
        ramps (np.ndarray): 3D array of ramp data with shape (n_reads, ny, nx).
        num_coadd (int): Number of coadds.

    Returns:
        dict[str, np.ndarray]: Dictionary with keys 'slope' and 'slope_err' containing the fitted slopes and errors.
    """
    result = _fit_ramps_mcds(times, ramps, num_coadd=num_coadd)
    return dict(slope=result[0], slope_err=result[1])

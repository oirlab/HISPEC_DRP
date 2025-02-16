from numba import jit, njit, prange
import numpy as np

# TODO: Implement jump detection, look into more sophisticated methods
# NOTE: If we want to use numba, we may want to cache some arrays like np.sum(times**2), etc.


@njit
def fit_ramp_utr(times, ramp):
    n = len(times)
    slope = np.sum(ramp * times) / np.sum(times**2)
    residuals = ramp - slope * times
    slope_err = np.sqrt(np.sum(residuals**2) / (n - 1) / np.sum(times**2))
    return slope, slope_err


@njit
def fit_ramp_mcds(times, ramp, num_coadd):
    n = len(times)
    n1 = np.sum(ramp[:num_coadd]) / num_coadd
    n2 = np.sum(ramp[-num_coadd:]) / num_coadd
    d1 = np.sum(times[:num_coadd]) / num_coadd
    d2 = np.sum(times[-num_coadd:]) / num_coadd
    slope = (n2 - n1) / (d2 - d1)
    residuals = ramp - slope * times
    slope_err = np.sqrt(np.sum(residuals**2) / (n - 1) / np.sum(times**2))
    return slope, slope_err


@njit
def fit_ramps_utr(times, ramps):
    ny, nx, _ = ramps.shape
    slope_image = np.zeros(shape=(ny, nx), dtype=np.float32)
    slope_errors = np.zeros(shape=(ny, nx), dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            slope_image[i, j], slope_errors[i, j] = fit_ramp_utr(times, ramps[i, j, :])
    return slope_image, slope_errors


@njit
def fit_ramps_mcds(times, ramps):
    ny, nx, _ = ramps.shape
    slope_image = np.zeros(shape=(ny, nx), dtype=np.float32)
    slope_errors = np.zeros(shape=(ny, nx), dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            slope_image[i, j], slope_errors[i, j] = fit_ramp_utr(times, ramps[i, j, :])
    return slope_image, slope_errors
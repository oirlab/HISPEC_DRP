import numpy as np
import scipy.constants
from numpy.polynomial.chebyshev import chebvander2d
from numba import njit

__all__ = [
    'delta_wave_to_delta_vel', 'delta_vel_to_delta_wave', 'delta_pix_to_delta_wave',
    'polyval1d', 'solve_chebyshev2d',
]

def delta_wave_to_delta_vel(delta_wave : np.ndarray | float, wave : np.ndarray | float) -> np.ndarray | float:
    """
    Convert a change in wavelength to a change in velocity.

    Args:
        delta_wave (np.ndarray | float): Change in wavelength (any units).
        wave (np.ndarray | float): Wavelength (same units as delta_wave).

    Returns:
        np.ndarray | float: Change in velocity in m/s.
    """
    return delta_wave / wave * scipy.constants.c


def delta_vel_to_delta_wave(delta_vel : np.ndarray | float, wave : np.ndarray | float) -> np.ndarray | float:
    """
    Convert a change in velocity to a change in wavelength.

    Args:
        delta_vel (np.ndarray | float): Change in velocity in m/s.
        wave (np.ndarray | float): Wavelength (any units).

    Returns:
        np.ndarray | float: Change in wavelength in the same units as wave.
    """
    return delta_vel / scipy.constants.c * wave


def delta_pix_to_delta_wave(
    delta_pix : float,
    x : float,
    wavesol : np.ndarray | None = None,
    dispersion : np.ndarray | float = None
) -> np.ndarray | float:
    """
    Convert a change in pixels to a change in wavelength.

    Args:
        delta_pix (float): Change in pixels.
        x (float): Pixel position.
        wavesol (np.ndarray | None): Wavelength solution used to calculate the dispersion. If None, the dispersion must be provided.
        dispersion (np.ndarray | float): Dispersion in wavelength per pixel. If None, it will be calculated from the wavelength solution.

    Returns:
        np.ndarray | float: Change in wavelength in the same units as the wavelength solution.
    """
    if dispersion is None:
        dispersion = np.gradient(wavesol)
    disp = np.interp(x, np.arange(len(dispersion)), dispersion)
    dw = disp * delta_pix
    return dw


@njit(nogil=True)
def polyval1d(coeffs : np.ndarray, x : np.ndarray) -> np.ndarray:
    """
    Evaluate a polynomial with given coefficients at specified points using the Horner method.

    Args:
        coeffs (np.ndarray): Coefficients of the polynomial, ordered from lowest degree to highest (NOTE: numpy is reversed).
        x (np.ndarray): Points at which to evaluate the polynomial.
    """
    v = np.zeros(x.shape, dtype=coeffs.dtype)
    for coeff in coeffs[::-1]:
        v = v * x + coeff
    return v


def solve_chebyshev2d_ols(
    x : np.ndarray, y : np.ndarray, z : np.ndarray,
    deg_x : int, deg_y : int,
    weights : np.ndarray | None = None,
) -> np.ndarray:
    """
    Fit a 2D Chebyshev polynomial using ordinary (uncorrelated) weighted least squares.

    Args:
        x: 1D array of x coordinates (length N).
        y: 1D array of y coordinates (length N).
        z (np.ndarray): 1D array of values at coordinates (x,y) (length N).
        deg_x (int): X Degree of the Chebyshev polynomial.
        deg_y (int): Y Degree of the Chebyshev polynomial.
        weights (np.ndarray | None): 1D array of weights (length N). If None, uniform weights are used.

    Returns:
        coeffs: 2D array of Chebyshev coefficients of shape (deg_x+1, deg_y+1)
    """

    # Construct the 2D Vandermonde matrix for Chebyshev basis
    V = chebvander2d(x, y, [deg_x, deg_y])  # shape: (N, (deg_x+1)*(deg_y+1))

    # Apply weights, if provided
    if weights is not None:
        W = np.sqrt(weights)
        V *= W[:, None]
        z = z * W

    # Solve the least squares problem
    coeffs_flat, *_ = np.linalg.lstsq(V, z, rcond=None)
    coeffs = coeffs_flat.reshape((deg_x + 1, deg_y + 1))
    return coeffs


#@njit(nogil=True)
def robust_stats(
    x: np.ndarray,
    w: np.ndarray | None = None,
    n_sigma: float = 4.0,
    maxiters: int = 100
) -> tuple[float, float]:
    """
    Compute robust mean and standard deviation using iterative outlier rejection
    based on biweight location and midvariance. Final mean and standard deviation
    are computed using the remaining good data points based on the weighted mean
    and weighted standard deviation.

    Args:
        x (np.ndarray): Input array of values.
        w (np.ndarray, optional): Weights for each value. If None, uniform weights are used.
        n_sigma (float, optional): Sigma threshold for outlier rejection. Defaults to 4.0.
        maxiters (int, optional): Maximum iterations for convergence. Defaults to 100.

    Returns:
        float: Robust weighted mean.
        float: Robust weighted standard deviation.
    """
    if w is None:
        w = np.ones_like(x)

    good = np.isfinite(x) & (w > 0)
    last_good = good.copy()

    for _ in range(maxiters):
        x_good = x[good]
        w_good = w[good]

        if x_good.size == 0:
            return np.nan, np.nan, good
        if x_good.size == 1:
            return x_good[0], 0.0, good

        mu = biweight_location(x_good)
        sig = np.sqrt(biweight_midvariance(x_good, M=mu))

        if sig == 0 or not np.isfinite(sig):
           return mu, sig, good

        # Outlier rejection
        good = np.abs(x - mu) < n_sigma * sig
        good &= np.isfinite(x) & (w > 0)

        if all_sc(good == last_good):
            break
        last_good = good.copy()

    x_good = x[good]
    w_good = w[good]
    w_sum = np.sum(w_good)

    if x_good.size < 2 or w_sum == 0:
        return np.nan, np.nan, good
    mu_final = weighted_mean(x_good, w_good)
    sig_final = weighted_stddev(x_good, w_good, M=mu_final)

    return mu_final, sig_final, good


@njit(nogil=True)
def weighted_mean(x : np.ndarray, w : np.ndarray) -> float:
    """
    Calculate the weighted mean of an array.

    Parameters
    ----------
    x : np.ndarray
        Array of values.
    w : np.ndarray
        Array of weights.

    Returns
    -------
    float
        Weighted mean.
    """
    w_sum = np.sum(w)
    if w_sum == 0:
        return np.nan
    w = w / w_sum
    return np.sum(x * w)


@njit(nogil=True)
def weighted_stddev(x : np.ndarray, w : np.ndarray, mu : float | None = None):
    """
    Calculate the weighted standard deviation of an array.
    
    Parameters
    ----------
    x : np.ndarray
        Array of values.
    w : np.ndarray
        Array of weights.
    mu : float | None, optional
        Pre-computed weighted mean. If None, it will be calculated from x and w.

    Returns
    -------
    float
        Weighted standard deviation or NaN if no valid data.
    """
    w_sum = np.sum(w)
    if w_sum == 0:
        return np.nan
    w = w / w_sum
    if mu is None:
        mu = np.sum(x * w) / w_sum
    dev = x - mu
    bias_estimator = 1.0 - np.nansum(w**2)
    var = np.nansum(dev ** 2 * w) / bias_estimator
    return np.sqrt(var)

@njit(nogil=True)
def biweight_location(
    x : np.ndarray,
    M : float | None = None,
    c : float = 6.0,
) -> float:
    """
    Calculate the biweight location of an array.

    Args:
        x (np.ndarray): Array of values.
        c (float): Tuning constant for biweight estimator (default=9.0). See astropy documentation for details.
        M (float | None): Initial guess of the location. If None, it will be calculated as the median location of x.
    """
    
    # Flatten
    x = x.ravel()

    # Median value
    if M is None:
        M = np.nanmedian(x)

    # MAD
    mad_val = mad(x, M)
    if mad_val == 0.0 or not np.isfinite(mad_val):
        return M

    # Center the data
    n = x.size
    result_num = 0
    result_den = 0
    for i in range(n):
        u = (x[i] - M) / (c * mad_val)
        if np.abs(u) < 1:
            w = (1 - u**2)**2
            result_num += (x[i] - M) * w
            result_den += w

    if result_den == 0:
        return M

    return M + result_num / result_den


@njit(nogil=True)
def biweight_midvariance(
    x : np.ndarray,
    M : float | None = None,
    c : float = 9.0,
) -> float:
    """
    Calculate the biweight midvariance of an array.

    Args:
        x (np.ndarray): Array of values.
        c (float): Tuning constant for biweight estimator (default=9.0). See astropy documentation for details.
        M (float | None): Initial guess of the location. If None, it will be calculated as the median location of x.
    """
    
    # Flatten
    x = x.ravel()

    # Median value
    if M is None:
        M = np.nanmedian(x)

    # Center the data
    d = x - M
    mad_val = mad(x, M=M)

    if mad_val == 0.0 or not np.isfinite(mad_val):
        return mad_val

    u = d / (c * mad_val)
    u2 = u * u

    good = np.where((np.abs(u) < 1.0) & np.isfinite(u))[0]
    n_good = len(good)

    f1 = 0
    f2 = 0
    for i in range(n_good):
        ii = good[i]
        t = 1 - u2[ii]
        f1 += d[ii] * d[ii] * t ** 4
        f2 += t * (1.0 - 5.0 * u2[ii])
    return n_good * f1 / (f2 * f2)


@njit(nogil=True)
def mad(x : np.ndarray, M : float | None = None):
    """
    Calculate the median absolute deviation of an array.
    
    Args:
        x: Array of values.
        M: Optional median value. If None, it will be calculated as the median of x.
    
    Returns:
        np.ndarray: Median absolute deviation of the array.
    """
    if M is None:
        M = np.nanmedian(x)
    return np.nanmedian(np.abs(x - M))


@njit(nogil=True)
def weighted_quantile(x : np.ndarray, w : np.ndarray | None = None, q : float = 0.5):
    """
    Calculate the weighted quantile of an array.

    Args:
        x (np.ndarray): Array of values.
        w (np.ndarray): Array of weights (defaults to uniform weights).
        q (float): Quantile to compute (0 <= q <= 1).
    """

    # Default weights
    if w is None:
        w = np.ones(x.shape, dtype=int)

    # Flatten
    x = x.ravel()
    w = w.ravel()

    # Edge cases
    if q == 0:
        return np.nanmin(x)
    if q == 1:
        return np.nanmax(x)

    sorted_indices = np.argsort(x)
    sorted_values = x[sorted_indices]
    sorted_weights = w[sorted_indices]

    total_weight = np.nansum(sorted_weights)
    target_weight = q * total_weight
    
    weight_sum = 0.0
    i = 0
    
    while i < len(sorted_weights) and weight_sum < target_weight:
        weight_sum += sorted_weights[i]
        i += 1
    
    if weight_sum == target_weight and i < len(sorted_weights):
        return (sorted_values[i-1] + sorted_values[i]) / 2.0
    elif i > 0:
        return sorted_values[i-1]
    else:
        return sorted_values[0]


@njit(nogil=True)
def all_sc(arr : np.ndarray) -> bool:
    """
    Optimized version of np.all which short circuits, unlike numpy.all.
    """
    for item in arr.ravel():
        if item:
            continue
        else:
            return False
    return True

@njit(nogil=True, cache=True, parallel=False)
def bitwise_or_reduce_uint32(arr: np.ndarray) -> np.uint32:
    result = np.uint32(0)
    for k in range(arr.shape[0]):
        result |= arr[k]
    return result
import numpy as np
import scipy
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import savgol_filter
from ...utils.endian_utils import normalize_dtype_array
from numba import njit


def fit_lines_gauss(
    spec : np.ndarray, specerr : np.ndarray, dq : np.ndarray,
    lines : np.ndarray,
    sigma_bounds : tuple, mu_bounds : tuple = (-1, 1),
    max_fit_window : float | None = None,
) -> dict:
    """
    Fit LFC lines in a 1D spectrum using Gaussian profiles with a uniform background.

    Args:
        spec (np.ndarray): Input 1D spectrum.
        specerr (np.ndarray): Input 1D spectrum error used for optimal fit weights.
        dq (np.ndarray): Data quality flags for input spectrum.
        lines (np.ndarray): Initial array of lines (in pixels) to fit and refine.
        sigma_bounds (tuple): Bounds on the Gaussian sigma.
        mu_bounds (tuple): Bounds on the Gaussian mean (in pixels) relative to the initial guess.
        max_fit_window (float | None): Maximum fit window size in pixels. If None, the limit is estimated based on the line spacing.

    Returns:
        dict: A dictionary containing the fit results as 1D arrays for each line and fit parameter. Keys are 'pixel', 'amplitude', 'sigma', and 'b'.
    """
    nx = len(spec)
    spec = spec.copy()
    specerr = specerr.copy()
    n_lines = len(lines)
    xarr = np.arange(nx)
    bad = np.where((dq > 0) | ~np.isfinite(spec) | ~np.isfinite(specerr))
    spec[bad] = np.nan
    specerr[bad] = np.nan
    fit_results = []
    ds = np.gradient(lines)
    for i in range(n_lines):
        fit_window = (np.floor(-ds[i] / 2), np.ceil(ds[i] / 2))
        fit_window = np.clip(fit_window, -max_fit_window / 2, max_fit_window / 2) if max_fit_window is not None else fit_window
        ki = max(0, int(np.round(lines[i] + fit_window[0])))
        kf = min(nx - 1, int(np.round(lines[i] + fit_window[1])))
        x_fit = xarr[ki:kf+1].copy()
        y_fit = spec[ki:kf+1].copy()
        yerr_fit = specerr[ki:kf+1].copy()
        fit_result = gauss_background_fit(x_fit, y_fit, yerr_fit, mu_bounds=mu_bounds, mu_guess=lines[i], sigma_bounds=sigma_bounds)
        fit_results.append(fit_result)

    return dict(
        amplitude=np.array([fit['amplitude'] for fit in fit_results]),
        amplitude_err=np.array([fit['amplitude_err'] for fit in fit_results]),
        mu=np.array([fit['mu'] for fit in fit_results]),
        mu_err=np.array([fit['mu_err'] for fit in fit_results]),
        sigma=np.array([fit['sigma'] for fit in fit_results]),
        sigma_err=np.array([fit['sigma_err'] for fit in fit_results]),
        b=np.array([fit['b'] for fit in fit_results]),
        b_err=np.array([fit['b_err'] for fit in fit_results]),
    )

def estimate_lfc_background(spec : np.ndarray, min_line_spacing : int = 12, n_lines : int = 10) -> np.ndarray:
    """
    Estimates the background of an LFC spectrum.

    Args:
        spec (np.ndarray): Input 1D spectrum.
        min_line_spacing (int, optional): Minimum spacing between LFC lines. Defaults to 12.
        n_lines (int, optional): Number of lines to consider for background estimation. Defaults to 10.

    Returns:
        np.ndarray: The estimated background.
    """
    filter_width = int(min_line_spacing / 3 * n_lines)
    if filter_width % 2 == 0:
        filter_width += 1  # Ensure filter width is odd
    background = _nanmin_filter1d(spec, filter_width)
    background = savgol_filter(background, filter_width, polyorder=1, mode='nearest')
    return background

def _nanmin_filter1d(data: np.ndarray, width: int) -> np.ndarray:
    """
    Applies a 1D nanmin filter to the input data.

    Args:
        data (np.ndarray): Input 1D array.
        width (int): Width of the filter window.

    Returns:
        np.ndarray: Filtered array.
    """
    if width % 2 == 0:
        width += 1  # ensure odd
    pad = width // 2
    padded = np.pad(data, pad_width=pad, mode='edge')
    windows = sliding_window_view(padded, width)
    return np.nanmin(windows, axis=1)

def find_initial_lfc_lines(spec, dq, min_line_spacing : float = 10) -> dict:
    """
    Finds initial LFC lines in a spectrum.

    Args:
        spec (np.ndarray): Input 1D spectrum.
        dq (np.ndarray): Data quality flags for input spectrum.
        min_line_spacing (float, optional): Minimum spacing between LFC lines. Defaults to 10.

    Returns:
        dict: A dictionary containing the initial line positions.
    """
    
    # Estimate and remove background
    background = estimate_lfc_background(spec, min_line_spacing=min_line_spacing, n_lines=2)
    spec_nobkg = spec - background

    # Find peaks
    lines, _ = scipy.signal.find_peaks(spec_nobkg, distance=min_line_spacing - 2)
    # NOTE: Keep plots here for debugging.
    # import matplotlib.pyplot as plt
    #plt.plot(spec); [plt.axvline(l, color='r', linestyle='--') for m in lines]
    #plt.plot(spec); plt.plot(background, color='orange', label='Background'); plt.show()

    # Refine lines based on centroid
    w2 = 3
    lines = lines[1:-1] # Ignore first and last lines, possibly noise
    lines = lines.astype(np.float64)
    for i in range(len(lines)):
        ki = int(max(lines[i] - w2, 0))
        kf = int(min(lines[i] + w2, len(spec) - 1))
        good = np.where(dq[ki:kf+1] == 0)[0]
        if len(good) > 0:
            lines[i] = np.sum((good + ki) * spec[ki:kf+1][good]) / np.sum(spec[ki:kf+1][good])

    return lines

@njit(nogil=True)
def gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

@njit(nogil=True)
def _gauss_background(params, x):
    model_y = gauss(x, params[0], params[1], params[2])
    if len(params) > 3:
        model_y = model_y + params[3]
    return model_y
    
@njit(nogil=True)
def _gauss_background_chi2_residuals(params, data_x, data_y, data_yerr):
    model_y = _gauss_background(params, data_x)
    wres = ((data_y - model_y) / data_yerr)
    return wres

def gauss_background_fit(
    x : np.ndarray, y : np.ndarray, yerr : np.ndarray, sigma_bounds : tuple | None = None,
    mu_guess : float | None = None,
    mu_bounds : tuple | None = (-np.inf, np.inf),
) -> dict[str, float]:
    
    # Initial params
    ymin = np.nanmin(y)
    yy = y - ymin
    ymax = np.nanmax(yy)
    yy /= ymax
    yyerr = yerr / ymax
    if mu_guess is None:
        mu_guess = np.average(x, weights=yy)
    sigma0 = np.sqrt(np.average((x - mu_guess)**2, weights=yy))
    if sigma0 < sigma_bounds[0] or sigma0 > sigma_bounds[1]:
        sigma0 = np.mean(sigma_bounds)
    b0 = 0.01
    p0 = np.array([1.0, mu_guess, sigma0, b0])
    bounds = (
        np.array([0.5, mu_bounds[0] + mu_guess, sigma_bounds[0], -0.2]),
        np.array([1.5, mu_bounds[1] + mu_guess, sigma_bounds[1], 0.2]),
    )
    n_dof = len(x) - len(p0)

    # Fit
    x = normalize_dtype_array(x)
    y = normalize_dtype_array(y)
    yerr = normalize_dtype_array(yerr)
    result = scipy.optimize.least_squares(_gauss_background_chi2_residuals, p0, args=(x, yy, yyerr), bounds=bounds, ftol=1E-10, xtol=1E-10)
    pbest = result.x
    redchi2 = result.cost / n_dof
    try:
        cov = np.linalg.inv(result.jac.T @ result.jac) * np.sum(result.fun**2) / n_dof
        errors = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        pbest = np.full(len(p0), np.nan)
        errors = np.full(len(p0), np.nan)
        redchi2 = np.nan

    # Collect output
    out = {
        'amplitude': pbest[0],
        'amplitude_err': errors[0],
        'mu': pbest[1],
        'mu_err': errors[1],
        'sigma': pbest[2],
        'sigma_err': errors[2],
        'b' : pbest[3],
        'b_err': errors[3],
        'redchi2': redchi2,
    }

    # Return
    return out
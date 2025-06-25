import numpy as np
import scipy
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import savgol_filter

from .gaussfit import gauss_background_fit


def fit_modes(
    spec : np.ndarray, specerr : np.ndarray, dq : np.ndarray,
    modes : np.ndarray,
    sigma_bounds : tuple, mu_bounds : tuple = (-1, 1),
    max_fit_window : float | None = None,
) -> dict:
    """
    Fit LFC modes in a 1D spectrum using Gaussian profiles with a uniform background.

    Args:
        spec (np.ndarray): Input 1D spectrum.
        specerr (np.ndarray): Input 1D spectrum error used for optimal fit weights.
        dq (np.ndarray): Data quality flags for input spectrum.
        modes (np.ndarray): Initial array of modes (in pixels) to fit and refine.
        sigma_bounds (tuple): Bounds on the Gaussian sigma.
        mu_bounds (tuple): Bounds on the Gaussian mean (in pixels) relative to the initial guess.
        max_fit_window (float | None): Maximum fit window size in pixels. If None, the limit is estimated based on the mode spacing.

    Returns:
        dict: A dictionary containing the fit results as 1D arrays for each mode and fit parameter. Keys are 'pixel', 'amplitude', 'sigma', and 'b':
    """
    nx = len(spec)
    spec = spec.copy()
    specerr = specerr.copy()
    n_modes = len(modes)
    xarr = np.arange(nx)
    bad = np.where((dq > 0) | ~np.isfinite(spec) | ~np.isfinite(specerr))
    spec[bad] = np.nan
    specerr[bad] = np.nan
    fit_results = []
    ds = np.gradient(modes)
    for i in range(n_modes):
        fit_window = (np.floor(-ds[i] / 2), np.ceil(ds[i] / 2))
        fit_window = np.clip(fit_window, -max_fit_window / 2, max_fit_window / 2) if max_fit_window is not None else fit_window
        ki = max(0, int(np.round(modes[i] + fit_window[0])))
        kf = min(nx - 1, int(np.round(modes[i] + fit_window[1])))
        x_fit = xarr[ki:kf+1].copy()
        y_fit = spec[ki:kf+1].copy()
        yerr_fit = specerr[ki:kf+1].copy()
        fit_result = gauss_background_fit(x_fit, y_fit, yerr_fit, mu_bounds=mu_bounds, mu_guess=modes[i], sigma_bounds=sigma_bounds)
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

def estimate_lfc_background(spec : np.ndarray, min_mode_spacing : int = 12, n_modes : int = 10) -> np.ndarray:
    filter_width = int(min_mode_spacing / 3 * n_modes)
    if filter_width % 2 == 0:
        filter_width += 1  # Ensure filter width is odd
    background = _nanmin_filter1d(spec, filter_width)
    background = savgol_filter(background, filter_width, polyorder=1, mode='nearest')
    return background

def _nanmin_filter1d(data: np.ndarray, width: int) -> np.ndarray:
    if width % 2 == 0:
        width += 1  # ensure odd
    pad = width // 2
    padded = np.pad(data, pad_width=pad, mode='edge')
    windows = sliding_window_view(padded, width)
    return np.nanmin(windows, axis=1)

def find_initial_modes(spec, dq, min_mode_spacing : float = 10) -> dict:
    
    # Estimate and remove background
    background = estimate_lfc_background(spec, min_mode_spacing=min_mode_spacing, n_modes=2)
    spec_nobkg = spec - background

    # Find peaks
    modes, _ = scipy.signal.find_peaks(spec_nobkg, distance=min_mode_spacing - 2)
    # NOTE: Keep plots here for debugging.
    # import matplotlib.pyplot as plt
    #plt.plot(spec); [plt.axvline(m, color='r', linestyle='--') for m in modes]
    #plt.plot(spec); plt.plot(background, color='orange', label='Background'); plt.show()

    # Refine modes based on centroid
    w2 = 3
    modes = modes[1:-1] # Ignore first and last modes, possibly noise
    modes = modes.astype(np.float64)
    for i in range(len(modes)):
        ki = int(max(modes[i] - w2, 0))
        kf = int(min(modes[i] + w2, len(spec) - 1))
        good = np.where(dq[ki:kf+1] == 0)[0]
        if len(good) > 0:
            modes[i] = np.sum((good + ki) * spec[ki:kf+1][good]) / np.sum(spec[ki:kf+1][good])

    return modes
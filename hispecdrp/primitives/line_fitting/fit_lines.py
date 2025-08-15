import numpy as np
import scipy
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import savgol_filter
from ...utils.endian_utils import normalize_dtype_array
from numba import njit


def fit_lines(
    spec : np.ndarray, specerr : np.ndarray, dq : np.ndarray,
    lines : np.ndarray,
    fwhm_bounds : tuple = (1.5, 8), mu_bounds : tuple = (-1, 1),
    max_fit_window : float | None = None,
    profile_type : str = 'gaussian',
) -> list[dict[str, float]]:
    nx = len(spec)
    n_lines = len(lines)
    xarr = np.arange(nx)
    ds = np.gradient(lines)
    
    fit_results = []
    for i in range(n_lines):
        fit_window = (np.floor(-ds[i] / 2), np.ceil(ds[i] / 2))
        fit_window = np.clip(fit_window, -max_fit_window / 2, max_fit_window / 2) if max_fit_window is not None else fit_window
        ki = max(0, int(np.round(lines[i] + fit_window[0])))
        kf = min(nx - 1, int(np.round(lines[i] + fit_window[1])))
        x_fit = xarr[ki:kf+1].copy()
        y_fit = spec[ki:kf+1].copy()
        yerr_fit = specerr[ki:kf+1].copy()
        fit_result = profile_background_fit(
            x_fit, y_fit, yerr_fit,
            fwhm_bounds=fwhm_bounds,
            mu_bounds=mu_bounds,
            mu_guess=lines[i], 
            profile_type=profile_type
        )
        fit_results.append(fit_result)

    return fit_results


def calculate_emission_spectrum_background(spec : np.ndarray, min_line_spacing : int = 12, n_lines : int = 10) -> np.ndarray:
    filter_width = int(min_line_spacing / 3 * n_lines)
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

@njit(nogil=True)
def _gaussian(x, amp, mu, fwhm):
    """Gaussian profile parameterized by FWHM."""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


@njit(nogil=True)
def _lorentzian(x, amp, mu, fwhm):
    """Lorentzian profile parameterized by FWHM."""
    gamma = fwhm / 2  # Half-width at half-maximum
    return amp * gamma**2 / ((x - mu)**2 + gamma**2)


@njit(nogil=True)
def _profile_background(params, x, profile_func):
    model_y = profile_func(x, params[0], params[1], params[2])
    if len(params) > 3:
        model_y = model_y + params[3]
    return model_y
    

@njit(nogil=True)
def _profile_background_chi2_residuals(params, data_x, data_y, data_yerr, profile_func):
    """Calculate chi-squared residuals for profile + background model."""
    model_y = _profile_background(params, data_x, profile_func)
    wres = ((data_y - model_y) / data_yerr)
    return wres


def profile_background_fit(
    x : np.ndarray, y : np.ndarray, yerr : np.ndarray, 
    fwhm_bounds : tuple | None = None,
    mu_guess : float | None = None,
    mu_bounds : tuple | None = (-np.inf, np.inf),
    profile_type : str = 'gaussian',
) -> dict[str, float]:
    """
    Fit a profile (Gaussian or Lorentzian) plus background to data.
    
    Args:
        x: Independent variable array
        y: Dependent variable array  
        yerr: Uncertainties on y
        fwhm_bounds: Bounds on the FWHM parameter
        mu_guess: Initial guess for line center
        mu_bounds: Bounds on line center relative to mu_guess
        profile_type: 'gaussian' or 'lorentzian'
    
    Returns:
        Dictionary with fit parameters and uncertainties
    """
    
    # Validate profile type
    if profile_type not in ['gaussian', 'lorentzian']:
        raise ValueError("profile_type must be 'gaussian' or 'lorentzian'")

    if profile_type == 'gaussian':
        profile_func = _gaussian
    elif profile_type == 'lorentzian':
        profile_func = _lorentzian
    
    # Initial params
    ymin = np.nanmin(y)
    yy = y - ymin
    ymax = np.nanmax(yy)
    yy /= ymax
    yyerr = yerr / ymax
    
    if mu_guess is None:
        mu_guess = np.average(x, weights=yy)
    
    # Estimate initial FWHM from data
    sigma_equiv = np.sqrt(np.average((x - mu_guess)**2, weights=yy))
    fwhm0 = sigma_equiv * 2 * np.sqrt(2 * np.log(2))  # Convert sigma-equivalent to FWHM
    
    if fwhm_bounds is not None:
        if fwhm0 < fwhm_bounds[0] or fwhm0 > fwhm_bounds[1]:
            fwhm0 = np.mean(fwhm_bounds)
    
    b0 = 0.01
    p0 = np.array([1.0, mu_guess, fwhm0, b0])
    
    # Set up bounds
    if fwhm_bounds is None:
        fwhm_bounds = (fwhm0 * 0.1, fwhm0 * 10)  # Default bounds
    
    bounds = (
        np.array([0.5, mu_bounds[0] + mu_guess, fwhm_bounds[0], -0.2]),
        np.array([1.5, mu_bounds[1] + mu_guess, fwhm_bounds[1], 0.2]),
    )
    n_dof = len(x) - len(p0)

    # Fit
    x = normalize_dtype_array(x)
    y = normalize_dtype_array(y)
    yerr = normalize_dtype_array(yerr)
    
    try:
        result = scipy.optimize.least_squares(
            _profile_background_chi2_residuals, 
            p0,
            args=(x, yy, yyerr, profile_func), 
            bounds=bounds,
            ftol=1E-10, 
            xtol=1E-10
        )
    except:
        breakpoint()
        import matplotlib.pyplot as plt
        plt.plot(x, yy, label='Data')
    
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
        'amplitude': pbest[0] * ymax,  # Scale back to original units
        'amplitude_err': errors[0] * ymax,
        'mu': pbest[1],
        'mu_err': errors[1],
        'fwhm': pbest[2],
        'fwhm_err': errors[2],
        'b' : pbest[3] * ymax + ymin,  # Scale back to original units
        'b_err': errors[3] * ymax,
        'redchi2': redchi2,
        'profile_type': profile_type,
    }

    return out


def find_initial_lines(spec, dq, min_line_spacing : float = 10) -> dict:
    
    # Estimate and remove background
    background = calculate_emission_spectrum_background(spec, min_line_spacing=min_line_spacing, n_lines=2)
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
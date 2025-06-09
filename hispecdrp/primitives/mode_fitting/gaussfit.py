
import numpy as np
import scipy.optimize

from numba import njit

from ...utils.endian_utils import normalize_dtype_array

@njit(nogil=True)
def gauss(x, amp, mu, sigma):
    """
    Gaussian function.

    Args:
        x: 1D array of x values
        amp: amplitude of the Gaussian
        mu: mean (center) of the Gaussian
        sigma: standard deviation (width) of the Gaussian

    Returns:
        1D array of Gaussian values at x
    """
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

@njit(nogil=True)
def _gauss_background(params, x):
    model_y = gauss(x, params[0], params[1], params[2])
    if len(params) > 3:
        model_y = model_y + _polyval1d_horner_array(params[3:], x)
    return model_y

@njit(nogil=True)
def _polyval1d_horner(coeffs, x):
    y = 0.0
    for c in coeffs[::-1]:
        y = y * x + c
    return y

@njit(nogil=True)
def _polyval1d_horner_array(coeffs, x):
    y = np.full(x.size, np.nan, dtype=np.float64)
    for i in range(x.size):
        y[i] = _polyval1d_horner(coeffs, x[i])
    return y
    
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
    """
    Fit a Gaussian function to the data using a Levenberg Marquardt solver.

    Args:
        x (np.ndarray): 1D array of x values
        y (np.ndarray): 1D array of y values
        yerr (np.ndarray): 1D array of uncertainties in y values.
        sigma_bounds (tuple): Bounds on Gaussian sigma.
        background_poly_deg (int | None): Degree of polynomial for background fitting. If None, no background fitting is performed.
    """
    import matplotlib.pyplot as plt
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
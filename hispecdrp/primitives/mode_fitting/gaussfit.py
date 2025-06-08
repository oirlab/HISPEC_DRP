
import numpy as np
import scipy.optimize

from numba import njit

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
    x : np.ndarray, y : np.ndarray, yerr : np.ndarray,
    mu_bounds : tuple | None = None, sigma_bounds : tuple | None = None,
    mu_guess : float | None = None,
    max_fit_window : float | None = None,
) -> dict[str, float]:
    """
    Fit a Gaussian function to the data using a Levenberg Marquardt solver.

    Args:
        x (np.ndarray): 1D array of x values
        y (np.ndarray): 1D array of y values
        yerr (np.ndarray): 1D array of uncertainties in y values.
        mu_bounds (tuple): Bounds on Gaussian mean.
        sigma_bounds (tuple): Bounds on Gaussian sigma.
        background_poly_deg (int | None): Degree of polynomial for background fitting. If None, no background fitting is performed.
    """

    # Normalize inputs
    dx = x[-1] - x[0]
    xx = (x - x[0]) / dx
    ymin = np.nanmin(y)
    yy = y - ymin
    ymax = np.nanmax(yy)
    yy = yy / ymax
    yyerr = yerr / ymax
    if mu_guess is None:
        mu_guess = 0.01
    else:
        mu_guess = (mu_guess - x[0]) / dx
    mu_bounds_scaled = (mu_bounds[0] / dx + 0.5, mu_bounds[1] / dx + 0.5)
    sigma_bounds_scaled = (sigma_bounds[0] / dx, sigma_bounds[1] / dx)
    sigma0 = sigma_bounds_scaled[0] + (sigma_bounds_scaled[1] - sigma_bounds_scaled[0]) / 2

    # Initial params
    p0 = np.array([1.0, mu_guess, sigma0, 0.01])
    bounds = (
        np.array([0.5, mu_bounds_scaled[0], sigma_bounds_scaled[0], -0.2]),
        np.array([1.5, mu_bounds_scaled[1], sigma_bounds_scaled[1], 0.2])
    )
    n_dof = len(x) - len(p0)

    # Fit
    result = scipy.optimize.least_squares(_gauss_background_chi2_residuals, p0, args=(xx, yy, yyerr), bounds=bounds, ftol=1E-10)

    # Collect output
    out = {
        'amplitude': result.x[0] * ymax,
        'mu': result.x[1] * dx + x[0],
        'sigma': result.x[2] * dx,
        'b' : result.x[3] + ymin,
        'redchi2': result.cost / n_dof,
    }

    # Return
    return out
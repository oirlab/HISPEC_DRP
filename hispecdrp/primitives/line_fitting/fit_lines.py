from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.optimize
from numba import njit
from hispecdrp.primitives.utils.array_utils import polyval1d


import logging
logger = logging.getLogger(__name__)

SIGMA_TO_FWHM = 2 * np.sqrt(2 * np.log(2))

LINE_DQ_FLAGS = {
    'GOOD': np.uint8(0),
    'DO_NOT_USE': np.uint8(2**0),
    'FIT_FAILURE': np.uint8(2**1),
    'POOR_FIT': np.uint8(2**2),
    'LOW_SNR': np.uint8(2**3),
}

def fit_lines_gauss(
    x : np.ndarray,
    y : np.ndarray, y_err : np.ndarray, dq : np.ndarray,
    line_centers : np.ndarray,
    fwhm_bounds : tuple[float, float] | None = None,
    mu_bounds : tuple[float, float] | None = None,
    min_fit_window : int | None = None,
    max_fit_window : int | None = None,
    background_poly_deg : int = 0,
    absorption : bool = False,
) -> list[GaussianLineFitResult]:

    nx = len(y)
    n_lines = len(line_centers)
    line_spacing = np.gradient(line_centers)

    fit_results = []
    for i in range(n_lines):

        # Half-window from spacing to neighbors, then clamp to [min, max] (full-window widths in px)
        half = line_spacing[i] / 2
        if min_fit_window is not None:
            half = max(half, min_fit_window / 2)
        if max_fit_window is not None:
            half = min(half, max_fit_window / 2)
        fit_window = np.array([-half, half])

        # Only pass data in the fitting window
        ki = max(0, int(np.round(line_centers[i] + fit_window[0])))
        kf = min(nx - 1, int(np.round(line_centers[i] + fit_window[1])))
        if kf < ki: # swap if these have somehow gotten backwards
            ki, kf = kf, ki
        x_fit = x[ki:kf+1].copy()
        y_fit = y[ki:kf+1].copy()
        y_err_fit = y_err[ki:kf+1].copy() if y_err is not None else None
        dq_fit = dq[ki:kf+1].copy() if dq is not None else None

        # Fit the line
        fit_result = fit_line_gauss(
            x_fit, y_fit, y_err_fit, dq_fit,
            fwhm_bounds=fwhm_bounds,
            mu_bounds=mu_bounds,
            mu_guess=line_centers[i],
            background_poly_deg=background_poly_deg,
            absorption=absorption,
        )

        # Append dict result to output list
        fit_results.append(fit_result)

    return fit_results


@dataclass
class GaussianLineFitResult:
    amplitude: float
    amplitude_err: float
    mu: float
    mu_err: float
    fwhm: float
    fwhm_err: float
    background_coeffs: np.ndarray
    background_coeffs_err: np.ndarray
    redchi2: float
    pcov : np.ndarray | None = None

    @property
    def popt(self) -> np.ndarray:
        return np.concatenate(([self.amplitude, self.mu, self.fwhm], self.background_coeffs))

    @property
    def perr(self) -> np.ndarray:
        return np.concatenate(([self.amplitude_err, self.mu_err, self.fwhm_err], self.background_coeffs_err))

    def to_row(self) -> dict:
        return {
            "amplitude": self.amplitude,
            "amplitude_err": self.amplitude_err,
            "mu": self.mu,
            "mu_err": self.mu_err,
            "fwhm": self.fwhm,
            "fwhm_err": self.fwhm_err,
            "background_coeffs": self.background_coeffs,
            "background_coeffs_err": self.background_coeffs_err,
            "redchi2": self.redchi2,
        }


def fit_line_gauss(
    x : np.ndarray,
    y : np.ndarray,
    y_err : np.ndarray | None = None,
    dq : np.ndarray | None = None,
    fwhm_bounds : tuple[float, float] | None = None,
    mu_guess : float | None = None,
    mu_bounds : tuple[float, float] | None = None,
    background_poly_deg : int = 0,
    absorption : bool = False,
) -> GaussianLineFitResult:
    """
    Fit a Gaussian profile to an absorption or emission line plus polynomial background to data.

    For ``absorption=False`` the model is ``amplitude * gaussian + background``
    (a peak above the continuum). For ``absorption=True`` the model is
    ``background - amplitude * gaussian`` (a dip below the continuum). In both
    cases ``amplitude >= 0`` is the line strength (depth for absorption); the
    continuum is carried by the background polynomial.

    Parameters
    ----------
    x : np.ndarray
        X-axis values (e.g., pixel positions, wavelengths, or a velocity grid).
    y : np.ndarray
        Flux values.
    y_err : np.ndarray | None
        Uncertainties on y.
    dq : np.ndarray | None
        Data quality mask for data.
    fwhm_bounds : tuple | None
        Bounds on the FWHM parameter.
    mu_guess : float | None
        Initial guess for line center.
    mu_bounds : tuple | None
        Bounds on line center relative to mu_guess.
    background_poly_deg : int
        Degree of the polynomial background model. For absorption a continuum
        is required, so this must be >= 0 (degree 0 = flat continuum).
    absorption : bool
        Whether to fit an absorption line (True) or emission line (False).

    Returns
    -------
    GaussianLineFitResult
        Result of the Gaussian line fit. ``amplitude`` is always the
        (positive) line strength; ``background_coeffs`` is the continuum.
    """

    if not absorption:
        residuals_func = _gauss_emission_residuals
        jac_func = _gauss_emission_jac
    else:
        residuals_func = _gauss_absorption_residuals
        jac_func = _gauss_absorption_jac

    if dq is None:
        dq = (np.isfinite(y) & np.isfinite(y_err)) == 0
    good = np.where(dq == 0)[0]

    def _output_badfit():
        return GaussianLineFitResult(
            amplitude=np.nan,
            amplitude_err=np.nan,
            mu=np.nan,
            mu_err=np.nan,
            fwhm=np.nan,
            fwhm_err=np.nan,
            background_coeffs=np.full(background_poly_deg + 1, np.nan),
            background_coeffs_err=np.full(background_poly_deg + 1, np.nan),
            redchi2=np.nan
        )

    if len(good) < 3:
        logger.warning(f"Not enough good data points to perform fit. ({len(good)} good data points)")
        return _output_badfit()

    x_fit = x[good]
    y_fit = y[good]
    if y_err is not None:
        y_err_fit = y_err[good]
    else:
        y_err_fit = np.ones_like(y_fit)

    bad_errs = np.where((y_err_fit <= 0) | (~np.isfinite(y_err_fit)))[0]
    if len(bad_errs) == len(y_fit):
        y_err_fit = np.ones_like(y_fit)
    elif len(bad_errs) > 0:
        return _output_badfit()

    # Normalize. Emission: anchor peak to ~1 (subtract floor, divide by peak).
    # Absorption: a continuum is present by definition; anchor the continuum to
    # ~1 (divide by a robust continuum level) and let the dip go below 1. Do NOT
    # subtract the floor for absorption -- the floor IS the line bottom.
    if not absorption:
        y_min = np.min(y_fit)
        y_fit = y_fit - y_min
        norm = np.nanmax(y_fit)
    else:
        y_min = 0.0
        norm = np.nanpercentile(y_fit, 90)  # robust continuum estimate
    y_fit = y_fit / norm if norm != 0 else y_fit
    if y_err_fit is not None:
        y_err_fit = y_err_fit / norm if norm != 0 else None

    # Line-strength weights for the seeds: flux for emission, depth
    # (continuum - flux) for absorption, since for a dip the flux is
    # concentrated in the continuum rather than at the line.
    if not absorption:
        w_seed = np.clip(y_fit, 0, None)
    else:
        w_seed = np.clip(np.nanmax(y_fit) - y_fit, 0, None)

    if np.sum(w_seed) == 0. or ~np.isfinite(np.sum(w_seed)):
        logger.warning(f"Seed weights sum to zero or are not finite, cannot fit line.")
        return _output_badfit()

    # Default mu guess
    if mu_guess is None:
        mu_guess = np.average(x_fit, weights=w_seed)
    if mu_bounds is None:
        mu_bounds = (x_fit[-1] - x_fit[0]) * np.array([-0.1, 0.1])

    # Estimate initial FWHM from data
    sigma_equiv = np.sqrt(np.average((x_fit - mu_guess)**2, weights=w_seed))
    fwhm0 = sigma_equiv * SIGMA_TO_FWHM  # Convert sigma-equivalent to FWHM
    if fwhm_bounds is not None:
        if fwhm0 < fwhm_bounds[0] or fwhm0 > fwhm_bounds[1]:
            fwhm0 = np.mean(fwhm_bounds)

    # Default bounds
    if fwhm_bounds is None:
        fwhm_bounds = (fwhm0 * 0.2, fwhm0 * 5)

    # Init params: amp, mu, fwhm, b0, b1, b2.
    # Continuum guess: emission ~0 floor (data already floored); absorption ~1
    # (data divided by continuum). Seed b0 accordingly so the fit starts sane.
    n_bkg_coeff = background_poly_deg + 1
    bkg0 = [0.0] * n_bkg_coeff
    amp0 = 1.0
    if not absorption:
        bkg0[0] = 1.0  # continuum ~1 after normalization
        amp0 = np.clip(1.0 - np.nanmin(y_fit), 0.0, 1.5)  # depth guess
    p0 = np.array([amp0, mu_guess, fwhm0] + bkg0)
    lb = np.array([0.0, mu_bounds[0] + mu_guess, fwhm_bounds[0]] + [-np.inf] * n_bkg_coeff)
    ub = np.array([1.5, mu_bounds[1] + mu_guess, fwhm_bounds[1]] + [np.inf] * n_bkg_coeff)
    bounds = (lb, ub)

    # Deg of freedom
    n_dof = len(x_fit) - len(p0)

    # Fit
    try:
        result = scipy.optimize.least_squares(
            residuals_func,
            p0,
            jac=jac_func,
            args=(x_fit, y_fit, y_err_fit),
            bounds=bounds,
            x_scale='jac',
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
    except Exception as e:
        logger.warning(f"Exception during fitting: {e}")
        return _output_badfit()

    pbest = result.x
    redchi2 = 2 * result.cost / n_dof

    try:
        cov = np.linalg.inv(result.jac.T @ result.jac)
    except np.linalg.LinAlgError:
        logger.exception("Covariance matrix could not be computed. Fit may be degenerate.")
        return _output_badfit()

    #breakpoint()
    # import matplotlib.pyplot as plt
    # if np.random.randint(0, 20) == 0:
    #     plt.plot(x_fit, y_fit, 'k.')
    #     plt.plot(x_fit, _gauss_emission_model(pbest, x_fit), 'r-')
    #     plt.show()

    # ``(J^T J)^{-1}`` is only approximately a covariance. On a poorly constrained
    # line the inverse can put negative values on the diagonal; ``sqrt`` of those
    # would warn and is not a real 1-sigma error. Keep the centroid and report nan.
    var = np.diag(np.asarray(cov, dtype=np.float64))
    errors = np.full(var.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(var) & (var >= 0.0)
    if np.any(ok):
        errors[ok] = np.sqrt(var[ok])

    # Scale amplitude and background back to original units. Emission subtracted
    # a floor then divided by norm; absorption only divided by norm. Restore the
    # floor into the continuum constant term for emission only.
    background_coeffs = pbest[3:] * norm
    if not absorption:
        background_coeffs = background_coeffs.copy()
        background_coeffs[0] = background_coeffs[0] + y_min
    out = GaussianLineFitResult(
        amplitude=pbest[0] * norm,
        amplitude_err=errors[0] * norm,
        mu=pbest[1],
        mu_err=errors[1],
        fwhm=pbest[2],
        fwhm_err=errors[2],
        background_coeffs=background_coeffs,
        background_coeffs_err=errors[3:] * norm,
        redchi2=redchi2,
        pcov=cov
    )

    return out


@njit(nogil=True)
def _gaussian(x, amp, mu, fwhm):
    sigma = fwhm / SIGMA_TO_FWHM  # Convert FWHM to sigma
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# --- Emission: model = amplitude * gaussian + background ---------------------

@njit(nogil=True)
def _gauss_emission_model(params, x):
    model = _gaussian(x, params[0], params[1], params[2])
    if len(params) > 3:
        model = model + polyval1d(params[3:], x - np.mean(x))
    return model


@njit(nogil=True)
def _gauss_emission_residuals(params, x, y, yerr):
    model = _gauss_emission_model(params, x)
    return (y - model) / yerr


@njit(nogil=True)
def _gauss_emission_jac(params, x, y, yerr):
    amp, mu, fwhm = params[0], params[1], params[2]
    n_poly = len(params) - 3
    sigma = fwhm / SIGMA_TO_FWHM  # Convert FWHM to sigma
    z = (x - mu) / sigma
    g = np.exp(-0.5 * z**2)
    J = np.empty((x.size, params.size))
    J[:, 0] = -g / yerr                              # d/dA
    J[:, 1] = -amp * g * z / sigma / yerr            # d/dmu
    J[:, 2] = -amp * g * z**2 / fwhm / yerr          # d/dfwhm
    # Polynomial background: polyval1d(x - mean(x), params[3:])
    xc = x - np.mean(x)
    p = np.ones(x.size)
    for k in range(n_poly):
        J[:, 3 + k] = -p / yerr                      # d/d(coeff_k)
        p = p * xc
    return J


# --- Absorption: model = background - amplitude * gaussian --------------------

@njit(nogil=True)
def _gauss_absorption_model(params, x):
    model = -1 * _gaussian(x, params[0], params[1], params[2])
    if len(params) > 3:
        model = model + polyval1d(params[3:], x - np.mean(x))
    return model


@njit(nogil=True)
def _gauss_absorption_residuals(params, x, y, yerr):
    model = _gauss_absorption_model(params, x)
    return (y - model) / yerr


@njit(nogil=True)
def _gauss_absorption_jac(params, x, y, yerr):
    amp, mu, fwhm = params[0], params[1], params[2]
    n_poly = len(params) - 3
    sigma = fwhm / SIGMA_TO_FWHM  # Convert FWHM to sigma
    z = (x - mu) / sigma
    g = np.exp(-0.5 * z**2)
    J = np.empty((x.size, params.size))
    # Profile enters with a minus sign, so residual derivatives flip sign.
    J[:, 0] = g / yerr                               # d/dA
    J[:, 1] = amp * g * z / sigma / yerr             # d/dmu
    J[:, 2] = amp * g * z**2 / fwhm / yerr           # d/dfwhm
    # Polynomial background: polyval1d(x - mean(x), params[3:])
    xc = x - np.mean(x)
    p = np.ones(x.size)
    for k in range(n_poly):
        J[:, 3 + k] = -p / yerr                      # d/d(coeff_k)
        p = p * xc
    return J
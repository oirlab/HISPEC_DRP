from dataclasses import dataclass

import numpy as np
from hispecdrp.primitives.line_fitting.fit_lines import SIGMA_TO_FWHM, fit_line_gauss, GaussianLineFitResult

from hispecdrp.primitives.utils.ccf_bisector import calc_bisector_inverse_slope

__all__ = ["calc_ccf_metrics"]

@dataclass
class CCFMetrics:
    """
    Dataclass to hold the results of a CCF fit.
    """
    contrast: float
    contrast_err: float
    rv: float
    rv_err: float
    fwhm: float
    fwhm_err: float
    offset: float
    offset_err: float
    bis_rv: float
    bis_rv_err: float
    bis_shallow : float | None = None
    bis_shallow_err : float | None = None
    bis_core : float | None = None
    bis_core_err : float | None = None
    bis_levels: np.ndarray | None = None
    bisector: np.ndarray | None = None
    bis_quant_shallow : tuple[float, float] | None = None
    bis_quant_core : tuple[float, float] | None = None

    @property
    def sigma(self) -> float:
        return self.fwhm / SIGMA_TO_FWHM

    @property
    def sigma_err(self) -> float:
        return self.fwhm_err / SIGMA_TO_FWHM

    @property
    def popt(self) -> np.ndarray:
        return np.array([self.contrast, self.rv, self.fwhm, self.offset])

    @property
    def perr(self) -> np.ndarray:
        return np.array([self.contrast_err, self.rv_err, self.fwhm_err, self.offset_err])

    def to_row(self) -> dict:
        return {
            "contrast": self.contrast,
            "contrast_err": self.contrast_err,
            "rv": self.rv,
            "rv_err": self.rv_err,
            "fwhm": self.fwhm,
            "fwhm_err": self.fwhm_err,
            "offset": self.offset,
            "offset_err": self.offset_err,
            "bis_rv": self.bis_rv,
            "bis_rv_err": self.bis_rv_err,
        }


def _calc_rverr_from_ccf(
    vel_grid: np.ndarray,
    ccf: np.ndarray,
    ccf_err: np.ndarray | None = None,
    rv0: float | None = None,
    fwhm: float | None = None,
    fwhm_factor: float = 0.85,
    pix_size : float | None = None,
) -> float:
    """
    Estimate the 1-sigma uncertainty on a CCF-derived radial velocity
    from the Fisher information:

        sigma_v^-2 = sum_i ( dCCF/dv |_i )^2 / sigma_eff_i^2

    where sigma_eff_i is the per-point CCF error, inflated to account
    for the correlation between adjacent CCF velocity bins introduced
    by top-hat binning of detector pixels onto an oversampled velocity
    grid:

        sigma_eff_i = ccf_err_i * sqrt(nsp)
        nsp = tophat_width / grid_spacing

    Parameters
    ----------
    vel_grid : ndarray
        1-D float ndarray: velocity grid (m/s) on which the CCF was
        evaluated, i.e. the centers of the top-hat bins. Must already be
        a 1-D numpy array (no list/Series coercion or flattening is
        performed).
    ccf : ndarray
        1-D float ndarray, same shape as `vel_grid`: CCF values at each
        point in `vel_grid`. Used to compute the local derivative
        dCCF/dv; also used as a noise proxy (sqrt(ccf)) if `ccf_err` is
        not provided.
    rv0 : float, optional
        Center velocity (m/s) of the window, typically the fitted RV.
        If None, estimated as the first moment (intensity-weighted
        centroid) of the inverted CCF: weight_i = max(ccf) - ccf_i.
    fwhm : float, optional
        Width (m/s) of the window around rv0 to use for the fit.
        Typically the fitted CCF FWHM. If None, estimated from the
        second moment (intensity-weighted variance) of the inverted
        CCF, converted to FWHM assuming a Gaussian-like core:
        fwhm = 2*sqrt(2*ln2) * sqrt(variance).
    fwhm_factor : float, default 0.85
        Factor multiplied by the FWHM to define the window around rv0 to use for the fit.
        Example: fwhm_factor=0.85 uses a window of +/- 0.85*FWHM around rv0.
    ccf_err : ndarray, optional
        1-D float ndarray, same shape as `ccf`: 1-sigma uncertainty on
        each CCF value. Assumed diagonal prior to the nsp inflation (no
        covariance matrix available). If None, falls back to a
        photon/Poisson noise proxy: sqrt(ccf).
    pix_size : float | None
        Size of a pixel in the velocity grid (m/s) to account for oversampling the CCF.
        If None, no oversampling correction is applied (nsp=1).

    Returns
    -------
    float
        Estimated 1-sigma RV uncertainty in m/s. Returns 0.0 if the
        inputs don't support a calculation (too few points, bad window,
        non-positive errors, non-positive tophat_width, etc.).
    """

    # Ensure correct shapes
    if vel_grid.shape != ccf.shape:
        raise ValueError("vel_grid and ccf must have the same shape")

    # Check that CCF and errors are positive and finite and have enough points for a fit
    if ccf_err is None:
        if np.any(ccf < 0):
            return 0.0
        ccf_err = np.sqrt(np.maximum(ccf, 1e-30))
    elif ccf_err.shape != ccf.shape:
        raise ValueError("ccf_err must have the same shape as ccf")

    # Mask out any non-finite or non-positive points in ccf and ccf_err
    good = np.isfinite(vel_grid) & np.isfinite(ccf) & np.isfinite(ccf_err)
    vel_grid, ccf, ccf_err = vel_grid[good], ccf[good], ccf_err[good]

    # Can't fit with fewer than 3 points in the window
    if vel_grid.size < 3:
        return 0.0

    # Compute zeroth and first moments if not provided
    if rv0 is None or fwhm is None:
        weight = np.maximum(np.max(ccf) - ccf, 0.0)
        weight_sum = np.sum(weight)
        if weight_sum <= 0 or not np.isfinite(weight_sum):
            return 0.0

        if rv0 is None:
            rv0 = float(np.sum(weight * vel_grid) / weight_sum)

        if fwhm is None:
            variance = float(np.sum(weight * (vel_grid - rv0) ** 2) / weight_sum)
            if variance <= 0 or not np.isfinite(variance):
                return 0.0
            fwhm = float(2.0 * np.sqrt(2.0 * np.log(2.0)) * np.sqrt(variance))

    # FWHM must be positive
    if fwhm <= 0:
        return np.nan

    # Correction factor for oversampling
    delta_vel = vel_grid[1] - vel_grid[0]
    nsp = max(pix_size / delta_vel, 1.0)

    # Restrict to the window around rv0
    fwhm_factor_half = fwhm_factor / 2.0
    in_window = (vel_grid >= rv0 - fwhm * fwhm_factor_half) & (vel_grid <= rv0 + fwhm * fwhm_factor_half)
    vel_window = vel_grid[in_window]
    ccf_window = ccf[in_window]
    ccf_err_window = ccf_err[in_window]

    # Check again that we have enough valid points
    if vel_window.size < 3:
        return np.nan
    if np.any(ccf_err_window <= 0) or np.any(~np.isfinite(ccf_err_window)):
        return np.nan

    # Inflate errors
    effective_err = ccf_err_window * np.sqrt(nsp)

    # dCCF/dv via finite differences
    gradient = np.gradient(ccf_window, vel_window)

    # FISHER info
    fisher_info = np.sum((gradient / effective_err) ** 2)

    if fisher_info <= 0 or not np.isfinite(fisher_info):
        return np.nan

    # Compute RV uncertainty from Fisher information
    rv_err = 1.0 / np.sqrt(fisher_info)

    return rv_err


def calc_ccf_metrics(
    vel_grid : np.ndarray,
    ccf : np.ndarray,
    ccf_err : np.ndarray | None = None,
    pix_size : float | None = None,
    absorption : bool = False,
    calc_bis : bool = True,
    n_levels_bis : int = 100,
    bis_quant_shallow : tuple[float, float] | None = (0.10, 0.40),
    bis_quant_core : tuple[float, float] | None = (0.55, 0.90),
    override_rv_err_fisher : bool = True,
) -> GaussianLineFitResult:
    """
    Fit a Gaussian plus flat background to a CCF.

    Parameters
    ----------
    vel_grid : ndarray
        Array of velocity trial values for the CCF.
    ccf : ndarray
        Array of CCF values corresponding to `vel_grid`.
    ccf_err : ndarray, optional
        Array of 1-sigma uncertainties on the CCF values.
    absorption : bool, optional
        Whether to fit an absorption line (True) or emission line (False). Default False.
    pix_size : float, optional
        Size of a pixel in the velocity grid (m/s) to account for oversampling the CCF.
    calc_bis : bool, optional
        Whether to calculate the bisector inverse slope. Default True.
    n_levels_bis : int, optional
        Number of levels to use for the bisector calculation. Default 100.
    bis_quant_shallow : tuple[float, float], optional
        Lower/upper depth fractions (0-1, measured from the continuum) bounding
        the shallow (near-continuum) region used for ``v_shallow``.
        Defaults to (0.10, 0.40).
    bis_quant_core : tuple[float, float], optional
        Lower/upper depth fractions (0-1, measured from the continuum) bounding
        the core (near-core) region used for ``v_core``.
        Defaults to (0.55, 0.90).
    override_rv_err_fisher : bool, optional
        Whether to override the RV error from the Gaussian fit with the Fisher information estimate. Default True.

    Returns
    -------
    result : GaussianLineFitResult
        Dataclass containing the fitted parameters and their uncertainties.
        Bisector inverse slope results are included if `calc_bis` is True.
    """

    # Index of the peak (emission line) or trough (abosrption line) in the CCF
    idx_mu = np.argmax(ccf) if not absorption else np.argmin(ccf)
    mu_guess = vel_grid[idx_mu]

    # Default CCF error
    if ccf_err is None:
        ccf_err = np.sqrt(np.maximum(ccf, 1e-30))

    # Oversample correction (naive)
    if pix_size is not None:
        delta_vel = vel_grid[1] - vel_grid[0]
        nsp = max(pix_size / delta_vel, 1.0)
        ccf_err_eff = ccf_err * np.sqrt(nsp)
    else:
        ccf_err_eff = ccf_err

    # Fit CCF with gaussian
    fit_result = fit_line_gauss(
        vel_grid, ccf, ccf_err_eff,
        mu_guess=mu_guess,
        background_poly_deg=0,
        absorption=absorption,
    )

    # Override the RV error
    if override_rv_err_fisher:
        rv_err = _calc_rverr_from_ccf(
            vel_grid, ccf, ccf_err,
            rv0=fit_result.mu,
            fwhm=fit_result.fwhm,
            pix_size=pix_size
        )
    else:
        rv_err = fit_result.mu_err

    if calc_bis:
        (
            bis_rv, bis_rv_err,
            bis_shallow, bis_shallow_err,
            bis_core, bis_core_err,
            bis_levels, bisector
        ) = calc_bisector_inverse_slope(
            vel_grid,
            ccf, ccf_err_eff,
            n_levels=n_levels_bis,
            quant_shallow=bis_quant_shallow,
            quant_core=bis_quant_core
        )
    else:
        bis_rv = None
        bis_rv_err = None
        bis_shallow = None
        bis_shallow_err = None
        bis_core = None
        bis_core_err = None
        bis_levels = None
        bisector = None

    ccf_result = CCFMetrics(
        contrast=fit_result.amplitude,
        contrast_err=fit_result.amplitude_err,
        rv=fit_result.mu,
        rv_err=rv_err,
        fwhm=fit_result.fwhm,
        fwhm_err=fit_result.fwhm_err,
        offset=fit_result.background_coeffs[0],
        offset_err=fit_result.background_coeffs_err[0],
        bis_rv=bis_rv,
        bis_rv_err=bis_rv_err,
        bis_shallow=bis_shallow,
        bis_shallow_err=bis_shallow_err,
        bis_core=bis_core,
        bis_core_err=bis_core_err,
        bis_levels=bis_levels,
        bisector=bisector,
        bis_quant_shallow=bis_quant_shallow,
        bis_quant_core=bis_quant_core
    )

    return ccf_result
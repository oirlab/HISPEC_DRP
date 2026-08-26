from ..base_primitive import BasePrimitive
from ... import datamodels
from ...calibrations import GainSelector,ReadNoiseSelector
from ...datamodels.dqflags import DQ_FLAGS
import logging

from .fit_ramp_numba import fit_ramps_ols, fit_ramps_mcds
from .fit_ramp_jwst import fit_ramps_jwst

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    'FitRampPrimitive',
    'sanitize_ramp_fit_rate_err',
    'clip_noise_driven_poisson_variance',
]

_DO_NOT_USE = int(DQ_FLAGS["DO_NOT_USE"])

# Optional detection gate for mode='detect_nsigma' (legacy empty-sky policy).
_POISSON_DETECT_NSIGMA = 3.0

# smooth_detect: gate on a spatially smoothed rate so empty sky stays RN-only
# while faint illuminated structure (orders) keeps pixel Poisson.
_POISSON_SMOOTH_SIGMA = 1.0
_POISSON_SMOOTH_NSIGMA = 1.0

# Default balances empty-sky pulls with faint illuminated emp/mean(err²).
_POISSON_VARIANCE_MODE_DEFAULT = "smooth_detect"
_POISSON_VARIANCE_MODES = frozenset({
    "positive_only",
    "detect_nsigma",
    "smooth_detect",
    "stcal",
})


def _nan_safe_gaussian_filter(rate: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smooth that ignores non-finite pixels (normalized by weight)."""
    from scipy.ndimage import gaussian_filter

    sig = float(sigma)
    if sig <= 0.0:
        return np.asarray(rate, dtype=np.float64)

    rate_a = np.asarray(rate, dtype=np.float64)
    finite = np.isfinite(rate_a)
    fill = np.where(finite, rate_a, 0.0)
    weight = finite.astype(np.float64)
    # Support 1-D unit tests and 2-D detector images.
    num = gaussian_filter(fill, sigma=sig, mode="nearest")
    den = gaussian_filter(weight, sigma=sig, mode="nearest")
    out = np.zeros_like(num)
    np.divide(num, den, out=out, where=den > 1e-6)
    return out


def clip_noise_driven_poisson_variance(
    rate,
    rate_var_poisson,
    rate_var_rnoise,
    *,
    mode: str = _POISSON_VARIANCE_MODE_DEFAULT,
    n_sigma: float = _POISSON_DETECT_NSIGMA,
    smooth_sigma: float = _POISSON_SMOOTH_SIGMA,
    smooth_nsigma: float = _POISSON_SMOOTH_NSIGMA,
    rate_err=None,
):
    """
    Adjust reported Poisson variance after the ramp fit.

    Ramp fits (including ``stcal`` likely) evaluate photon-noise variance at a
    non-negative function of the fitted rate (Wald plug-in). Modes:

    - ``'smooth_detect'`` (default): keep the fit Poisson term where
      ``rate > 0`` **and** a Gaussian-smoothed rate exceeds
      ``smooth_nsigma * sqrt(rate_var_rnoise)``. Empty sky stays RN-limited
      (isolated positive noise spikes are dropped); faint illuminated
      structure keeps Poisson because neighbors hold the smooth rate up.
    - ``'positive_only'``: zero Poisson only where ``rate <= 0``.
    - ``'detect_nsigma'``: also zero Poisson unless
      ``rate > n_sigma * sqrt(rate_var_rnoise)`` (legacy; underestimates
      faint-source ``rate_err``).
    - ``'stcal'``: leave ``rate_var_poisson`` unchanged; only rebuild
      ``rate_err = sqrt(var_poisson + var_rnoise)``.

    If ``rate_err`` is provided, it is preserved wherever ``var_poisson`` /
    ``var_rnoise`` cannot form a finite rebuilt uncertainty (e.g. legacy
    fitters that omit the variance split).

    Returns
    -------
    rate_var_poisson_out, rate_err_out, n_clipped
    """
    rate_a = np.asarray(rate, dtype=np.float64)
    vp = np.asarray(rate_var_poisson, dtype=np.float64).copy()
    vr = np.asarray(rate_var_rnoise, dtype=np.float64)
    if rate_a.shape != vp.shape or rate_a.shape != vr.shape:
        raise ValueError(
            f"shape mismatch: rate={rate_a.shape} var_poisson={vp.shape} "
            f"var_rnoise={vr.shape}"
        )

    mode_key = str(mode).strip().lower()
    if mode_key not in _POISSON_VARIANCE_MODES:
        raise ValueError(
            f"poisson variance mode must be one of {sorted(_POISSON_VARIANCE_MODES)}, "
            f"got {mode!r}"
        )

    # Only gate / rebuild where variance planes are usable. Fitters that leave
    # NaN var_* (historical cython_likely) must not wipe a finite rate_err.
    var_ok = np.isfinite(vp) & np.isfinite(vr) & (vr >= 0.0) & (vp >= 0.0)

    n_clipped = 0
    if mode_key != "stcal":
        nondetect = ~np.isfinite(rate_a) | (rate_a <= 0.0)
        if mode_key == "detect_nsigma":
            sig_rn = np.sqrt(np.maximum(vr, 0.0))
            thresh = float(n_sigma) * sig_rn
            nondetect = nondetect | ~np.isfinite(sig_rn) | (rate_a < thresh)
        elif mode_key == "smooth_detect":
            sig_rn = np.sqrt(np.maximum(vr, 0.0))
            # Smooth only where rates are finite; invalid vr already in ~var_ok.
            rate_s = _nan_safe_gaussian_filter(rate_a, smooth_sigma)
            thresh = float(smooth_nsigma) * sig_rn
            nondetect = (
                nondetect
                | ~np.isfinite(sig_rn)
                | ~np.isfinite(rate_s)
                | (rate_s < thresh)
            )
        clip_mask = nondetect & var_ok & (vp > 0.0)
        n_clipped = int(np.count_nonzero(clip_mask))
        vp = vp.copy()
        vp[clip_mask] = 0.0

    rebuilt = np.sqrt(np.maximum(vp, 0.0) + np.maximum(vr, 0.0))
    rebuild_ok = var_ok & np.isfinite(rebuilt) & (rebuilt >= 0.0)

    if rate_err is not None:
        rate_err_out = np.asarray(rate_err, dtype=np.float64).copy()
        if rate_err_out.shape != rate_a.shape:
            raise ValueError(
                f"shape mismatch: rate={rate_a.shape} rate_err={rate_err_out.shape}"
            )
        rate_err_out[rebuild_ok] = rebuilt[rebuild_ok]
    else:
        rate_err_out = rebuilt
        rate_err_out = np.where(rebuild_ok, rate_err_out, np.nan)

    bad = ~(np.isfinite(rate_err_out) & (rate_err_out >= 0.0))
    rate_err_out = rate_err_out.astype(
        np.result_type(rate_var_poisson, np.float32), copy=False
    )
    vp_out = vp.astype(np.asarray(rate_var_poisson).dtype, copy=False)
    if np.any(bad):
        rate_err_out = np.array(rate_err_out, copy=True)
        rate_err_out[bad] = np.nan
    return vp_out, rate_err_out, n_clipped


def sanitize_ramp_fit_rate_err(
    rate,
    rate_err,
    dq,
    *,
    rate_var_rnoise=None,
    do_not_use: int = _DO_NOT_USE,
):
    """
    Make ramp-fit ``rate_err`` safe for downstream extraction.

    Ramp fits often leave empty / inter-order pixels as ``rate==0`` with NaN
    ``rate_err``. Flagging those as ``DO_NOT_USE`` and letting box extract
    renormalize the aperture caused large spectral spikes for undersampled
    PSFs. Instead:

    - Where ``rate`` is finite and ``rate_err`` is not: fill ``rate_err`` with
      ``sqrt(rate_var_rnoise)`` when that variance is finite and positive,
      otherwise ``0``. Do **not** mark ``DO_NOT_USE``.
    - Where ``rate`` is non-finite: OR ``do_not_use`` into ``dq`` and set
      ``rate_err`` to ``0`` for array hygiene.

    Returns
    -------
    rate_err_out, dq_out, n_filled_err, n_flagged_rate
        ``rate_err`` / ``dq`` are copied only when modified.
    """
    rate_a = np.asarray(rate)
    rate_err_a = np.asarray(rate_err)
    dq_a = np.asarray(dq)

    rate_finite = np.isfinite(rate_a)
    err_bad = ~np.isfinite(rate_err_a)
    fill = rate_finite & err_bad
    flag = ~rate_finite
    n_filled = int(np.count_nonzero(fill))
    n_flagged = int(np.count_nonzero(flag))

    if n_filled == 0 and n_flagged == 0:
        return rate_err_a, dq_a, 0, 0

    rate_err_out = rate_err_a.copy()
    dq_out = dq_a

    if n_filled:
        fill_val = np.zeros(n_filled, dtype=rate_err_out.dtype)
        if rate_var_rnoise is not None:
            vr = np.asarray(rate_var_rnoise)
            if vr.shape == rate_a.shape:
                vr_fill = vr[fill]
                ok = np.isfinite(vr_fill) & (vr_fill > 0)
                fill_val = fill_val.astype(np.result_type(fill_val, np.float64), copy=False)
                fill_val[ok] = np.sqrt(vr_fill[ok]).astype(fill_val.dtype, copy=False)
        rate_err_out[fill] = fill_val

    if n_flagged:
        dq_out = dq_a.copy()
        dq_out[flag] = dq_out[flag] | int(do_not_use)
        rate_err_out[flag] = np.asarray(0, dtype=rate_err_out.dtype)

    return rate_err_out, dq_out, n_filled, n_flagged


def flag_nonfinite_rate_pixels(rate, rate_err, dq, *, do_not_use: int = _DO_NOT_USE):
    """
    Backward-compatible wrapper around :func:`sanitize_ramp_fit_rate_err`.

    Historically this ORed ``DO_NOT_USE`` for non-finite ``rate`` **or**
    ``rate_err``. That mistreated empty pixels (``rate==0``, NaN err) and
    caused extraction spikes. The sanitizer now fills finite ``rate_err`` for
    those pixels and only flags non-finite ``rate``.

    Returns ``(dq, n_flagged_rate)`` for compatibility with existing callers.
    """
    _rate_err, dq_out, _n_filled, n_flagged = sanitize_ramp_fit_rate_err(
        rate, rate_err, dq, do_not_use=do_not_use
    )
    return dq_out, n_flagged


class FitRampPrimitive(BasePrimitive):
    """
    This step fits the slope (DN/s) from the UTR for each pixel.

    While fitting the ramps, this step will also calculate the wavelength dependent mid-exposure times, which is then used to calculate the wavelength dependent barycentric correction.
    HISPEC leverages the UTR instead of having an exposure meter for this purpose.

    The entire ramp does not need to be discarded for a single jump, which is a significant advantage of UTR.
    Each ramp might be divided into separate chunks after the jump detection step.
    The final slope will be the combined slope estimate from every valid chunk in that pixel.

    Descriptions of the different ramp fitting algorithms:

    - **Ordinary Least Squares ('ols'):** This method fits a straight line to the ramp data using the ordinary least squares approach. This method is simple and computationally efficient but is not optimal with significant photon noise.
    - **Correlated Double Sampling ('cds'):** This method uses the difference between last read and the first read to estimate the slope. It is very fast but not optimal for readnoise dominated ramps.
    - **Multiple Correlated Double Sampling ('mcds'):** This method generalizes CDS by using multiple reads (see argument mcds_num_coadd) at the beginning and end of the ramp to reduce the readnoise.
    - **JWST Fixsen ('jwst_fixsen'):** This method assumes that the covariance is piece-wise constant by dividing 9 different regimes of various ratios of readnoise and photon noise. This is referenced as OLS in the JWST pipeline. Fixsen+2000 https://ui.adsabs.harvard.edu/abs/2000PASP..112.1350F/abstract
    - **JWST Likely ('jwst_likely'):** Uses the maximum-likelihood from Brandt (2024) to fit the ramps, which is the optimal approach (but assuming constant illumination). Reference: Brandt, T. D., PASP, 136, 045005 (2024) https://ui.adsabs.harvard.edu/abs/2024PASP..136d5005B/abstract
    - **Cython Likely ('cython_likely'):** Same as JWST Likely but implemented in Cython for better performance (author: Brandt, Tim).

    todo: As a future research avenue, although not a DRP requirement, we plan to explore using the UTR and the ATC images to optimize the slope fitting to account for the variable slopes due to cloud or seeing variability. For example, a simple strategy could be that reads acquired during spikes in seeing (as monitored in the ATC data) could simply be masked in the same manner as cosmic rays. This could be particularly beneficial for high contrast spectroscopy of companions because spikes in seeing means that there is more speckles and less planet signal getting through the fiber.

    Parameters
    ----------
    dataset : str or RampModel
        Input file path or data model to process. This must be a raw
        up-the-ramp (UTR) ramp data product in which ramps are to be fitted.
    suffix : str or None, optional
        Suffix to append to the output filename. Default is ``'rampfit'``.
    method : {'ols', 'cds', 'mcds', 'jwst_likely', 'jwst_fixsen', 'cython_likely'}, optional
        Ramp fitting algorithm to use. See descriptions above for details.
        Default is ``'cython_likely'``.
    mcds_num_coadd : int or None, optional
        Number of reads to coadd for MCDS ramp fitting. If None, it will be set to n_reads/4.
        Only used if method is set to 'mcds'. Default is None.
    max_cores : int, optional
        Maximum number of CPU cores to use for parallel processing. Default
        is 1 (no parallel execution).
    jwst_jump_detection : bool, optional
        If True, use shared memory optimization for jump detection in combination with JWST ramp fitting methods.
        Default is False.
    jwst_jump_rejection_threshold : float, optional
        Rejection threshold (approximately in signal-to-noise) above which
        discontinuities are flagged as jumps when using JWST ramp fitting methods.
        Default is 4.0.
    poisson_variance_mode : {'smooth_detect', 'positive_only', 'detect_nsigma', 'stcal'}, optional
        How to adjust reported ``rate_var_poisson`` / ``rate_err`` after the fit.
        ``'smooth_detect'`` (default) gates Poisson on a smoothed rate so empty
        sky stays RN-limited while faint illuminated structure keeps Poisson;
        ``'positive_only'`` clears Poisson only for ``rate <= 0``;
        ``'detect_nsigma'`` uses a per-pixel ``poisson_detect_nsigma`` gate
        (legacy); ``'stcal'`` leaves the fit Poisson term unchanged.
    poisson_detect_nsigma : float, optional
        Per-pixel detection threshold for ``poisson_variance_mode='detect_nsigma'``.
        Default is 3.0.
    poisson_smooth_sigma : float, optional
        Gaussian smoothing scale (pixels) for ``smooth_detect``. Default is 1.0.
    poisson_smooth_nsigma : float, optional
        Smoothed-rate detection threshold in RN σ for ``smooth_detect``.
        Default is 1.0.

    Calibrations
    ------------
    gain : GainModel
        - selector: :class:`~hispecdrp.calibrations.selectors.GainSelector`
    rn : ReadNoiseModel
        - selector: :class:`~hispecdrp.calibrations.selectors.ReadNoiseSelector`

    Returns
    -------
    output : Level0Model
        Level 0 data model with fitted slope (rate) and associated arrays.


    Notes
    -----
    Roman calibration pipeline reference for ramp fitting algorithms:
    https://github.com/spacetelescope/romancal

    Ramp fitting roman reference:
    https://www.stsci.edu/files/live/sites/www/files/home/roman/_documents/Roman-STScI-000394_DeterminingTheBestFittingSlope.pdf

    See JWST primitive:
    https://jwst-pipeline.readthedocs.io/en/latest/jwst/ramp_fitting/description.html

    CHARIS reference:
    https://ui.adsabs.harvard.edu/abs/2017JATIS...3d8002B/abstract

    HPF DRP:
    https://github.com/indiajoe/HxRGproc

    """

    PARAMETERS = {
        'suffix' : 'rampfit',
        'method' : 'cython_likely',
        'mcds_num_coadd' : None,
        'max_cores' : 1,
        'output_model' : 'L0Model',
        'jwst_jump_detection': False,
        'jwst_jump_rejection_threshold': 4.0,
        'poisson_variance_mode': _POISSON_VARIANCE_MODE_DEFAULT,
        'poisson_detect_nsigma': _POISSON_DETECT_NSIGMA,
        'poisson_smooth_sigma': _POISSON_SMOOTH_SIGMA,
        'poisson_smooth_nsigma': _POISSON_SMOOTH_NSIGMA,
    }

    CALIBRATIONS = {
        'gain': {
            'selector': GainSelector,
            'selector_kwargs': {}
        },
        # 'dark': { # would need to be defined if we implement dark subtraction before ramp fitting.
        #     'selector': DarkSelector,
        #     'selector_kwargs': {}
        # },
        'rn': {
            'selector': ReadNoiseSelector,
            'selector_kwargs': {}
        }
    }
    def _perform(self):

        if self.jwst_jump_detection and self.method.lower() not in ('jwst_likely', 'jwst_fixsen'):
            raise ValueError("JWST jump detection is only for shared memory optimization in combination with JWST ramp fitting methods. "
                             "Please set jwst_jump_detection to False or change ramp fitting algorithm.")

        input_model = datamodels.open(self.dataset, copy=False)
        logger.info(f"Fitting ramp to {input_model} with method={self.method.lower()}")

        gain_cal, _ = self.get_calibration(input_model, 'gain')
        gain_model = datamodels.open(gain_cal)
        # Dark would need to be defined if we implement dark subtraction before ramp fitting.
        # dark_cal, _ = self.get_calibration(input_model, 'dark')
        # dark_model = datamodels.open(dark_cal)
        rn_cal, _ = self.get_calibration(input_model, 'rn')
        rn_model = datamodels.open(rn_cal)

        _mcds_num_coadd = self.mcds_num_coadd

        delta_readtimes = input_model.get_delta_readtimes()
        first_read_mjd = input_model.get_first_read_mjd()

        max_cores = self.max_cores or 1
        if self.method.lower() == 'ols':
            result = fit_ramps_ols(
                delta_readtimes,
                input_model.data,
                dq_raw=input_model.dq_raw,
                dq=input_model.dq,
                rdnoise=rn_model.rn,
                gain=gain_model.gain,
                start_times=first_read_mjd,
                dark_current=None,  # Not used because we assume dark subtraction is done in rate maps.
                max_cores=max_cores
            )
        elif self.method.lower() in ('cds'):
            result = fit_ramps_mcds(
                delta_readtimes,
                input_model.data,
                dq_raw=input_model.dq_raw,
                dq=input_model.dq,
                rdnoise=rn_model.rn,
                gain=gain_model.gain,
                start_times=first_read_mjd,
                dark_current=None, # Not used because we assume dark subtraction is done in rate maps.
                num_coadd=1,
                max_cores=max_cores
            )
        elif self.method.lower() in ('mcds'):
            if _mcds_num_coadd is None:
                n_read = input_model.data.shape[0]
                _mcds_num_coadd = np.max([n_read // 4, 1]) # Ensure at least 1 read is used.
            result = fit_ramps_mcds(
                delta_readtimes,
                input_model.data,
                dq_raw=input_model.dq_raw,
                dq=input_model.dq,
                rdnoise=rn_model.rn,
                gain=gain_model.gain,
                start_times=first_read_mjd,
                dark_current=None, # Not used because we assume dark subtraction is done in rate maps.
                num_coadd=_mcds_num_coadd,
                max_cores=max_cores
            )

        elif self.method.lower() in ('jwst_likely', 'jwst_fixsen', 'cython_likely'):
            result = fit_ramps_jwst(
                delta_readtimes, input_model.data,
                dq_raw=input_model.dq_raw,
                dq=input_model.dq,
                rdnoise=rn_model.rn,
                gain=gain_model.gain,
                start_times=first_read_mjd,
                dark_current=None, # Not used because we assume dark subtraction is done in rate maps.
                max_cores=max_cores,
                algorithm=self.method.lower(),
                jump_detection = self.jwst_jump_detection,
                jump_rejection_threshold = self.jwst_jump_rejection_threshold
            )
        else:
            logger.error(f"Invalid ramp fit method {self.method.lower()}")
            raise ValueError(f"Invalid ramp fit method {self.method.lower()}")

        output_model = self.get_output_model(input_model)
        output_model.rate = result['rate']
        output_model.rate_err = result['rate_err']
        output_model.rate_var_poisson = result['rate_var_poisson']
        output_model.rate_var_rnoise = result['rate_var_rnoise']
        output_model.dq = result['dq']
        output_model.mid_time = result['mid_time']

        # Adjust reported Poisson variance (see clip_noise_driven_poisson_variance).
        # Default 'smooth_detect' keeps empty-sky RN pulls while preserving
        # faint illuminated Poisson; see poisson_variance_mode for alternatives.
        (
            output_model.rate_var_poisson,
            output_model.rate_err,
            n_pois_clipped,
        ) = clip_noise_driven_poisson_variance(
            output_model.rate,
            output_model.rate_var_poisson,
            output_model.rate_var_rnoise,
            mode=self.poisson_variance_mode,
            n_sigma=self.poisson_detect_nsigma,
            smooth_sigma=self.poisson_smooth_sigma,
            smooth_nsigma=self.poisson_smooth_nsigma,
            rate_err=output_model.rate_err,
        )
        if n_pois_clipped:
            logger.info(
                "Ramp fit: cleared Poisson variance on %d pixel(s) "
                "(mode=%s) for reported rate_err.",
                n_pois_clipped,
                self.poisson_variance_mode,
            )

        # Empty / unfitted pixels often return rate=0 with NaN rate_err. Fill a
        # finite rate_err (0 or RN floor) so box extract does not drop aperture
        # edge pixels and renormalize; only flag DO_NOT_USE when rate itself is
        # non-finite.
        (
            output_model.rate_err,
            output_model.dq,
            n_filled,
            n_flagged,
        ) = sanitize_ramp_fit_rate_err(
            output_model.rate,
            output_model.rate_err,
            output_model.dq,
            rate_var_rnoise=output_model.rate_var_rnoise,
        )
        if n_filled or n_flagged:
            logger.info(
                "Ramp fit: filled non-finite rate_err on %d finite-rate pixel(s); "
                "flagged %d non-finite-rate pixel(s) as DO_NOT_USE.",
                n_filled,
                n_flagged,
            )

        return dict(output=output_model, method=self.method, mcds_num_coadd=_mcds_num_coadd)
    

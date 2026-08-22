from ..base_primitive import BasePrimitive
from ... import datamodels
from ...calibrations import GainSelector,ReadNoiseSelector
from ...datamodels.dqflags import DQ_FLAGS
import logging

from .fit_ramp_numba import fit_ramps_ols, fit_ramps_mcds
from .fit_ramp_jwst import fit_ramps_jwst

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ['FitRampPrimitive']

_DO_NOT_USE = int(DQ_FLAGS["DO_NOT_USE"])


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
                max_cores=self.max_cores
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
                max_cores=self.max_cores
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
                max_cores=self.max_cores
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
                max_cores=self.max_cores,
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
    

from ..base_primitive import BasePrimitive
from ... import datamodels

from .fit_lines import fit_lines_gauss
from .find_lines import find_emission_lines
from ...utils.math import delta_pix_to_delta_wave

import numpy as np
import scipy.constants

from logging import getLogger
logger = getLogger(__name__)

__all__ = ['FitLFCLinesPrimitive']


class FitLFCLinesPrimitive(BasePrimitive):

    """
    NOTE: *Under development*
    
    Fit each laser frequency comb (LFC) line with a Gaussian profile. This is useful for calculating the wavelength solution and the the line spread function (LSF).

    Each LFC line is first identified with `scipy.signal.find_peaks <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_ then refined with a centroid. Then each line is fit with a Gaussian profile using `scipy.optimize.least_squares <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>`_. The fit parameters are stored in a table in the output data model's attribute ``lfc_fitparams_fib#``.

    Parameters
    ----------
    input
        A level 1 or 2 data product.
    fwhm_bounds : Sequence[float, float]
        The bounds for the FWHM of the fit. Default is (0.2, 8) pixels.
    mu_bounds : Sequence[float, float]
        The bounds for the mean of the fit (in pixels) relative the centroid.
        Default is (-1, 1) pixels.
    max_fit_window : float
        The maximum number of pixels to fit for a mode.
        Default is 14 pixels.
    do_orders : Sequence[int] | None
        Which order(s) to fit. If None, fit all available orders.
    do_fibers : Sequence[int] | None
        Which fiber(s) to fit. If None, fit all available fibers.
    lfc_nu0 : float
        The central frequency (pump line) of the LFC in GHz.
    lfc_delta_nu : float
        The frequency spacing of the LFC modes in GHz.
    line_source : str
        The source of the lines to fit. Options:
        - 'peaks' : Use only the lines found by `find_emission_lines`.
        - 'wavesol' : Use all lines that fall within the valid wavelength range.
    min_line_spacing : int
        Minimum spacing between lines in pixels. Default is 10 pixels.
    background_filter_width : int | None
        Number of pixels to use for background estimation. If None, no background subtraction is performed.

    Returns
    -------
    output : LFCFitParamsModel
        A data model containing the fit parameters for each LFC line.

    """

    ARGUMENTS = {
        'fwhm_bounds' : (1.0, 3.0),
        'mu_bounds' : (-1.0, 1.0),
        'max_fit_window' : 14,
        'do_fibers' : None,
        'do_orders' : None,
        'lfc_nu0' : None,
        'lfc_delta_nu' : None,
        'background_poly_deg' : 0,
        'line_source' : 'peaks', # 'wavesol' (all lines) or 'peaks' (only lines found by find_peaks)
        'min_line_spacing' : 10, # Minimum number of pixels between lines
        'background_filter_width' : 31, # Pixels
    }

    def _perform(self):
        input_model = datamodels.open(self.input)
        output_model = datamodels.LFCFitParamsModel.from_datamodels(
            input_model, include_data=True,
            primitive=self
        )

        # Which traces to process
        do_fibers, do_orders = self.get_do_traces(input_model, self.do_fibers, self.do_orders)

        # LFC parameters
        lfc_nu0 = self.lfc_nu0 if self.lfc_nu0 is not None else input_model.meta.lfc_nu0
        lfc_delta_nu = self.lfc_delta_nu if self.lfc_delta_nu is not None else input_model.meta.lfc_delta_nu

        fit_results = {fiber : [] for fiber in do_fibers}

        for fiber in do_fibers:
            wave = getattr(input_model, f'wave_fib{fiber}')
            spec = getattr(input_model, f'spec_fib{fiber}')
            spec_err = getattr(input_model, f'spec_err_fib{fiber}')
            spec_dq = getattr(input_model, f'spec_dq_fib{fiber}')
            for order in do_orders:

                # Relative order index
                oi = np.abs(order - input_model.meta.order_start)

                # Good pixels for this order
                good = np.where(spec_dq[oi, :] == 0)[0]
                if len(good) == 0:
                    logger.warning(f"No good pixels found in order {order} for fiber {fiber}. Skipping.")
                    continue

                # Pixel array
                # TODO: Lift this out of the loop once model.shape is reliable?
                pix_arr = np.arange(spec.size)

                # Find the initial lines
                if self.line_source.lower() == 'peaks':
                    line_centers, _ = find_emission_lines(
                        spec[oi, :], spec_dq[oi, :],
                        min_line_spacing=self.min_line_spacing,
                        background_filter_width=self.background_filter_width,
                        centroid_num_pixels=3,
                        drop_first_last=True,
                        refine_with_centroid=True,
                    )
                elif self.line_source.lower() == 'wavesol':

                    # Get the wavelength range for this order
                    # and the LFC lines that fall within it
                    wi, wf = wave[oi, good[0]], wave[oi, good[-1]]
                    line_centers, _, _ = gen_lfc_modes(
                        lfc_nu0, lfc_delta_nu,
                        wave_min=wi, wave_max=wf
                    )
                else:
                    logger.error(f"Invalid line_source '{self.line_source}'. Must be 'peaks' or 'wavesol'.")

                # Gaussian fit to lines
                _fit_results = fit_lines_gauss(
                    pix_arr, spec[oi, :], spec_err[oi, :], spec_dq[oi, :],
                    line_centers=line_centers,
                    fwhm_bounds=self.fwhm_bounds,
                    mu_bounds=self.mu_bounds,
                    max_fit_window=self.max_fit_window,
                )
                fit_results[fiber].append(_fit_results)

                # Store results in the model
                params_table = getattr(output_model, f'lfc_fitparams_fib{fiber}')
                if order in params_table['ORDER']:
                    setattr(input_model, f'lfc_fitparams_fib{fiber}', params_table[params_table['ORDER'] != order])
                n_lines = len(_fit_results)
                for i in range(n_lines):
                    r = {
                        'ORDER' : order,
                        'AMPLITUDE' : _fit_results[i].amplitude,
                        'AMPLITUDE_ERR' : _fit_results[i].amplitude_err,
                        'MU_PIX' : _fit_results[i].mu,
                        'MU_ERR_PIX' : _fit_results[i].mu_err,
                        'FWHM_PIX' : _fit_results[i].fwhm,
                        'FWHM_ERR_PIX' : _fit_results[i].fwhm_err,
                    }
                    wave_closest = wave[oi, int(r['MU_PIX'])]
                    line_wave, _, line_num = _get_closest_lfc_mode(
                        lfc_nu0, lfc_delta_nu, wave_closest
                    )
                    r['WAVE'] = line_wave
                    r['MODE_NUM'] = line_num
                    r['FWHM_WAVE'] = delta_pix_to_delta_wave(r['FWHM_PIX'], r['MU_PIX'], wave[oi, :])
                    r['FWHM_ERR_WAVE'] = delta_pix_to_delta_wave(r['FWHM_ERR_PIX'], r['MU_PIX'], wave[oi, :])
                    r['MU_ERR_WAVE'] = delta_pix_to_delta_wave(r['MU_ERR_PIX'], r['MU_PIX'], wave[oi, :])
                    params_table.add_row(r)
                params_table.sort(['ORDER', 'MU_PIX'])
        return dict(output=output_model, fit_results=fit_results)


def gen_lfc_modes(
    nu0 : float, delta_nu : float,
    wave_min : float, wave_max : float
):
    """
    Generate LFC modes based on central frequency, frequency spacing, and wavelength bounds.

    Parameters
    ----------
    nu0 : float
        The central frequency (pump line) of the LFC in GHz.
    delta_nu : float
        The frequency spacing of the LFC modes in GHz.
    wave_min : float
        The minimum wavelength of the LFC modes in microns.
    wave_max : float
        The maximum wavelength of the LFC modes in microns.

    Returns
    -------
    waves : np.ndarray
        The wavelengths of the LFC modes in microns.
    nus : np.ndarray
        The frequencies of the LFC modes in GHz.
    integers : np.ndarray
        The integer indices of the LFC modes.
    """
    nu_end = (scipy.constants.c / wave_min) * 1E6
    nu_start = (scipy.constants.c / wave_max) * 1E6
    n1 = int(np.ceil((nu_start - nu0) / delta_nu))
    n2 = int(np.ceil((nu_end - nu0) / delta_nu))
    integers = np.arange(min(n1, n2), max(n1, n2) + 1).astype(int)
    waves, nus = _gen_lfc_modes(nu0, delta_nu, integers)
    waves = waves[::-1]
    nus = nus[::-1]
    integers = integers[::-1]
    return waves, nus, integers

def _gen_lfc_modes(
    nu0 : float, delta_nu : float,
    integers : np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate LFC modes based on central frequency, frequency spacing, and integer indices.

    Parameters
    ----------
    nu0 : float
        The central frequency (pump line) of the LFC in GHz.
    delta_nu : float
        The frequency spacing of the LFC modes in GHz.
    integers : np.ndarray
        An array of integer indices representing the mode numbers.

    Returns
    -------
    waves : np.ndarray
        The wavelengths of the LFC modes in microns.
    nus : np.ndarray
        The frequencies of the LFC modes in GHz.
    """
    nus = integers * delta_nu + nu0
    waves = scipy.constants.c / nus * 1E6
    return waves, nus

def _get_closest_lfc_mode(
    nu0 : float, delta_nu : float,
    wave : float
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Get the closest LFC mode to a given wavelength value.

    Parameters
    ----------
    nu0 : float
        The central frequency (pump line) of the LFC in GHz.
    delta_nu : float
        The frequency spacing of the LFC modes in GHz.
    wave : float
        The wavelength value in microns.

    Returns
    -------
    np.ndarray
        The closest wavelength of the LFC mode in microns.
    np.ndarray
        The corresponding frequency of the LFC mode in GHz.
    int
        The integer index of the closest LFC mode.
    """
    nu = scipy.constants.c / wave * 1E6
    n = int(np.rint((nu - nu0) / delta_nu))
    nu_mode = nu0 + n * delta_nu
    wave_mode = scipy.constants.c / nu_mode * 1E6
    return wave_mode, nu_mode, n
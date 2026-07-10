from ..base_primitive import BasePrimitive
from ... import datamodels

from .fit_lines import fit_lines_gauss
from .find_lines import find_emission_lines
from ...utils.math import delta_pix_to_delta_wave

import numpy as np
import scipy.constants

__all__ = ['FitEtalonLinesPrimitive']


class FitEtalonLinesPrimitive(BasePrimitive):
    """
    Fit each Fabry-Pérot etalon line with a Gaussian or Lorentzian (default) profile.

    Each etalon line is first identified with `scipy.signal.find_peaks <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_ then refined with a centroid. Then each line is fit with a Gaussian or Lorentzian (default) profile using `scipy.optimize.least_squares <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>`_. The fit parameters are stored in a table in the output data model's attribute ``etalon_fitparams_fib#``.

    Parameters
    ----------
    input : str or L2Model
        Input level 2 model or path to a level 2 file.
    fwhm_bounds : tuple, optional
        Bounds on the full width at half maximum (FWHM) of the etalon lines in pixels, given as a tuple (min, max).
    suffix : str, optional
        Suffix to add to the output filename when saving the result. Default is 'etalon_fit'.
    mu_bounds : tuple, optional
        Bounds on the mean (mu) of the etalon lines in pixels relative to the centroid, given as a tuple (min, max) in units of pixels.
    fsr_estimate : float, optional
        An estimate of the free spectral range (fsr) of the etalon in GHz.
    max_fit_window : int, optional
        Maximum number of pixels to include in the fit for each line.
    do_fibers : list of int, optional
        List of fiber numbers to process. If None, all fibers will be processed.
    do_orders : list of int, optional
        List of echelle orders to process. If None, all orders will be processed.
    profile_type : str, optional
        The profile type to use for fitting the etalon lines. Options are 'gaussian' or 'lorentzian'. Default is 'lorentzian'.
    
    Returns
    -------
    output_model
        The input model with the Table attribute ``etalon_fitparams_fib#`` populated.
    """

    ARGUMENTS = {
        'suffix' : 'etalon_fit',
        'fwhm_bounds' : None,
        'mu_bounds' : (-1, 1),
        'fsr_estimate' : None,  # GHz
        'max_fit_window' : None,
        'do_fibers' : None,
        'do_orders' : None,
    }

    def _perform(self):

        input_model = datamodels.open(self.input)
        output_model = self.get_output_model(input_model)

        # Which traces to process
        do_fibers, do_orders = self.get_do_traces(input_model, self.do_fibers, self.do_orders)
        
        fit_results = {fiber : [] for fiber in do_fibers}

        for fiber in do_fibers:
            wave = getattr(input_model, f'wave_fib{fiber}')
            spec = getattr(input_model, f'spec_fib{fiber}')
            spec_err = getattr(input_model, f'spec_err_fib{fiber}')
            spec_dq = getattr(input_model, f'spec_dq_fib{fiber}')
            pixarr = np.arange(wave.shape[1])
            for order in do_orders:

                # Relative order index
                oi = np.abs(order - input_model.meta.order_start)

                good = np.where(spec_dq[oi, :] > 0)[0]
                ki, kf = good[0], good[-1]
                min_line_spacing = _get_etalon_spacing_pixels(self.fsr_estimate, wave[oi, ki:kf+1])
                min_line_spacing = int(np.nanmin(min_line_spacing))
                lines, _ = find_emission_lines(spec[oi, :], spec_dq[oi, :], min_line_spacing=min_line_spacing)

                # Lorentzian fit to lines
                _fit_results = fit_lines_gauss(
                    pixarr, spec[oi, :], spec_err[oi, :], spec_dq[oi, :],
                    line_centers=lines,
                    fwhm_bounds=self.fwhm_bounds, mu_bounds=self.mu_bounds,
                    max_fit_window=self.max_fit_window,
                    absorption=False,
                )
                fit_results[fiber].append(_fit_results)

                # Store results in the model
                params_table = getattr(output_model, f'etalon_fitparams_fib{fiber}')
                if order in params_table['ORDER']:
                    setattr(input_model, f'etalon_fitparams_fib{fiber}', params_table[params_table['ORDER'] != order])
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
                    r['WAVE'] = np.interp(r['MU_PIX'], pixarr, wave[oi, :], left=np.nan, right=np.nan)
                    r['FWHM_WAVE'] = delta_pix_to_delta_wave(r['FWHM_PIX'], r['MU_PIX'], wave[oi, :])
                    r['FWHM_ERR_WAVE'] = delta_pix_to_delta_wave(r['FWHM_ERR_PIX'], r['MU_PIX'], wave[oi, :])
                    r['MU_ERR_WAVE'] = delta_pix_to_delta_wave(r['MU_ERR_PIX'], r['MU_PIX'], wave[oi, :])
                    params_table.add_row(r)
                params_table.sort(['ORDER', 'MU_PIX'])

        return dict(output=output_model, fit_results=fit_results)
    

def _get_etalon_spacing_pixels(fsr : float, wave : np.ndarray) -> float:
    """
    Calculate the per-pixel etalon line spacing in pixels based on
    the free spectral range (fsr) and the wavelength solution.

    Parameters
    ----------
    fsr : float
        Free spectral range in GHz.
    wave : np.ndarray
        Wavelength solution in microns.
    """
    fsr_hz = fsr * 1E9
    wave_m = wave * 1E-6
    dispersion = np.gradient(wave_m)
    fsr_wave = wave_m**2 * fsr_hz / scipy.constants.c
    spacing_pix = fsr_wave / dispersion
    return spacing_pix
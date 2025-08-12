from ..base_primitive import BasePrimitive
from ... import datamodels

from .fit_lines import fit_lines, find_initial_lines
from ...utils.math import delta_pix_to_delta_wave

import numpy as np
import scipy.constants

__all__ = ['FitEtalonLinesPrimitive']


class FitEtalonLinesPrimitive(BasePrimitive):
    """
    Given a 1D spectrum, finds and fits the Etalon lines with a Gaussian + background profile.
    """

    ARGUMENTS = {
        'fwhm_bounds' : None,
        'mu_bounds' : (-1, 1),
        'fsr_estimate' : None,  # GHz
        'max_fit_window' : None,
        'do_fibers' : None,
        'do_orders' : None,
        'profile_type' : 'lorentzian',
    }

    def _perform(self):

        input_model = datamodels.open(self.input)
        output_model = self.get_output_model(input_model)
        fit_results = {fiber : [] for fiber in self.do_fibers}

        for fiber in self.do_fibers:
            wave = getattr(input_model, f'wave_fib{fiber}')
            spec = getattr(input_model, f'spec_fib{fiber}')
            spec_err = getattr(input_model, f'spec_err_fib{fiber}')
            spec_dq = getattr(input_model, f'spec_dq_fib{fiber}')
            xarr = np.arange(wave.shape[1])
            for order in self.do_orders:

                # Relative order index
                oi = np.abs(order - input_model.meta.order_start)

                good = np.where(spec_dq[oi, :] > 0)[0]
                ki, kf = good[0], good[-1]
                min_line_spacing = etalon_spacing_pixels(self.fsr_estimate, wave[oi, ki:kf+1])
                min_line_spacing = int(np.nanmin(min_line_spacing)) # NOTE: Consider rescaling this to a smaller value if needed
                lines = find_initial_lines(spec[oi, :], spec_dq[oi, :], min_line_spacing=min_line_spacing)

                # Lorentzian fit to lines
                _fit_results = fit_lines(
                    spec[oi, :], spec_err[oi, :], spec_dq[oi, :],
                    lines=lines,
                    fwhm_bounds=self.fwhm_bounds, mu_bounds=self.mu_bounds,
                    max_fit_window=self.max_fit_window,
                    profile_type=self.profile_type,
                )
                #except:
                #import matplotlib.pyplot as plt
                    #breakpoint()
                fit_results[fiber].append(_fit_results)

                # Store results in the model
                params_table = getattr(output_model, f'etalon_fitparams_fib{fiber}')
                if order in params_table['ORDER']:
                    setattr(input_model, f'etalon_fitparams_fib{fiber}', params_table[params_table['ORDER'] != order])
                n_lines = len(_fit_results)
                for i in range(n_lines):
                    r = {
                        'ORDER' : order,
                        'AMPLITUDE' : _fit_results[i]['amplitude'],
                        'AMPLITUDE_ERR' : _fit_results[i]['amplitude_err'],
                        'MU_PIX' : _fit_results[i]['mu'],
                        'MU_ERR_PIX' : _fit_results[i]['mu_err'],
                        'FWHM_PIX' : _fit_results[i]['fwhm'],
                        'FWHM_ERR_PIX' : _fit_results[i]['fwhm_err'],
                    }
                    r['WAVE'] = np.interp(r['MU_PIX'], xarr, wave[oi, :], left=np.nan, right=np.nan)
                    r['FWHM_WAVE'] = delta_pix_to_delta_wave(r['FWHM_PIX'], r['MU_PIX'], wave[oi, :])
                    r['FWHM_ERR_WAVE'] = delta_pix_to_delta_wave(r['FWHM_ERR_PIX'], r['MU_PIX'], wave[oi, :])
                    r['MU_ERR_WAVE'] = delta_pix_to_delta_wave(r['MU_ERR_PIX'], r['MU_PIX'], wave[oi, :])
                    params_table.add_row(r)
                params_table.sort(['ORDER', 'MU_PIX'])

        return dict(output=output_model, fit_results=fit_results)
    

def etalon_spacing_pixels(fsr : float, wave : np.ndarray) -> float:
    """
    Calculate the minimum etalon line spacing in pixels based on the free spectral range (fsr) and the wavelength solution.

    Args:
        fsr (float): Free spectral range in GHz.
        wave (np.ndarray): Wavelength solution in microns.
    """
    fsr_hz = fsr * 1E9
    wave_m = wave * 1E-6
    dispersion = np.gradient(wave_m)
    fsr_wave = wave_m**2 * fsr_hz / scipy.constants.c
    spacing_pix = fsr_wave / dispersion
    spacing_pix = np.nanmin(spacing_pix)
    return spacing_pix
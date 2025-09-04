from ..base_primitive import BasePrimitive
from ... import datamodels

from .fit_lines import fit_lines, find_initial_lines
from ...utils.math import delta_pix_to_delta_wave

import numpy as np
import scipy.constants

__all__ = ['FitLFCLinesPrimitive']


class FitLFCLinesPrimitive(BasePrimitive):

    ARGUMENTS = {
        'fwhm_bounds' : (0.2, 8),
        'mu_bounds' : (-1, 1),
        'max_fit_window' : 14,
        'do_fibers' : None,
        'do_orders' : None,
        'lfc_nu0' : None,
        'lfc_delta_nu' : None,
        'profile_type' : 'gaussian',
    }

    def _perform(self):
        input_model = datamodels.open(self.input)
        output_model = self.get_output_model(input_model, output_model='LFCFitParamsModel')
        fit_results = {fiber : [] for fiber in self.do_fibers}

        for fiber in self.do_fibers:
            wave = getattr(input_model, f'wave_fib{fiber}')
            spec = getattr(input_model, f'spec_fib{fiber}')
            spec_err = getattr(input_model, f'spec_err_fib{fiber}')
            spec_dq = getattr(input_model, f'spec_dq_fib{fiber}')
            for order in self.do_orders:
                # Relative order index
                oi = np.abs(order - input_model.meta.order_start)
                # Find the initial lines
                lines = find_initial_lines(spec[oi, :], spec_dq[oi, :])
                # Gaussian fit to lines
                _fit_results = fit_lines(
                    spec[oi, :], spec_err[oi, :], spec_dq[oi, :],
                    lines=lines,
                    fwhm_bounds=self.fwhm_bounds, mu_bounds=self.mu_bounds,
                    max_fit_window=self.max_fit_window,
                    profile_type=self.profile_type,
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
                        'AMPLITUDE' : _fit_results[i]['amplitude'],
                        'AMPLITUDE_ERR' : _fit_results[i]['amplitude_err'],
                        'MU_PIX' : _fit_results[i]['mu'],
                        'MU_ERR_PIX' : _fit_results[i]['mu_err'],
                        'FWHM_PIX' : _fit_results[i]['fwhm'],
                        'FWHM_ERR_PIX' : _fit_results[i]['fwhm_err'],
                    }
                    wave_closest = wave[oi, int(r['MU_PIX'])]
                    line_wave, _, line_num = get_closest_lfc_mode(self.lfc_nu0, self.lfc_delta_nu, wave_closest)
                    r['WAVE'] = line_wave
                    r['MODE_NUM'] = line_num
                    r['FWHM_WAVE'] = delta_pix_to_delta_wave(r['FWHM_PIX'], r['MU_PIX'], wave[oi, :])
                    r['FWHM_ERR_WAVE'] = delta_pix_to_delta_wave(r['FWHM_ERR_PIX'], r['MU_PIX'], wave[oi, :])
                    r['MU_ERR_WAVE'] = delta_pix_to_delta_wave(r['MU_ERR_PIX'], r['MU_PIX'], wave[oi, :])
                    params_table.add_row(r)
                params_table.sort(['ORDER', 'MU_PIX'])
        return dict(output=output_model, fit_results=fit_results)


def gen_lfc_modes(nu0 : float, delta_nu : float, wave_start : float, wave_end : float):
    nu_end = (scipy.constants.c / wave_start) * 1E6
    nu_start = (scipy.constants.c / wave_end) * 1E6
    n1 = int(np.ceil((nu_start - nu0) / delta_nu))
    n2 = int(np.ceil((nu_end - nu0) / delta_nu))
    integers = np.arange(min(n1, n2), max(n1, n2) + 1).astype(int)
    waves, nus = _gen_lfc_modes(nus, delta_nu, integers)
    waves = waves[::-1]
    nus = nus[::-1]
    integers = integers[::-1]
    return waves, nus, integers

def _gen_lfc_modes(nu0 : float, delta_nu : float, integers : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nus = integers * delta_nu + nu0
    waves = scipy.constants.c / nus * 1E6
    return waves, nus

def get_closest_lfc_mode(nu0 : float, delta_nu : float, wave : float):
    nu = (scipy.constants.c / wave) * 1E6
    n = int(np.round((nu - nu0) / delta_nu))
    _wave, nu = _gen_lfc_modes(nu0, delta_nu, n)
    return _wave, nu, n
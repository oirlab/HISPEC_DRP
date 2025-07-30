from ..base_primitive import BasePrimitive
from ... import datamodels

from keckdrpframework.models.arguments import Arguments

from .fit_lines_gauss import fit_lines_gauss, find_initial_lfc_lines
from ...utils.math import delta_pix_to_delta_wave

import numpy as np
import scipy.constants

__all__ = ['FitLFCLinesPrimitive']


class FitLFCLinesPrimitive(BasePrimitive):
    """
    Given a 1D spectrum, finds and fits the LFC lines with a Gaussian + background profile.
    """

    ARGUMENTS = {
        'sigma_bounds' : None,
        'mu_bounds' : (-1, 1),
        'max_fit_window' : None,
        'do_fibers' : None,
        'do_orders' : None,
        'lfc_nu0' : None,
        'lfc_delta_nu' : None,
    }

    def _perform(self):

        input_model = datamodels.open(self.args.name)
        output_model = self.get_output_model(input_model, model_suffix='LFCFitParamsModel')

        for fiber in self.args.do_fibers:
            fit_results = []
            wave = getattr(input_model, f'wave_fib{fiber}')
            spec = getattr(input_model, f'spec_fib{fiber}')
            spec_err = getattr(input_model, f'spec_err_fib{fiber}')
            spec_dq = getattr(input_model, f'spec_dq_fib{fiber}')
            for order in self.args.do_orders:

                # Relative order index
                oi = np.abs(order - input_model.meta.order_start)

                # Find the initial lines
                lines = find_initial_lfc_lines(spec[oi, :], spec_err[oi, :])

                # Gaussin fit to lines
                _fit_results = fit_lines_gauss(
                    spec[oi, :], spec_err[oi, :], spec_dq[oi, :],
                    lines=lines,
                    sigma_bounds=self.args.sigma_bounds, mu_bounds=self.args.mu_bounds,
                    max_fit_window=self.args.max_fit_window
                )
                fit_results.append(_fit_results)

                # Store results in the model
                params_table = getattr(output_model, f'lfc_fitparams_fib{fiber}')
                if order in params_table['ORDER']:
                    setattr(input_model, f'lfc_fitparams_fib{fiber}', params_table[params_table['ORDER'] != order])
                n_lines = len(_fit_results['mu'])
                for i in range(n_lines):
                    r = {
                        'ORDER' : order,
                        'AMPLITUDE' : _fit_results['amplitude'][i],
                        'AMPLITUDE_ERR' : _fit_results['amplitude_err'][i],
                        'MU_PIX' : _fit_results['mu'][i],
                        'MU_ERR_PIX' : _fit_results['mu_err'][i],
                        'SIGMA_PIX' : _fit_results['sigma'][i],
                        'SIGMA_ERR_PIX' : _fit_results['sigma_err'][i],
                    }
                    wave_closest = wave[oi, int(r['MU_PIX'])]
                    line_wave, _, line_num = get_closest_lfc_mode(self.args.lfc_nu0, self.args.lfc_delta_nu, wave_closest)
                    r['WAVE'] = line_wave
                    r['MODE_NUM'] = line_num
                    r['SIGMA_WAVE'] = delta_pix_to_delta_wave(r['SIGMA_PIX'], r['MU_PIX'], wave[oi, :])
                    r['SIGMA_ERR_WAVE'] = delta_pix_to_delta_wave(r['SIGMA_ERR_PIX'], r['MU_PIX'], wave[oi, :])
                    r['MU_ERR_WAVE'] = delta_pix_to_delta_wave(r['MU_ERR_PIX'], r['MU_PIX'], wave[oi, :])
                    params_table.add_row(r)
                params_table.sort(['ORDER', 'MU_PIX'])

        return Arguments(name=output_model, fit_results=fit_results)


def gen_lfc_modes(nu0 : float, delta_nu : float, wave_start : float, wave_end : float):
    """
    Generates LFC modes (wavelengths and frequencies) within a given wavelength range.

    Args:
        nu0 (float): Reference frequency.
        delta_nu (float): Frequency spacing.
        wave_start (float): Start wavelength in meters.
        wave_end (float): End wavelength in meters.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing arrays of wavelengths, frequencies, and mode numbers.
    """
    nu_end = (scipy.constants.c / wave_start) * 1E6
    nu_start = (scipy.constants.c / wave_end) * 1E6
    n1 = int(np.ceil((nu_start - nu0) / delta_nu))
    n2 = int(np.ceil((nu_end - nu0) / delta_nu))
    integers = np.arange(min(n1, n2), max(n1, n2) + 1).astype(int)
    waves, nus = _gen_lfc_modes(nu0, delta_nu, integers)
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
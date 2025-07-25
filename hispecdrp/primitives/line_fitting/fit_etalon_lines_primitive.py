from ..base_primitive import BasePrimitive
from ... import datamodels

from keckdrpframework.models.arguments import Arguments

from .fit_lines import fit_lines, find_initial_lines
from ...utils.math import delta_pix_to_delta_wave

import numpy as np

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

        input_model = datamodels.open(self.args.name)
        output_model = self.get_output_model(input_model, model_suffix='EtalonFitParamsModel')

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
                def etalon_spacing_pixels(fsr_ghz, wavelengths_um):
                    c = 3e8  # m/s
                    fsr_hz = fsr_ghz * 1e9
                    wavelengths_m = wavelengths_um * 1e-6  # convert μm → m
                    dispersion_nm_per_pix = np.gradient(wavelengths_um) * 1e3  # μm/pix → nm/pix

                    delta_lambda_nm = (wavelengths_m**2 * fsr_hz / c) * 1e9  # m → nm
                    spacing_pix = delta_lambda_nm / dispersion_nm_per_pix
                    return spacing_pix

                min_line_spacing = etalon_spacing_pixels(self.args.fsr_estimate, wave[oi, :])
                min_line_spacing = int(np.nanmin(min_line_spacing))
                lines = find_initial_lines(spec[oi, :], spec_dq[oi, :], min_line_spacing=min_line_spacing)

                # Gaussin fit to lines
                _fit_results = fit_lines(
                    spec[oi, :], spec_err[oi, :], spec_dq[oi, :],
                    lines=lines,
                    fwhm_bounds=self.args.fwhm_bounds, mu_bounds=self.args.mu_bounds,
                    max_fit_window=self.args.max_fit_window,
                    profile_type=self.args.profile_type,
                )
                fit_results.append(_fit_results)

                xarr = np.arange(len(wave[oi, :]))

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

        return Arguments(name=output_model, fit_results=fit_results)
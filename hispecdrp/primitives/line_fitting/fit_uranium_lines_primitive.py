from ..base_primitive import BasePrimitive
from ... import datamodels

from .fit_lines import fit_lines_gauss
from .find_lines import find_emission_lines
from ..utils.spectral_utils import delta_pix_to_delta_wave

import numpy as np
import scipy.constants
import os
from astropy.table import Table

from pathlib import Path

from ...calibrations import WavelengthSolutionSelector
import logging

logger = logging.getLogger(__name__)


__all__ = ['FitUraniumLinesPrimitive']

LINELIST_DIR = (Path(__file__).resolve().parents[2] / "data" / "wavesol").resolve()

class FitUraniumLinesPrimitive(BasePrimitive):
    """
  
    ----------
    input : str or L2Model
        Input level 2 model or path to a level 2 file.
    fwhm_bounds : tuple, optional
        Bounds on the full width at half maximum (FWHM) of the lamp lines in pixels, given as a tuple (min, max).
    suffix : str, optional
        Suffix to add to the output filename when saving the result. Default is 'etalon_fit'.
    mu_bounds : tuple, optional
        Bounds on the mean (mu) of the etalon lines in pixels relative to the centroid, given as a tuple (min, max) in units of pixels.
    max_fit_window : int, optional
        Maximum number of pixels to include in the fit for each line.
    do_fibers : list of int, optional
        List of fiber numbers to process. If None, all fibers will be processed.
    do_orders : list of int, optional
        List of echelle orders to process. If None, all orders will be processed.
    profile_type : str, optional
        The profile type to use for fitting lines. Options are 'gaussian' or 'lorentzian'. Default is 'lorentzian'.
    
    Returns
    -------
    output_model
        The input model with the Table attribute ``lamp_fitparams_fib#`` populated.
    """

    PARAMETERS = {
        'suffix' : 'lamp_fit',
        'fwhm_bounds' : (1,8),
        'mu_bounds' : (-1, 1),
        'max_fit_window' : None,
        'do_fibers' : None,
        'do_orders' : None,
    }

    def _perform(self):

        input_model = datamodels.open(self.dataset)
        output_model = self.get_output_model(input_model)

        # Which traces to process
        do_fibers, do_orders = self.get_do_traces(input_model, self.do_fibers, self.do_orders)
        
        fit_results = {fiber : [] for fiber in do_fibers}

        # read and parse line list
        linelist = Table.read(os.path.join(LINELIST_DIR,"UNe_lines_nir.txt"), format="ascii.cds")
        lamp_line_wvl = linelist['WaveLen'] / 1e3 # was in nm now in um
        lamp_line_rflux = linelist['RFlux']

        # predict blended lines from line list
        blended = _flag_blended_lines(
            np.asarray(lamp_line_wvl),
            np.asarray(lamp_line_rflux),
            tol=3e-5, # ~approximatelly twice the FWHM
            min_flux_frac=0.05, # neighbor >= 5% of this line's flux
        )

        clean_lamp_wvl = lamp_line_wvl[~blended]
        clean_lamp_rflux = lamp_line_rflux[~blended]

        for fiber in do_fibers:
            # parse input data
            wave = getattr(input_model, f'wave_fib{fiber}')
            spec = getattr(input_model, f'spec_fib{fiber}')
            spec_err = getattr(input_model, f'spec_err_fib{fiber}')
            spec_dq = getattr(input_model, f'spec_dq_fib{fiber}')
            xarr = np.arange(wave.shape[1])

            for order in do_orders:

                # Relative order index
                oi = np.abs(order - input_model.meta.order_start)

                # find lines in order
                lines_wvl = clean_lamp_wvl[(clean_lamp_wvl > np.nanmin(wave[oi,:])) & (clean_lamp_wvl < np.nanmax(wave[oi,:]))]
                lines_pixel = np.interp(lines_wvl, wave, xarr)

                # Lorentzian fit to lines
                _fit_results = fit_lines_gauss(
                    xarr, spec[oi, :], spec_err[oi, :], spec_dq[oi, :], lines_pixel,
                    fwhm_bounds=self.fwhm_bounds, mu_bounds=self.mu_bounds,
                    max_fit_window=self.max_fit_window,
                    absorption=False,
                )
                fit_results[fiber].append(_fit_results)

                # Store results in the model
                params_table = getattr(output_model, f'lamp_fitparams_fib{fiber}')
                if order in params_table['ORDER']:
                    setattr(input_model, f'lamp_fitparams_fib{fiber}', params_table[params_table['ORDER'] != order])
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
                    r['WAVE'] = np.interp(r['MU_PIX'], xarr, wave[oi, :], left=np.nan, right=np.nan)
                    r['FWHM_WAVE'] = delta_pix_to_delta_wave(r['FWHM_PIX'], r['MU_PIX'], wave[oi, :])
                    r['FWHM_ERR_WAVE'] = delta_pix_to_delta_wave(r['FWHM_ERR_PIX'], r['MU_PIX'], wave[oi, :])
                    r['MU_ERR_WAVE'] = delta_pix_to_delta_wave(r['MU_ERR_PIX'], r['MU_PIX'], wave[oi, :])
                    params_table.add_row(r)
                params_table.sort(['ORDER', 'MU_PIX'])

        return dict(output=output_model, fit_results=fit_results)



def _flag_blended_lines(wvl, flux, tol=3e-5, min_flux_frac=0.05):
    """
    Flag lines that have a neighbor within `tol` whose flux is significant
    relative to the line's own flux.

    Parameters
    ----------
    wvl : array-like
        Line wavelengths (same units as `tol`, e.g. um).
    flux : array-like
        Line relative fluxes (same length as `wvl`).
    tol : float
        Maximum wavelength separation for two lines to be considered blended.
    min_flux_frac : float
        Neighbor must have flux >= min_flux_frac * (this line's flux) to count
        as a blend. E.g. 0.05 = neighbor is at least 5% as bright as the line.

    Returns
    -------
    blended : ndarray of bool
        True where the line is likely blended. Same order as the input.
    """
    wvl = np.asarray(wvl, dtype=float)
    flux = np.asarray(flux, dtype=float)
    n = len(wvl)
    blended = np.zeros(n, dtype=bool)

    # Sort by wavelength so neighbors within tol are a contiguous slice
    order = np.argsort(wvl)
    w = wvl[order]
    f = flux[order]

    # Window of candidate neighbors for each line: [lo, hi)
    lo = np.searchsorted(w, w - tol, side="left")
    hi = np.searchsorted(w, w + tol, side="right")

    flagged_sorted = np.zeros(n, dtype=bool)
    for i in range(n):
        if hi[i] - lo[i] <= 1:      # only the line itself
            continue
        nbr_f = f[lo[i]:hi[i]]
        # Max flux among neighbors, excluding the line itself
        # (set own flux to -inf temporarily via mask)
        mask = np.ones(len(nbr_f), dtype=bool)
        mask[i - lo[i]] = False
        if nbr_f[mask].max() >= min_flux_frac * f[i]:
            flagged_sorted[i] = True

    # Undo the sort
    blended[order] = flagged_sorted
    return blended



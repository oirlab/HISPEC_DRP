from ..base_primitive import BasePrimitive
from ... import datamodels
from ...utils.math import delta_pix_to_delta_wave, delta_wave_to_delta_vel, robust_stats, weighted_mean
from ...calibrations import ReferenceEtalonSelector

from koa_middleware import CalibrationStore

import numpy as np
import scipy.constants

__all__ = ['CalculateEtalonDriftCorrectionPrimitive', 'ApplyDriftCorrectionPrimitive']

class CalculateEtalonDriftCorrectionPrimitive(BasePrimitive):

    ARGUMENTS = {
        'do_fibers' : None,
        'do_orders' : None,
    }

    CALIBRATIONS = {
        'ref_etalon' : {
            'selector' : ReferenceEtalonSelector,
            'selector_kwargs' : {}
        },
    }

    def _perform(self):

        input_model = datamodels.open(self.input)
        ref_model, _ = self.get_calibration(input_model, 'ref_etalon')
        output_model = self.get_output_model(input_model)

        # Get fibers and orders
        do_fibers, do_orders = self.get_do_traces(input_model, self.do_fibers, self.do_orders)

        drift_results = {fiber : [] for fiber in do_fibers}

        for fiber in do_fibers:

            wave = getattr(input_model, f'wave_fib{fiber}')
            xarr = np.arange(wave.shape[1])

            params_table = getattr(input_model, f'etalon_fitparams_fib{fiber}')
            params_table_ref = getattr(ref_model, f'etalon_fitparams_fib{fiber}')

            line_centers = params_table['MU_PIX']
            line_centers_ref = params_table_ref['MU_PIX']
            line_center_errs = params_table['MU_ERR_PIX']
            line_center_errs_ref = params_table_ref['MU_ERR_PIX']

            drift_table = getattr(output_model, f'drift_fib{fiber}')

            for order in do_orders:

                order_mask = params_table['ORDER'] == order
                order_mask_ref = params_table_ref['ORDER'] == order
                lcs = line_centers[order_mask]
                lfc_ref = line_centers_ref[order_mask_ref]
                lcs_err = line_center_errs[order_mask]
                lcs_ref_err = line_center_errs_ref[order_mask_ref]

                if len(lcs) == 0 or len(lfc_ref) == 0:
                    self.log.info(f'No etalon lines found for {order=}, {fiber=}. Skipping.')
                    continue

                drifts = calculate_etalon_drifts_pix(lcs, lcs_err, lfc_ref, lcs_ref_err)
                n_lines = len(drifts)

                for i in range(n_lines):
                    r = {
                        'PIXEL' : drifts[i]['PIXEL'],
                        'ORDER' : order,
                        'DRIFT_PIX' : drifts[i]['DRIFT'],
                        'DRIFT_ERR_PIX' : drifts[i]['DRIFT_ERR']
                    }
                    # Relative order index
                    oi = np.abs(order - input_model.meta.order_start)
                    r['WAVE'] = np.interp(r['PIXEL'], xarr, wave[oi, :], left=wave[oi, 0], right=wave[oi, -1])
                    r['DRIFT_WAVE'] = delta_pix_to_delta_wave(r['DRIFT_PIX'], r['PIXEL'], wave[oi, :])
                    r['DRIFT_ERR_WAVE'] = delta_pix_to_delta_wave(r['DRIFT_ERR_PIX'], r['PIXEL'], wave[oi, :])

                    r['DRIFT_VEL'] = delta_wave_to_delta_vel(r['DRIFT_WAVE'], r['WAVE'])
                    r['DRIFT_ERR_VEL'] = delta_wave_to_delta_vel(r['DRIFT_ERR_WAVE'], r['WAVE'])
                
                    drift_results[fiber].append(r)

                # Calculate per order drifts from distribution
                drifts_pix = np.array([d['DRIFT_PIX'] for d in drift_results[fiber] if d['ORDER'] == order])
                drift_err_pix = np.array([d['DRIFT_ERR_PIX'] for d in drift_results[fiber] if d['ORDER'] == order])
                w = 1 / drift_err_pix**2
                good = np.isfinite(w) & (w > 0)
                if np.count_nonzero(good) > 0:
                    drifts_pix = drifts_pix[good]
                    w = w[good]
                    drift_pix_order, drift_order_err, drift_mask = robust_stats(drifts_pix, w=w)
                    drift_order_err /= np.sqrt(np.count_nonzero(good) - 1)
                    pixels = np.array([d['PIXEL'] for d in drift_results[fiber] if d['ORDER'] == order])
                    
                    pix_eff = weighted_mean(pixels[drift_mask], w=w[drift_mask])
                    wave_eff = np.interp(pix_eff, xarr, wave[oi, :], left=wave[oi, 0], right=wave[oi, -1])
                else:
                    drift_pix_order, drift_order_err = np.nan, np.nan
                    pix_eff = np.nan
                    wave_eff = np.nan

                # Update drift table with per order drift
                drift_wave_order = delta_pix_to_delta_wave(drift_pix_order, pix_eff, wave[oi, :])
                drift_err_wave_order = delta_pix_to_delta_wave(drift_order_err, pix_eff, wave[oi, :])
                drift_vel_order = delta_wave_to_delta_vel(drift_wave_order, wave_eff)
                drift_err_vel_order = delta_wave_to_delta_vel(drift_err_wave_order, wave_eff)
                drift_table.add_row({
                    'PIXEL' : pix_eff,
                    'ORDER' : order,
                    'WAVE' : wave_eff,
                    'DRIFT_PIX' : drift_pix_order,
                    'DRIFT_ERR_PIX' : drift_order_err,
                    'DRIFT_WAVE' : drift_wave_order,
                    'DRIFT_ERR_WAVE' : drift_err_wave_order,
                    'DRIFT_VEL' : drift_vel_order,
                    'DRIFT_ERR_VEL' : drift_err_vel_order,
                })

                drift_table.sort(['ORDER', 'PIXEL'])

        return dict(output=output_model, drift_results=drift_results)
    

class ApplyDriftCorrectionPrimitive(BasePrimitive):
    """
    fiber_reference_map: A mapping of fiber numbers to their reference fibers.
        Dict. <fiber_to_correct_for> : <from_fiber>.
        Example:

            >>> fiber_reference_map = {
                    '1' : '1',
                    '2' : '1',
                    '3' : '1'
                }

        to apply drift from fiber 1 to fibers 1-3.
    """

    ARGUMENTS = {
        'suffix' : 'driftcorr',
        'fiber_reference_map' : None,
        'do_orders' : None,
    }

    def _perform(self):
        input_model = datamodels.open(self.input)
        # Get fibers and orders
        _, do_orders = self.get_do_traces(input_model, None, self.do_orders)
        for fiber, ref_fiber in self.fiber_reference_map.items():
            wave = getattr(input_model, f'wave_fib{fiber}')
            for order in do_orders:
                drift_table = getattr(input_model, f'drift_fib{ref_fiber}')
                k = np.where(drift_table['ORDER'] == order)[0]
                if len(k) == 0:
                    self.log.warning(f'No drift correction found for {fiber=} {order=}. Skipping.')
                    continue
                k = k[0]
                drift_vel = drift_table['DRIFT_VEL'][k]
                oi = np.abs(order - input_model.meta.order_start)
                wave[oi, :] *= (1 - drift_vel / scipy.constants.c) # NOTE: For line_pos_new - line_pos_ref

        return dict(output=input_model)
    

def calculate_etalon_drifts_pix(
    line_centers : np.ndarray,
    line_centers_err : np.ndarray,
    line_centers_ref : np.ndarray,
    line_centers_ref_err : float = 0.0,
    max_sep : int = 5
) -> list[dict[str, float]]:
    drifts = []
    for i, lc in enumerate(line_centers):
        diffs = np.abs(line_centers_ref - lc)
        if np.count_nonzero(diffs < max_sep) == 1:
            k = np.argmin(diffs)
            drift_err = np.sqrt(line_centers_err[i]**2 + line_centers_ref_err[k]**2)
            drifts.append({
                'PIXEL': lc,
                'PIXEL_REF': line_centers_ref[k],
                'DRIFT': lc - line_centers_ref[k],
                'DRIFT_ERR': drift_err
            })
    return drifts
from ..base_primitive import BasePrimitive
from ... import datamodels
from ..utils.spectral_utils import delta_pix_to_delta_wave, delta_wave_to_delta_vel
from ..utils.stats import robust_stats, weighted_quantile
from ...calibrations import ReferenceEtalonSelector
import logging

from koa_middleware import CalibrationStore

import numpy as np
import scipy.constants

logger = logging.getLogger(__name__)

__all__ = ['CalculateEtalonDriftCorrectionPrimitive', 'ApplyDriftCorrectionPrimitive']

class CalculateEtalonDriftCorrectionPrimitive(BasePrimitive):
    """
    Determines the per-etalon-peak drift by matching each etalon peak based on a proximity threshold in wavelength space.

    **For now, a per-order drift is calculated** as the weighted average of the drifts measured in detector pixels. The drift is then converted to effective drifts in wavelength and velocity units according to the local dispersion.

    Corrects the wavelength drifts. Currently only the Fabry-Pérot etalons are used to measure and correct for any drifts. Drifts are also currently calculated on a per-order basis which is subject to change as the instrument is characterized.

    *Calculation and applying the WLS are separate primitives while other WLS sources are under development.*

    Parameters
    ----------
    input
        Level 1 or 2 data product.
    suffix : str, optional
        Suffix for the output data product. Default is 'driftcalc'.
    do_fibers : Sequence[str], optional
        List of fibers to perform drift correction for. If None, all fibers will be used.
    do_orders : Sequence[int], optional
        List of orders to perform drift correction for. If None, all orders will be used.

    Calibrations
    ------------
    ref_etalon : ReferenceEtalonModel
        - selector: :class:`~hispecdrp.calibrations.selectors.ReferenceEtalonSelector`
    """
# 
    PARAMETERS = {
        'suffix' : 'driftcalc',
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

        input_model = datamodels.open(self.dataset)
        ref_model, _ = self.get_calibration(input_model, 'ref_etalon')
        ref_model = datamodels.open(ref_model)
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
                lcs_ref = line_centers_ref[order_mask_ref]
                lcs_err = line_center_errs[order_mask]
                lcs_ref_err = line_center_errs_ref[order_mask_ref]

                if len(lcs) == 0 or len(lcs_ref) == 0:
                    logger.info(f'No etalon lines found for {order=}, {fiber=}. Skipping.')
                    continue

                drifts = calculate_etalon_drifts_pix(lcs, lcs_err, lcs_ref, lcs_ref_err)
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
                    
                    pix_eff = weighted_quantile(pixels[drift_mask], w=w[drift_mask])
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
    Applies the calculated drift corrections to the input data's wavelength grid as a classical Doppler shift.

    Parameters
    ----------
    input
        Level 1 or 2 data product.
    suffix : str, optional
        Suffix for the output data product. Default is 'driftcorr'.
    fiber_reference_map : dict[str, str]
        A mapping of fiber numbers to their reference fibers.
        *Example to apply drift from fiber 1 to fibers 1-3.:*

            >>> fiber_reference_map = {
                    '1' : '1',
                    '2' : '1',
                    '3' : '1'
                }
    do_orders : Sequence[int], optional
        List of orders to perform drift correction for. If None, all orders will be used.
    """

    PARAMETERS = {
        'suffix' : 'driftcorr',
        'fiber_reference_map' : None,
        'do_orders' : None,
    }

    def _perform(self):
        input_model = datamodels.open(self.dataset)
        # Get fibers and orders
        _, do_orders = self.get_do_traces(input_model, None, self.do_orders)
        for fiber, ref_fiber in self.fiber_reference_map.items():
            wave = getattr(input_model, f'wave_fib{fiber}')
            for order in do_orders:
                drift_table = getattr(input_model, f'drift_fib{ref_fiber}')
                k = np.where(drift_table['ORDER'] == order)[0]
                if len(k) == 0:
                    logger.warning(f'No drift correction found for {fiber=} {order=}. Skipping.')
                    continue
                k = k[0]
                drift_vel = drift_table['DRIFT_VEL'][k]
                oi = np.abs(order - input_model.meta.order_start)
                wave[oi, :] *= (1 - drift_vel / scipy.constants.c) # NOTE: For line_pos_new - line_pos_ref

        return dict(output=input_model)
    

def calculate_drifts_from_line_centers(
    line_centers : np.ndarray,
    line_centers_err : np.ndarray,
    line_centers_ref : np.ndarray,
    line_centers_ref_err : np.ndarray,
    max_sep : float = 1
) -> list[dict[str, float]]:
    drifts = []
    for i_ref in range(len(line_centers_ref)):
        lc_ref = line_centers_ref[i_ref]
        lc_ref_err = line_centers_ref_err[i_ref]
        diffs = line_centers - lc_ref
        k = np.argmin(np.abs(diffs))
        if abs(diffs[k]) < max_sep:
            lc = line_centers[k]
            lc_err = line_centers_err[k]
            drift = lc - lc_ref
            drift_err = np.sqrt(lc_err**2 + lc_ref_err**2)
            drifts.append({
                'MU_REF': lc_ref,
                'MU': lc,
                'DRIFT': drift,
                'DRIFT_ERR': drift_err
            })
        else:
            drifts.append({
                'MU_REF': lc_ref,
                'MU': np.nan,
                'DRIFT': np.nan,
                'DRIFT_ERR': np.nan
            })

    return drifts
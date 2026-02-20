from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np
from typing import Sequence

from scipy.interpolate import LSQUnivariateSpline, CubicSpline
from scipy.ndimage import median_filter, gaussian_filter
from astropy.stats import mad_std

from ...calibrations import FiberFlatSelector

__all__ = ['BlazeCorrectionPrimitive']


class BlazeCorrectionPrimitive(BasePrimitive):
    
    ARGUMENTS = {
        'suffix' : 'blaze_correct',
        'max_cores' : 1,
        'do_fibers' : None,
        'do_orders' : None,
        'remove_lamp_blackbody' : True,
        'normalize_blaze' : True,
        'output_model' : 'L2Model',
    }

    CALIBRATIONS = {
        'fiberflat': {
            'selector': FiberFlatSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):

        if isinstance(self.input, (datamodels.DataModel, str)):
            output = None
            input = [self.input]
        elif isinstance(self.input, Sequence):
            output = []
            input = self.input

        # Loop over inputs (now a list)
        for _input in input:
        
            # Open the input model
            input_model = datamodels.open(_input)

            # Get calibrations
            fiberflat, _ = self.get_calibration(input_model, 'fiberflat')
            fiberflat_model = datamodels.open(fiberflat)

            # Init output model
            output_model = self.get_output_model(input_model)
            
            # Which fibers and orders to process
            do_fibers, do_orders = self.get_do_traces(input_model, self.do_fibers, self.do_orders)
            order_vec = input_model.make_order_vec()
            n_orders = len(order_vec)

            for fiber in do_fibers:
                npx = getattr(input_model, f'SPEC_UNCAL_FIB{fiber}').shape[1]
                if getattr(output_model, f'SPEC_FIB{fiber}') is None:
                    setattr(output_model, f'SPEC_FIB{fiber}', 
                            np.full((n_orders, npx), np.nan, dtype=input_model.traces[int(fiber)]['spec_uncal'].dtype))
                    setattr(output_model, f'SPEC_ERR_FIB{fiber}',
                            np.full((n_orders, npx), np.nan, dtype=input_model.traces[int(fiber)]['spec_uncal_err'].dtype))
                    setattr(output_model, f'SPEC_VAR_POISSON_FIB{fiber}', 
                            np.full((n_orders, npx), np.nan, dtype=input_model.traces[int(fiber)]['spec_uncal_var_poisson'].dtype))
                    setattr(output_model, f'SPEC_VAR_RNOISE_FIB{fiber}', 
                            np.full((n_orders, npx), np.nan, dtype=input_model.traces[int(fiber)]['spec_uncal_var_rnoise'].dtype))
                    setattr(output_model, f'SPEC_DQ_FIB{fiber}', 
                            np.zeros((n_orders, npx), dtype=input_model.traces[int(fiber)]['spec_dq'].dtype))
                    #TODO do we need to set an attribute to carry the fiber flat data used for blaze correction?

                # get aray of smoothed blaze values for this fiber and all orders in do_orders
                blaze_array = get_smooth_blaze(fiberflat_model.traces[int(fiber)]['spec'],
                                              fiberflat_model.traces[int(fiber)]['spec_err'],
                                              fiberflat_model.traces[int(fiber)]['spec_dq'],
                                              order_vec, do_orders,
                                              clipping_filter_size=15)
                blaze_std_array = fiberflat_model.traces[int(fiber)]['spec_err']
                
                # remove lamp blackbody from blaze array
                if self.remove_lamp_blackbody:
                    pass #TODO

                # normalize blaze array across orders with spline fit to order peaks
                if self.normalize_blaze:
                    blaze_array, blaze_std_array = normalize_blaze_array(blaze_array, 
                                                                blaze_std_array,  
                                                                fiberflat_model.traces[int(fiber)]['wave'], 
                                                                order_vec, do_orders)
                    pass #TODO

                # apply blaze correction to science spectrum using blaze array, and propagate errors
                apply_blaze_array(output_model, blaze_array, blaze_std_array, fiber, do_orders)

        return dict(output=output_model)
    

def get_smooth_blaze(fiberflat_flux, fiberflat_std, fiberflat_dq, order_vec, do_orders, clipping_filter_size=15):

    blaze_array = np.ones((len(order_vec), len(fiberflat_flux[0])), dtype=fiberflat_flux[0].dtype)
    for o, order in enumerate(order_vec):
        if order not in do_orders:
            continue
        flat_flux = fiberflat_flux[o]
        flat_std = fiberflat_std[o]
        dq_flags = fiberflat_dq[o]

        smooth_blaze = estimate_blaze_spline(flat_flux, flat_std, dq_flags, clipping_filter_size)
        blaze_array[o] = smooth_blaze

    return blaze_array
    

def estimate_blaze_spline(
    flat_flux,
    flat_std,
    dq_flags,
    clipping_filter_size=15
):
    """
    Estimate blaze using weighted cubic smoothing spline.

    Parameters
    ----------
    flat_flux : ndarray
        Fiber flat flux.
    flat_std : ndarray
        Fiber flat standard deviation.
    filter_size : int
        Size of the median and Gaussian filters to use.


    Returns
    -------
    blaze : ndarray
    blaze_std : ndarray
    """
    px = np.arange(len(flat_flux))
    weights = 1 / flat_std**2

    mask = (flat_flux > 0) & (flat_std > 0) & (dq_flags == 0) 

    # trend = gaussian_filter(median_filter(flat_flux[mask], size=clipping_filter_size), clipping_filter_size)
    # sigma_cut = 20.0
    # try:
    #     mask[~mask] = np.abs(flat_flux[~mask]-trend)/mad_std(flat_flux[~mask]-trend) > sigma_cut
    # except RuntimeWarning:
    #     mask[~mask] = np.abs(flat_flux[~mask]-trend)/np.std(flat_flux[~mask]-trend) > sigma_cut

    num_knots = max(10, len(px[mask]) // 16)  # heuristic: one knot per ~16 pixels, with a minimum of 10
    knots = np.linspace(px[mask].min(), px[mask].max(), num_knots)[1:-1]

    spline = LSQUnivariateSpline(
        px[mask],
        flat_flux[mask],
        t=knots,
        k=3,
        w=weights[mask],
        ext=3
    )

    smooth_flux = spline(px)

    return smooth_flux


def normalize_blaze_array(blaze_array, blaze_std, wavelengths, order_vec, do_orders):
    """Normalize blaze array across orders."""
    do_orders_idx = [i for i, x in enumerate(order_vec) if x in do_orders]
    px_max = np.argmax(blaze_array[do_orders_idx], axis=1)
    blaze_max = np.max(blaze_array[do_orders_idx], axis=1)
    wave_max = wavelengths[do_orders_idx,px_max]
    spline = CubicSpline(wave_max, blaze_max, bc_type='natural')
    normalized_blaze = blaze_array / spline(wavelengths)
    blaze_std_normalized = blaze_std / spline(wavelengths)
    return normalized_blaze, blaze_std_normalized



def apply_blaze_array(output_model, blaze_array, blaze_std_array, fiber, do_orders):
    """Apply blaze correction to science spectrum and propagate errors."""
    
    spec_blaze_cal_array = np.full_like(getattr(output_model, f'SPEC_UNCAL_FIB{fiber}'), np.nan)
    spec_blaze_cal_err_array = np.full_like(getattr(output_model, f'SPEC_UNCAL_ERR_FIB{fiber}'), np.nan)
    for o, order in enumerate(do_orders):
        spec_uncal = getattr(output_model, f'SPEC_UNCAL_FIB{fiber}')[o]
        spec_uncal_err = getattr(output_model, f'SPEC_UNCAL_ERR_FIB{fiber}')[o]
        blaze = blaze_array[o]
        blaze_std = blaze_std_array[o]

        spec_blaze_cal, spec_blaze_cal_err = apply_blaze(spec_uncal, spec_uncal_err, blaze, blaze_std)
        spec_blaze_cal_array[o] = spec_blaze_cal
        spec_blaze_cal_err_array[o] = spec_blaze_cal_err

    setattr(output_model, f'SPEC_FIB{fiber}', spec_blaze_cal_array)
    setattr(output_model, f'SPEC_ERR_FIB{fiber}', spec_blaze_cal_err_array)
        #TODO propagate DQ flags


    
def apply_blaze(science_flux, science_std, blaze, blaze_std):

    flux_corr = science_flux / blaze

    frac_science = science_std / science_flux
    frac_blaze   = blaze_std / blaze

    std_corr = np.abs(flux_corr) * np.sqrt(
        frac_science**2 + frac_blaze**2
    )

    return flux_corr, std_corr


    

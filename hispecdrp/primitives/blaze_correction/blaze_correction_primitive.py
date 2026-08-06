from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np
from typing import Sequence

from scipy.interpolate import LSQUnivariateSpline, CubicSpline
from scipy.ndimage import median_filter, gaussian_filter
from astropy.stats import mad_std

from ...calibrations import FiberFlatSelector
import logging

logger = logging.getLogger(__name__)

__all__ = ['BlazeCorrectionPrimitive']


class BlazeCorrectionPrimitive(BasePrimitive):
    
    ARGUMENTS = {
        'suffix' : 'blaze_correct',
        'max_cores' : 1,
        'do_fibers' : None,
        'do_orders' : None,
        'remove_lamp_blackbody' : False,
        'normalize_for_stitching' : False,
        'smooth_blaze' : False,
        'pixels_per_spline_knot' : 16,
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
            output_model = self._model_types(input_model)

            spec_key = 'spec_uncal' if isinstance(input_model, datamodels.level1.L1Model) else 'spec_'
            
            # Which fibers and orders to process
            do_fibers, do_orders = self.get_do_traces(input_model, self.do_fibers, self.do_orders)
            order_vec = input_model.make_order_vec()
            n_orders = len(order_vec)

            n_blazed = 0
            for fiber in do_fibers:
                ff_trace = fiberflat_model.traces.get(int(fiber)) if hasattr(fiberflat_model, "traces") else None
                ff_spec = None if ff_trace is None else ff_trace.get("spec")
                if ff_spec is None:
                    logger.warning(
                        "Skipping blaze for fiber %s: fiberflat has no SPEC_FIB%s "
                        "(restrict do_fibers or use a fiberflat covering this fiber).",
                        fiber,
                        fiber,
                    )
                    continue

                npx = getattr(input_model, f'SPEC_UNCAL_FIB{fiber}').shape[1]
                if getattr(output_model, f'SPEC_FIB{fiber}') is None:
                    setattr(output_model, f'SPEC_FIB{fiber}', 
                            np.full((n_orders, npx), np.nan, dtype=input_model.traces[int(fiber)][f'{spec_key}'].dtype))
                    setattr(output_model, f'SPEC_ERR_FIB{fiber}',
                            np.full((n_orders, npx), np.nan, dtype=input_model.traces[int(fiber)][f'{spec_key}_err'].dtype))
                    setattr(output_model, f'SPEC_VAR_POISSON_FIB{fiber}', 
                            np.full((n_orders, npx), np.nan, dtype=input_model.traces[int(fiber)]['spec_uncal_var_poisson'].dtype))
                    setattr(output_model, f'SPEC_VAR_RNOISE_FIB{fiber}', 
                            np.full((n_orders, npx), np.nan, dtype=input_model.traces[int(fiber)]['spec_uncal_var_rnoise'].dtype))
                    setattr(output_model, f'SPEC_DQ_FIB{fiber}', 
                            np.zeros((n_orders, npx), dtype=np.uint32))

                # get aray of smoothed blaze values for this fiber and all orders in do_orders
                blaze_array = get_smooth_blaze(ff_trace['spec'],
                                              ff_trace['spec_err'],
                                              ff_trace['spec_dq'],
                                              order_vec, do_orders, pixels_per_spline_knot=self.pixels_per_spline_knot)
                blaze_std_array = ff_trace['spec_err']
                
                # remove lamp blackbody from blaze array
                if self.remove_lamp_blackbody:
                    #TODO
                    raise NotImplementedError('remove_lamp_blackbody=True is not yet') 
                
                # normalize blaze array, either with simple median, or fitting spline to wavelength/blaze peaks
                if self.normalize_for_stitching:
                    # only necessary if stitching
                    blaze_array, blaze_std_array = normalize_blaze_array(blaze_array, 
                                                                blaze_std_array,  
                                                                ff_trace['wave'], 
                                                                order_vec, do_orders)
                else:
                    blaze_norm = np.nanmax(blaze_array, axis=1)[:, np.newaxis]
                    blaze_array /= blaze_norm
                    blaze_std_array /= blaze_norm

                # apply blaze correction to science spectrum using blaze array, and propagate errors
                apply_blaze_array(output_model, blaze_array, blaze_std_array,
                                fiber, order_vec, do_orders,
                                pixels_per_spline_knot=self.pixels_per_spline_knot)
                n_blazed += 1
                
                if isinstance(output, list):
                    output.append(output_model)
                else:
                    output = output_model

            if n_blazed == 0:
                raise ValueError(
                    "Blaze correction found no fibers with fiberflat SPEC data. "
                    f"Requested do_fibers={do_fibers}; fiberflat fibers with data: "
                    f"{sorted(f for f, t in fiberflat_model.traces.items() if t.get('spec') is not None)}."
                )

        return dict(output=output)
    
    def _model_types(self, input_model):
        if isinstance(input_model, datamodels.level1.L1Model):
            output_model = self.get_output_model(input_model, output_model='L2Model', include_data=True)
        elif isinstance(input_model, datamodels.level2.L2Model):
            output_model = self.get_output_model(input_model, output_model='input', include_data=True)
        else:
            raise TypeError(f"Unsupported input model type: {type(input_model)}. Expected L1Model or L2Model.")
        return output_model
    

def _longest_contiguous_true_segment(mask: np.ndarray) -> slice:
    """Return a slice covering the longest run of True values in ``mask``."""
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return slice(0, 0)
    padded = np.concatenate(([False], m, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    i = int(np.argmax(ends - starts))
    return slice(int(starts[i]), int(ends[i]))


def get_smooth_blaze(fiberflat_flux, fiberflat_std, fiberflat_dq, order_vec, do_orders, pixels_per_spline_knot=16):
    """
    Collect smoothed blaze function (not normalized) for each order and return in 2D array

    Parameters
    ----------
    fibarflat_flux : ndarray
        Fiber flat flux for all orders.
    fiberflat_std : ndarray
        Fiber flat standard deviation for all orders.
    fiberflat_dq : ndarray
        Data quality flags for all orders
    order_vec : list
        Contains all orders
    do_orders: List
        Which orders to get blaze function 
    pixels_per_spline_knot : int
        Number of pixels between spline nodes, used to adjust smoothing

    Returns
    -------
    smooth_flux : ndarray
        Smoothed fiber flat spectrum

    """

    blaze_array = np.ones((len(order_vec), len(fiberflat_flux[0])), dtype=fiberflat_flux[0].dtype)
    for o, order in enumerate(order_vec):
        if order not in do_orders:
            continue
        flat_flux = fiberflat_flux[o]
        flat_std = fiberflat_std[o]
        dq_flags = fiberflat_dq[o]

        smooth_blaze = estimate_blaze_spline(flat_flux, flat_std, dq_flags, pixels_per_spline_knot=pixels_per_spline_knot)
        blaze_array[o] = smooth_blaze 

    return blaze_array
    

def estimate_blaze_spline(
    flat_flux,
    flat_std,
    dq_flags,
    pixels_per_spline_knot=16,
):
    """
    Estimate blaze using weighted cubic lsq smoothing spline.

    Parameters
    ----------
    flat_flux : ndarray
        Fiber flat flux.
    flat_std : ndarray
        Fiber flat standard deviation.
    dq_flags : ndarray
        Data quality flags
    pixels_per_spline_knot : int
        Number of pixels between spline nodes, used to adjust smoothing

    Returns
    -------
    smooth_flux : ndarray
        Smoothed fiber flat spectrum
    """
    # TODO: make smoothing resistant to lines in the lamp spectrum
    px = np.arange(len(flat_flux), dtype=np.float64)
    flat_flux = np.asarray(flat_flux, dtype=np.float64)
    flat_std = np.asarray(flat_std, dtype=np.float64)

    mask = (
        np.isfinite(flat_flux)
        & np.isfinite(flat_std)
        & (flat_flux > 0)
        & (flat_std > 0)
        & (dq_flags == 0)
    )
    if not np.any(mask):
        return np.full(len(flat_flux), np.nan, dtype=flat_flux.dtype)

    # Trace gaps (NaN extraction) leave holes in the flat; fit the blaze spline on the
    # longest contiguous good segment only, then evaluate on the full pixel grid.
    seg = _longest_contiguous_true_segment(mask)
    fit_mask = np.zeros(len(flat_flux), dtype=bool)
    fit_mask[seg] = mask[seg]
    px_fit = px[fit_mask]
    if px_fit.size < 4:
        return np.full(len(flat_flux), np.nan, dtype=flat_flux.dtype)

    weights = 1.0 / flat_std[fit_mask]
    num_knots = max(0, int(px_fit.size // pixels_per_spline_knot))
    if num_knots >= 2:
        knots = np.linspace(px_fit.min(), px_fit.max(), num_knots)[1:-1]
    else:
        knots = np.empty(0, dtype=np.float64)

    smooth_flux = np.full(len(flat_flux), np.nan, dtype=flat_flux.dtype)
    try:
        if knots.size > 0:
            spline = LSQUnivariateSpline(
                px_fit,
                flat_flux[fit_mask],
                t=knots,
                k=3,
                w=weights,
                ext=3,
            )
        else:
            # Too few points for interior knots — cubic smoothing spline without LSQ knots.
            from scipy.interpolate import UnivariateSpline

            spline = UnivariateSpline(
                px_fit,
                flat_flux[fit_mask],
                w=weights,
                k=min(3, px_fit.size - 1),
                s=len(px_fit),
            )
        smooth_flux = np.asarray(spline(px), dtype=flat_flux.dtype)
    except ValueError:
        # Last resort: wide median filter on the contiguous segment, constant extrapolation.
        seg_flux = flat_flux[seg].copy()
        seg_flux[~mask[seg]] = np.nan
        filled = np.nan_to_num(seg_flux, nan=np.nanmedian(seg_flux))
        win = min(max(3, pixels_per_spline_knot * 2 + 1), len(filled) // 2 * 2 + 1)
        smooth_seg = median_filter(filled, size=win, mode="nearest")
        smooth_flux[seg] = smooth_seg

    smooth_flux[~mask] = np.nan
    return smooth_flux


def normalize_blaze_array(blaze_array, blaze_std, wavelengths, order_vec, do_orders):
    """Normalize blaze array across orders using a CubicSpline fit to wavelength/blaze peak.
    Necessary for stitching order by order spectra.
    
    Parameters
    ----------
    blaze_array : ndarray
        Array of blaze functions per order, not yet normalized.
    blaze_std : ndarray
        Array of standard deviation per order from the fiber flats.
    wavelengths : ndarray
        Wavelength solution per order for blaze function.
    order_vec : List
        Contains all orders
    do_orders:
        Which orders to perform normalization

    Returns
    -------
    normalized_blaze : ndarray
        Normalized blaze per order ready to be used to correct science spectra
    blaze_std_normalized : ndarray
        Normalized fiber flat std per order to be used when correcting science spectra
    """
    #TODO fix to perform the normalization in flux density (per wavelength) space instead of pixel rate
    do_orders_idx = [i for i, x in enumerate(order_vec) if x in do_orders]

    # calculate blaze function max and corresponding wavelength
    px_max = np.argmax(blaze_array[do_orders_idx], axis=1)
    blaze_max = np.max(blaze_array[do_orders_idx], axis=1)
    wave_max = wavelengths[do_orders_idx,px_max]

    # sort and interpolate
    wv_sort_idx = np.argsort(wave_max)
    spline = CubicSpline(wave_max[wv_sort_idx], blaze_max[wv_sort_idx], bc_type='natural')

    # normalize blaze functions
    normalized_blaze = blaze_array / spline(wavelengths)
    blaze_std_normalized = blaze_std / spline(wavelengths)
    return normalized_blaze, blaze_std_normalized


def apply_blaze_array(output_model, blaze_array, blaze_std_array, fiber, order_vec, do_orders, pixels_per_spline_knot=16):
    """Apply blaze correction to science spectrum and propagate errors for all orders, then add to output model.
    
    Parameters
    ----------
    output_model : datamodels.DataModel
        Output model to insert calibrated spectra, should already contain uncalibrated science spectra
    blaze_array : ndarray
        Normalized blaze functions per order ready to use for blaze correction
    blaze_std_array : ndarray
        Normalized blaze function stds per order ready to be used for error propagation
    fiber : int
        Current fiber
    do_orders: List
        Which orders to perform blaze correction
    pixels_per_spline_knot : int
        Number of pixels between spline nodes, used here to set dq flags for edges where smoothing works poorly
    """
    
    spec_blaze_cal_array = np.full_like(getattr(output_model, f'SPEC_UNCAL_FIB{fiber}'), np.nan)
    spec_blaze_cal_err_array = np.full_like(getattr(output_model, f'SPEC_UNCAL_ERR_FIB{fiber}'), np.nan)
    for o, order in enumerate(order_vec):
        if order in do_orders:
            spec_uncal = getattr(output_model, f'SPEC_UNCAL_FIB{fiber}')[o]
            spec_uncal_err = getattr(output_model, f'SPEC_UNCAL_ERR_FIB{fiber}')[o]
            blaze = blaze_array[o]
            blaze_std = blaze_std_array[o]

            spec_blaze_cal, spec_blaze_cal_err = apply_blaze(spec_uncal, spec_uncal_err, blaze, blaze_std)
            spec_blaze_cal_array[o] = spec_blaze_cal
            spec_blaze_cal_err_array[o] = spec_blaze_cal_err

    output_dq = output_model.traces[int(fiber)]['spec_dq']
    # pixel edges along dispersion (per order) cannot be smoothed properly with spline
    n_edge = int(pixels_per_spline_knot * 10)
    output_dq[:, :n_edge] = 1
    output_dq[:, -n_edge:] = 1 

    setattr(output_model, f'SPEC_FIB{fiber}', spec_blaze_cal_array)
    setattr(output_model, f'SPEC_ERR_FIB{fiber}', spec_blaze_cal_err_array)
    setattr(output_model, f'SPEC_DQ_FIB{fiber}', output_dq )

    
def apply_blaze(science_flux, science_std, blaze, blaze_std):
    """Apply blaze correction to science spectrum and propagate errors for all orders, then add to output model.
    
    Parameters
    ----------
    science_flux : ndarray
        science flux for single order
    science_std : ndarray
        science error for single order
    blaze : ndarray
        Normalized blaze function to be removed from science_flux
    blaze_std : ndarray
        Normalized blaze error used in error propagation
    """

    flux_corr = science_flux / blaze

    frac_science = science_std / science_flux
    frac_blaze   = blaze_std / blaze

    std_corr = np.abs(flux_corr) * np.sqrt(
        frac_science**2 + frac_blaze**2
    )

    return flux_corr, std_corr


    

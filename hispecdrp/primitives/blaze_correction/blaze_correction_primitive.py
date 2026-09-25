from ..base_primitive import BasePrimitive
from ... import datamodels
DQ_FLAGS = datamodels.DQ_FLAGS
import numpy as np
from typing import Sequence

from scipy.interpolate import LSQUnivariateSpline, CubicSpline
from scipy.ndimage import median_filter, gaussian_filter
from astropy.stats import mad_std

from ...calibrations import FiberFlatSelector
import logging

logger = logging.getLogger(__name__)

__all__ = [
    'BlazeCorrectionPrimitive',
    'lamp_spectrum_shape',
    'divide_out_lamp_spectrum',
]


class BlazeCorrectionPrimitive(BasePrimitive):
    
    PARAMETERS = {
        'suffix' : 'blaze_correct',
        'do_fibers' : None,
        'do_orders' : None,
        # Divide fiberflat by analytic lamp SED before blaze-correcting science.
        # Default shape matches simulated flats (1000 K blackbody); see lamp_temperature_k.
        'remove_lamp_spectrum' : False,
        'lamp_temperature_k' : 1000.0,
        # Deprecated alias for remove_lamp_spectrum (kept for older configs).
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
        
        # Open the input model
        input_model = datamodels.open(self.dataset)

        # Get calibrations
        fiberflat, _ = self.get_calibration(input_model, 'fiberflat')
        fiberflat_model = datamodels.open(fiberflat)

        # Init output model
        output_model = self.get_output_model(input_model)

        # Which version of the spectrum to blaze-correct
        spec_key = 'spec_uncal' if isinstance(input_model, datamodels.L1Model) else 'spec_'
        
        # Which fibers and orders to process
        do_fibers, do_orders = self.get_do_traces(input_model, self.do_fibers, self.do_orders)
        order_vec = input_model.make_order_vec()

        # Check that requested fibers have valid fiberflat data.
        missing_fibers = [fiber for fiber in do_fibers if not fiberflat_model.has_extension_value(f'SPEC_FIB{fiber}')]
        if missing_fibers:
            raise ValueError(
                f"Blaze correction requested fibers {do_fibers} but fiberflat has no SPEC data for fibers {missing_fibers}."
            )

        # Loop over fibers
        for fiber in do_fibers:

            # Skip if fiberflat has no data for this fiber
            if fiberflat_model.has_extension_value(f'SPEC_FIB{fiber}'):
                ff_spec = fiberflat_model[fiber].spec
                ff_spec_err = fiberflat_model[fiber].spec_err
            else:
                logger.warning(
                    f"Skipping blaze for fiber {fiber}: fiberflat has no SPEC_FIB{fiber} "
                    "(restrict do_fibers or use a fiberflat covering this fiber)."
                )
                continue

            # Initialize blaze array from fiber flat spectrum
            blaze_array = ff_spec.copy()
            blaze_err_array = ff_spec_err.copy()
            blaze_dq_array = fiberflat_model[fiber].spec_dq.copy()

            # Remove the lamp BB spectrum or not
            remove_lamp = bool(self.remove_lamp_spectrum) or bool(
                self.remove_lamp_blackbody
            )
            if self.remove_lamp_blackbody and not self.remove_lamp_spectrum:
                logger.warning(
                    "remove_lamp_blackbody is deprecated; use remove_lamp_spectrum=True"
                )
            if remove_lamp:
                if fiberflat_model.has_extension(f'WAVE_FIB{fiber}'):
                    wave_ff = fiberflat_model[fiber].wave
                else:
                    raise ValueError(
                        "remove_lamp_spectrum=True requires fiberflat WAVE "
                        f"(missing wave for fiber {fiber})"
                    )
                blaze_array, blaze_err_array = divide_out_lamp_spectrum(
                    blaze_array,
                    blaze_err_array,
                    wave_ff,
                    order_vec,
                    do_orders,
                    temperature_k=float(self.lamp_temperature_k),
                )

            # Smooth instrument blaze after optional lamp detrend
            if self.smooth_blaze:
                blaze_array = get_smooth_blaze(
                    blaze_array,
                    blaze_err_array,
                    fiberflat_model[fiber].spec_dq,
                    order_vec,
                    do_orders,
                    pixels_per_spline_knot=self.pixels_per_spline_knot,
                )

            # Normalize blaze array, either with simple median, or fitting spline to wavelength/blaze peaks
            if self.normalize_for_stitching:
                # only necessary if stitching
                blaze_array, blaze_err_array = normalize_blaze_array(
                    blaze_array,
                    blaze_err_array,
                    fiberflat_model[fiber].wave,
                    order_vec, do_orders
                )
            else:
                blaze_norm = np.nanmax(blaze_array, axis=1)[:, None]
                blaze_array /= blaze_norm
                blaze_err_array /= blaze_norm

            # Apply blaze correction to science spectrum using blaze array, and propagate errors
            apply_blaze_array(
                output_model,
                blaze_array, blaze_err_array, blaze_dq_array,
                fiber, do_orders,
            )

            # Do not use pixels at edge of the orders where smoothing is unreliable.
            if self.smooth_blaze:
                n_edge = int(self.pixels_per_spline_knot * 10)
                output_model[fiber].spec_dq[:, :n_edge] |= DQ_FLAGS['DO_NOT_USE']
                output_model[fiber].spec_dq[:, -n_edge:] |= DQ_FLAGS['DO_NOT_USE']

        return dict(output=output_model)
    
    def get_output_model(self, input_model, **kwargs):
        if isinstance(input_model, datamodels.L1Model):
            output_model = 'L2Model'
        elif isinstance(input_model, datamodels.L2Model):
            output_model = 'input'
        else:
            raise TypeError(f"Unsupported input model type: {type(input_model)}. Expected L1Model or L2Model.")
        output_model = super().get_output_model(
            input_model,
            output_model=output_model,
            include_data=True,
            **kwargs
        )
        return output_model


def lamp_spectrum_shape(
    wave_um: np.ndarray,
    *,
    temperature_k: float = 1000.0,
) -> np.ndarray:
    """
    Analytic lamp SED matching simulated fiberflats (energy density).

    Uses the same blackbody construction as
    :func:`~hispecdrp.simulations.simulate_up_the_ramp.create_dummy_fiberflat_rate_files`:
    ``BlackBody`` radiance per Å, then ×1e4 to per-µm energy density. Only the
    relative shape is used when dividing the fiberflat.
    """
    import astropy.units as u
    from astropy.modeling.models import BlackBody

    wave_um = np.asarray(wave_um, dtype=np.float64)
    out = np.full(wave_um.shape, np.nan, dtype=np.float64)
    good = np.isfinite(wave_um) & (wave_um > 0.0)
    if not np.any(good):
        return out

    wl_aa = (wave_um[good] * u.micron).to(u.AA)
    bb = BlackBody(
        temperature=float(temperature_k) * u.K,
        scale=1.0 * u.erg / (u.cm**2 * u.s * u.AA * u.sr),
    )
    # Match dummy-fiberflat: B_λ [per Å] → per µm energy density (×1e4).
    out[good] = np.asarray(bb(wl_aa).value, dtype=np.float64) * 1.0e4
    return out


def divide_out_lamp_spectrum(
    blaze_array: np.ndarray,
    blaze_err_array: np.ndarray,
    wavelengths: np.ndarray,
    order_vec,
    do_orders,
    *,
    temperature_k: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Divide fiberflat blaze (and its uncertainty) by an analytic lamp spectrum.

    Per-order lamp shapes are applied directly (no median renormalization) so
    the subsequent per-order max-normalization step preserves the correct
    instrument-blaze shape after dividing out BB×blaze fiberflats.
    """
    blaze = np.array(blaze_array, dtype=np.float64, copy=True)
    blaze_err = np.array(blaze_err_array, dtype=np.float64, copy=True)
    wave = np.asarray(wavelengths, dtype=np.float64)
    if wave.shape != blaze.shape:
        raise ValueError(
            f"fiberflat WAVE shape {wave.shape} != blaze shape {blaze.shape}"
        )

    for o, order in enumerate(order_vec):
        if order not in do_orders:
            continue
        sed = lamp_spectrum_shape(wave[o], temperature_k=temperature_k)
        valid = np.isfinite(sed) & (sed > 0.0) & np.isfinite(blaze[o])
        if not np.any(valid):
            logger.warning(
                "remove_lamp_spectrum: no valid lamp SED samples for order %s; skipping",
                order,
            )
            continue
        sed_norm = sed
        with np.errstate(divide="ignore", invalid="ignore"):
            blaze[o] = blaze[o] / sed_norm
            blaze_err[o] = blaze_err[o] / sed_norm
    return blaze, blaze_err


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


def get_smooth_blaze(fiberflat_flux, fiberflat_err, fiberflat_dq, order_vec, do_orders, pixels_per_spline_knot=16):
    """
    Collect smoothed blaze function (not normalized) for each order and return in 2D array

    Parameters
    ----------
    fibarflat_flux : ndarray
        Fiber flat flux for all orders.
    fiberflat_err : ndarray
        Fiber flat uncertainty for all orders.
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
        flat_err = fiberflat_err[o]
        dq_flags = fiberflat_dq[o]

        smooth_blaze = estimate_blaze_spline(flat_flux, flat_err, dq_flags, pixels_per_spline_knot=pixels_per_spline_knot)
        blaze_array[o] = smooth_blaze 

    return blaze_array
    

def estimate_blaze_spline(
    flat_flux,
    flat_err,
    dq_flags,
    pixels_per_spline_knot=16,
):
    """
    Estimate blaze using weighted cubic lsq smoothing spline.

    Parameters
    ----------
    flat_flux : ndarray
        Fiber flat flux.
    flat_err : ndarray
        Fiber flat uncertainty.
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
    flat_err = np.asarray(flat_err, dtype=np.float64)

    mask = (
        np.isfinite(flat_flux)
        & np.isfinite(flat_err)
        & (flat_flux > 0)
        & (flat_err > 0)
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

    weights = 1.0 / flat_err[fit_mask]
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


def normalize_blaze_array(
    blaze_array : np.ndarray, blaze_err : np.ndarray,
    wavelengths : np.ndarray,
    order_vec : list, do_orders : list
):
    """Normalize blaze array across orders using a CubicSpline fit to wavelength/blaze peak.
    Necessary for stitching order by order spectra.
    
    Parameters
    ----------
    blaze_array : ndarray
        Array of blaze functions per order, not yet normalized.
    blaze_err : ndarray
        Array of uncertainty per order from the fiber flats.
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
    blaze_err_normalized : ndarray
        Normalized fiber flat err per order to be used when correcting science spectra
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
    blaze_err_normalized = blaze_err / spline(wavelengths)
    return normalized_blaze, blaze_err_normalized


def apply_blaze_array(
    output_model : datamodels.L2Model,
    blaze_array : np.ndarray,
    blaze_err_array : np.ndarray,
    blaze_dq_array : np.ndarray,
    fiber : int,
    do_orders : list,
):
    """
    Apply blaze correction to science spectrum and propagate errors for all orders, then add to output model. Correction is performed to the "_spec" attribute and corresponding error attributes.
    
    Parameters
    ----------
    output_model : datamodels.L2Model
        Output model to correct blaze for. Array shapes = (N_ORDERS, N_PIXELS)
    blaze_array : ndarray
        Normalized blaze with shape = (N_ORDERS, N_PIXELS).
    blaze_err_array : ndarray
        Normalized blaze errors with shape = (N_ORDERS, N_PIXELS).
    blaze_dq_array : ndarray
        Blaze data quality flags with shape = (N_ORDERS, N_PIXELS).
    fiber : int
        Which fiber to correct in output_model.
    do_orders: list[int]
        Which orders to perform blaze correction
    """
    order_vec = output_model.make_order_vec()
    for o, order in enumerate(order_vec):
        if order in do_orders:
            apply_blaze(
                output_model[fiber].spec[o],
                output_model[fiber].spec_err[o],
                output_model[fiber].spec_var_rnoise[o],
                output_model[fiber].spec_var_poisson[o],
                output_model[fiber].spec_dq[o],
                blaze_array[o],
                blaze_err_array[o],
                blaze_dq_array[o]
            )

    
def apply_blaze(
    science_flux : np.ndarray,
    science_err : np.ndarray | None = None,
    science_var_rnoise : np.ndarray | None = None,
    science_var_poisson : np.ndarray | None = None,
    science_dq : np.ndarray | None = None,
    blaze : np.ndarray | None = None,
    blaze_err : np.ndarray | None = None,
    blaze_dq : np.ndarray | None = None,
):
    """
    Apply blaze correction to science spectrum and propagate errors for a single order. Corrections are performed in place.
    
    Parameters
    ----------
    science_flux : ndarray
        science flux for single order
    science_err : ndarray | None
        science error for single order
    science_var_rnoise : ndarray | None
        Optional science read noise variance for single order.
    science_var_poisson : ndarray | None
        Optional science poisson variance for single order.
    science_dq : ndarray | None
        science data quality flags for single order.
    blaze : ndarray | None
        Normalized blaze function to be removed from science_flux
    blaze_err : ndarray | None
        Normalized blaze error used in error propagation.
    blaze_dq : ndarray | None
        Normalized blaze dq used to propagate dq flags to science spectrum.
    """

    # Correct the science flux
    science_flux /= blaze

    # Propagate errors
    if science_err is not None and blaze_err is not None:
        frac_science = science_err / science_flux
        frac_blaze = blaze_err / blaze
        science_err[:] = np.abs(science_flux) * np.sqrt(
            frac_science**2 + frac_blaze**2
        )

    # Scale variance arrays
    if science_var_rnoise is not None:
        science_var_rnoise /= blaze**2
    if science_var_poisson is not None:
        science_var_poisson /= blaze**2

    # Propagate DQ flags
    if science_dq is not None and blaze_dq is not None:
        science_dq |= blaze_dq

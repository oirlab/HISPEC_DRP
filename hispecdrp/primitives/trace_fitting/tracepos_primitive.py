from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np
from numba import njit
import scipy.signal
from typing import Sequence
from ..utils.array_utils import median_filter_2d, nanmedian_axis0_ignore_empty_columns
import matplotlib.pyplot as plt

from ..spectral_extraction.extraction_utils import get_extraction_window

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

__all__ = ['TracePositionPrimitive']


class TracePositionPrimitive(BasePrimitive):

    """
    Calculate the trace positions for each fiber/order.

    Parameters
    ----------
    input : DataModel, str, Sequence[DataModel or str], or dict
        Input data. Can be a single DataModel or filename, a sequence of DataModels or filenames, or a dict mapping fiber numbers to DataModels or filenames.
    deg_pos : int
        Degree of polynomial to fit for trace position as a function of pixel.
    deg_width : int
        Degree of polynomial to fit for trace width as a function of pixel.
    window_size : int
        Size of window (in pixels) to use when computing centroids and widths.
    xdisp_range : tuple or dict
        Range in the cross dispersion direction (in pixels) to restrict trace finding for the central slice.
        Can be a single tuple for all fibers or a dict mapping fiber numbers to tuples.
    disp_range : tuple or dict
        Range in the dispersion direction (in pixels) to restrict trace finding.
        Can be a single tuple for all fibers or a dict mapping fiber numbers to tuples.
    prominence_factor : float
        Initial peak-finding threshold as a fraction of the maximum cross-dispersion
        profile value (default 0.05). Lower values recover fainter echelle orders on
        high dynamic-range fiber flats.
    min_flux_frac : float
        Per-order signal floor for the trace walker (default 0.05). Rows whose windowed
        flux falls below this fraction of the order's peak windowed flux are dropped from
        the polynomial fit, so faint blaze ends and partially illuminated orders cannot
        pull the trace into unphysical curves.

    Returns
    -------
    output : TracePosModel
        Output model containing fitted trace positions and widths for each fiber.
    """

    PARAMETERS = {
        'deg_pos' : None,
        'deg_width' : None,
        'window_size' : 8,
        'xdisp_range' : None,
        'disp_range' : None,
        'prominence_factor' : 0.05,
        'min_flux_frac' : 0.05,
        'output_model' : 'TracePosModel',
    }

    def _perform(self):

        # Associate inputs by fiber number
        if isinstance(self.dataset, (datamodels.DataModel, str)):
            self._input_by_fiber = datamodels.group_by_fiber_combination([self.dataset])
        if isinstance(self.dataset, Sequence):
            self._input_by_fiber = datamodels.group_by_fiber_combination(self.dataset)
        elif isinstance(self.dataset, dict):
            self._input_by_fiber = self.dataset
        else:
            raise ValueError(f"Unsupported input type: {type(self.dataset)}")
        
        first_key = next(iter(self._input_by_fiber))

        # Current keys: ('WHITELIGHT', 'BLOCKED', 'BLOCKED') : file, etc.
        # Want: ('1' : file, '2' : file, '3' : file)
        if not str(first_key).isdigit():
            self._input_by_fiber = {
                str(t.index(next(x for x in t if x.upper() != "BLOCKED")) + 1) : v
                for t, v in self._input_by_fiber.items()
            }

        # Prepare output model
        first_model = datamodels.open(list(self._input_by_fiber.values())[0])
        output_model = datamodels.TracePosModel.from_datamodels(list(self._input_by_fiber.values()), primitive=self)

        # Init trace dict
        trace_results = {}

        # Pixel array pos(xarr), xarr either rows or columns depending on orientation of orders
        ny, nx = first_model.rate.shape
        if first_model.meta.orders_are_vertical:
            pixel_arr = np.arange(ny)
        else:
            pixel_arr = np.arange(nx)

        # Order array
        order_vec = first_model.make_order_vec()
        n_orders = len(order_vec)

        # Loop over fibers and fit traces
        for fiber in self._input_by_fiber.keys():

            # Open input model for this fiber
            input_model = datamodels.open(self._input_by_fiber[fiber])

            # Get xdisp_range, disp_range, and prominence for this fiber.
            xdisp_range = self.xdisp_range[fiber] if isinstance(self.xdisp_range, dict) else self.xdisp_range
            disp_range = self.disp_range[fiber] if isinstance(self.disp_range, dict) else self.disp_range
            if isinstance(self.prominence_factor, dict):
                prominence_factor = self.prominence_factor[fiber]
            else:
                prominence_factor = self.prominence_factor

            # Transpose if needed
            if input_model.meta.orders_are_vertical:
                _input_rate, _input_dq = input_model.rate, input_model.dq
            else:
                _input_rate, _input_dq = input_model.rate.T, input_model.dq.T

            logger.info(f"Performing trace finding on {input_model} fiber {fiber}")

            # Find trace positions
            r = find_trace_positions(
                data=_input_rate, dq=_input_dq,
                disp_range=disp_range, xdisp_range=xdisp_range,
                deg_pos=self.deg_pos, deg_width=self.deg_width,
                window_size=self.window_size,
                prominence_factor=prominence_factor,
                min_flux_frac=self.min_flux_frac,
                n_expected_orders=n_orders,
            )

            # Sanity check
            num_orders_found = r['coeffs_pos'].shape[0]
            assert num_orders_found == n_orders, f"Found {num_orders_found} traces, expected {n_orders} for fiber {fiber}"
            logger.info(f"Found {num_orders_found} traces")

            # Store results in output model
            # Evaluate polynomials
            trace_results[fiber] = r
            pos = np.full((n_orders, len(pixel_arr)), np.nan)
            fwhm = np.full((n_orders, len(pixel_arr)), np.nan)
            for i in range(num_orders_found):
                cent_row = r['trace_centroids'][i, : len(pixel_arr)]
                width_row = r['trace_widths'][i, : len(pixel_arr)]
                pos[i, :] = np.polyval(r['coeffs_pos'][i, :], pixel_arr)
                fwhm[i, :] = np.polyval(r['coeffs_width'][i, :], pixel_arr)
                # Do not extrapolate trace polynomials into rows where the walker failed.
                pos[i, ~np.isfinite(cent_row)] = np.nan
                fwhm[i, ~np.isfinite(width_row)] = np.nan
            setattr(output_model, f'COEFF_POS_FIB{fiber}', r['coeffs_pos'])
            setattr(output_model, f'COEFF_FWHM_FIB{fiber}', r['coeffs_width'])
            setattr(output_model, f'POS_FIB{fiber}', pos)
            setattr(output_model, f'FWHM_FIB{fiber}', fwhm)

        # Return
        return dict(output=output_model, trace_results=trace_results)
        
    def _create_preview(self):

        active_fibers = list(self._input_by_fiber.keys())
        if not active_fibers:
            logger.warning("No trace position data found for preview.")
            return
        
        tracepos_model = self._output['output']

        # Create subplots
        n_fibers = len(active_fibers)
        fig, axes = plt.subplots(
            nrows=1, ncols=n_fibers,
            figsize=(4 * n_fibers, 4),
            squeeze=False,
            dpi=150,
        )

        for i, fiber in enumerate(active_fibers):

            ax = axes[0, i]
            ax.set_title(f"Fiber {fiber}")

            input_model = datamodels.open(self._input_by_fiber[fiber])
            im_data = input_model.rate
            ny, nx = im_data.shape
            zmin = 0
            zmax = np.nanquantile(im_data, 0.98)

            # Downsample image for preview to save space
            # Use slice factor of 4 (16x smaller size)
            res = 2
            im_data_small = im_data[::res, ::res]
            # Adjust x/y coordinates to center the large pixels
            x_small = np.arange(0, nx, res) + res/2 - 0.5
            y_small = np.arange(0, ny, res) + res/2 - 0.5

            # Heatmap
            extent = [x_small[0] - res/2, x_small[-1] + res/2,
                    y_small[-1] + res/2, y_small[0] - res/2]
            ax.imshow(
                im_data_small,
                origin='upper',
                extent=extent,
                vmin=zmin, vmax=zmax,
                cmap='viridis',
                interpolation='nearest',
                aspect='equal'
            )

            # Get traces
            pos = getattr(tracepos_model, f'POS_FIB{fiber}')
            n_orders = pos.shape[0]
            orders_are_vertical = tracepos_model.meta.orders_are_vertical
            pixarr = np.arange(pos.shape[1])
            
            for ord_idx in range(n_orders):
                
                # Center line
                trace_y = pos[ord_idx, :]
                
                # Check orientation for plotting
                if orders_are_vertical:
                    # If vertical, x is the spatial dimension (trace position), y is spectral (pixel index)
                    # The pos array stores position as function of pixel index.
                    
                    # Handle nan values for filled region
                    valid = np.isfinite(trace_y)
                    if not np.any(valid):
                        continue
                    
                    curr_trace_pos = trace_y[valid]
                    curr_pixels = pixarr[valid]

                    # Center line coords
                    x_trace = curr_trace_pos
                    y_trace = curr_pixels

                else:
                    # Horizontal traces
                    # x=pixel_index, y=pos

                    # Handle nan values for filled region
                    valid = np.isfinite(trace_y)
                    if not np.any(valid):
                        continue
                    
                    curr_trace_pos = trace_y[valid]
                    curr_pixels = pixarr[valid]

                    # Center line coords
                    x_trace = curr_pixels
                    y_trace = curr_trace_pos

                # Add center line
                ax.plot(
                    x_trace,
                    y_trace,
                    color='red', linewidth=1,
                    label=f'Order {ord_idx}'
                )

            ax.grid(False)

        # Update layout
        fig.suptitle(f"Trace Positions: {tracepos_model.meta.filename}")
        fig.tight_layout()
        
        # File paths (same directory as FITS products from save_output)
        output_path = self.generate_output_path(tracepos_model).replace(
            ".fits", "_preview.png"
        )

        if self.save_preview:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')

        if self.show_preview:
            logger.info("Showing trace position preview...")
            plt.show()

        plt.close(fig)
        self._output['preview'] = output_path



def find_trace_positions(
    data : np.ndarray,
    dq : np.ndarray,
    deg_pos : int,
    deg_width : int,
    xdisp_range : tuple = None,
    disp_range : tuple = None,
    fwhm : float = 2.5,
    window_size : int | None = None,
    distance_factor : float = 0.5,
    prominence_factor : float = 0.05,
    min_flux_frac : float = 0.05,
    max_step : float | None = None,
    n_expected_orders : int | None = None,
) -> dict:

    # Initial parameters
    ny, nx = data.shape
    if xdisp_range is None:
        xdisp_range = (16, nx - 16)
    if disp_range is None:
        disp_range = (16, ny - 16)
    if window_size is None:
        window_size = int(np.round(fwhm * 2.5))
    window_size_half = int(np.round(window_size / 2))
    min_distance = max(1, int(fwhm * distance_factor))
    # When the expected order count is known, require detected peaks to be separated by at
    # least a fraction of the mean inter-order spacing. This suppresses spurious split
    # peaks on bright orders (a single order resolving into two adjacent maxima) without
    # merging genuinely distinct orders, which are spaced far wider than this floor.
    if n_expected_orders is not None and n_expected_orders > 0:
        span = max(1, int(xdisp_range[1]) - int(xdisp_range[0]))
        spacing_distance = int(0.35 * span / n_expected_orders)
        min_distance = max(min_distance, spacing_distance)
    # Maximum allowed centroid shift between adjacent rows. Real echelle traces curve by
    # well under a pixel per row, so a tight cap rejects order-hopping and runaway drift
    # into inter-order gaps while still following genuine trace curvature.
    if max_step is None:
        max_step = float(window_size_half)

    # Fix negative values in data
    data = data.copy()
    data[data < 0] = 0

    # Find peaks for center column
    y0 = ny // 2
    ki, kf = max(0, y0 - 3), min(ny, y0 + 3)
    y = nanmedian_axis0_ignore_empty_columns(
        median_filter_2d(data[ki:kf+1, :], dq[ki:kf+1, :], window_size=(3, 3))
    )
    y = y[xdisp_range[0]:xdisp_range[1] + 1]
    # Interpolate over NaNs (masked / flagged pixels) before peak finding: scipy.signal
    # find_peaks silently drops nearly all peaks when NaNs are present because comparisons
    # against NaN are False, so a handful of scattered bad pixels can collapse the detected
    # order count. Linear interpolation across finite samples preserves the order profile.
    finite = np.isfinite(y)
    if finite.any() and not finite.all():
        idx = np.arange(y.size)
        y = np.interp(idx, idx[finite], y[finite])
    y_max = np.nanmax(y) if np.any(np.isfinite(y)) else 0.0
    min_prominence = y_max * prominence_factor
    peaks0, _ = scipy.signal.find_peaks(
        y,
        prominence=min_prominence,
        distance=min_distance
    )
    peaks0 += xdisp_range[0]
    num_traces = len(peaks0)

    coeffs_pos = np.full((num_traces, deg_pos + 1), np.nan)
    coeffs_width = np.full((num_traces, deg_width + 1), np.nan)
    trace_centroids = np.full((num_traces, ny), np.nan)
    trace_widths = np.full((num_traces, ny), np.nan)
    trace_flux = np.full((num_traces, ny), np.nan)

    pixel_arr = np.arange(nx)

    # Replace NaN / non-finite pixels (e.g. dq-flagged) with zero before walking: a single
    # flagged pixel inside the centroid window makes the windowed flux non-finite, which
    # aborts the walk for the whole order. Treating bad pixels as zero signal lets the walk
    # bridge isolated flags while contributing negligibly to the local centroid.
    data_for_walk = np.where(np.isfinite(data), data, 0.0)

    # Walk each trace up/down from initial peak position
    for i in range(num_traces):
        trace_centroids[i, :], trace_widths[i, :], trace_flux[i, :] = walk_trace_up_down(
            pixel_arr,
            data_for_walk,
            dq,
            y0,
            peaks0[i],
            disp_range,
            window_size_half,
            max_step,
        )
        # Reject low-signal rows: where the windowed flux falls below a fraction of the
        # order's peak windowed flux, the centroid is dominated by noise / inter-order
        # background and must not enter the polynomial fit (otherwise faint blaze ends and
        # partially illuminated orders pull the fit into wild, unphysical curves).
        flux_row = trace_flux[i, :]
        if np.any(np.isfinite(flux_row)):
            peak_flux = np.nanmax(flux_row)
            if np.isfinite(peak_flux) and peak_flux > 0:
                low = ~(flux_row > (min_flux_frac * peak_flux))
                trace_centroids[i, low] = np.nan
                trace_widths[i, low] = np.nan
        good = np.where(np.isfinite(trace_centroids[i, :]))[0]
        if len(good) >= deg_pos + 1:
            coeffs_pos[i, :] = np.polyfit(good, trace_centroids[i, good], deg_pos)
            #breakpoint()
            #plt.plot(good, trace_centroids[i, good], 'o')
            #plt.plot(np.arange(ny), np.polyval(coeffs_pos[i, :], np.arange(ny)))
            # plt.plot(good, trace_centroids[i, good] - np.polyval(coeffs_pos[i, :], good), 'o'); plt.show()
            #plt.show()
        good = np.where(np.isfinite(trace_widths[i, :]))[0]
        if len(good) >= deg_width + 1:
            coeffs_width[i, :] = np.polyfit(good, trace_widths[i, good], deg_width)

    return dict(
        coeffs_pos=coeffs_pos,
        coeffs_width=coeffs_width,
        trace_centroids=trace_centroids,
        trace_widths=trace_widths,
        trace_flux=trace_flux,
    )


def walk_trace_up_down(
    pixels : np.ndarray,
    data : np.ndarray,
    dq : np.ndarray,
    disp0 : float,
    xdisp0 : float,
    disp_range : tuple,
    window_size_half : float,
    max_step : float,
) -> np.ndarray:
    trace_centroids = np.full(data.shape[0], np.nan)
    trace_widths = np.full(data.shape[0], np.nan)
    trace_flux = np.full(data.shape[0], np.nan)
    _walk_trace_up_down(
        pixels,
        data,
        dq,
        int(disp0),
        float(xdisp0),
        disp_range,
        window_size_half,
        max_step,
        trace_centroids,
        trace_widths,
        trace_flux,
    )
    return trace_centroids, trace_widths, trace_flux



@njit(nogil=True, cache=True)
def _walk_trace_up_down(
    pixels: np.ndarray,
    data: np.ndarray,
    dq: np.ndarray,
    y0: int,
    xc0: float,
    disp_range: tuple,
    window_size_half: float,
    max_step: float,
    trace_centroids: np.ndarray,
    trace_widths: np.ndarray,
    trace_flux: np.ndarray,
):

    # Central slice
    c, w, f = _compute_centroid_and_width_2pass(
        pixels,
        data[y0, :],
        dq[y0, :],
        xc0,
        window_size_half,
    )
    trace_centroids[y0], trace_widths[y0], trace_flux[y0] = c, w, f
    anchor0 = c if np.isfinite(c) else xc0

    # Walk down. The window is anchored on the last accepted centroid (not the previous
    # row's raw value) so that a row with no real signal cannot seed a runaway drift; a
    # centroid that jumps by more than ``max_step`` is rejected (order hop / noise spike).
    anchor = anchor0
    for y in range(y0 - 1, disp_range[0] - 1, -1):
        c, w, f = _compute_centroid_and_width_2pass(
            pixels,
            data[y, :],
            dq[y, :],
            anchor,
            window_size_half,
        )
        trace_flux[y] = f
        if np.isfinite(c) and abs(c - anchor) <= max_step:
            trace_centroids[y], trace_widths[y] = c, w
            anchor = c
        else:
            trace_centroids[y], trace_widths[y] = np.nan, np.nan

    # Walk up
    anchor = anchor0
    for y in range(y0 + 1, disp_range[1] + 1):
        c, w, f = _compute_centroid_and_width_2pass(
            pixels,
            data[y, :],
            dq[y, :],
            anchor,
            window_size_half,
        )
        trace_flux[y] = f
        if np.isfinite(c) and abs(c - anchor) <= max_step:
            trace_centroids[y], trace_widths[y] = c, w
            anchor = c
        else:
            trace_centroids[y], trace_widths[y] = np.nan, np.nan


# @njit(nogil=True, cache=True)
# def _compute_centroid_and_width(
#     pixels : np.ndarray,
#     trace_slice : np.ndarray,
#     trace_dq : np.ndarray,
#     center_loc : float,
#     window_size_half : float,
# ):
#     """
#     Compute the centroid, width, and integrated flux of a trace slice.

#     Returns
#     -------
#     centroid : float
#         Flux-weighted centroid, or NaN when the window has no usable signal.
#     fwhm : float
#         Gaussian-equivalent FWHM, or NaN when the window has no usable signal.
#     total_flux : float
#         Summed flux inside the window (0.0 when the window is out of bounds / empty).
#     """

#     if not np.isfinite(center_loc):
#         return np.nan, np.nan, 0.0

#     xi = int(np.round(center_loc - window_size_half))
#     xf = int(np.round(center_loc + window_size_half + 1))
#     if xi < 0 or xf > trace_slice.size - 1:
#         return np.nan, np.nan, 0.0

#     slice_x = pixels[xi : xf + 1]
#     slice_y = trace_slice[xi : xf + 1]
#     slice_dq = trace_dq[xi : xf + 1]

#     n_good = np.sum(slice_dq == 0)
#     if n_good < slice_dq.size:
#         return np.nan, np.nan, 0.0

#     total_flux = np.sum(slice_y)

#     if total_flux > 0:
#         centroid = np.sum(slice_x * slice_y) / total_flux
#         variance = np.sum(((slice_x - centroid) ** 2) * slice_y) / total_flux
#         if variance < 0:
#             return np.nan, np.nan, total_flux
#         fwhm = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(variance)
#         return centroid, fwhm, total_flux
#     else:
#         return np.nan, np.nan, 0.0


@njit(nogil=True, cache=True)
def _compute_centroid_and_width(
    pixels : np.ndarray,
    trace_slice : np.ndarray,
    trace_dq : np.ndarray,
    center_loc : float,
    window_size_half : float,
):
    """
    Compute the centroid, width, and integrated flux of a trace slice.

    The window is the continuous interval ``[center_loc - window_size_half,
    center_loc + window_size_half]``; ``get_extraction_window`` supplies the fractional
    overlap of each pixel with that interval, so the pixels straddling the window edges
    contribute partially. No interpolation is performed — pixel values are used as-is and
    only their weights vary.

    Returns
    -------
    centroid : float
        Flux-weighted centroid, or NaN when the window has no usable signal.
    fwhm : float
        Gaussian-equivalent FWHM, or NaN when the window has no usable signal.
    total_flux : float
        Weighted flux inside the window (0.0 when the window is out of bounds / empty).
    """

    if not np.isfinite(center_loc):
        return np.nan, np.nan, 0.0

    n = trace_slice.size

    # Reject windows that are not fully on the detector. get_extraction_window would
    # silently clip them to the edge, which biases the centroid inward; the walker wants
    # these rows dropped instead.
    if center_loc - window_size_half < -0.5 or center_loc + window_size_half > n - 0.5:
        return np.nan, np.nan, 0.0

    idx, W = get_extraction_window(center_loc, 2.0 * window_size_half, n)
    if idx.size == 0:
        return np.nan, np.nan, 0.0

    # Any flagged pixel touching the window invalidates the measurement.
    for j in range(idx.size):
        if W[j] > 0.0 and trace_dq[idx[j]] != 0:
            return np.nan, np.nan, 0.0

    total_flux = 0.0
    moment1 = 0.0
    for j in range(idx.size):
        i = idx[j]
        fi = W[j] * trace_slice[i]
        total_flux += fi
        moment1 += pixels[i] * fi

    if total_flux <= 0:
        return np.nan, np.nan, 0.0

    centroid = moment1 / total_flux

    variance = 0.0
    for j in range(idx.size):
        i = idx[j]
        d = pixels[i] - centroid
        variance += d * d * W[j] * trace_slice[i]
    variance /= total_flux

    if variance < 0:
        return np.nan, np.nan, total_flux

    fwhm = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(variance)
    return centroid, fwhm, total_flux


@njit(nogil=True, cache=True)
def _compute_centroid_and_width_2pass(
    pixels : np.ndarray,
    trace_slice : np.ndarray,
    trace_dq : np.ndarray,
    center_loc : float,
    window_size_half : float,
):
    """
    Two-pass centroid: the first pass centers the window on ``center_loc`` (the anchor
    carried from the previous row), the second re-centers it on the first pass's centroid
    so the fractional edge weights are symmetric about the measured trace rather than about
    the anchor. Falls back to the first-pass result if the refinement fails (e.g. the
    re-centered window runs off the edge or picks up a flagged pixel).
    """

    c, w, f = _compute_centroid_and_width(
        pixels, trace_slice, trace_dq, center_loc, window_size_half,
    )
    if not np.isfinite(c):
        return c, w, f

    c2, w2, f2 = _compute_centroid_and_width(
        pixels, trace_slice, trace_dq, c, window_size_half,
    )
    if not np.isfinite(c2):
        return c, w, f
    return c2, w2, f2
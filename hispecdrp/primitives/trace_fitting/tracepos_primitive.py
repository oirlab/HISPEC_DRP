from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np
from numba import njit
import scipy.signal
from typing import Sequence
from ...utils.math import median_filter_2d
import matplotlib.pyplot as plt

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

    Returns
    -------
    output : TracePosModel
        Output model containing fitted trace positions and widths for each fiber.
    """

    ARGUMENTS = {
        'deg_pos' : None,
        'deg_width' : None,
        'window_size' : 8,
        'xdisp_range' : None,
        'disp_range' : None,
        'output_model' : 'TracePosModel',
    }

    def _perform(self):

        # Associate inputs by fiber number
        if isinstance(self.input, (datamodels.DataModel, str)):
            self._input_by_fiber = datamodels.group_by_fiber_combination([self.input])
        if isinstance(self.input, Sequence):
            self._input_by_fiber = datamodels.group_by_fiber_combination(self.input)
        elif isinstance(self.input, dict):
            self._input_by_fiber = self.input
        else:
            raise ValueError(f"Unsupported input type: {type(self.input)}")
        
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

            # Get xdisp_range and disp range for this fiber.
            xdisp_range = self.xdisp_range[fiber] if isinstance(self.xdisp_range, dict) else self.xdisp_range
            disp_range = self.disp_range[fiber] if isinstance(self.disp_range, dict) else self.disp_range

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
                pos[i, :] = np.polyval(r['coeffs_pos'][i, :], pixel_arr)
                fwhm[i, :] = np.polyval(r['coeffs_width'][i, :], pixel_arr)
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
            zmax = np.nanquantile(im_data, 0.999)

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
        
        # File paths
        output_path = tracepos_model._generate_output_path(
            suffix=self.suffix
        ).replace(".fits", "_preview.png")

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
    prominence_factor : float = 0.05
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

    # Fix negative values in data
    data = data.copy()
    data[data < 0] = 0

    # Find peaks for center column
    y0 = ny // 2
    ki, kf = max(0, y0 - 3), min(ny, y0 + 3)
    y = np.nanmedian(median_filter_2d(data[ki:kf+1, :], dq[ki:kf+1, :], window_size=(3, 3)), axis=0)
    y = y[xdisp_range[0]:xdisp_range[1] + 1]
    min_prominence = np.nanmax(y) * prominence_factor
    peaks0, _ = scipy.signal.find_peaks(
        y,
        prominence=min_prominence,
        distance=min_distance
    )
    peaks0 += xdisp_range[0]
    num_traces = len(peaks0)

    # NOTE: Keep for debugging
    # breakpoint()
    #plt.imshow(data, vmin=0, vmax=10); plt.ylim(0, ny); plt.show()
    #
    # xd_pix = np.arange(nx); plt.plot(xd_pix[xdisp_range[0]:xdisp_range[1] + 1], y); [plt.axvline(p, color='r', ls='--') for p in peaks0]; plt.show()

    coeffs_pos = np.full((num_traces, deg_pos + 1), np.nan)
    coeffs_width = np.full((num_traces, deg_width + 1), np.nan)
    trace_centroids = np.full((num_traces, ny), np.nan)
    trace_widths = np.full((num_traces, ny), np.nan)

    pixel_arr = np.arange(ny)

    # Walk each trace up/down from initial peak position
    for i in range(num_traces):
        trace_centroids[i, :], trace_widths[i, :] = walk_trace_up_down(
            pixel_arr,
            data, dq, y0,
            peaks0[i],
            disp_range,
            window_size_half
        )
        good = np.where(np.isfinite(trace_centroids[i, :]))[0]
        if len(good) >= deg_pos + 1:
            coeffs_pos[i, :] = np.polyfit(good, trace_centroids[i, good], deg_pos)
        good = np.where(np.isfinite(trace_widths[i, :]))[0]
        if len(good) >= deg_width + 1:
            coeffs_width[i, :] = np.polyfit(good, trace_widths[i, good], deg_width)

    # NOTE: Keep for debugging
    #import matplotlib.pyplot as plt
    #breakpoint()
    #plt.imshow(data, origin='lower', aspect='auto')
    #for i in range(num_traces):
        #plt.plot(np.polyval(coeffs_pos[i, :], pixel_arr), pixel_arr, color='r', lw=0.5)
    #plt.show()

    return dict(
        coeffs_pos=coeffs_pos,
        coeffs_width=coeffs_width,
        trace_centroids=trace_centroids,
        trace_widths=trace_widths
    )


def walk_trace_up_down(
    pixels : np.ndarray,
    data : np.ndarray,
    dq : np.ndarray,
    disp0 : float,
    xdisp0 : float,
    disp_range : tuple,
    window_size_half : float,
) -> np.ndarray:
    trace_centroids = np.full(data.shape[0], np.nan)
    trace_widths = np.full(data.shape[0], np.nan)
    _walk_trace_up_down(
        pixels,
        data,
        dq,
        disp0, xdisp0,
        disp_range,
        window_size_half,
        trace_centroids, trace_widths
    )
    return trace_centroids, trace_widths


@njit(nogil=True, cache=True)
def _walk_trace_up_down(
    pixels, data, dq,
    disp0, xdisp0,
    disp_range, window_size_half,
    trace_centroids, trace_widths,
):
    # Central slice
    c, w = _compute_centroid_and_width(
        pixels, data[disp0, :], dq[disp0, :], xdisp0, window_size_half
    )
    trace_centroids[disp0], trace_widths[disp0] = c, w

    # Walk down — carry forward last good centroid as the guess
    xdisp_last = xdisp0 if np.isnan(c) else c
    for disp in range(disp0 - 1, disp_range[0] - 1, -1):
        c, w = _compute_centroid_and_width(
            pixels, data[disp, :], dq[disp, :], xdisp_last, window_size_half
        )
        trace_centroids[disp], trace_widths[disp] = c, w
        if not np.isnan(c):
            xdisp_last = c

    # Walk up
    xdisp_last = xdisp0 if np.isnan(trace_centroids[disp0]) else trace_centroids[disp0]
    for disp in range(disp0 + 1, disp_range[1] + 1):
        c, w = _compute_centroid_and_width(
            pixels, data[disp, :], dq[disp, :], xdisp_last, window_size_half
        )
        trace_centroids[disp], trace_widths[disp] = c, w
        if not np.isnan(c):
            xdisp_last = c


@njit(nogil=True, cache=True)
def _compute_centroid_and_width(
    pixels : np.ndarray,
    trace_slice : np.ndarray,
    trace_dq : np.ndarray,
    center_loc : float,
    window_size_half : float,
):
    """
    Compute the centroid and width of a trace slice.

    Parameters
    ----------
    pixels : np.ndarray
        Pixel coordinates corresponding to the trace_slice.
    trace_slice : np.ndarray
        1D array of pixel values along the slice of the trace.
    trace_dq : np.ndarray
        1D array of data quality flags corresponding to the trace_slice.
    center_loc : float
        Initial guess for the centroid position (in pixel coordinates).
    window_size_half : float
        Half the size of the window (in pixels) to use when computing the centroid and width.
    """

    xi = int(np.round(center_loc - window_size_half))
    xf = int(np.round(center_loc + window_size_half + 1))
    if xi < 0 or xf > trace_slice.size - 1:
        return np.nan, np.nan

    slice_x = pixels[xi:xf+1]
    slice_y = trace_slice[xi:xf+1]
    slice_dq = trace_dq[xi:xf+1]

    n_good = np.sum(slice_dq == 0)
    if n_good < slice_dq.size:
        return np.nan, np.nan

    total_flux = np.sum(slice_y)

    if total_flux > 0:
        centroid = np.sum(slice_x * slice_y) / total_flux
        variance = np.sum(((slice_x - centroid) ** 2) * slice_y) / total_flux
        if variance < 0:
            return np.nan, np.nan
        fwhm = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(variance)
        return centroid, fwhm
    else:
        return np.nan, np.nan
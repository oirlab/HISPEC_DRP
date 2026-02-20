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
    yrange : tuple or dict
        Optional y range (in pixels) to restrict trace finding. Can be a single tuple for all fibers or a dict mapping fiber numbers to tuples.
    xrange : tuple or dict
        Optional x range (in pixels) to restrict trace finding. Can be a single tuple for all fibers or a dict mapping fiber numbers to tuples.

    Returns
    -------
    output : TracePosModel
        Output model containing fitted trace positions and widths for each fiber.
    """

    ARGUMENTS = {
        'deg_pos' : None,
        'deg_width' : None,
        'window_size' : 8,
        'xrange' : None,
        'yrange' : None,
        'output_model' : 'TracePosModel',
        'save_preview' : True,
        'show_preview' : False,
    }

    def _perform(self):

        # Associate inputs by fiber number
        if isinstance(self.input, (datamodels.DataModel, str)):
            self._input_by_fiber = datamodels.group_by_illuminated_fiber([self.input])
        if  isinstance(self.input, Sequence):
            self._input_by_fiber = datamodels.group_by_illuminated_fiber(self.input)
        elif isinstance(self.input, dict):
            self._input_by_fiber = self.input
        else:
            raise ValueError(f"Unsupported input type: {type(self.input)}")

        # Prepare output model
        first_model = datamodels.open(list(self._input_by_fiber.values())[0])
        output_model = datamodels.TracePosModel.from_datamodels(list(self._input_by_fiber.values()))

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

            # Get y range for this fiber
            yrange = self.yrange[fiber] if isinstance(self.yrange, dict) else self.yrange

            # Find trace positions
            logger.info(f"Performing trace finding on {input_model} fiber {fiber}")
            if input_model.meta.orders_are_vertical:
                r = find_trace_positions(
                    input_model.rate.T, input_model.dq.T,
                    #xrange=self.xrange, yrange=yrange,
                    xrange=self.yrange, yrange=self.xrange,
                    deg_pos=self.deg_pos, deg_width=self.deg_width, window_size=self.window_size,
                )
            else:
                r = find_trace_positions(
                    input_model.rate, input_model.dq,
                    xrange=self.xrange, yrange=yrange,
                    deg_pos=self.deg_pos, deg_width=self.deg_width, window_size=self.window_size,
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
        fig = make_subplots(
            rows=n_fibers, cols=1,
            subplot_titles=[f"Fiber {f}" for f in active_fibers],
            vertical_spacing=0.05
        )
            
        for i, fiber in enumerate(active_fibers):

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
            fig.add_trace(
                go.Heatmap(
                    z=im_data_small,
                    x=x_small,
                    y=y_small,
                    zmin=zmin, zmax=zmax,
                    colorscale='Viridis',
                    showscale=False
                ),
                row=i+1, col=1
            )
            
            # Force square pixels
            # Note: valid constraints for scaleanchor are 'x', 'x2', 'x3' etc. 
            # We need to anchor to the x-axis of the CURRENT subplot.
            xaxis_name = 'x' if i == 0 else f'x{i+1}'
            
            fig.update_yaxes(
                scaleanchor=xaxis_name,
                scaleratio=1,
                row=i+1, col=1
            )
            fig.update_xaxes(
                constrain='domain',
                row=i+1, col=1
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
                fig.add_trace(
                    go.Scatter(
                        x=x_trace,
                        y=y_trace,
                        mode='lines',
                        line=dict(color='red', width=1),
                        showlegend=False,
                        name=f'Order {ord_idx}'  
                    ),
                    row=i+1, col=1
                )

                #breakpoint()

        # Update layout
        height = 400 * n_fibers
        fig.update_layout(
            height=height,
            width=1000,
            title_text=f"Trace Positions: {tracepos_model.meta.filename}",
            showlegend=False
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        
        # File paths
        base_path = tracepos_model.get_output_path(output_dir=self.output_dir, suffix=self.suffix)
        output_path_html = base_path.replace(".fits", "_preview.html")
        output_path_png = base_path.replace(".fits", "_preview.png")

        if self.save_preview:
            fig.write_html(output_path_html, include_plotlyjs='cdn')
            fig.write_image(output_path_png, scale=2)

        if self.show_preview:
            logger.info("Showing trace position preview...")
            fig.show()

        self._output['preview'] = {
            "html": output_path_html,
            "png": output_path_png,
            "fig": fig
        }



def find_trace_positions(
    data : np.ndarray,
    dq : np.ndarray,
    deg_pos : int,
    deg_width : int,
    xrange : tuple = None,
    yrange : tuple = None,
    fwhm : float = 2.5,
    window_size : int | None = None,
    distance_factor : float = 0.5,
    prominence_factor : float = 0.05
) -> dict:

    # Initial parameters
    ny, nx = data.shape
    if xrange is None:
        xrange = (16, nx - 16)
    if yrange is None:
        yrange = (8, ny - 8)
    if window_size is None:
        window_size = int(np.round(fwhm * 2.5))
    window_size_half = int(np.round(window_size / 2))
    min_distance = max(1, int(fwhm * distance_factor))

    # Find peaks for center column
    x0 = nx // 2
    ki, kf = max(0, x0 - 3), min(nx, x0 + 3)
    y = np.nanmedian(median_filter_2d(data[:, ki:kf+1], dq[:, ki:kf+1], window_size=(3, 3)), axis=1)
    y = y[yrange[0]:yrange[1] + 1]
    min_prominence = np.nanmax(y) * prominence_factor
    peaks0, _ = scipy.signal.find_peaks(y, prominence=min_prominence, distance=min_distance)
    peaks0 += yrange[0]
    num_traces = len(peaks0)

    # NOTE: Keep for debugging
    #import matplotlib
    #matplotlib.use("QTAGG")
    #if num_traces != 39:
    #    breakpoint()
        #plt.imshow(data, vmin=0, vmax=10); plt.ylim(0, ny); plt.show()
    #xarr = np.arange(nx); plt.plot(xarr[yrange[0]:yrange[1] + 1], y); [plt.axvline(p, color='r', ls='--') for p in peaks0]; plt.show()

    coeffs_pos = np.full((num_traces, deg_pos + 1), np.nan)
    coeffs_width = np.full((num_traces, deg_width + 1), np.nan)
    y_centroids = np.full((num_traces, nx), np.nan)
    y_widths = np.full((num_traces, nx), np.nan)

    # Walk each trace left/right from initial peak position
    for i in range(num_traces):
        y_centroids[i, :], y_widths[i, :] = walk_trace_left_right(data, x0, peaks0[i], xrange, window_size_half)
        good = np.where(np.isfinite(y_centroids[i, :]))[0]
        if len(good) >= deg_pos + 1:
            coeffs_pos[i, :] = np.polyfit(good, y_centroids[i, good], deg_pos)
        good = np.where(np.isfinite(y_widths[i, :]))[0]
        if len(good) >= deg_width + 1:
            coeffs_width[i, :] = np.polyfit(good, y_widths[i, good], deg_width)

    # NOTE: Keep for debugging
    # plt.imshow(data, origin='lower', aspect='auto')
    # xarr = np.arange(nx)
    # for i in range(num_traces):
    #     plt.plot(xarr, np.polyval(coeffs_pos[i, :], xarr), color='r', lw=0.5)
    # plt.title("Traced orders")
    # plt.show()
    # breakpoint()

    return dict(
        coeffs_pos=coeffs_pos,
        coeffs_width=coeffs_width,
        y_centroids=y_centroids,
        y_widths=y_widths
    )


def walk_trace_left_right(
    data : np.ndarray,
    x0 : float, yc0 : float,
    xrange : tuple,
    window_size_half : int,
) -> np.ndarray:
    ycs_out = np.full(data.shape[1], np.nan)
    widths_out = np.full(data.shape[1], np.nan)
    _walk_trace_left_right(
        data, x0, yc0, xrange, window_size_half, ycs_out, widths_out
    )
    return ycs_out, widths_out

@njit(nogil=True, parallel=False, cache=True)
def _walk_trace_left_right(
    data : np.ndarray,
    x0 : float, yc0 : float,
    xrange : tuple,
    window_size_half : int,
    ycs_out : np.ndarray,
    widths_out : np.ndarray,
):
    ny, nx = data.shape
    yc = yc0
    ycs_out[x0] = yc
    xarr = np.arange(nx)
    ycs_out[x0], widths_out[x0] = _compute_centroid_and_width(xarr, data[:, x0], yc, window_size_half)

    # Walk left
    for x in range(x0 - 1, xrange[0] - 1, -1):
        yc = ycs_out[x + 1]
        ycs_out[x], widths_out[x] = _compute_centroid_and_width(xarr, data[:, x], yc, window_size_half)

    # Walk right
    yc = yc0
    for x in range(x0 + 1, xrange[1] + 1):
        yc = ycs_out[x - 1]
        ycs_out[x], widths_out[x] = _compute_centroid_and_width(xarr, data[:, x], yc, window_size_half)

@njit(nogil=True, parallel=False, cache=True)
def _compute_centroid_and_width(
    xarr : np.ndarray,
    data : np.ndarray,
    yc : float,
    window_size_half : int,
):
    yi = int(max(0, yc - window_size_half))
    yf = int(min(data.shape[0], yc + window_size_half + 1))

    slice_x = xarr[yi:yf+1]
    slice_y = data[yi:yf+1]
    total_flux = np.nansum(slice_y)
    SIGMA_TO_FWHM = 2 * np.sqrt(2 * np.log(2))

    if total_flux > 0:
        centroid = np.nansum(slice_x * slice_y) / total_flux
        variance = np.nansum(((slice_x - centroid) ** 2) * slice_y) / total_flux
        fwhm = SIGMA_TO_FWHM * np.sqrt(variance)
        return centroid, fwhm
    else:
        return np.nan, np.nan
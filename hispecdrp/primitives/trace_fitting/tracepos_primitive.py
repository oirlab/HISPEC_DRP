from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np
from numba import njit
import scipy.signal
from ...utils.math import median_filter_2d
import matplotlib.pyplot as plt

__all__ = ['TracePositionPrimitive']


class TracePositionPrimitive(BasePrimitive):

    ARGUMENTS = {
        'suffix' : 'tracepos',
        'deg_pos' : None,
        'deg_width' : None,
        'window_size' : 8,
        'xrange' : None,
        'yrange' : None,
        'output_model' : 'TracePosModel',
    }

    def _perform(self):
        input_type = type(self.input)
        if input_type in (datamodels.DataModel, str):
            input = self.associate_fibers([self.input])
        elif input_type in (list, tuple):
            input = self.associate_fibers(self.input)
        elif input_type is dict:
            input = self.input
        else:
            raise ValueError(f"Unsupported input type: {input_type}")

        first_model = datamodels.open(input[list(input.keys())[0]])
        output_model = self.get_output_model(first_model)
        trace_results = {}
        nx = first_model.rate.shape[1]
        xarr = np.arange(nx)
        order_vec = output_model.make_order_vec()
        n_orders = len(order_vec)
        for fiber in input.keys():
            input_model = datamodels.open(input[fiber])
            yrange = self.yrange[fiber] if isinstance(self.yrange, dict) else self.yrange
            self.log.info(f"Performing trace finding on {input_model} fiber {fiber}")
            r = find_trace_positions(
                input_model.rate, input_model.dq,
                xrange=self.xrange, yrange=yrange,
                deg_pos=self.deg_pos, deg_width=self.deg_width, window_size=self.window_size,
            )
            num_orders_found = r['coeffs_pos'].shape[0]
            #assert num_orders_found == n_orders, f"Found {num_orders_found} traces, expected {n_orders} for fiber {fiber}"
            self.log.info(f"Found {num_orders_found} traces")
            trace_results[fiber] = r
            pos = np.full((n_orders, nx), np.nan)
            fwhm = np.full((n_orders, nx), np.nan)
            for i in range(num_orders_found):
                pos[i, :] = np.polyval(r['coeffs_pos'][i, :], xarr)
                fwhm[i, :] = np.polyval(r['coeffs_width'][i, :], xarr)
            setattr(output_model, f'COEFF_POS_FIB{fiber}', r['coeffs_pos'])
            setattr(output_model, f'COEFF_FWHM_FIB{fiber}', r['coeffs_width'])
            setattr(output_model, f'POS_FIB{fiber}', pos)
            setattr(output_model, f'FWHM_FIB{fiber}', fwhm)
            self.add_data_ancestors(output_model, input[fiber])

        # Return
        return dict(output=output_model, trace_results=trace_results)


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
    #breakpoint()
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
    #breakpoint()

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


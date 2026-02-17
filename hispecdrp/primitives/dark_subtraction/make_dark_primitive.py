from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np
from ...datamodels.dqflags import DQ_FLAGS
from typing import Sequence
from copy import copy
from ...utils.parallelization_utils import numba_thread_scope

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from numba import njit, prange

__all__ = ['MakeDarkPrimitive']

HOT_FLAG =  DQ_FLAGS['HOT']
DO_NOT_USE_FLAG = DQ_FLAGS["DO_NOT_USE"]

class MakeDarkPrimitive(BasePrimitive):
    """
    Create a dark calibration model from a combined rate map.
    This primitive expects the coadd primitive to have already been run on a list of darks exposures.

    Parameters
    ----------
    input : L0Model
        Input dark rate map to converted to dark calibration. Do not input a list.
    max_cores : int, optional
        Maximum number of CPU cores to use for parallel processing. Default is 1.
    hot_pixel_threshold : float, optional
        Threshold dark rate above which pixels are flagged as hot in the DQ array. Default is 5 e-/s.

    Returns
    -------
    output : DarkModel
        The derived dark calibration model containing the dark rate map and updated DQ array.

    """

    ARGUMENTS = {
        'output_model' : 'DarkModel',
        'max_cores' : 1,
        'hot_pixel_threshold' : 5,  # e-/s,  Threshold above which pixels are bad
    }

    def _perform(self):
        if  isinstance(self.input, Sequence):
            raise ValueError("Rate maps must be combined before computing the detector dark. Input should not be a list.")

        input_model = datamodels.open(self.input)
        # Define output model from inputs
        output_dark = self.get_output_model(input_model)

        # Median dark rate for instrument monitoring
        mask = input_model.dq == 0
        median_dark_level = np.nanmedian(input_model.rate[mask])
        output_dark.meta.median_dark_current = median_dark_level

        output_dark.rate = input_model.rate
        output_dark.rate_err = input_model.rate_err
        output_dark.rate_var_rnoise = input_model.rate_var_rnoise
        output_dark.rate_var_poisson = input_model.rate_var_poisson
        output_dark.dq = copy(input_model.dq)

        with numba_thread_scope(self.max_cores):
            _detect_hot_pixels(output_dark.rate, output_dark.dq,self.hot_pixel_threshold)

        return dict(output=output_dark)

    def _create_preview(self):
        output_dark = self._output['output']
        dq = output_dark.dq
        dark_rate = output_dark.rate.copy()
        dark_rate[dq != 0] = np.nan

        median = np.nanmedian(dark_rate)
        sigma = np.nanstd(dark_rate)

        data = dark_rate[~np.isnan(dark_rate)].ravel()
        heatmap_range = [max(median - 5 * sigma, 0), median + 5 * sigma]
        hist_range = [max(median - 5 * sigma, 0), median + 5 * sigma]

        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.1
        )

        # --- Heatmap ---
        fig.add_trace(
            go.Heatmap(
                z=dark_rate,
                x=np.arange(dark_rate.shape[1]),
                y=np.arange(dark_rate.shape[0]),
                colorscale="Inferno",
                zmin=heatmap_range[0],
                zmax=heatmap_range[1],
                colorbar=dict(
                    title=dict(text="DN/s", side="right"),
                    thickness=20
                ),
            ),
            row=1,
            col=1
        )

        # --- Histogram ---
        fig.add_trace(
            go.Histogram(
                x=data,
                xbins=dict(
                    start=hist_range[0],
                    end=hist_range[1],
                    size=(hist_range[1] - hist_range[0]) / 100
                ),
                marker=dict(color="#555555"),
                opacity=0.8,
                showlegend=False
            ),
            row=2,
            col=1
        )

        # --- Layout & Axes ---
        fig.update_layout(
            height=1000,
            width=800,
            title=dict(
                text=f"Dark Rate Map<br><sup>Median: {median:.2f} | σ: {sigma:.2f}</sup>",
                x=0.5
            ),
            template="plotly_white"
        )

        # Heatmap axes: keep aspect ratio and pixel-aligned
        fig.update_xaxes(title_text="X (pixels)", row=1, col=1, range=[-0.5, dark_rate.shape[1]-0.5])
        fig.update_yaxes(title_text="Y (pixels)", row=1, col=1, range=[-0.5, dark_rate.shape[0]-0.5], scaleanchor="x", scaleratio=1)

        # Histogram axes
        fig.update_xaxes(title_text="Dark Rate (DN/s)", row=2, col=1, range=hist_range)
        fig.update_yaxes(title_text="Count", row=2, col=1)

        # Vertical line for median
        fig.add_vline(x=median, line=dict(color="Red", dash="dash"), row=2, col=1)

        # File paths
        base_path = output_dark.get_output_path(output_dir=self.output_dir, suffix=self.suffix)
        output_path_html = base_path.replace(".fits", "_preview.html")
        output_path_png = base_path.replace(".fits", "_preview.png")

        if self.save_preview:
            fig.write_html(output_path_html, include_plotlyjs='cdn')
            fig.write_image(output_path_png, scale=2)

        if self.show_preview:
            fig.show()

        self._output['preview'] = {
            "html": output_path_html,
            "png": output_path_png,
            "fig": fig
        }


@njit(nogil=True, parallel=True, cache=True)
def _detect_hot_pixels(dark_rate: np.ndarray, dq: np.ndarray, hot_pixel_threshold: float = 0.9):
    ny, nx = dark_rate.shape
    for j in prange(ny):
        for i in range(nx):
            if dark_rate[j, i] > hot_pixel_threshold:
                dq[j, i] |= HOT_FLAG


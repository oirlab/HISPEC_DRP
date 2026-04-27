from ..base_primitive import BasePrimitive
from ...datamodels.dqflags import DQ_FLAGS
import numpy as np
from copy import copy
from typing import Sequence
from ... import datamodels
from ..jump_detection.jump_detection_difference import median_axis0_3d
import logging
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from numba import njit, prange
from ...utils.parallelization_utils import numba_thread_scope

logger = logging.getLogger(__name__)

__all__ = ['MakeBiasPrimitive']

UNRELIABLE_BIAS_FLAG =  DQ_FLAGS['UNRELIABLE_BIAS']
DO_NOT_USE_FLAG = DQ_FLAGS["DO_NOT_USE"]

class MakeBiasPrimitive(BasePrimitive):
    """
    Derive the detector bias and kTC noise from a set of dark exposures.
    1) Median combine the first read to create the bias frame.
    2) Compute the kTC noise as the standard deviation across the first reads, using sigma-clipping to reject outliers.
    3) Flag pixels with high bias level or high kTC noise in the DQ array.

    Parameters
    ----------
    input : list of str or list of RampModel
        List of input file paths or ramp models to process.
    max_cores : int, optional
        Maximum number of CPU cores to use for parallel processing. Default is 1.
    kTC_threshold : float, optional
        Threshold kTC above which pixels are flagged as bad in the DQ array. Default is 100 DN.
    bias_threshold : float, optional
        Threshold bias above which pixels are flagged as bad in the DQ array. Default is 1000 DN.

    Returns
    -------
    output : BiasModel
        The derived bias calibration model containing the bias frame, kTC noise map, and updated DQ array.

    """

    ARGUMENTS = {
        'output_model' : 'BiasModel',
        'max_cores' : 1,
        'kTC_threshold' : 100,  # DN,  Threshold kTC above which pixels are bad
        'bias_threshold' : 1000,  # DN,  Threshold bias above which pixels are bad
    }

    def _perform(self):
        if isinstance(self.input, (datamodels.DataModel, str)):
            _input_list_tmp = [self.input]
        if  isinstance(self.input, Sequence):
            _input_list_tmp = self.input
        nfiles = len(_input_list_tmp)
        logger.info(f"Deriving the detector bias and kTC noise from a list of {nfiles} inputs.")

        input_models = [datamodels.open(f) for f in _input_list_tmp]

        # Define output model from inputs
        output_bias = self.get_output_model(input_models)

        bias_arr = []
        for _input in input_models:
            bias_arr.append(_input.data[0,:,:])
        bias_arr = np.array(bias_arr)

        with numba_thread_scope(self.max_cores):
            output_bias.bias = median_axis0_3d(bias_arr)
            output_bias.ktc = std_axis0_3d_sigclip(bias_arr,nsig=5,min_keep=10)
            output_bias.dq = copy(input_models[0].dq)
            _find_bad_pixels_bias(output_bias.bias, output_bias.ktc , output_bias.dq,
                                    self.kTC_threshold, self.bias_threshold)

        median_bias_level = np.nanmedian(output_bias.bias)
        output_bias.meta.median_bias_level = median_bias_level
        median_kTC_level = np.nanmedian(output_bias.ktc)
        output_bias.meta.median_kTC_level = median_kTC_level

        return dict(output=output_bias)

    def _create_preview(self):
        """Create preview visualization of the derived bias calibration."""
        
        try:
            # Get the bias model (output of _perform)
            bias_model = self._output.get('output')
            if not bias_model:
                logger.warning("No bias model available for preview.")
                return
            
        except Exception as e:
            logger.warning(f"Could not load bias model for preview: {e}")
            return
        
        # Extract bias and kTC data
        bias_data = bias_model.bias
        ktc_data = bias_model.ktc if hasattr(bias_model, 'ktc') and bias_model.ktc is not None else np.zeros_like(bias_data)
        
        ny, nx = bias_data.shape
        
        # Create subplots for bias
        fig_bias = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Bias Map (DN)", "Bias Histogram", "Bias Statistics"),
            specs=[[{}, {"type": "histogram"}, {"type": "histogram"}]]
        )
        
        # Bias heatmap
        fig_bias.add_trace(
            go.Heatmap(
                z=bias_data,
                colorscale='Inferno',
                showscale=True,
                name='Bias',
                colorbar=dict(x=0.16, y=-0.15, len=0.2, orientation='h', xanchor='center', yanchor='top')
            ),
            row=1, col=1
        )
        
        # Bias histogram
        valid_bias = bias_data[np.isfinite(bias_data)].flatten()
        fig_bias.add_trace(
            go.Histogram(
                x=valid_bias,
                nbinsx=50,
                name='Bias',
                marker=dict(color='#1f77b4'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # DQ flag histogram
        valid_dq = bias_model.dq[np.isfinite(bias_model.dq)].flatten()
        fig_bias.add_trace(
            go.Histogram(
                x=valid_dq,
                nbinsx=50,
                name='DQ Flags',
                marker=dict(color='#d62728'),
                showlegend=False
            ),
            row=1, col=3
        )
        
        fig_bias.update_xaxes(title_text="Bias (DN)", row=1, col=2, showgrid=False)
        fig_bias.update_yaxes(title_text="Frequency", row=1, col=2, showgrid=False)
        fig_bias.update_xaxes(title_text="DQ Flags", row=1, col=3, showgrid=False)
        fig_bias.update_yaxes(title_text="Frequency", row=1, col=3, showgrid=False)
        fig_bias.update_xaxes(showgrid=False, row=1, col=1)
        fig_bias.update_yaxes(showgrid=False, row=1, col=1)
        
        median_bias = np.nanmedian(bias_data)
        fig_bias.update_layout(
            height=500,
            width=1200,
            title_text=f"Bias Calibration (Median: {median_bias:.2f} DN)",
            showlegend=False,
            margin=dict(b=100)
        )
        
        # Create subplots for kTC
        fig_ktc = make_subplots(
            rows=1, cols=3,
            subplot_titles=("kTC Noise Map (DN)", "kTC Histogram", "Bias vs kTC"),
            specs=[[{}, {"type": "histogram"}, {}]]
        )
        
        # kTC heatmap
        fig_ktc.add_trace(
            go.Heatmap(
                z=ktc_data,
                colorscale='Plasma',
                showscale=True,
                name='kTC',
                colorbar=dict(x=0.16, y=-0.15, len=0.2, orientation='h', xanchor='center', yanchor='top')
            ),
            row=1, col=1
        )
        
        # kTC histogram
        valid_ktc = ktc_data[np.isfinite(ktc_data)].flatten()
        fig_ktc.add_trace(
            go.Histogram(
                x=valid_ktc,
                nbinsx=50,
                name='kTC',
                marker=dict(color='#ff7f0e'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 2D scatter plot of bias vs kTC
        valid_mask = np.isfinite(bias_data) & np.isfinite(ktc_data)
        fig_ktc.add_trace(
            go.Scatter(
                x=bias_data[valid_mask].flatten(),
                y=ktc_data[valid_mask].flatten(),
                mode='markers',
                marker=dict(
                    size=3,
                    color=ktc_data[valid_mask].flatten(),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(x=0.84, y=-0.15, len=0.2, orientation='h', xanchor='center', yanchor='top')
                ),
                name='Bias vs kTC',
                showlegend=False
            ),
            row=1, col=3
        )
        
        fig_ktc.update_xaxes(title_text="kTC (DN)", row=1, col=2, showgrid=False)
        fig_ktc.update_yaxes(title_text="Frequency", row=1, col=2, showgrid=False)
        fig_ktc.update_xaxes(title_text="Bias (DN)", row=1, col=3, showgrid=False)
        fig_ktc.update_yaxes(title_text="kTC (DN)", row=1, col=3, showgrid=False)
        fig_ktc.update_xaxes(showgrid=False, row=1, col=1)
        fig_ktc.update_yaxes(showgrid=False, row=1, col=1)
        
        median_ktc = np.nanmedian(ktc_data)
        fig_ktc.update_layout(
            height=500,
            width=1200,
            title_text=f"kTC Noise (Median: {median_ktc:.2f} DN)",
            showlegend=False,
            margin=dict(b=100)
        )
        
        # Save if configured
        if self.save_preview:
            if not self.output_dir:
                self.output_dir = '.'
            
            os.makedirs(self.output_dir, exist_ok=True)
            
            bias_html = os.path.join(self.output_dir, f'bias_preview_{self.suffix}.html')
            ktc_html = os.path.join(self.output_dir, f'ktc_preview_{self.suffix}.html')
            
            fig_bias.write_html(bias_html, include_plotlyjs='cdn')
            fig_ktc.write_html(ktc_html, include_plotlyjs='cdn')
            
            logger.info(f"Bias preview saved to {bias_html}")
            logger.info(f"kTC preview saved to {ktc_html}")
            
            # Try to save PNG versions
            try:
                bias_png = bias_html.replace('.html', '.png')
                ktc_png = ktc_html.replace('.html', '.png')
                fig_bias.write_image(bias_png, scale=2)
                fig_ktc.write_image(ktc_png, scale=2)
                logger.info(f"PNG versions saved")
            except Exception as e:
                logger.debug(f"Could not save PNG preview: {e}")
        
        if self.show_preview:
            logger.info("Showing bias calibration preview...")
            fig_bias.show()
            fig_ktc.show()
        
        # Store in output
        self._output['preview'] = {
            'bias_fig': fig_bias,
            'ktc_fig': fig_ktc,
        }


@njit(nogil=True, parallel=True, cache=True)
def _find_bad_pixels_bias(bias: np.ndarray, kTC: np.ndarray, dq: np.ndarray,
                       kTC_threshold: float = 100, bias_threshold: float = 1000):
    ny, nx = bias.shape
    for j in prange(ny):
        for i in range(nx):
            if kTC[j, i] > kTC_threshold:
                dq[j, i] |= UNRELIABLE_BIAS_FLAG
            if bias[j, i] > bias_threshold:
                dq[j, i] |= UNRELIABLE_BIAS_FLAG

@njit(nogil=True, parallel=True, cache=True)
def std_axis0_3d_sigclip(bias_arr, nsig=5.0, min_keep=10):
    """
    Sigma-clipped std over axis=0 for bias_arr with shape (n_files, ny, nx)

    Parameters
    ----------
    bias_arr : float array (n_files, ny, nx)
    nsig     : sigma threshold
    min_keep : minimum number of samples required after clipping

    Returns
    -------
    std : (ny, nx) array
    """
    n_files, ny, nx = bias_arr.shape
    std = np.zeros((ny, nx), dtype=bias_arr.dtype)

    for y in prange(ny):
        for x in range(nx):

            # ---------
            # Pass 1: initial mean/std (Welford)
            mean = 0.0
            M2 = 0.0
            count = 0

            for f in range(n_files):
                v = bias_arr[f, y, x]
                count += 1
                delta = v - mean
                mean += delta / count
                M2 += delta * (v - mean)

            if count < 2:
                std[y, x] = 0.0
                continue

            sigma = np.sqrt(M2 / count)

            # ---------
            # Pass 2: sigma clipping + recompute stats
            mean2 = 0.0
            M2_2 = 0.0
            count2 = 0
            thresh = nsig * sigma

            for f in range(n_files):
                v = bias_arr[f, y, x]
                if abs(v - mean) <= thresh:
                    count2 += 1
                    delta = v - mean2
                    mean2 += delta / count2
                    M2_2 += delta * (v - mean2)

            if count2 < min_keep:
                # fallback: unclipped std
                std[y, x] = sigma
            else:
                std[y, x] = np.sqrt(M2_2 / count2)

    return std
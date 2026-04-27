from ..base_primitive import BasePrimitive
from ...calibrations import BiasSelector
from ... import datamodels
from ...utils.parallelization_utils import numba_thread_scope
from numba import njit, prange
import numpy as np
import logging
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os 

logger = logging.getLogger(__name__)

__all__ = ['BiasSubtractionPrimitive']


class BiasSubtractionPrimitive(BasePrimitive):
    """
    Subtract the bias level from the input data.

    This is the base signal from which accumulated charges are measured.
    The bias voltage is to prevent negative values to be read given the
    unsigned nature of the values produced by the ADC. Bias subtraction
    is useful to set the y-intercept term of the UTR data close to 0.

    This primitive applies the appropriate bias calibration to the
    input data product. The bias frame is subtracted from the science data,
    the associated uncertainties are propagated by adding errors in
    quadrature, and data quality (DQ) flags are combined using a bitwise OR.

    Parameters
    ----------
    input : str or RampModel
        Input file path or data model to process. Currently, this is expected
        to be a raw (ramp) data product.
    suffix : str or None, optional
        Suffix to append to the output filename. Default is ``'biassub'``.
    max_cores : int, optional
        Maximum number of CPU cores to use for parallel processing. Default is 1 (no parallelization).

    Calibrations
    ------------
    bias : BiasModel
        - selector: :class:`~hispecdrp.calibrations.selectors.BiasSelector`

    Returns
    -------
    output : DataModel
        Bias-subtracted data model with updated error and data quality
        arrays.
    """

    ARGUMENTS = {
        'suffix' : 'biassub',
        'max_cores' : 1
    }

    CALIBRATIONS = {
        'bias': {
            'selector': BiasSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):

        # Load the input model
        input_model = datamodels.open(self.input)

        # Get the bias model
        bias, _ = self.get_calibration(input_model, 'bias')
        bias_model = datamodels.open(bias)

        # Output
        result = self.get_output_model(input_model)

        if not hasattr(result, 'DATA'):
            raise Exception("DATA extension not found in input model for bias subtraction.")
        if not hasattr(result, 'DQ'):
            raise Exception("DQ extension not found in input model for bias subtraction.")

        # Log
        logger.info(f"Applying bias correction with {bias} to {input_model}")

        if result.data.dtype != np.float32:
            logger.info(f"Converting UTR data from {result.data.dtype} to float32")
            result.data = result.data.astype(np.float32)

        # Perform bias subtraction
        _bias_subtraction(
            result.data, result.dq,
            bias_model.bias, bias_model.dq,
            max_cores=self.max_cores
        )

        # Output
        return dict(output=result, bias=bias)

    def _create_preview(self):
        """Create preview visualization of bias calibration applied."""
    
        try:
            # Try to get the bias calibration model
            bias_filepath = self._output.get('bias')
            if not bias_filepath:
                logger.warning("No bias calibration file available for preview.")
                return
            
            bias_model = datamodels.open(bias_filepath)
            
        except Exception as e:
            logger.warning(f"Could not load bias model for preview: {e}")
            return
        
        # Extract bias and kTC data
        bias_data = bias_model.bias
        ktc_data = bias_model.ktc if hasattr(bias_model, 'ktc') and bias_model.ktc is not None else np.zeros_like(bias_data)
        
        ny, nx = bias_data.shape
        
        # Create subplots for bias
        fig_bias = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Bias Calibration Map (DN)", "Bias Histogram"),
            specs=[[{}, {"type": "histogram"}]]
        )
        
        # Bias heatmap
        fig_bias.add_trace(
            go.Heatmap(
                z=bias_data,
                colorscale='Inferno',
                showscale=True,
                name='Bias',
                colorbar=dict(x=0.46, y=0.5, len=0.4, orientation='h', xanchor='center', yanchor='middle')
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
        
        fig_bias.update_xaxes(title_text="Bias (DN)", row=1, col=2, showgrid=False)
        fig_bias.update_yaxes(title_text="Frequency", row=1, col=2, showgrid=False)
        fig_bias.update_xaxes(showgrid=False, row=1, col=1)
        fig_bias.update_yaxes(showgrid=False, row=1, col=1)
        
        fig_bias.update_layout(
            height=500,
            width=1000,
            title_text=f"Bias Calibration: {os.path.basename(bias_filepath)}",
            showlegend=False,
            margin=dict(b=80)
        )
        
        # Create subplots for kTC
        fig_ktc = make_subplots(
            rows=1, cols=2,
            subplot_titles=("kTC Noise Map (DN)", "kTC Histogram"),
            specs=[[{}, {"type": "histogram"}]]
        )
        
        # kTC heatmap
        fig_ktc.add_trace(
            go.Heatmap(
                z=ktc_data,
                colorscale='Plasma',
                showscale=True,
                name='kTC',
                colorbar=dict(x=0.46, y=0.5, len=0.4, orientation='h', xanchor='center', yanchor='middle')
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
        
        fig_ktc.update_xaxes(title_text="kTC (DN)", row=1, col=2, showgrid=False)
        fig_ktc.update_yaxes(title_text="Frequency", row=1, col=2, showgrid=False)
        fig_ktc.update_xaxes(showgrid=False, row=1, col=1)
        fig_ktc.update_yaxes(showgrid=False, row=1, col=1)
        
        fig_ktc.update_layout(
            height=500,
            width=1000,
            title_text=f"kTC Calibration: {os.path.basename(bias_filepath)}",
            showlegend=False,
            margin=dict(b=80)
        )
        
        # Save if configured
        if self.save_preview:
            if not self.output_dir:
                self.output_dir = os.path.dirname(self.input) if isinstance(self.input, str) else '.'
            
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

def _bias_subtraction(
    input_data : np.ndarray,
    input_dq : np.ndarray,
    bias_data : np.ndarray,
    bias_dq : np.ndarray,
    max_cores : int = 1
):
    """
    Subtract the bias from the ramps.

    Parameters
    ----------
    input_data
        Input ramp data with shape (Nreads, Ny, Nx).
    input_dq
        Input DQ array with shape (Ny, Nx).
    bias_data
        Bias data with shape (Ny, Nx).
    bias_dq
        Bias DQ array with shape (Ny, Nx).
    max_cores
        Maximum number of CPU cores to use.
    """
    with numba_thread_scope(max_cores):
        _bias_subtraction_numba(input_data, input_dq, bias_data, bias_dq)
    return input_data, input_dq


@njit(nogil=True, parallel=True, cache=True)
def _bias_subtraction_numba(data : np.ndarray, dq : np.ndarray, bias : np.ndarray, bias_dq : np.ndarray):
    n_read, ny, nx = data.shape
    for j in prange(ny):
        for i in range(nx):
            data[:, j, i] = data[:, j, i] - bias[j, i]
            dq[j, i] |= bias_dq[j, i]

from ..base_primitive import BasePrimitive
from ... import datamodels
from ...calibrations import GainSelector
from ...utils.parallelization_utils import numba_thread_scope
import numpy as np
from numba import njit, prange
import logging
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

__all__ = ['GainCorrectionPrimitive', 'gain_correction']


class GainCorrectionPrimitive(BasePrimitive):
    """
    Convert rate maps from DN/s to electrons/s using the gain calibration.

    The conversion is performed by multiplying the input
    data and error arrays by the per-pixel gain values provided in the gain
    calibration file, which are expressed in units of e-/DN.

    Parameters
    ----------
    input : str or RampModel
        Input file path or data model to process. This is expected to be a
        level 0 (rate map) data product.
    suffix : str or None, optional
        Suffix to append to the output filename. Default is ``'gaincorr'``.

    Calibrations
    ------------
    gain : GainModel
        - selector: :class:`~hispecdrp.calibrations.selectors.GainSelector`

    Returns
    -------
    output : DataModel
        Data model with values and uncertainties converted to units of
        electrons.
    """

    ARGUMENTS = {
        'suffix' : 'gaincorr',
        'max_cores' : 1,
    }

    CALIBRATIONS = {
        'gain': {
            'selector': GainSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):

        # Load the input model
        input_model = datamodels.open(self.input)

        # Get the gain model
        gain, _ = self.get_calibration(input_model, 'gain')
        gain_model = datamodels.open(gain)

        # Output
        result = self.get_output_model(input_model)

        if not hasattr(input_model, 'RATE'):
            raise Exception("RATE extension not found in input model for detector gain correction.")
        if not hasattr(input_model, 'RATE_ERR'):
            raise Exception("RATE_ERR extension not found in input model for detector gain correction.")
        if not hasattr(input_model, 'DQ'):
            raise Exception("DQ extension not found in input model for detector gain correction.")
        if not hasattr(gain_model, 'GAIN'):
            raise Exception("GAIN extension not found in gain model for dark subtraction.")
        if not hasattr(gain_model, 'DQ'):
            raise Exception("DQ extension not found in gain model for dark subtraction.")

        # Log
        logger.info(f"Applying detector gain correction with {gain} to {input_model}")

        # Perform detector flat correction
        gain_correction(
            result.rate, result.rate_err, result.dq,
            gain_model.gain, np.sqrt(gain_model.cov[0,0]), gain_model.dq,
            max_cores=self.max_cores
        )

        # Output
        return dict(output=result, gain=gain)

    def _create_preview(self):
        """Create preview visualization of gain calibration applied."""
        
        try:
            # Try to get the gain calibration model
            gain_filepath = self._output.get('gain')
            if not gain_filepath:
                logger.warning("No gain calibration file available for preview.")
                return
            
            gain_model = datamodels.open(gain_filepath)
            
        except Exception as e:
            logger.warning(f"Could not load gain model for preview: {e}")
            return
        
        # Extract gain data and error
        gain_data = gain_model.gain
        gain_err = np.sqrt(gain_model.cov[0, 0]) if hasattr(gain_model, 'cov') and gain_model.cov is not None else None
        
        ny, nx = gain_data.shape
        
        # Create figure with 2x2 subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Master Gain (Full Frame)",
                "Center 50x50 Pixels (Zoomed)",
                "Histogram of Gain Values",
                "Histogram of Gain Errors"
            ),
            specs=[
                [{}, {}],
                [{"type": "histogram"}, {"type": "histogram"}]
            ]
        )
        
        # Plot 1: Full gain map
        fig.add_trace(
            go.Heatmap(
                z=gain_data,
                colorscale='Viridis',
                showscale=True,
                name='Gain',
                colorbar=dict(x=0.46, y=0.75, len=0.15, orientation='h', xanchor='center', yanchor='top')
            ),
            row=1, col=1
        )
        
        # Plot 2: Zoomed center (50x50 pixels)
        center_y, center_x = ny // 2, nx // 2
        half_size = min(25, min(center_y, center_x, ny - center_y, nx - center_x))
        gain_zoom = gain_data[center_y-half_size:center_y+half_size, center_x-half_size:center_x+half_size]
        
        fig.add_trace(
            go.Heatmap(
                z=gain_zoom,
                colorscale='Viridis',
                showscale=True,
                name='Gain (Zoomed)',
                colorbar=dict(x=1.02, y=0.75, len=0.15, orientation='h', xanchor='center', yanchor='top')
            ),
            row=1, col=2
        )
        
        # Plot 3: Histogram of gain values
        valid_gain = gain_data[np.isfinite(gain_data)].flatten()
        fig.add_trace(
            go.Histogram(
                x=valid_gain,
                nbinsx=100,
                name='Gain',
                marker=dict(color='#1f77b4'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Plot 4: Histogram of gain errors
        if gain_err is not None:
            valid_err = gain_err[np.isfinite(gain_err)].flatten()
            fig.add_trace(
                go.Histogram(
                    x=valid_err,
                    nbinsx=100,
                    name='Error',
                    marker=dict(color='#ff7f0e'),
                    showlegend=False
                ),
                row=2, col=2
            )
            fig.update_xaxes(title_text="Gain Error (e-/ADU)", row=2, col=2, showgrid=False)
        else:
            logger.warning("No gain error data available for preview histogram")
        
        # Update axes labels and properties
        fig.update_xaxes(title_text="Gain (e-/ADU)", row=2, col=1, showgrid=False)
        fig.update_yaxes(title_text="Frequency", row=2, col=1, showgrid=False)
        fig.update_yaxes(title_text="Frequency", row=2, col=2, showgrid=False)
        
        fig.update_xaxes(showgrid=False, row=1, col=1)
        fig.update_yaxes(showgrid=False, row=1, col=1)
        fig.update_xaxes(showgrid=False, row=1, col=2)
        fig.update_yaxes(showgrid=False, row=1, col=2)
        
        fig.update_layout(
            height=900,
            width=1200,
            title_text=f"Gain Calibration: {os.path.basename(gain_filepath)}",
            showlegend=False,
            margin=dict(b=80)
        )
        
        # Save if configured
        if self.save_preview:
            if not self.output_dir:
                self.output_dir = os.path.dirname(self.input) if isinstance(self.input, str) else '.'
            
            os.makedirs(self.output_dir, exist_ok=True)
            
            gain_html = os.path.join(self.output_dir, f'gain_preview_{self.suffix}.html')
            
            fig.write_html(gain_html, include_plotlyjs='cdn')
            
            logger.info(f"Gain preview saved to {gain_html}")
            
            # Try to save PNG version
            try:
                gain_png = gain_html.replace('.html', '.png')
                fig.write_image(gain_png, scale=2)
                logger.info(f"PNG preview saved")
            except Exception as e:
                logger.debug(f"Could not save PNG preview: {e}")
        
        if self.show_preview:
            logger.info("Showing gain calibration preview...")
            fig.show()
        
        # Store in output
        self._output['preview'] = {'gain_fig': fig}


def gain_correction(
    input_rate : np.ndarray, input_err : np.ndarray, input_dq : np.ndarray,
    gain : np.ndarray, gain_err : np.ndarray, gain_dq : np.ndarray,
    max_cores : int = 1
):
    """
    Apply detector gain correction to 2D rate image.

    Parameters
    ----------
    input_rate : np.ndarray
        2D array of input rates.
    input_err : np.ndarray
        2D array of input rate errors.
    input_dq : np.ndarray
        2D array of input data quality flags.
    gain : np.ndarray
        2D array of gain values.
    gain_err : np.ndarray
        2D array of gain errors.
    gain_dq : np.ndarray
        2D array of gain data quality flags.
    max_cores : int, optional
        Maximum number of CPU cores to use for parallel processing. Default is 1.

    Returns
    -------
    input_rate : np.ndarray
        Gain-corrected rate array.
    input_err : np.ndarray
        Updated rate error array.
    input_dq : np.ndarray
        Updated data quality flags array.
    """
    with numba_thread_scope(max_cores):
        _gain_correction_numba(
            input_rate, input_err, input_dq,
            gain, gain_err, gain_dq
        )
    return input_rate, input_err, input_dq

@njit(nogil=True, parallel=True, cache=True)
def _gain_correction_numba(
    input_rate : np.ndarray, input_err : np.ndarray, input_dq : np.ndarray,
    gain : np.ndarray, gain_err : np.ndarray, gain_dq : np.ndarray
):
    ny, nx = input_rate.shape
    for j in prange(ny):
        for i in range(nx):
            g = gain[j, i]
            r = input_rate[j, i]
            input_rate[j, i] = r * g
            gr = np.abs(input_rate[j, i])
            e_in = input_err[j, i]
            ge_in = gain_err[j, i]

            # Approximate error propagation for division of random variables
            # https://en.wikipedia.org/wiki/Propagation_of_uncertainty
            input_err[j, i] = gr * np.sqrt((e_in / r) ** 2 + (ge_in / g) ** 2)

            input_dq[j, i] |= gain_dq[j, i]
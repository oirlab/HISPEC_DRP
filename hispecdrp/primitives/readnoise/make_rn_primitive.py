from ..base_primitive import BasePrimitive
from ... import datamodels
from ...calibrations import GainSelector,ReadNoiseSelector
import numpy as np
from typing import Sequence
from ...datamodels.dqflags import DQ_FLAGS
from ..gain.make_gain_primitive import fit_gain_and_readnoise
import logging
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

__all__ = ['MakeReadnoisePrimitive']

SATURATED_FLAG_uint8 =  np.uint8(DQ_FLAGS["SATURATED"])
DO_NOT_USE_FLAG_uint8 = np.uint8(DQ_FLAGS["DO_NOT_USE"])
JUMP_DET_FLAG_uint8 = np.uint8(DQ_FLAGS["JUMP_DET"])

class MakeReadnoisePrimitive(BasePrimitive):
    """
    Estimate detector read noise from up-the-ramp data.

    This primitive fits the detector read noise parameter from raw
    **dark** up-the-ramp (UTR) ramps using the maximum-likelihood framework
    described in Brandt (2025; PASP, 137, 125006). In this case, the gain is fixed and only the readnoise is fitted.
    Three fitting strategies are supported: a grid-based maximum-likelihood search, a local quadratic
    approximation, or a hybrid approach that uses the grid search to locate
    the optimum and then refines it with a quadratic approximation.

    An initial read-noise value may be provided explicitly (e.g., as an
    array defining a search grid). If not provided, the primitive is
    expected to obtain an appropriate default from the calibration database.
    An optional argument allows treating all ramps for a given pixel as having identical count
    rates.

    The fitting may be performed on a per-pixel basis or using a single
    model for the full detector. Although the full detector mode is only for
    testing.


    Parameters
    ----------
    input : list of str or list of RampModel
        List of input file paths or ramp models to process. Use dark exposures.
    max_cores : int, optional
        Maximum number of CPU cores to use for parallel processing. Default
        is 1.
    init_RN : None or array-like, optional
        Initial read noise value(s) used to define the maximum-likelihood
        grid search. If None, obtain an initial value from the calibration
        database. If provided, this may be a scalar or an array defining the
        read-noise sampling (e.g., ``np.linspace(1, 20, 101)``).
    mode : {'per_pixel', 'channel', 'full_detector'}, optional
        Mode used to fit the parameters. ``'per_pixel'`` fits an independent
        read noise for each pixel. ``'full_detector'`` fits a single read
        noise value for the detector. ``'channel'`` is reserved for future
        implementation.
    method : {'quad', 'grid', 'both'}, optional
        Fitting strategy. ``'grid'`` performs a maximum-likelihood grid
        search. ``'quad'`` uses a quadratic approximation around the initial
        parameter. ``'both'`` runs a grid search first and then refines the
        solution with a quadratic approximation.
    init_delta_RNs : array-like or None, optional
        Read-noise grid offsets used for the maximum-likelihood grid search.
        The read-noise sampling is defined as ``init_RN + init_delta_RNs``.
    quad_delta_RN : float, optional
        Only used when ``method='quad'``. Defines the initial read-noise step
        size around ``init_RN`` for the quadratic approximation. Default is
        0.5.
    countratesidentical : bool, optional
        If True, assume all ramps have identical count rates. Default is
        False.

    Calibrations
    ------------
    gain : GainModel
        - selector: :class:`~hispecdrp.calibrations.selectors.GainSelector`
    rn_init : ReadNoiseModel
        - selector: :class:`~hispecdrp.calibrations.selectors.ReadNoiseSelector`

    Returns
    -------
    output : DataModel
        Data model containing the fitted read noise parameter and associated
        fit diagnostics.

    References
    ----------
    Brandt, T. D. (2025), *Computing the Electronic Gain for Detectors Read
    Out Up-the-ramp*, PASP, 137, 125006.
    https://ui.adsabs.harvard.edu/abs/2025PASP..137l5006B/abstract

    Roman calibration pipeline reference:
    https://github.com/spacetelescope/romancal
    """

    ARGUMENTS = {
        'max_cores' : 1,
        'init_RN' : None,  # None | in,  Define the initial readnoise value used for the maximum likelihood grid search.
        # If None, use calibration database, otherwise specify a np.array (eg, np.linspace(1, 20, 101)).
        'mode': 'per_pixel', # str, 'per_pixel' | 'channel' | 'full_detector', mode to fit the parameters.
        # 'per_pixel' compute readnoise for each pixel independently,
        # 'channel' (Not implemented yet) compute readnoise for each readout channel independently,
        # 'full_detector' compute readnoise for the full detector at once. This can be used when a single exposure is available.
        'method': 'quad',# str, 'quad' | 'grid' | 'both', method to fit the parameters.
        # 'quad' uses quadratic approximation,
        # 'grid' uses maximum likelihood grid search.
        # 'both' start with grid search and then apply quadratic approximation.
        'init_delta_RNs' : None,  # np.array, the grid sampling is defined as init_RN + init_delta_RNs for the maximum likelihood grid search.
        # specify a np.array (eg, np.linspace(1, 20, 101)).
        'quad_delta_RN' : 0.5, # float, used only in 'quad' mode, defines the initial delta read noise around init_RN for the quadratic approximation.
        'countratesidentical': False, # bool, if True, assume all ramps have identical count rates.
    }

    CALIBRATIONS = {
        'gain': {
            'selector': GainSelector,
            'selector_kwargs': {}
        },
        'rn_init': {
            'selector': ReadNoiseSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):
        if self.mode.lower() != 'full_detector' and self.mode.lower() != 'per_pixel':
            raise NotImplementedError(
                f"Only 'full_detector' and 'per_pixel' modes are implemented currently. Not {self.mode}")

        if isinstance(self.input, (datamodels.DataModel, str)):
            _input_list_tmp = [self.input]
        if isinstance(self.input, Sequence):
            _input_list_tmp = self.input
        nfiles = len(_input_list_tmp)
        logger.info(f"Deriving the detector readnoise from a list of {nfiles} inputs.")

        input_models = [datamodels.open(f) for f in _input_list_tmp]
        datamodel0 = input_models[0]
        nreads, ny, nx = datamodel0.data.shape

        gain, _ = self.get_calibration(datamodel0, 'gain')
        gain_model = datamodels.open(gain)
        gain_arr = gain_model.gain

        if self.init_RN is None:
            rn_cal_init, _ = self.get_calibration(datamodel0, 'rn_init')
            rn_model_init = datamodels.open(rn_cal_init)
            _init_center_RN = rn_model_init.RN
            logger.info(f"Using {rn_cal_init} as initial read noise values with median value {np.nanmedian(_init_center_RN)} DN.")
        elif isinstance(self.init_RN, float):
            _init_center_RN = self.init_RN
            logger.info(f"Using user-defined initial read noise values: {self.init_RN}.")

        delta_readtimes = datamodel0.get_delta_readtimes()

        if self.method == 'quad':
            _init_delta_RNs = np.array([-self.quad_delta_RN,0,+self.quad_delta_RN])
        elif self.method == 'grid' or self.method == 'both':
            if self.init_delta_RNs is None:
                _init_delta_RNs = np.linspace(-10,10, 51)
            else:
                _init_delta_RNs = self.init_delta_RNs

        if self.mode.lower() == 'full_detector':
            logger.info(
                f"Running in full_detector mode. All detector pixels will be fitted jointly and a single gain and/or readnoise value will be returned for the entire chip.")
        elif self.mode.lower() == 'per_pixel':
            logger.info(
                f"Running in single_pixel mode. Each detector pixel will be fitted independently and a map of gain and/or readnoise will be returned.")

        # Define output model from inputs
        output_detrn = self.get_output_model(input_models, "ReadNoiseModel")

        # Make all ramps from all files into a big cube
        ramp_arr = []
        dq_raw_arr = []
        for opened_input in input_models:
            if nreads != opened_input.data.shape[0]:
                raise ValueError("All input ramps must have the same number of reads.")
            ramp_arr.append(opened_input.data)
            dq_raw_arr.append(opened_input.dq_raw)
        ramp_arr = np.array(ramp_arr)
        dq_raw_arr = np.array(dq_raw_arr)

        _out = fit_gain_and_readnoise(delta_readtimes, ramp_arr, dq_raw_arr, gain_arr, _init_center_RN,None,_init_delta_RNs,
                                      countratesidentical=self.countratesidentical,
                                      mode = self.mode, method= self.method, max_cores =self.max_cores)
        _, bestfit_rn, cov = _out

        # Output
        output_detrn.rn = bestfit_rn
        output_detrn.rn_err = np.sqrt(cov[1,1])
        output_detrn.dq = np.zeros(bestfit_rn.shape, dtype=np.uint32)

        return dict(output=output_detrn)

    def _create_preview(self):
        """Create preview visualization of read noise calibration results."""
        try:
            rn_model = self._output.get('output')
            if not rn_model:
                logger.warning("No read noise model available for preview.")
                return
            
            rn_data = rn_model.rn
            rn_err = rn_model.rn_err if hasattr(rn_model, 'rn_err') and rn_model.rn_err is not None else None
            
        except Exception as e:
            logger.warning(f"Could not load read noise model for preview: {e}")
            return
        
        ny, nx = rn_data.shape
        
        # Create figure with 2x2 subplots for read noise
        fig_rn = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Fitted Read Noise (Full Frame)",
                "Center 50x50 Pixels (Zoomed)",
                "Histogram of Read Noise Values",
                "Bad Pixel Map"
            ),
            specs=[
                [{}, {}],
                [{"type": "histogram"}, {}]
            ]
        )
        
        # Plot 1: Full read noise map
        fig_rn.add_trace(
            go.Heatmap(
                z=rn_data,
                colorscale='Magma',
                showscale=True,
                name='RN',
                colorbar=dict(x=0.46, y=0.75, len=0.15, orientation='h', xanchor='center', yanchor='top')
            ),
            row=1, col=1
        )
        
        # Plot 2: Zoomed center (50x50 pixels)
        center_y, center_x = ny // 2, nx // 2
        half_size = min(25, min(center_y, center_x, ny - center_y, nx - center_x))
        rn_zoom = rn_data[center_y-half_size:center_y+half_size, center_x-half_size:center_x+half_size]
        
        fig_rn.add_trace(
            go.Heatmap(
                z=rn_zoom,
                colorscale='Magma',
                showscale=True,
                name='RN (Zoomed)',
                colorbar=dict(x=1.02, y=0.75, len=0.15, orientation='h', xanchor='center', yanchor='top')
            ),
            row=1, col=2
        )
        
        # Plot 3: Histogram of read noise values
        valid_rn = rn_data[np.isfinite(rn_data)].flatten()
        fig_rn.add_trace(
            go.Histogram(
                x=valid_rn,
                nbinsx=100,
                name='RN',
                marker=dict(color='#1f77b4'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Plot 4: Bad pixel map (DQ flags)
        # Create a map where bad pixels are marked and good pixels are blank
        bad_pixel_map = np.full_like(rn_model.dq, np.nan, dtype=np.float32)
        bad_pixel_map[rn_model.dq != 0] = 1  # Mark bad pixels
        
        fig_rn.add_trace(
            go.Heatmap(
                z=bad_pixel_map,
                colorscale=[[0, 'white'], [1, 'red']],
                showscale=True,
                name='Bad Pixels',
                colorbar=dict(x=1.02, y=0.25, len=0.15, orientation='h', xanchor='center', yanchor='top', tickvals=[1], ticktext=['Bad']),
                hovertemplate='Bad Pixel<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig_rn.update_xaxes(title_text="Read Noise (DN)", row=2, col=1, showgrid=False)
        fig_rn.update_yaxes(title_text="Frequency", row=2, col=1, showgrid=False)
        fig_rn.update_xaxes(title_text="X Pixel", row=2, col=2, showgrid=False)
        fig_rn.update_yaxes(title_text="Y Pixel", row=2, col=2, showgrid=False)
        fig_rn.update_xaxes(showgrid=False, row=1, col=1)
        fig_rn.update_yaxes(showgrid=False, row=1, col=1)
        fig_rn.update_xaxes(showgrid=False, row=1, col=2)
        fig_rn.update_yaxes(showgrid=False, row=1, col=2)
        
        median_rn = np.nanmedian(rn_data)
        fig_rn.update_layout(
            height=900,
            width=1200,
            title_text=f"Fitted Read Noise (Median: {median_rn:.2f} DN)",
            showlegend=False,
            margin=dict(b=80)
        )
        
        # Save if configured
        if self.save_preview:
            if not self.output_dir:
                self.output_dir = '.'
            
            os.makedirs(self.output_dir, exist_ok=True)
            
            rn_html = os.path.join(self.output_dir, f'readnoise_preview_{self.suffix}.html')
            
            fig_rn.write_html(rn_html, include_plotlyjs='cdn')
            
            logger.info(f"Read Noise preview saved to {rn_html}")
            
            # Try to save PNG version
            try:
                rn_png = rn_html.replace('.html', '.png')
                fig_rn.write_image(rn_png, scale=2)
                logger.info(f"PNG preview saved")
            except Exception as e:
                logger.debug(f"Could not save PNG preview: {e}")
        
        if self.show_preview:
            logger.info("Showing read noise calibration preview...")
            fig_rn.show()
        
        # Store in output
        self._output['preview'] = {
            'rn_fig': fig_rn,
        }

from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np
from typing import Sequence


from ...calibrations import LSFSelector
from .telluric_models import airmass_scale_singlemodel

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import logging

logger = logging.getLogger(__name__)

__all__ = ['TelluricCorrectionPrimitive']

class TelluricCorrectionPrimitive(BasePrimitive):
    """


    Parameters
    ----------
    input : str or L1Model/L2Model
        Input file path or data model to process. Currently, this is expected
        to be either a level 1 (uncalibrated) or level 2 (calibrated) data product.
    suffix : str or None, optional
        Suffix to append to the output filename. Default is ``'tell'``.
    method : str, optional
        Method to use for telluric correction. Currently, only ``'airmass_scaling'`` is implemented. Default is ``'airmass_scaling'``.
    do_orders : list of int or None, optional
        List of orders to process. If None, all orders will be processed. Default is None.
    sci_fiber : int or None, optional
        Science fiber number to use for telluric correction. If None, the science fiber will be determined from the input model's metadata. Default is None.

    Calibrations
    ------------
    lsf : GaussianLSFModel
        - selector: :class:`~hispecdrp.calibrations.selectors.LSFSelector`


    Returns
    -------
    output
        The telluric corrected level 2 data model with updated error and data quality arrays.
    """
    ARGUMENTS = {
        'suffix' : 'tell',
        'do_orders' : None,
        'method' : 'airmass_scaling',
        'sci_fiber' : None,
    }

    CALIBRATIONS = {
        'lsf': {
            'selector': LSFSelector,
            'selector_kwargs': {}
        }
    }


    def _perform(self):
        
        if isinstance(self.input, (datamodels.DataModel, str)):
            output = None
            input = [self.input]
        elif isinstance(self.input, Sequence):
            output = []
            input = self.input

        # Loop over inputs (now a list)
        for _input in input:
            # Load the input model
            input_model = datamodels.open(_input)

             # Init output model
            output_model = self.get_output_model(input_model, output_model='L2Model', include_data=True)

            # Get calibrations
            lsf, _ = self.get_calibration(input_model, 'lsf')
            lsf_model = datamodels.open(lsf)
                       
            # Which orders to process
            _, do_orders = self.get_do_traces(input_model, [1,2,3], self.do_orders)
            order_vec = input_model.make_order_vec()
            n_orders = len(order_vec)

            spec_key = 'SPEC_UNCAL_' if isinstance(input_model, datamodels.level1.L1Model) else 'SPEC_'

            # get science/sky fiber numbers, sky fiber defaults to farthest fiber from tracking axis
            for fiber in range(1,4):
                if input_model.meta.get(f'object_fiber{fiber}', '').upper() == 'SCIENCE' and self.sci_fiber == None:
                    sci_fib_num = fiber
                elif self.sci_fiber != None:
                    sci_fib_num = self.sci_fiber
                    logger.info(f'Science fiber manually set to fiber {sci_fib_num}')

            if self.method == 'airmass_scaling':
                output_model = airmass_scale_singlemodel(
                    input_model,
                    output_model,
                    lsf_model,
                    sci_fib_num,
                    spec_key,
                    order_vec,
                    do_orders,
                )
            else:
                raise ValueError(self.method+' is not a recognized telluric correction method.')


            if isinstance(output, list):
                output.append(output_model)
            else:
                output = output_model

        return dict(output=output)
    
    

    def _create_preview(self):
        """
        Build an interactive Plotly preview of the telluric correction.

        Assumes the correction routine (e.g. airmass_scale_singlemodel) stores,
        for the science fiber:
            SPEC_FIB{sci}         -> telluric-corrected science spectrum
            SPEC_ERR_FIB{sci}     -> corrected error
            WAVE_FIB{sci}         -> wavelength grid
            TELLURIC_FIB{sci}     -> telluric transmission/correction curve applied

        and that the *input* model (still reachable via output_model, or cached
        separately) has the pre-correction spectrum under SPEC_FIB{sci} /
        SPEC_UNCAL_FIB{sci}. Adjust the attribute names below if your
        airmass_scale_singlemodel implementation names things differently.
        """
        output_model = self._output['output']

        # Normalize to a list
        if not isinstance(output_model, list):
            output_models = [output_model]
        else:
            output_models = output_model

        # --- resolve science fiber number per model ---
        def resolve_sci_fiber(model, sci_fiber_override):
            sci_fib_num = None
            for fiber in range(1, 4):
                if model.meta.get(f'object_fiber{fiber}', '').upper() == 'SCIENCE' and sci_fiber_override is None:
                    sci_fib_num = fiber
                elif sci_fiber_override is not None:
                    sci_fib_num = sci_fiber_override
            return sci_fib_num

        # Pre-compute per-model metadata so we can build traces in one pass
        model_meta = []
        for model in output_models:
            sci_fib_num = resolve_sci_fiber(model, self.sci_fiber)
            order_vec   = model.make_order_vec()
            filename    = model.meta.get('filename', f'Model {len(model_meta) + 1}')
            model_meta.append(dict(
                model       = model,
                sci_fib_num = sci_fib_num,
                order_vec   = order_vec,
                n_orders    = len(order_vec),
                filename    = filename,
            ))

        # All models must share the same order vector for the slider to be coherent;
        # use the first model's order_vec as the reference.
        ref_order_vec = model_meta[0]['order_vec']
        n_orders      = model_meta[0]['n_orders']
        # Traces per order: corrected spectrum, uncorrected spectrum, telluric curve
        traces_per_order = 3

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
            vertical_spacing=0.06,
            subplot_titles=('Science Spectrum: Corrected vs. Uncorrected', 'Telluric Correction Applied'),
        )

        traces_per_model = n_orders * traces_per_order
        total_traces     = len(output_models) * traces_per_model

        for m_idx, meta in enumerate(model_meta):
            model       = meta['model']
            sci_fib_num = meta['sci_fib_num']
            order_vec   = meta['order_vec']

            wave           = getattr(model, f'WAVE_FIB{sci_fib_num}')
            spec_corrected = getattr(model, f'SPEC_FIB{sci_fib_num}')
            spec_uncorr    = getattr(model, f'SPEC_UNCAL_FIB{sci_fib_num}', None)
            if spec_uncorr is None:
                # Fall back if the uncorrected spectrum wasn't preserved separately
                spec_uncorr = spec_corrected
            telluric_curve = getattr(model, f'TEL_FIB{sci_fib_num}', None)

            for i, order in enumerate(order_vec):
                # Only the first model's first order is visible on load
                visible = (m_idx == 0 and i == 0)
                w = wave[i]

                fig.add_trace(go.Scatter(
                    x           = w,
                    y           = spec_uncorr[i],
                    mode        = 'lines',
                    name        = f'Uncorrected (fiber {sci_fib_num})',
                    line        = dict(color='lightslategray', width=1.2, dash='dot'),
                    legendgroup = 'uncorrected',
                    showlegend  = True,
                    visible     = visible,
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x           = w,
                    y           = spec_corrected[i],
                    mode        = 'lines',
                    name        = f'Telluric-corrected (fiber {sci_fib_num})',
                    line        = dict(color='royalblue', width=1.5),
                    legendgroup = 'corrected',
                    showlegend  = True,
                    visible     = visible,
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x           = w,
                    y           = telluric_curve[i] if telluric_curve is not None else np.full_like(w, np.nan),
                    mode        = 'lines',
                    name        = 'Telluric correction',
                    line        = dict(color='seagreen', width=1.5),
                    legendgroup = 'telluric',
                    showlegend  = True,
                    visible     = visible,
                ), row=2, col=1)

        # --- Order slider ---
        def make_visibility(active_model: int, active_order: int) -> list:
            """Return a full visibility list with only the chosen model+order shown."""
            vis = [False] * total_traces
            base = active_model * traces_per_model + active_order * traces_per_order
            for t in range(traces_per_order):
                vis[base + t] = True
            return vis

        slider_steps = []
        for i, order in enumerate(ref_order_vec):
            slider_steps.append(dict(
                method = 'update',
                label  = str(order),
                args   = [
                    {'visible': make_visibility(0, i)},
                    {'title': f'Telluric Correction — Order {order}  |  {model_meta[0]["filename"]}'}
                ],
            ))

        # --- Model dropdown ---
        dropdown_buttons = []
        for m_idx, meta in enumerate(model_meta):
            model_slider_steps = []
            for i, order in enumerate(meta['order_vec']):
                model_slider_steps.append(dict(
                    method = 'update',
                    label  = str(order),
                    args   = [
                        {'visible': make_visibility(m_idx, i)},
                        {'title': f'Telluric Correction — Order {order}  |  {meta["filename"]}'}
                    ],
                ))

            dropdown_buttons.append(dict(
                label  = meta['filename'],
                method = 'update',
                args   = [
                    {'visible': make_visibility(m_idx, 0)},
                    {
                        'title': f'Telluric Correction — Order {ref_order_vec[0]}  |  {meta["filename"]}',
                        'sliders': [dict(
                            active       = 0,
                            currentvalue = dict(prefix='Order: ', font=dict(size=14)),
                            pad          = dict(t=50),
                            steps        = model_slider_steps,
                        )],
                    },
                ],
            ))

        fig.update_layout(
            title = f'Telluric Correction — Order {ref_order_vec[0]}  |  {model_meta[0]["filename"]}',
            updatemenus = [dict(
                buttons    = dropdown_buttons,
                direction  = 'down',
                showactive = True,
                x          = 1.0,
                xanchor    = 'right',
                y          = 1.15,
                yanchor    = 'top',
                bgcolor    = '#f0f0f0',
                bordercolor = '#cccccc',
            )] if len(output_models) > 1 else [],
            sliders = [dict(
                active       = 0,
                currentvalue = dict(prefix='Order: ', font=dict(size=14)),
                pad          = dict(t=50),
                steps        = slider_steps,
            )],
            plot_bgcolor  = '#ffffff',
            paper_bgcolor = '#ffffff',
            font          = dict(color='#000000'),
            legend        = dict(bgcolor='#ffffff'),
            height        = 700,
        )
        fig.update_xaxes(title_text='Wavelength (micron)', showgrid=True, gridcolor='rgba(200,200,200,0.3)', row=2, col=1)
        fig.update_yaxes(title_text='Flux (e-/s)', showgrid=True, gridcolor='rgba(200,200,200,0.3)', row=1, col=1)
        fig.update_yaxes(title_text='Transmission', showgrid=True, gridcolor='rgba(200,200,200,0.3)', row=2, col=1)

        # File paths — use the first model as the naming anchor
        base_path        = output_models[0]._generate_output_path(output_dir=self.output_dir, suffix=self.suffix)
        output_path_html = base_path.replace('.fits', '_preview.html')
        output_path_png  = base_path.replace('.fits', '_preview.png')

        if self.save_preview:
            fig.write_html(output_path_html, include_plotlyjs='cdn')
            fig.write_image(output_path_png, scale=2)

        if self.show_preview:
            fig.show()

        self._output['preview'] = {
            'html': output_path_html,
            'png':  output_path_png,
            'fig':  fig,
        }
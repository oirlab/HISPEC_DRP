from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np
from typing import Sequence


from ...calibrations import FiberFlatSelector

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import logging

logger = logging.getLogger(__name__)

__all__ = ['SkySubtractionPrimitive']

class SkySubtractionPrimitive(BasePrimitive):
    """
    Subtract the sky background from the input data.

    This primitive applies the appropriate sky calibration to the
    input data product. The sky frame is subtracted from the science data,
    the associated uncertainties are propagated by adding errors in
    quadrature, and data quality (DQ) flags are combined using a bitwise OR.

    Parameters
    ----------
    input : str or L2Model
        Input file path or data model to process. Currently, this is expected
        to be either a level 1 (uncalibrated) or level 2 (calibrated) data product.
    suffix : str or None, optional
        Suffix to append to the output filename. Default is ``'skysub'``.
    sky_fiber : int, optional
        Manually override the sky fiber to a specific fiber
    sci_fiber : int, optional
        Manually override the science fiber to a specific fiber

    Calibrations
    ------------
    fiberflat : FiberFlatModel
        - selector: :class:`~hispecdrp.calibrations.selectors.FiberFlatSelector`

    Returns
    -------
    output
        The sky-subtracted level 2 data model with updated error and data quality arrays.
    """
    PARAMETERS = {
        'suffix' : 'skysub',
        'do_orders' : None,
        'sky_fiber' : None,
        'sci_fiber' : None,
    }

    CALIBRATIONS = {
        'fiberflat': {
            'selector': FiberFlatSelector,
            'selector_kwargs': {}
        }
    }


    def _perform(self):

        if isinstance(self.dataset, (datamodels.DataModel, str)):
            output = None
            input = [self.dataset]
        elif isinstance(self.dataset, Sequence):
            output = []
            input = self.dataset

        # Loop over inputs (now a list)
        for _input in input:
            # Load the input model
            input_model = datamodels.open(_input)

             # Init output model
            output_model = self._model_types(input_model)
                       
            # Which orders to process
            _, do_orders = self.get_do_traces(input_model, [1,2,3], self.do_orders)
            order_vec = input_model.make_order_vec()
            n_orders = len(order_vec)

            # Get calibrations
            fiberflat, _ = self.get_calibration(input_model, 'fiberflat')
            fiberflat_model = datamodels.open(fiberflat)

            spec_key = 'SPEC_UNCAL_' if isinstance(input_model, datamodels.level1.L1Model) else 'SPEC_'

            # get science/sky fiber numbers, sky fiber defaults to farthest fiber from tracking axis
            sci_fib_num, sky_fib_num = self._resolve_fibers(input_model, self.sci_fiber, self.sky_fiber)

            # get sky, warn if sky fiber seems unusually close or bright
            sky_spec = getattr(input_model, f'{spec_key}FIB{sky_fib_num}')
            sky_spec_err = getattr(input_model, f'{spec_key}ERR_FIB{sky_fib_num}')
            sky_wave = getattr(input_model, f"WAVE_FIB{sky_fib_num}")
            if sky_spec is None or sky_spec_err is None or sky_wave is None:
                raise ValueError(f"Sky fiber {sky_fib_num} data is missing. Cannot perform sky subtraction.")
            if input_model.meta.get(f'distance_fiber{sky_fib_num}') is not None:
                if input_model.meta.get(f'distance_fiber{sky_fib_num}') < 1.0: #TODO check if warning distance is appropriate number
                    logger.warning(f"Sky fiber {sky_fib_num} is unusually close to the tracking axis. Sky subtraction may be inaccurate.")
                elif np.nanmedian(sky_spec) > 0.3:  # arbitrary threshold for "too bright" #TODO check brightness number with observed sky brightnesses
                    logger.warning(f"Sky fiber {sky_fib_num} seems too bright. Sky subtraction may be inaccurate.")

            # get science spectrum
            sci_spec = getattr(input_model, f'{spec_key}FIB{sci_fib_num}')
            sci_spec_err = getattr(input_model, f'{spec_key}ERR_FIB{sci_fib_num}')
            sci_wave = getattr(input_model, f"WAVE_FIB{sci_fib_num}")
            if sci_spec is None or sci_spec_err is None or sci_wave is None:
                raise ValueError(f"Science fiber {sci_fib_num} data is missing. Cannot perform sky subtraction.")

            # get variance and DQ arrays for science and sky fibers
            sci_var_poisson = getattr(input_model, f'{spec_key}VAR_POISSON_FIB{sci_fib_num}')
            sci_var_rnoise  = getattr(input_model, f'{spec_key}VAR_RNOISE_FIB{sci_fib_num}')
            sci_dq          = getattr(input_model, f'SPEC_DQ_FIB{sci_fib_num}')
            sky_dq          = getattr(input_model, f'SPEC_DQ_FIB{sky_fib_num}')

            # get fiberflat spectrum from same fibers as sky/science
            fiberflat_sky_spec = getattr(fiberflat_model, f"SPEC_FIB{sky_fib_num}")
            fiberflat_sky_wave = getattr(fiberflat_model, f"WAVE_FIB{sky_fib_num}")
            fiberflat_sci_spec = getattr(fiberflat_model, f"SPEC_FIB{sci_fib_num}")
            fiberflat_sci_wave = getattr(fiberflat_model, f"WAVE_FIB{sci_fib_num}")

            sci_spec_sky_subtracted     = np.empty_like(sci_spec)
            sci_spec_sky_subtracted_err = np.empty_like(sci_spec_err)
            sky_spec_scaled             = np.empty_like(sci_spec)
            sky_spec_err_scaled         = np.empty_like(sci_spec_err)
            out_dq                      = np.empty_like(sci_dq)

            for order in range(n_orders):
                if order_vec[order] not in do_orders:
                    continue

                # get relative transmission ratio from science/sky from fiberflat calibration
                if fiberflat_sky_spec is None or fiberflat_sci_spec is None:
                    logger.warning(f"Fiberflat calibration missing for science or sky fiber for order {order_vec[order]}. Sky subtraction may be inaccurate.")
                    fiberflat_ratio = 1.0
                else:
                    fiberflat_sky_spec_interp = np.interp(fiberflat_sci_wave[order], fiberflat_sky_wave[order], fiberflat_sky_spec[order])
                    fiberflat_ratio = fiberflat_sci_spec[order] / fiberflat_sky_spec_interp


                sky_spec_interp = np.interp(sci_wave[order], sky_wave[order], sky_spec[order])
                sky_spec_scaled[order] = sky_spec_interp * fiberflat_ratio

                # Interpolate sky error onto science wavelength grid and scale by fiberflat
                sky_spec_err_interp = np.interp(sci_wave[order], sky_wave[order], sky_spec_err[order])
                sky_spec_err_scaled[order] = sky_spec_err_interp * fiberflat_ratio

                sci_spec_sky_subtracted[order] = sci_spec[order] - sky_spec_scaled[order]

                # Propagate flux errors in quadrature: err = sqrt(sci_err^2 + (scaled_sky_err)^2)
                sci_spec_sky_subtracted_err[order] = np.sqrt(
                    sci_spec_err[order]**2 + sky_spec_err_scaled[order]**2
                )

                out_of_bounds_mask = np.logical_or((sci_wave[order] < np.nanmin(sky_wave[order])), (sci_wave[order] > np.nanmax(sky_wave[order])))
                out_of_bounds_dq = np.zeros(len(sci_wave[order]), dtype=np.int32) 
                out_of_bounds_dq[out_of_bounds_mask] = np.int32(1)
                out_dq[order] = sci_dq[order] | out_of_bounds_dq

            setattr(output_model, f'SPEC_FIB{sci_fib_num}',             sci_spec_sky_subtracted)
            setattr(output_model, f'SPEC_ERR_FIB{sci_fib_num}',         sci_spec_sky_subtracted_err)
            setattr(output_model, f'SPEC_VAR_POISSON_FIB{sci_fib_num}', sci_var_poisson.copy())
            setattr(output_model, f'SPEC_VAR_RNOISE_FIB{sci_fib_num}',  sci_var_rnoise.copy())
            setattr(output_model, f'SPEC_DQ_FIB{sci_fib_num}',          out_dq)
            setattr(output_model, f'SKY_FIB{sci_fib_num}',              sky_spec_scaled)
            setattr(output_model, f'SKY_ERR_FIB{sci_fib_num}',          sky_spec_err_scaled)

            if isinstance(output, list):
                output.append(output_model)
            else:
                output = output_model

        return dict(output=output)
    

    def _model_types(self, input_model):
        if isinstance(input_model, datamodels.level1.L1Model):
            output_model = self.get_output_model(input_model, output_model='L2Model', include_data=True)
        elif isinstance(input_model, datamodels.level2.L2Model):
            output_model = self.get_output_model(input_model, output_model='input', include_data=True)
        else:
            raise TypeError(f"Unsupported input model type: {type(input_model)}. Expected L1Model or L2Model.")
        return output_model
    

    def _create_preview(self):
        output_model = self._output['output']

        # Normalize to a list
        if not isinstance(output_model, list):
            output_models = [output_model]
        else:
            output_models = output_model


        # Pre-compute per-model metadata so we can build traces in one pass
        model_meta = []
        for model in output_models:
            sci_fib_num, sky_fib_num = self._resolve_fibers(model, self.sci_fiber, self.sky_fiber)
            order_vec   = model.make_order_vec()
            filename    = model.meta.get('filename', f'Model {len(model_meta) + 1}')
            model_meta.append(dict(
                model       = model,
                sci_fib_num = sci_fib_num,
                sky_fib_num = sky_fib_num,
                order_vec   = order_vec,
                n_orders    = len(order_vec),
                filename    = filename,
            ))

        # All models must share the same order vector for the slider to be coherent;
        # use the first model's order_vec as the reference.
        ref_order_vec = model_meta[0]['order_vec']
        n_orders      = model_meta[0]['n_orders']
        traces_per_order = 2  # sky + sky_err band

        fig = go.Figure()

        # Build traces: outer loop = models, inner loop = orders
        # Layout: [model0_order0_sky, model0_order0_err, model0_order1_sky, ...]
        #          [model1_order0_sky, model1_order0_err, ...]
        traces_per_model = n_orders * traces_per_order
        total_traces     = len(output_models) * traces_per_model

        for m_idx, meta in enumerate(model_meta):
            model       = meta['model']
            sci_fib_num = meta['sci_fib_num']
            sky_fib_num = meta['sky_fib_num']
            order_vec   = meta['order_vec']

            sky_wave = getattr(model, f'WAVE_FIB{sci_fib_num}')
            sky_spec = getattr(model, f'SKY_FIB{sci_fib_num}')
            sky_err  = getattr(model, f'SKY_ERR_FIB{sci_fib_num}')

            for i, order in enumerate(order_vec):
                # Only the first model's first order is visible on load
                visible = (m_idx == 0 and i == 0)
                w_sky   = sky_wave[i]

                fig.add_trace(go.Scatter(
                    x         = w_sky,
                    y         = sky_spec[i],
                    mode      = 'lines',
                    name      = f'Sky (fiber {sky_fib_num})',
                    line      = dict(color='tomato', width=1.5),
                    legendgroup = 'sky',
                    showlegend  = True,
                    visible     = visible,
                ))
                fig.add_trace(go.Scatter(
                    x    = np.concatenate([w_sky, w_sky[::-1]]),
                    y    = np.concatenate([
                        sky_spec[i] + sky_err[i],
                        (sky_spec[i] - sky_err[i])[::-1]
                    ]),
                    fill        = 'toself',
                    fillcolor   = 'rgba(255,99,71,0.15)',
                    line        = dict(color='rgba(0,0,0,0)'),
                    name        = 'Sky ±1σ',
                    legendgroup = 'sky_err',
                    showlegend  = True,
                    visible     = visible,
                ))

        # --- Order slider ---
        # Each step shows the currently-selected model's traces for one order.
        # The active model index is tracked via a mutable closure updated by the
        # dropdown buttons below (Plotly sliders can't directly read button state,
        # so each dropdown button embeds a full visibility list for order 0 of that
        # model, and the slider steps are recomputed per-model inside each button).
        #
        # Strategy: slider steps always operate on ALL traces.  For a given
        # (model_idx, order_idx) pair, exactly `traces_per_order` traces are True.

        def make_visibility(active_model: int, active_order: int) -> list:
            """Return a full visibility list with only the chosen model+order shown."""
            vis = [False] * total_traces
            base = active_model * traces_per_model + active_order * traces_per_order
            for t in range(traces_per_order):
                vis[base + t] = True
            return vis

        # Slider steps fix model=0 by default; dropdown buttons override this by
        # supplying their own visibility list for every step via restyle calls.
        slider_steps = []
        for i, order in enumerate(ref_order_vec):
            slider_steps.append(dict(
                method = 'update',
                label  = str(order),
                args   = [
                    {'visible': make_visibility(0, i)},
                    {'title': f'Sky — Order {order}  |  {model_meta[0]["filename"]}'}
                ],
            ))

        # --- Model dropdown ---
        # Each button sets visibility for order 0 of that model AND rewrites the
        # slider steps so subsequent slider moves stay within the same model.
        dropdown_buttons = []
        for m_idx, meta in enumerate(model_meta):
            # Build slider steps for this model
            model_slider_steps = []
            for i, order in enumerate(meta['order_vec']):
                model_slider_steps.append(dict(
                    method = 'update',
                    label  = str(order),
                    args   = [
                        {'visible': make_visibility(m_idx, i)},
                        {'title': f'Sky — Order {order}  |  {meta["filename"]}'}
                    ],
                ))

            dropdown_buttons.append(dict(
                label  = meta['filename'],
                method = 'update',
                args   = [
                    # Show order 0 for this model
                    {'visible': make_visibility(m_idx, 0)},
                    {
                        'title': f'Sky — Order {ref_order_vec[0]}  |  {meta["filename"]}',
                        # Replace the slider steps so it now controls this model
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
            title = f'Sky — Order {ref_order_vec[0]}  |  {model_meta[0]["filename"]}',
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
            xaxis      = dict(title='Wavelength (micron)', showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
            yaxis      = dict(title='Flux (e-/s)',          showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
            plot_bgcolor  = '#ffffff',
            paper_bgcolor = '#ffffff',
            font          = dict(color='#000000'),
            legend        = dict(bgcolor='#ffffff'),
            height        = 550,
        )

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

    def _resolve_fibers(self, model, sci_fiber_override=None, sky_fiber_override=None):
        """Resolve science and sky fiber numbers from overrides or metadata."""
        sci_fib_num = sci_fiber_override
        if sci_fib_num is not None:
            logger.info(f'Science fiber manually set to fiber {sci_fib_num}')
        else:
            for fiber in range(1, 4):
                if model.meta.get(f'object_fiber{fiber}', '').upper() == 'SCIENCE':
                    sci_fib_num = fiber
                    break
            if sci_fib_num is None:
                sci_fib_num = 1
                logger.info(f'Could not find SCIENCE keyword in fiber meta data, science fiber set to {sci_fib_num}')

        sky_fib_num = sky_fiber_override
        if sky_fib_num is not None:
            logger.info(f'Sky fiber manually set to fiber {sky_fib_num}')
        else:
            sky_fib_dist = 0.
            for fiber in range(1, 4):
                fib_dist = model.meta.get(f'distance_fiber{fiber}')
                if fib_dist is None:
                    logger.info(f'Could not find distance_fiber keyword in meta data for fiber {fiber}')
                elif fib_dist > sky_fib_dist:
                    sky_fib_num = fiber
                    sky_fib_dist = fib_dist
            if sky_fib_num is None:
                sky_fib_num = 3
                logger.info(f'No distance_fiber keywords found in meta data for any fibers, sky fiber will be set to {sky_fib_num}.')

        return sci_fib_num, sky_fib_num
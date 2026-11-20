from ..base_primitive import BasePrimitive
from ... import datamodels

from .fit_lines import fit_lines_gauss
from .find_lines import find_emission_lines
from ..utils.spectral_utils import delta_pix_to_delta_wave

import numpy as np
import scipy.constants
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from logging import getLogger
logger = getLogger(__name__)

__all__ = ['FitLFCLinesPrimitive']


class FitLFCLinesPrimitive(BasePrimitive):

    """
    NOTE: *Under development*
    
    Fit each laser frequency comb (LFC) line with a Gaussian profile. This is useful for calculating the wavelength solution and the the line spread function (LSF).

    Each LFC line is first identified with `scipy.signal.find_peaks <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_ then refined with a centroid. Then each line is fit with a Gaussian profile using `scipy.optimize.least_squares <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>`_. The fit parameters are stored in a table in the output data model's attribute ``lfc_fitparams_fib#``.

    Parameters
    ----------
    input
        A level 1 or 2 data product.
    fwhm_bounds : Sequence[float, float]
        The bounds for the FWHM of the fit. Default is (0.2, 8) pixels.
    mu_bounds : Sequence[float, float]
        The bounds for the mean of the fit (in pixels) relative the centroid.
        Default is (-1, 1) pixels.
    max_fit_window : float
        The maximum number of pixels to fit for a mode.
        Default is 14 pixels.
    do_orders : Sequence[int] | None
        Which order(s) to fit. If None, fit all available orders.
    do_fibers : Sequence[int] | None
        Which fiber(s) to fit. If None, fit all available fibers.
    lfc_nu0 : float
        The central frequency (pump line) of the LFC in GHz.
    lfc_delta_nu : float
        The frequency spacing of the LFC modes in GHz.
    line_source : str
        The source of the lines to fit. Options:
        - 'peaks' : Use only the lines found by `find_emission_lines`.
        - 'wavesol' : Use all lines that fall within the valid wavelength range.
    min_line_spacing : int
        Minimum spacing between lines in pixels. Default is 10 pixels.
    background_filter_width : int | None
        Number of pixels to use for background estimation. If None, no background subtraction is performed.

    Returns
    -------
    output : LFCFitParamsModel
        A data model containing the fit parameters for each LFC line.

    """

    PARAMETERS = {
        'fwhm_bounds' : (1.5, 7.0),
        'mu_bounds' : (-1.0, 1.0),
        'max_fit_window' : 20,
        'do_fibers' : None,
        'do_orders' : None,
        'lfc_nu0' : None,
        'lfc_delta_nu' : None,
        'background_poly_deg' : 0,
        'line_source' : 'wavesol', # 'wavesol' (all lines, seeded from the existing wavesol) or 'peaks' (only lines found by find_peaks)
        'min_line_spacing' : 10, # Minimum number of pixels between lines
        'background_filter_width' : 31, # Pixels
        'preview_n_lines_per_order' : 25, # Max lines shown per order in the preview slider. None = show all.
    }

    def _perform(self):
        input_model = datamodels.open(self.dataset)
        output_model = datamodels.LFCFitParamsModel.from_datamodels(
            input_model, include_data=True,
            primitive=self
        )

        # Which traces to process
        do_fibers, do_orders = self.get_do_traces(input_model, self.do_fibers, self.do_orders)

        # LFC parameters
        lfc_nu0 = self.lfc_nu0 if self.lfc_nu0 is not None else input_model.meta.lfc_nu0
        lfc_delta_nu = self.lfc_delta_nu if self.lfc_delta_nu is not None else input_model.meta.lfc_delta_nu

        fit_results = {fiber : [] for fiber in do_fibers}

        for fiber in do_fibers:

            # Get extensions for this fiber (blaze-corrected preferred when present)
            wave = input_model[fiber].wave
            if input_model.has_extension(f'spec_fib{fiber}'):
                spec = input_model[fiber].spec
                spec_err = input_model[fiber].spec_err
                spec_dq = input_model[fiber].spec_dq
            else:
                spec = input_model[fiber].spec_uncal
                spec_err = input_model[fiber].spec_uncal_err
                spec_dq = input_model[fiber].spec_uncal_dq

            for order in do_orders:

                # Relative order index
                oi = np.abs(order - input_model.meta.order_start)

                # Good pixels for this order
                good = np.where(spec_dq[oi, :] == 0)[0]
                if len(good) == 0:
                    logger.warning(f"No good pixels found in order {order} for fiber {fiber}. Skipping.")
                    continue

                _wave = wave[oi, :]
                _spec = spec[oi, :]
                _spec_err = spec_err[oi, :]
                _spec_dq = spec_dq[oi, :]

                line_spacing_um = (1E-6 * np.mean(_wave[good]))**2 * lfc_delta_nu / scipy.constants.c * 1E6
                line_spacing_pix = line_spacing_um / np.max(np.abs(np.gradient(_wave)))
                adaptive_spacing = float(line_spacing_pix) * 0.8
                configured_spacing = float(self.min_line_spacing)
                # Prefer the tighter of adaptive vs configured spacing so both
                # LFC densification and explicit test/config floors work;
                # find_peaks requires distance >= 1.
                candidates = [
                    s for s in (adaptive_spacing, configured_spacing)
                    if np.isfinite(s) and s > 0.0
                ]
                min_line_spacing = max(1.0, min(candidates) if candidates else 1.0)
                # Pixel array along the spectral axis (not spec.size, which is 2-D)
                pix_arr = np.arange(spec.shape[-1])

                # Find the initial lines
                if self.line_source.lower() == 'peaks':
                    line_centers, _ = find_emission_lines(
                        _spec, _spec_dq,
                        min_line_spacing=min_line_spacing,
                        background_filter_width=self.background_filter_width,
                        centroid_num_pixels=3,
                        drop_first_last=True,
                        refine_with_centroid=True,
                    )
                elif self.line_source.lower() == 'wavesol':

                    # Get the wavelength range for this order
                    # and the LFC lines that fall within it
                    wi, wf = np.min(wave[oi, good]), np.max(wave[oi, good])
                    line_centers_um, _, _ = gen_lfc_modes(
                        lfc_nu0, lfc_delta_nu,
                        wave_min=wi, wave_max=wf
                    )
                    line_centers = np.interp(line_centers_um, wave[oi, good], pix_arr[good])
                    line_centers = line_centers[1:-1] # Drop first and last lines to avoid edge effects
                else:
                    logger.error(f"Invalid line_source '{self.line_source}'. Must be 'peaks' or 'wavesol'.")

                # Gaussian fit to lines
                _fit_results = fit_lines_gauss(
                    pix_arr, _spec, _spec_err, _spec_dq,
                    line_centers=line_centers,
                    fwhm_bounds=self.fwhm_bounds,
                    mu_bounds=self.mu_bounds,
                    max_fit_window=self.max_fit_window,
                )
                fit_results[fiber].append(_fit_results)

                # Store results in the model
                params_table = output_model[fiber].lfc_fitparams
                if order in params_table['ORDER']:
                    input_model[fiber].lfc_fitparams.remove_rows(
                        np.where(input_model[fiber].lfc_fitparams['ORDER'] == order)[0]
                    )
                n_lines = len(_fit_results)
                n_pix = spec.shape[-1]
                for i in range(n_lines):
                    mu_pix = _fit_results[i].mu
                    skip_reason = _unusable_lfc_line_fit_reason(
                        _fit_results[i], fwhm_bounds=self.fwhm_bounds
                    )
                    if skip_reason is not None:
                        logger.warning(
                            "Skipping failed LFC line fit in order %s fiber %s (%s).",
                            order,
                            fiber,
                            skip_reason,
                        )
                        continue
                    pix_idx = int(np.round(mu_pix))
                    if pix_idx < 0 or pix_idx >= n_pix:
                        logger.warning(
                            "Skipping LFC line fit in order %s fiber %s "
                            "(MU_PIX=%s outside 0..%s).",
                            order,
                            fiber,
                            mu_pix,
                            n_pix - 1,
                        )
                        continue
                    r = {
                        'ORDER' : order,
                        'AMPLITUDE' : _fit_results[i].amplitude,
                        'AMPLITUDE_ERR' : _fit_results[i].amplitude_err,
                        'MU_PIX' : mu_pix,
                        'MU_ERR_PIX' : _fit_results[i].mu_err,
                        'FWHM_PIX' : _fit_results[i].fwhm,
                        'FWHM_ERR_PIX' : _fit_results[i].fwhm_err,
                    }
                    wave_closest = wave[oi, pix_idx]
                    if not np.isfinite(wave_closest):
                        logger.warning(
                            "Skipping LFC line fit in order %s fiber %s "
                            "(WAVE at MU_PIX=%s is not finite).",
                            order,
                            fiber,
                            mu_pix,
                        )
                        continue
                    line_wave, _, line_num = _get_closest_lfc_mode(
                        lfc_nu0, lfc_delta_nu, wave_closest
                    )
                    r['WAVE'] = line_wave
                    r['MODE_NUM'] = line_num
                    # WAVE can decrease along the detector; FWHM / errors are widths.
                    r['FWHM_WAVE'] = abs(delta_pix_to_delta_wave(r['FWHM_PIX'], r['MU_PIX'], wave[oi, :]))
                    r['FWHM_ERR_WAVE'] = abs(delta_pix_to_delta_wave(r['FWHM_ERR_PIX'], r['MU_PIX'], wave[oi, :]))
                    r['MU_ERR_WAVE'] = abs(delta_pix_to_delta_wave(r['MU_ERR_PIX'], r['MU_PIX'], wave[oi, :]))
                    params_table.add_row(r)
                params_table.sort(['ORDER', 'MU_PIX'])
            # A single comb mode can be fit twice (e.g. a structured ZEMAX
            # PSF split into two peaks a few pixels apart, both assigned
            # the same MODE_NUM). Two knots with one wavelength wreck any
            # downstream wavesol interpolant, so keep only the strongest
            # fit per (ORDER, MODE_NUM).
            output_model[fiber].lfc_fitparams = _drop_duplicate_lfc_modes(
                output_model[fiber].lfc_fitparams
            )
        return dict(output=output_model, fit_results=fit_results)

    def _create_preview(self):
        """
        Build an interactive preview comparing each fitted LFC line's Gaussian
        model to the underlying spectral data.
 
        Navigation is three-level: a *fiber* dropdown, an *order* dropdown
        (options depend on the selected fiber), and a *line* slider (options
        depend on the selected fiber+order).
 
        For speed and file size, this keeps exactly two persistent traces in
        the figure (the data points and the Gaussian fit curve) and swaps
        their x/y data on each dropdown/slider interaction, rather than
        pre-building a trace pair per line and toggling visibility across
        thousands of traces (which scales quadratically with the number of
        lines and is what made this slow). Additionally, at most
        ``preview_n_lines_per_order`` lines (evenly sampled) are included per
        order; set it to None to include every fitted line.
        """
        output_model = self._output['output']
        fit_results = self._output['fit_results']
 
        # Recompute the same fiber/order selection used in _perform so that
        # fit_results[fiber][i] lines up with do_orders[i].
        do_fibers, do_orders = self.get_do_traces(output_model, self.do_fibers, self.do_orders)
 
        sigma_scale = 2 * np.sqrt(2 * np.log(2))  # FWHM -> sigma
        half_win = self.max_fit_window / 2.0
        n_preview = self.preview_n_lines_per_order
 
        # --- Build a nested fiber -> order -> line structure (sampled per order) ---
        fiber_meta = []
        for fiber in do_fibers:
            if fiber not in fit_results:
                continue
 
            wave     = getattr(output_model, f'wave_fib{fiber}')
            spec     = getattr(output_model, f'spec_fib{fiber}')
            spec_err = getattr(output_model, f'spec_err_fib{fiber}')
            n_pix    = spec.shape[1]
            pix_arr  = np.arange(n_pix)
 
            order_entries = []
            for oi_idx, order in enumerate(do_orders):
                oi = int(np.abs(order - output_model.meta.order_start))
                order_fits = fit_results[fiber][oi_idx] if oi_idx < len(fit_results[fiber]) else []
 
                # Cheaply filter to fits with a finite centroid before doing any
                # array slicing / curve reconstruction work.
                valid_fits = [lf for lf in order_fits if np.isfinite(getattr(lf, 'mu', np.nan))]
                if not valid_fits:
                    continue
 
                n_total = len(valid_fits)
                if n_preview is not None and n_total > n_preview:
                    sample_idx = np.unique(np.linspace(0, n_total - 1, n_preview).astype(int))
                    sampled_fits = [valid_fits[i] for i in sample_idx]
                else:
                    sampled_fits = valid_fits
 
                lines = []
                for line_fit in sampled_fits:
                    mu       = line_fit.mu
                    amp      = getattr(line_fit, 'amplitude', np.nan)
                    amp_err  = getattr(line_fit, 'amplitude_err', np.nan)
                    mu_err   = getattr(line_fit, 'mu_err', np.nan)
                    fwhm     = getattr(line_fit, 'fwhm', np.nan)
                    fwhm_err = getattr(line_fit, 'fwhm_err', np.nan)
 
                    # Data window around the line, padded a bit beyond the fit window for context
                    lo = max(int(np.floor(mu - 1.5 * half_win)), 0)
                    hi = min(int(np.ceil(mu + 1.5 * half_win)), n_pix - 1)
                    if hi <= lo:
                        continue
                    x_data = pix_arr[lo:hi + 1]
                    y_data = spec[oi, lo:hi + 1]
                    yerr_data = spec_err[oi, lo:hi + 1]
 
                    # Reconstructed Gaussian model curve, densely sampled
                    x_model = np.linspace(x_data[0], x_data[-1], 200)
                    sigma = fwhm / sigma_scale if np.isfinite(fwhm) and fwhm > 0 else np.nan
                    if np.isfinite(sigma) and np.isfinite(amp):
                        y_model = amp * np.exp(-0.5 * ((x_model - mu) / sigma) ** 2)
                    else:
                        y_model = np.full_like(x_model, np.nan)
 
                    wave_val = np.interp(mu, pix_arr, wave[oi, :]) if wave is not None else np.nan
 
                    lines.append(dict(
                        mu=mu, mu_err=mu_err,
                        amp=amp, amp_err=amp_err,
                        fwhm=fwhm, fwhm_err=fwhm_err,
                        wave=wave_val,
                        x_data=x_data, y_data=y_data, yerr_data=yerr_data,
                        x_model=x_model, y_model=y_model,
                    ))
 
                if lines:
                    order_entries.append(dict(order=order, lines=lines, n_total=n_total))
 
            if order_entries:
                fiber_meta.append(dict(fiber=fiber, orders=order_entries))
            else:
                logger.warning(f"No valid line fits to preview for fiber {fiber}.")
 
        fig = go.Figure()
 
        if not fiber_meta:
            logger.warning("No LFC line fits available to preview.")
            fig.update_layout(title='No LFC line fits available to preview.')
            self._output['preview'] = dict(html=None, png=None, fig=fig)
            return
 
        def _line_title(fiber, order, line, line_idx, n_shown, n_total):
            wave_str = f", λ={line['wave']:.5f} µm" if np.isfinite(line['wave']) else ""
            sample_note = f" (sampled {n_shown} of {n_total} fit)" if n_total > n_shown else ""
            return (
                f"LFC Line Fit — Fiber {fiber}, Order {order}, "
                f"Line {line_idx + 1}/{n_shown}{sample_note}{wave_str}<br>"
                f"µ = {line['mu']:.3f} ± {line['mu_err']:.3f} px  |  "
                f"FWHM = {line['fwhm']:.3f} ± {line['fwhm_err']:.3f} px  |  "
                f"Amp = {line['amp']:.4g} ± {line['amp_err']:.4g}"
            )
 
        def _line_data_update(line):
            # Only 2 traces ever exist (data, fit), so these arrays map 1:1 to them.
            return {
                'x': [line['x_data'], line['x_model']],
                'y': [line['y_data'], line['y_model']],
                'error_y.array': [line['yerr_data'], line['yerr_data']],
            }
 
        # --- The only two traces in the figure; their data gets swapped on interaction ---
        first_line = fiber_meta[0]['orders'][0]['lines'][0]
        fig.add_trace(go.Scatter(
            x           = first_line['x_data'],
            y           = first_line['y_data'],
            error_y     = dict(type='data', array=first_line['yerr_data'], visible=True),
            mode        = 'markers',
            name        = 'Data',
            marker      = dict(color='steelblue', size=7),
        ))
        fig.add_trace(go.Scatter(
            x           = first_line['x_model'],
            y           = first_line['y_model'],
            mode        = 'lines',
            name        = 'Gaussian fit',
            line        = dict(color='crimson', width=2),
        ))
 
        # --- Line slider steps (scrolls through the (sampled) lines within one fiber+order) ---
        def build_line_slider_steps(f_idx, o_idx):
            meta = fiber_meta[f_idx]
            order_entry = meta['orders'][o_idx]
            n_shown = len(order_entry['lines'])
            steps = []
            for l_idx, line in enumerate(order_entry['lines']):
                steps.append(dict(
                    method = 'update',
                    label  = str(l_idx + 1),
                    args   = [
                        _line_data_update(line),
                        {'title': _line_title(meta['fiber'], order_entry['order'], line, l_idx, n_shown, order_entry['n_total'])},
                    ],
                ))
            return steps
 
        # --- Order dropdown buttons (options depend on the selected fiber) ---
        def build_order_dropdown_buttons(f_idx):
            meta = fiber_meta[f_idx]
            buttons = []
            for o_idx, order_entry in enumerate(meta['orders']):
                n_shown = len(order_entry['lines'])
                label = f"Order {order_entry['order']}"
                if order_entry['n_total'] > n_shown:
                    label += f" ({n_shown}/{order_entry['n_total']})"
                buttons.append(dict(
                    label  = label,
                    method = 'update',
                    args   = [
                        _line_data_update(order_entry['lines'][0]),
                        {
                            'title': _line_title(meta['fiber'], order_entry['order'], order_entry['lines'][0], 0, n_shown, order_entry['n_total']),
                            # Update only the line slider (leave both dropdowns untouched)
                            'sliders[0].steps':  build_line_slider_steps(f_idx, o_idx),
                            'sliders[0].active': 0,
                        },
                    ],
                ))
            return buttons
 
        # --- Fiber dropdown buttons (rewrite the order dropdown + line slider) ---
        fiber_buttons = []
        for f_idx, meta in enumerate(fiber_meta):
            first_order_entry = meta['orders'][0]
            n_shown = len(first_order_entry['lines'])
            fiber_buttons.append(dict(
                label  = f'Fiber {meta["fiber"]}',
                method = 'update',
                args   = [
                    _line_data_update(first_order_entry['lines'][0]),
                    {
                        'title': _line_title(meta['fiber'], first_order_entry['order'], first_order_entry['lines'][0], 0, n_shown, first_order_entry['n_total']),
                        # Rebuild the order dropdown's own buttons for this fiber...
                        'updatemenus[1].buttons': build_order_dropdown_buttons(f_idx),
                        'updatemenus[1].active':  0,
                        # ...and reset the line slider to this fiber's first order
                        'sliders[0].steps':  build_line_slider_steps(f_idx, 0),
                        'sliders[0].active': 0,
                    },
                ],
            ))
 
        fig.update_layout(
            title = _line_title(
                fiber_meta[0]['fiber'], fiber_meta[0]['orders'][0]['order'],
                fiber_meta[0]['orders'][0]['lines'][0], 0,
                len(fiber_meta[0]['orders'][0]['lines']), fiber_meta[0]['orders'][0]['n_total'],
            ),
            updatemenus = [
                dict(  # [0] fiber dropdown
                    buttons     = fiber_buttons,
                    direction   = 'down',
                    showactive  = True,
                    x           = 1.0,
                    xanchor     = 'right',
                    y           = 1.15,
                    yanchor     = 'top',
                    bgcolor     = '#f0f0f0',
                    bordercolor = '#cccccc',
                ),
                dict(  # [1] order dropdown — options for the initial fiber
                    buttons     = build_order_dropdown_buttons(0),
                    direction   = 'down',
                    showactive  = True,
                    x           = 1.0,
                    xanchor     = 'right',
                    y           = 1.0,
                    yanchor     = 'top',
                    bgcolor     = '#f0f0f0',
                    bordercolor = '#cccccc',
                ),
            ],
            sliders = [dict(
                active       = 0,
                currentvalue = dict(prefix='Line: ', font=dict(size=14)),
                pad          = dict(t=50),
                steps        = build_line_slider_steps(0, 0),
            )],
            xaxis         = dict(title='Pixel', showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
            yaxis         = dict(title='Flux (e-/s)', showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
            plot_bgcolor  = '#ffffff',
            paper_bgcolor = '#ffffff',
            font          = dict(color='#000000'),
            legend        = dict(bgcolor='#ffffff'),
            height        = 550,
            margin        = dict(t=140),
        )
 
        base_path        = output_model._generate_output_path(output_dir=self.output_dir, suffix=self.suffix)
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



def _drop_duplicate_lfc_modes(params_table):
    """Keep the highest-AMPLITUDE row per (ORDER, MODE_NUM) pair."""
    if len(params_table) == 0:
        return params_table
    order_col = np.asarray(params_table['ORDER'])
    mode_col = np.asarray(params_table['MODE_NUM'])
    amp_col = np.asarray(params_table['AMPLITUDE'], dtype=float)
    keys = order_col.astype(np.int64) * 1_000_000_000 + mode_col.astype(np.int64)
    by_amp = np.argsort(-amp_col, kind='stable')
    seen: set[int] = set()
    keep: list[int] = []
    for idx in by_amp:
        key = int(keys[idx])
        if key not in seen:
            seen.add(key)
            keep.append(int(idx))
    out = params_table[np.sort(keep)]
    out.sort(['ORDER', 'MU_PIX'])
    return out


def _unusable_lfc_line_fit_reason(fit, *, fwhm_bounds) -> str | None:
    """Return a skip reason for a collapsed / bound-hitting Gaussian, else None."""
    mu = fit.mu
    amp = fit.amplitude
    fwhm = fit.fwhm
    fwhm_err = fit.fwhm_err
    if not np.isfinite(mu):
        return "MU_PIX is not finite"
    if (not np.isfinite(amp)) or amp <= 0.0:
        return f"AMPLITUDE={amp} is not a positive line"
    if not np.isfinite(fwhm):
        return "FWHM_PIX is not finite"
    if fwhm_bounds is not None:
        lo, hi = float(fwhm_bounds[0]), float(fwhm_bounds[1])
        span = hi - lo
        tol = 1.0e-3 * span if span > 0.0 else 1.0e-3
        if fwhm <= lo + tol or fwhm >= hi - tol:
            return f"FWHM_PIX={fwhm} is at the fit bound {fwhm_bounds}"
    if (not np.isfinite(fwhm_err)) or fwhm_err > 1.0:
        return f"FWHM_ERR_PIX={fwhm_err} is unusable"
    return None


def gen_lfc_modes(
    nu0 : float, delta_nu : float,
    wave_min : float, wave_max : float
):
    """
    Generate LFC modes based on central frequency, frequency spacing, and wavelength bounds.

    Parameters
    ----------
    nu0 : float
        The central frequency (pump line) of the LFC in GHz.
    delta_nu : float
        The frequency spacing of the LFC modes in GHz.
    wave_min : float
        The minimum wavelength of the LFC modes in microns.
    wave_max : float
        The maximum wavelength of the LFC modes in microns.

    Returns
    -------
    waves : np.ndarray
        The wavelengths of the LFC modes in microns.
    nus : np.ndarray
        The frequencies of the LFC modes in GHz.
    integers : np.ndarray
        The integer indices of the LFC modes.
    """
    nu_end = (scipy.constants.c / wave_min) * 1E6
    nu_start = (scipy.constants.c / wave_max) * 1E6
    n1 = int(np.ceil((nu_start - nu0) / delta_nu))
    n2 = int(np.floor((nu_end - nu0) / delta_nu))
    integers = np.arange(min(n1, n2), max(n1, n2) + 1).astype(int)
    waves, nus = _gen_lfc_modes(nu0, delta_nu, integers)
    waves = waves[::-1]
    nus = nus[::-1]
    integers = integers[::-1]
    return waves, nus, integers

def _gen_lfc_modes(
    nu0 : float, delta_nu : float,
    integers : np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate LFC modes based on central frequency, frequency spacing, and integer indices.

    Parameters
    ----------
    nu0 : float
        The central frequency (pump line) of the LFC in GHz.
    delta_nu : float
        The frequency spacing of the LFC modes in GHz.
    integers : np.ndarray
        An array of integer indices representing the mode numbers.

    Returns
    -------
    waves : np.ndarray
        The wavelengths of the LFC modes in microns.
    nus : np.ndarray
        The frequencies of the LFC modes in GHz.
    """
    nus = integers * delta_nu + nu0
    waves = scipy.constants.c / nus * 1E6
    return waves, nus

def _get_closest_lfc_mode(
    nu0 : float, delta_nu : float,
    wave : float
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Get the closest LFC mode to a given wavelength value.

    Parameters
    ----------
    nu0 : float
        The central frequency (pump line) of the LFC in Hz.
    delta_nu : float
        The frequency spacing of the LFC modes in Hz.
    wave : float
        The wavelength value in microns.

    Returns
    -------
    np.ndarray
        The closest wavelength of the LFC mode in microns.
    np.ndarray
        The corresponding frequency of the LFC mode in GHz.
    int
        The integer index of the closest LFC mode.
    """
    nu = scipy.constants.c / wave * 1E6
    n = int(np.rint((nu - nu0) / delta_nu))
    nu_mode = nu0 + n * delta_nu
    wave_mode = scipy.constants.c / nu_mode * 1E6
    return wave_mode, nu_mode, n
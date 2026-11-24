from ... import datamodels
from ..base_primitive import BasePrimitive
import logging

from ..utils.spectral_utils import delta_wave_to_delta_vel
from ..utils.array_utils import auto_deg_inter_order, solve_chebyshev2d_ols
from ..line_fitting.fit_lines import LINE_DQ_FLAGS
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

__all__ = ['CalculateChebyshev2DWLSPrimitive', 'wavesol_norm_bounds']

WAVE_MODEL_CHEBYSHEV2D = "chebyshev2d"
WAVE_MODEL_CUBIC_SPLINE_1D = "cubic_spline_1d"
WAVE_MODELS = (WAVE_MODEL_CHEBYSHEV2D, WAVE_MODEL_CUBIC_SPLINE_1D)
DEFAULT_DEG_INTRA_ORDER = 6


# ---------------------------------------------------------------------------
# Per-source configuration: the only place source-specific names/defaults live.
# ---------------------------------------------------------------------------
# input_model_class : name of the input datamodel class in `datamodels` (used for source detection)
# table_prefix      : input table attribute is f'{table_prefix}_fitparams_fib{fiber}'
# model_class       : name of the output datamodel class in `datamodels`
# defaults          : fit settings used when the user leaves a parameter as None.
#                     LFC values match the previous primitive defaults. Etalon and
#                     uranium values are PLACEHOLDERS: tune on real data.
SOURCE_CONFIG = {
    'lfc': dict(
        input_model_class='LFCFitParamsModel',  
        table_prefix='lfc',
        model_class='LFCWavesolModel',
        defaults=dict(wave_model=WAVE_MODEL_CHEBYSHEV2D, deg_intra_order=DEFAULT_DEG_INTRA_ORDER,
                      deg_inter_order='auto', max_iters=100, sigma_cut=4, max_vel_cut=1E3,
                      opt_weights=True),
    ),
    'etalon': dict(
        input_model_class='EtalonRefModel',  
        table_prefix='etalon',
        model_class='EtalonWavesolModel',
        defaults=dict(wave_model=WAVE_MODEL_CHEBYSHEV2D, deg_intra_order=DEFAULT_DEG_INTRA_ORDER,
                      deg_inter_order='auto', max_iters=100, sigma_cut=4, max_vel_cut=1E3,
                      opt_weights=True),
    ),
    'uranium': dict(
        input_model_class='UraniumFitParamsModel', 
        table_prefix='uranium',
        model_class='UraniumWavesolModel',
        # m*lambda varies more along an order than across orders, so intra > inter
        # ('auto' would give inter=10 for 39 orders). Outliers are removed by sigma_cut alone.
        defaults=dict(wave_model=WAVE_MODEL_CHEBYSHEV2D, deg_intra_order=5,
                      deg_inter_order=4, max_iters=100, sigma_cut=4, max_vel_cut=None,
                      opt_weights=True),
    ),
}


def wavesol_norm_bounds(Ncols : int, order_vec) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Normalization bounds of the 2D Chebyshev wavelength solution, fixed by the detector and the model's orders.

    The pixel range (0, Ncols - 1) is padded by 10 and the full order range of the model (``make_order_vec``)
    by 1 on each side, so the bounds do not depend on which orders or lines are fit and are not stored.

    Returns
    -------
    pixel_bounds, order_bounds : tuple[float, float]
    """
    return (-10.0, Ncols - 1 + 10.0), (float(np.min(order_vec)) - 1.0, float(np.max(order_vec)) + 1.0)


class CalculateChebyshev2DWLSPrimitive(BasePrimitive):

    """
    Fit a wavelength solution from fitted line centroids (LFC, etalon, or uranium lamp).

    The calibration source is autodetected from the class of the input datamodel. The source sets the input table, the output datamodel, and the default fit settings.

    The default ``wave_model='chebyshev2d'`` fits a 2D Chebyshev polynomial
    to ``{pixel, order, m*λ}`` (a grating-equation expansion).
    ``wave_model='cubic_spline_1d'`` instead interpolates each order with a
    cubic spline through ``(MU_PIX, WAVE)``. The spline is optional and is
    not the default.

    Any parameter left as None uses the default for the detected source (see SOURCE_CONFIG).

    Parameters
    ----------
    source : {'lfc', 'etalon', 'uranium'} or None
        Calibration source. If None, autodetected from the input model class. Set explicitly to override.
    wave_model : {'chebyshev2d', 'cubic_spline_1d'} or None
        Wavelength model used to fill ``wave_fib*``.
    deg_intra_order : int, 'auto' or None
        Degree of the polynomial within an order (Chebyshev model only).
        ``'auto'`` uses :data:`DEFAULT_DEG_INTRA_ORDER`.
    deg_inter_order : int, 'auto' or None
        Degree of the polynomial between orders. If 'auto', it will be set based on the number of orders being fit.
    max_iters : int or None
        Maximum number of iterations for the 2D fitting process.
    sigma_cut : float or None
        Lines whose residual, divided by its error, is more than ``sigma_cut`` times the robust
        scatter (1.4826 x MAD) of those normalized residuals are dropped in the next iteration.
        Use float('inf') to disable.
    max_vel_cut : float or None
        Lines with |residual| above this many m/s are also dropped. None or inf disables it.
    opt_weights : bool or None
        Weight each line by 1 / (m * MU_ERR_WAVE)**2, i.e. by its own centroid error. If False,
        uniform weights are used.
    debug : bool
        If True, run self-consistency checks after each fiber's fit and log the results (see
        ``_debug_check_wavesol``). The results are also stored in ``fit_result['debug']``.
        When the input wavelength solution is (near) exact, as in simulations, the
        "final vs. input" numbers measure the accuracy of the returned solution.
    fit_orders : list of int or None
        List of orders to fit. If None, all orders will be fit.
    do_fibers : list of int or None
        List of fibers to fit. If None, all fibers will be fit (separately in series).
    reject_flags : list of str or None
        List of LINE_DQ flags (e.g. ['DO_NOT_USE', 'FIT_FAILURE', 'POOR_FIT', 'LOW_SNR']), which will be ignored in line fitting. Rekects all if set to None. Use empty list for no rejection.

    Returns
    -------
    output : datamodel (type depends on source)
        The wavelength solution model.

    """

    # None means "use the per-source default from SOURCE_CONFIG"
    PARAMETERS = {
        'source' : None,
        'wave_model' : None,
        'deg_intra_order' : None,
        'deg_inter_order' : None,
        'max_iters' : None,
        'sigma_cut' : None,
        'max_vel_cut' : None,
        'opt_weights' : None,
        'fit_orders' : None,
        'do_fibers' : None,
        'reject_flags': None,
        'debug' : False,
    }

    def _detect_source(self, input_model):
        """Return the source name, from the user override or the input model's class."""
        if self.source is not None:
            if self.source not in SOURCE_CONFIG:
                raise ValueError(f"Unknown source {self.source!r}; options: {list(SOURCE_CONFIG)}")
            return self.source

        found = []
        for name, cfg in SOURCE_CONFIG.items():
            cls = getattr(datamodels, cfg['input_model_class'], None)
            if cls is not None and isinstance(input_model, cls):
                found.append(name)

        if len(found) == 1:
            return found[0]
        if not found:
            raise ValueError(
                f"Could not detect a calibration source from input model type "
                f"{type(input_model).__name__}. Expected one of "
                f"{[c['input_model_class'] for c in SOURCE_CONFIG.values()]}, or set the 'source' parameter."
            )
        raise ValueError(f"Input model matches several sources {found}. Set the 'source' parameter.")

    def _perform(self):

        # Open the input model
        input_model = datamodels.open(self.dataset)

        # Get fibers and orders
        do_fibers, fit_orders = self.get_do_traces(input_model, self.do_fibers, self.fit_orders)

        # Detect source and resolve settings (user value wins, else source default)
        source = self._detect_source(input_model)
        cfg = SOURCE_CONFIG[source]
        s = {k: (getattr(self, k) if getattr(self, k) is not None else v) for k, v in cfg['defaults'].items()}

        wave_model = str(s['wave_model']).lower()
        if wave_model not in WAVE_MODELS:
            raise ValueError(
                f"wave_model={s['wave_model']!r} is invalid; expected one of {WAVE_MODELS}"
            )

        if s['deg_intra_order'] == 'auto':
            deg_intra_order = DEFAULT_DEG_INTRA_ORDER
            logger.info(f"deg_intra_order='auto', using {deg_intra_order=}.")
        else:
            deg_intra_order = int(s['deg_intra_order'])

        # Inter order deg (Chebyshev model only)
        deg_inter_order = s['deg_inter_order']
        if wave_model == WAVE_MODEL_CHEBYSHEV2D:
            if deg_inter_order == 'auto':
                deg_inter_order = auto_deg_inter_order(len(fit_orders))
                logger.info(f"deg_inter_order='auto', using {deg_inter_order=}.")
            deg_inter_order = int(deg_inter_order)
            max_allowed = max(len(fit_orders) - 1, 0)
            if deg_inter_order > max_allowed:
                logger.warning(
                    f"deg_inter_order={deg_inter_order} exceeds what {len(fit_orders)} order(s) "
                    f"can constrain; using {max_allowed}."
                )
                deg_inter_order = max_allowed

        logger.info(f"Detected source '{source}', {wave_model=}, {deg_intra_order=}, {deg_inter_order=}, "
                    f"max_iters={s['max_iters']}, sigma_cut={s['sigma_cut']}, max_vel_cut={s['max_vel_cut']}.")

        fit_results = []
        line_data = {}  # per fiber: every line in the fit orders with its residual, for the preview

        # lines to flag out
        reject_mask = np.uint8(0)
        if self.reject_flags is None:
            reject_flags = [name for name in LINE_DQ_FLAGS if name != 'GOOD']
            self.reject_flags = reject_flags
        for flag in self.reject_flags:
            reject_mask |= LINE_DQ_FLAGS[flag]

        # Output model type depends on the source
        model_cls = getattr(datamodels, cfg['model_class'])
        output_model = model_cls.from_datamodels(
            input_model, include_data=True, primitive=self
        )
        if hasattr(output_model.meta, "wave_model"):
            output_model.meta.wave_model = wave_model

        for fiber in do_fibers:

            # Get the table for the fiber
            params_table = getattr(input_model, f"{cfg['table_prefix']}_fitparams_fib{fiber}")

            # Setup input vectors
            pixels = np.asarray(params_table['MU_PIX'], dtype=float)
            orders = np.asarray(params_table['ORDER'], dtype=float)
            waves = np.asarray(params_table['WAVE'], dtype=float)
            try:
                line_dq = np.asarray(params_table['LINE_DQ'], dtype=np.uint8)
            except (KeyError, ValueError):
                logger.warning("LINE_DQ column not found; no lines will be rejected by flag.")
                line_dq = np.zeros(len(params_table), dtype=np.uint8)
            mask = np.isin(orders, fit_orders) & ((line_dq & reject_mask) == 0)
            pixels = pixels[mask]
            orders = orders[mask]
            waves = waves[mask]
            if s['opt_weights']:
                # The fit is in m*lambda, so the error on each point is m * sigma_lambda.
                # Lines without a valid error get weight NaN and are left out of the fit.
                err_wave = np.asarray(params_table['MU_ERR_WAVE'], dtype=float)[mask]
                with np.errstate(divide='ignore', invalid='ignore'):
                    weights = 1 / (orders * err_wave)**2
            else:
                weights = None

            nx = getattr(output_model, f'spec_fib{fiber}').shape[1]
            pixel_vec = np.arange(nx)
            order_vec = output_model.make_order_vec()

            if wave_model == WAVE_MODEL_CUBIC_SPLINE_1D:
                fit_result = fit_wavesol_spline_1d(
                    pixels, orders, waves, pixel_vec, order_vec, weights=weights
                )
                wavesol_full = fit_result["wave_grid"]
            else:
                pixel_bounds, order_bounds = wavesol_norm_bounds(nx, order_vec)
                fit_result = fit_wavesol_chebyshev2d(
                    pixels, orders, waves,
                    deg_intra_order=deg_intra_order, deg_inter_order=deg_inter_order,
                    pixel_bounds=pixel_bounds, order_bounds=order_bounds,
                    max_iters=s['max_iters'], sigma_cut=s['sigma_cut'], max_vel_cut=s['max_vel_cut'],
                    weights=weights,
                )
                if fit_result["coeffs"] is not None:
                    setattr(output_model, f'cheb_coeffs_fib{fiber}', fit_result['coeffs'])
                wavesol_full = _build_wavesol_poly2d_structured(
                    fit_result['coeffs'],
                    pixel_vec,
                    order_vec,
                    fit_result['pixel_bounds'], fit_result['order_bounds']
                )
            fit_results.append(fit_result)

            # Residuals of every line in the fit orders (kept, clipped and flag-rejected) against
            # the final solution, for the preview.
            line_data[fiber] = _collect_preview_lines(
                params_table, mask, fit_result, wavesol_full, np.asarray(order_vec), fit_orders,
            )

            if self.debug and wave_model != WAVE_MODEL_CUBIC_SPLINE_1D and fit_result['coeffs'] is not None:
                # Must run before wave_fib is overwritten: it compares against the input solution.
                fit_result['debug'] = _debug_check_wavesol(
                    fiber=fiber,
                    pixels=pixels, orders=orders, waves=waves, weights=weights,
                    fit_result=fit_result, wavesol_full=wavesol_full,
                    wave_input=getattr(output_model, f'wave_fib{fiber}', None),
                    order_vec=np.asarray(order_vec), deg_intra_order=deg_intra_order,
                    deg_inter_order=deg_inter_order,
                )

            setattr(output_model, f'wave_fib{fiber}', wavesol_full)

            # Set unreliable WLS in DQ
            bad_orders = np.setdiff1d(order_vec, fit_orders)
            # Copy: flagging must not mutate the input model's DQ plane.
            dq = np.array(getattr(output_model, f'spec_dq_fib{fiber}'), copy=True)
            flag_value = datamodels.DQ_FLAGS['UNRELIABLE_WS']
            for order in bad_orders:
                oi = np.abs(order - output_model.meta.order_start)
                dq[oi, :] |= flag_value

            # fit_result['good'] indexes the masked (fit_orders only) arrays, so apply the same mask first
            params_table_good = params_table[mask][fit_result['good']]
            for order in fit_orders:
                order_mask = params_table_good['ORDER'] == order
                if np.sum(order_mask) > 0:
                    oi = np.abs(order - output_model.meta.order_start)
                    line_centers_pix = params_table_good['MU_PIX'][order_mask]
                    k1 = int(np.floor(np.min(line_centers_pix)))
                    k2 = int(np.ceil(np.max(line_centers_pix)))
                    if k1 > 0:
                        dq[oi, :k1] |= flag_value
                    if k2 < nx - 1:
                        dq[oi, k2+1:] |= flag_value
                    if wave_model == WAVE_MODEL_CUBIC_SPLINE_1D:
                        # Spline WAVE in the first / last knot interval
                        # rests on a boundary segment of the spline and
                        # shows the largest interpolation error; flag it
                        # as unreliable like the out-of-range pixels.
                        centers = np.sort(np.asarray(line_centers_pix, dtype=float))
                        if centers.size >= 2:
                            b1 = int(np.floor(centers[1])) + 1
                            dq[oi, k1:b1] |= flag_value
                            b2 = int(np.ceil(centers[-2]))
                            dq[oi, b2:k2 + 1] |= flag_value
            setattr(output_model, f'spec_dq_fib{fiber}', dq)

        return dict(output=output_model, fit_results=fit_results, source=source, line_data=line_data)


    def _create_preview(self):
        output_model = self._output['output']
        line_data = self._output.get('line_data') or {}
        if not line_data:
            logger.warning("No line data available; skipping wavelength solution preview.")
            return

        source = self._output.get('source') or ''
        src_label = {'lfc': 'LFC', 'etalon': 'Etalon', 'uranium': 'Uranium'}.get(source, source)
        filename = output_model.meta.get('filename', 'Wavelength solution')
        wave_model = getattr(output_model.meta, 'wave_model', None) or ''

        fibers = list(line_data)
        orders_by_fiber = {f: [int(o) for o in line_data[f]['fit_orders']] for f in fibers}

        # Trace layout. Per fiber: 2 summary traces (bottom panel: per-order RMS bars and mean
        # offset markers), then per order: kept, clipped, flag-rejected, ±RMS band, highlight.
        n_summary = 2
        traces_per_order = 5
        fiber_start, n_total = [], 0
        for f in fibers:
            fiber_start.append(n_total)
            n_total += n_summary + traces_per_order * len(orders_by_fiber[f])

        styles = {
            _STATUS_KEPT:    dict(label='Used in fit',       color='royalblue',  symbol='circle',   size=6),
            _STATUS_CLIPPED: dict(label='Clipped by fit',    color='darkorange', symbol='x',        size=7),
            _STATUS_FLAGGED: dict(label='Rejected (LINE_DQ)', color='lightgray',  symbol='circle-open', size=6),
        }

        hovertemplate = (
            'pixel = %{x:.2f}<br>'
            'Δv = %{y:.1f} m/s<br>'
            'Δ = %{customdata[1]:+.4f} px<br>'
            'λ = %{customdata[0]:.6f} µm<br>'
            'σ(centroid) = %{customdata[2]:.1f} m/s<br>'
            'LINE_DQ = %{customdata[3]:d}'
            '<extra>%{meta}</extra>'
        )

        # Per (fiber, order) statistics, titles and y-ranges, computed once
        stats = {}
        fiber_rms = {}
        for f in fibers:
            d = line_data[f]
            kept_all = d['resid_vel'][(d['status'] == _STATUS_KEPT) & np.isfinite(d['resid_vel'])]
            fiber_rms[f] = float(np.sqrt(np.mean(kept_all**2))) if kept_all.size else np.nan
            for order in orders_by_fiber[f]:
                in_o = (d['order'] == order) & np.isfinite(d['resid_vel'])
                r_kept = d['resid_vel'][in_o & (d['status'] == _STATUS_KEPT)]
                rms = float(np.sqrt(np.mean(r_kept**2))) if r_kept.size else np.nan
                mean = float(np.mean(r_kept)) if r_kept.size else np.nan
                # y-range from the kept lines, so a few wild outliers do not flatten the plot
                ref = r_kept if r_kept.size else d['resid_vel'][in_o]
                if ref.size:
                    lim = 1.25 * max(float(np.max(np.abs(ref))), 3 * (rms if np.isfinite(rms) else 0.0), 1.0)
                else:
                    lim = 10.0
                n_off = int(np.sum(np.abs(d['resid_vel'][in_o]) > lim))
                stats[(f, order)] = dict(
                    rms=rms, mean=mean, lim=lim, n_off=n_off,
                    n=[int(np.sum(in_o & (d['status'] == s))) for s in (_STATUS_KEPT, _STATUS_CLIPPED, _STATUS_FLAGGED)],
                )

        def make_title(fiber, order):
            st = stats[(fiber, order)]
            head = f'{src_label} wavelength solution residuals — Fiber {fiber}, Order {order}  |  {filename}'
            sub = (f"order: {st['n'][0]} lines used, {st['n'][1]} clipped, {st['n'][2]} rejected by flag; "
                   f"RMS {st['rms']:.1f} m/s, mean {st['mean']:+.1f} m/s  |  "
                   f"fiber RMS {fiber_rms[fiber]:.1f} m/s")
            if wave_model:
                sub += f"  |  model: {wave_model}"
            if st['n_off']:
                sub += f"  |  {st['n_off']} point(s) beyond the y-range (double-click to autoscale)"
            return f'{head}<br><sup>{sub}</sup>'

        def make_yrange(fiber, order):
            lim = stats[(fiber, order)]['lim']
            return [-lim, lim]

        fig = make_subplots(
            rows=2, cols=1, row_heights=[0.68, 0.32], vertical_spacing=0.13,
            subplot_titles=('Line residuals along the order (λ_line − λ_solution)',
                            'Per-order summary of lines used in the fit'),
        )

        for f_idx, fiber in enumerate(fibers):
            d = line_data[fiber]
            orders = orders_by_fiber[fiber]
            first_fiber = f_idx == 0

            # --- bottom panel: per-order summary for this fiber (visible for every order) ---
            rms_arr = np.array([stats[(fiber, o)]['rms'] for o in orders])
            mean_arr = np.array([stats[(fiber, o)]['mean'] for o in orders])
            counts = np.array([stats[(fiber, o)]['n'] for o in orders]).reshape(-1, 3)
            fig.add_trace(go.Bar(
                x=orders, y=rms_arr, name='RMS per order',
                marker=dict(color='rgba(65,105,225,0.45)'),
                customdata=counts,
                hovertemplate=('Order %{x}<br>RMS = %{y:.1f} m/s<br>'
                               'used %{customdata[0]}, clipped %{customdata[1]}, '
                               'rejected %{customdata[2]}<extra></extra>'),
                legendgroup='summary', visible=first_fiber,
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=orders, y=mean_arr, mode='markers', name='Mean offset per order',
                marker=dict(color='black', size=6, symbol='diamond'),
                hovertemplate='Order %{x}<br>mean = %{y:+.1f} m/s<extra></extra>',
                legendgroup='summary', visible=first_fiber,
            ), row=2, col=1)

            # --- top panel: one set of traces per order ---
            for o_idx, order in enumerate(orders):
                visible = first_fiber and o_idx == 0
                in_o = (d['order'] == order) & np.isfinite(d['resid_vel'])

                for status in (_STATUS_FLAGGED, _STATUS_CLIPPED, _STATUS_KEPT):
                    sty = styles[status]
                    sel = in_o & (d['status'] == status)
                    customdata = np.column_stack([
                        d['wave'][sel], d['resid_px'][sel], d['err_vel'][sel], d['line_dq'][sel],
                    ])
                    fig.add_trace(go.Scatter(
                        x=d['pixel'][sel], y=d['resid_vel'][sel],
                        mode='markers',
                        name=f"{sty['label']} ({int(sel.sum())})",
                        marker=dict(color=sty['color'], symbol=sty['symbol'], size=sty['size'],
                                    line=dict(width=1, color=sty['color'])),
                        error_y=dict(type='data', array=d['err_vel'][sel], visible=status == _STATUS_KEPT,
                                     thickness=0.8, width=0, color='rgba(65,105,225,0.5)'),
                        customdata=customdata, meta=sty['label'],
                        hovertemplate=hovertemplate,
                        legendgroup=sty['label'], visible=visible,
                    ), row=1, col=1)

                # ±RMS of the kept lines, as two dashed segments in one trace
                rms = stats[(fiber, order)]['rms']
                xs, ys = ([0, d['nx'] - 1, np.nan, 0, d['nx'] - 1],
                          [rms, rms, np.nan, -rms, -rms]) if np.isfinite(rms) else ([], [])
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode='lines', name='±RMS (order)',
                    line=dict(color='royalblue', width=1, dash='dash'),
                    hoverinfo='skip', legendgroup='rms', visible=visible,
                ), row=1, col=1)

                # Highlight the current order in the summary panel
                fig.add_trace(go.Scatter(
                    x=[order], y=[rms if np.isfinite(rms) else 0.0], mode='markers',
                    name='Current order', showlegend=False,
                    marker=dict(color='rgba(0,0,0,0)', size=16, symbol='circle-open',
                                line=dict(color='red', width=2)),
                    hoverinfo='skip', visible=visible,
                ), row=2, col=1)

        def make_visibility(f_idx, o_idx):
            vis = [False] * n_total
            base = fiber_start[f_idx]
            for t in range(n_summary):
                vis[base + t] = True
            ob = base + n_summary + o_idx * traces_per_order
            for t in range(traces_per_order):
                vis[ob + t] = True
            return vis

        def make_relayout(fiber, order):
            return {'title.text': make_title(fiber, order), 'yaxis.range': make_yrange(fiber, order)}

        def make_slider(f_idx):
            fiber = fibers[f_idx]
            return dict(
                active=0,
                currentvalue=dict(prefix='Order: ', font=dict(size=14)),
                pad=dict(t=50),
                steps=[
                    dict(method='update', label=str(order),
                         args=[{'visible': make_visibility(f_idx, o_idx)}, make_relayout(fiber, order)])
                    for o_idx, order in enumerate(orders_by_fiber[fiber])
                ],
            )

        dropdown_buttons = [
            dict(
                label=f'Fiber {fiber}', method='update',
                args=[
                    {'visible': make_visibility(f_idx, 0)},
                    {**make_relayout(fiber, orders_by_fiber[fiber][0]), 'sliders': [make_slider(f_idx)]},
                ],
            )
            for f_idx, fiber in enumerate(fibers)
        ]

        f0, o0 = fibers[0], orders_by_fiber[fibers[0]][0]
        grid = dict(showgrid=True, gridcolor='rgba(200,200,200,0.3)')
        fig.add_hline(y=0, line=dict(color='black', width=1), row=1, col=1)
        fig.add_hline(y=0, line=dict(color='gray', width=1, dash='dot'), row=2, col=1)
        fig.update_layout(
            title=dict(text=make_title(f0, o0)),
            updatemenus=[dict(
                buttons=dropdown_buttons, direction='down', showactive=True,
                x=1.0, xanchor='right', y=1.17, yanchor='top',
                bgcolor='#f0f0f0', bordercolor='#cccccc',
            )] if len(fibers) > 1 else [],
            sliders=[make_slider(0)],
            xaxis=dict(title='Pixel', range=[0, line_data[f0]['nx'] - 1], **grid),
            yaxis=dict(title='Δv (m/s)', range=make_yrange(f0, o0), zeroline=False, **grid),
            xaxis2=dict(title='Order', dtick=1 if len(orders_by_fiber[f0]) <= 50 else None, **grid),
            yaxis2=dict(title='m/s', **grid),
            bargap=0.15,
            hovermode='closest',
            plot_bgcolor='#ffffff',
            paper_bgcolor='#ffffff',
            font=dict(color='#000000'),
            legend=dict(bgcolor='#ffffff'),
            margin=dict(t=110),
            height=750,
        )

        suffix = getattr(self, 'suffix', None) or 'wavesol'
        base_path = output_model._generate_output_path(output_dir=self.output_dir, suffix=suffix)
        output_path_html = base_path.replace('.fits', '_preview.html')
        output_path_png = base_path.replace('.fits', '_preview.png')

        if self.save_preview:
            fig.write_html(output_path_html, include_plotlyjs='cdn')
            fig.write_image(output_path_png, scale=2)

        if self.show_preview:
            fig.show()

        self._output['preview'] = {
            'html': output_path_html,
            'png': output_path_png,
            'fig': fig,
        }


# Line status codes used by the preview
_STATUS_KEPT, _STATUS_CLIPPED, _STATUS_FLAGGED = 0, 1, 2


def _filled_float(col):
    """Table column as a float ndarray with masked entries set to NaN."""
    return np.asarray(np.ma.filled(np.ma.asarray(col, dtype=float), np.nan), dtype=float)


def _collect_preview_lines(params_table, mask, fit_result, wavesol_full, order_vec, fit_orders):
    """Residual of every line in the fit orders against the final solution, for the preview.

    ``mask`` selects the rows passed to the fit (fit orders, not rejected by LINE_DQ) and
    ``fit_result['good']`` indexes those rows. Each returned line gets a status: kept in the
    fit, passed to the fit but clipped, or rejected by LINE_DQ before the fit. Residuals of
    rejected lines are evaluated from the final model (exactly for Chebyshev, by interpolating
    ``wavesol_full`` for the spline).
    """
    colnames = params_table.colnames
    pix = _filled_float(params_table['MU_PIX'])
    ords = _filled_float(params_table['ORDER'])
    waves = _filled_float(params_table['WAVE'])
    n = len(pix)
    line_dq = (np.asarray(np.ma.filled(params_table['LINE_DQ'], 0), dtype=int)
               if 'LINE_DQ' in colnames else np.zeros(n, dtype=int))
    err_wave = _filled_float(params_table['MU_ERR_WAVE']) if 'MU_ERR_WAVE' in colnames else np.full(n, np.nan)

    status = np.full(n, _STATUS_FLAGGED, dtype=np.int8)
    idx_fit = np.flatnonzero(mask)
    status[idx_fit] = _STATUS_CLIPPED
    status[idx_fit[np.asarray(fit_result['good'], dtype=int)]] = _STATUS_KEPT

    use = np.isin(ords, np.asarray(fit_orders, dtype=float)) & np.isfinite(pix) & np.isfinite(waves)
    nx = wavesol_full.shape[1]
    xg = np.arange(nx, dtype=float)
    row_of = {float(m): i for i, m in enumerate(order_vec)}

    model = np.full(n, np.nan)
    disp = np.full(n, np.nan)
    coeffs = fit_result.get('coeffs')
    if coeffs is not None:
        sel = use & (ords != 0)
        model[sel] = _build_wavesol_poly2d_unstructured(
            coeffs, pix[sel], ords[sel], fit_result['pixel_bounds'], fit_result['order_bounds'])
    for m in np.unique(ords[use]):
        r = row_of.get(float(m))
        if r is None:
            continue
        sel = use & (ords == m)
        row = wavesol_full[r]
        fin = np.isfinite(row)
        if fin.sum() < 2:
            continue
        disp[sel] = np.interp(pix[sel], xg, np.gradient(row))
        if coeffs is None:
            model[sel] = np.interp(pix[sel], xg[fin], row[fin], left=np.nan, right=np.nan)
    # Lines that went into the fit: use the fit's own model values (identical for Chebyshev)
    model[idx_fit] = np.asarray(fit_result['model'], dtype=float)

    resid_wave = waves - model
    resid_vel = delta_wave_to_delta_vel(resid_wave, waves)
    with np.errstate(divide='ignore', invalid='ignore'):
        resid_px = resid_wave / disp
    err_vel = np.abs(delta_wave_to_delta_vel(err_wave, waves))

    return dict(
        order=ords[use].astype(int), pixel=pix[use], wave=waves[use],
        resid_vel=np.asarray(resid_vel, dtype=float)[use], resid_px=resid_px[use],
        err_vel=np.asarray(err_vel, dtype=float)[use], line_dq=line_dq[use],
        status=status[use], fit_orders=[int(o) for o in fit_orders], nx=nx,
    )


def _normalize_val(x, lo, hi):
    mu = (lo + hi) / 2
    s = hi - lo
    if s == 0:
        return np.zeros_like(np.asarray(x, dtype=float))
    return 2 * (x - mu) / s

def fit_wavesol_spline_1d(
    pixels: np.ndarray,
    orders: np.ndarray,
    waves: np.ndarray,
    pixel_vec: np.ndarray,
    order_vec: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> dict:
    """Per-order cubic spline through fitted line ``(MU_PIX, WAVE)``."""
    from scipy.interpolate import CubicSpline

    pixels = np.asarray(pixels, dtype=float)
    orders = np.asarray(orders, dtype=int)
    waves = np.asarray(waves, dtype=float)
    pixel_vec = np.asarray(pixel_vec, dtype=float)
    order_vec = np.asarray(order_vec, dtype=int)
    if weights is None:
        weights = np.ones(pixels.shape, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    wave_grid = np.full((order_vec.size, pixel_vec.size), np.nan, dtype=float)
    model_at_lines = np.full(pixels.shape, np.nan, dtype=float)
    good_mask = np.zeros(pixels.size, dtype=bool)

    for i, od in enumerate(order_vec):
        sel = (
            (orders == int(od))
            & np.isfinite(pixels)
            & np.isfinite(waves)
            & np.isfinite(weights)
            & (weights > 0)
        )
        n_sel = int(sel.sum())
        if n_sel < 4:
            logger.warning(
                "Order %s has %s usable lines; need 4 for a cubic spline.",
                od,
                n_sel,
            )
            continue
        x = pixels[sel]
        y = waves[sel]
        order = np.argsort(x)
        x, y = x[order], y[order]
        _, uniq = np.unique(np.round(x, 6), return_index=True)
        x, y = x[uniq], y[uniq]
        if x.size < 4:
            logger.warning(
                "Order %s has %s unique line pixels; need 4 for a cubic spline.",
                od,
                x.size,
            )
            continue
        fn = CubicSpline(x, y, extrapolate=False)
        inside = (pixel_vec >= x[0]) & (pixel_vec <= x[-1])
        wave_grid[i, inside] = fn(pixel_vec[inside])
        at_x = fn(pixels[sel])
        model_at_lines[sel] = at_x
        good_mask[sel] = np.isfinite(at_x)

    residuals = waves - model_at_lines
    residuals_vel = delta_wave_to_delta_vel(residuals, waves)
    if pixels.size == 0:
        pixel_bounds = (0.0, 1.0)
        order_bounds = (0.0, 1.0)
    else:
        pixel_bounds = (float(np.nanmin(pixels)) - 10.0, float(np.nanmax(pixels)) + 10.0)
        order_bounds = (float(np.nanmin(orders)) - 1.0, float(np.nanmax(orders)) + 1.0)
    return dict(
        coeffs=None,
        good=np.where(good_mask)[0],
        iterations=1,
        pixels=pixels,
        wave=waves,
        model=model_at_lines,
        residuals=residuals,
        residuals_vel=residuals_vel,
        order_bounds=order_bounds,
        pixel_bounds=pixel_bounds,
        wave_grid=wave_grid,
        wave_model=WAVE_MODEL_CUBIC_SPLINE_1D,
    )


def fit_wavesol_chebyshev2d(
    pixels : np.ndarray,
    orders : np.ndarray,
    waves : np.ndarray,
    deg_intra_order : int,
    deg_inter_order : int,
    pixel_bounds : tuple[float] | None = None,
    order_bounds : tuple[float] | None = None,
    max_iters : int | None = None,
    sigma_cut : float | None = None,
    max_vel_cut : float | None = None,
    weights : np.ndarray | None = None,
) -> dict:
    """
    Fit a 2D Chebyshev polynomial to the wavelength solution data.

    Args:
        pixels (np.ndarray): Pixel values.
        orders (np.ndarray): Echelle order values.
        waves (np.ndarray): Wavelength values.
        deg_intra_order (int): Degree of the polynomial within an order.
        deg_inter_order (int): Degree of the polynomial between orders.
        pixel_bounds (tuple[float] | None): Bounds for pixel values for normalization of the data.
        order_bounds (tuple[float] | None): Bounds for order values for normalization of the data.
        max_iters (int | None): Maximum number of iterations (None means a single fit).
        sigma_cut (float | None): Cutoff for the residuals in terms of standard deviation.
        max_vel_cut (float | None): Maximum velocity cutoff for the residuals.
        weights (np.ndarray | None): Fit weights. Defaults to uniform weights.

    Returns:
        dict : Dictionary containing the results of the fit.
    """

    # Copy to not modify original data
    pixels = np.copy(pixels)
    orders = np.copy(orders)
    waves = np.copy(waves)
    if weights is None:
        weights = np.ones(pixels.shape)
    else:
        weights = np.copy(weights)
    if max_iters is None:
        max_iters = 1

    if pixel_bounds is None:
        pixel_bounds = (np.nanmin(pixels) - 10, np.nanmax(pixels) + 10)
    if order_bounds is None:
        order_bounds = (np.nanmin(orders) - 1, np.nanmax(orders) + 1)

    # Get normalized vals
    pixels_norm = _normalize_val(pixels, *pixel_bounds)
    orders_norm = _normalize_val(orders, *order_bounds)

    # Actual vals to fit
    orders_times_waves = orders * waves

    # Good vals
    usable = np.isfinite(weights) & (weights > 0) & np.isfinite(pixels) & np.isfinite(waves) & np.isfinite(orders)
    good = np.flatnonzero(usable)
    coeffs = None
    n_coeffs = (deg_intra_order + 1) * (deg_inter_order + 1)

    # Iteratively fit
    for i in range(max_iters):

        if len(good) < n_coeffs:
            raise RuntimeError(
                f"Only {len(good)} usable lines for {n_coeffs} coefficients "
                f"({deg_intra_order=}, {deg_inter_order=}). Lower the degrees or loosen the clipping."
            )

        # Fit
        x = pixels_norm[good]
        y = orders_norm[good]
        z = orders_times_waves[good]
        w = weights[good]
        coeffs = solve_chebyshev2d_ols(x, y, z, deg_x=deg_intra_order, deg_y=deg_inter_order, weights=w)

        # Model and residuals
        model_flat = _build_wavesol_poly2d_unstructured(coeffs, pixels, orders, pixel_bounds, order_bounds)
        residuals = waves - model_flat
        residuals_vel = delta_wave_to_delta_vel(residuals, waves)

        # Clipping. Residuals are compared with each line's own error (the fit residual in
        # m*lambda times sqrt(weight)), and the threshold uses a robust scatter of the lines
        # currently in the fit, so a few outliers cannot inflate it.
        keep = usable & np.isfinite(residuals_vel)
        if sigma_cut is not None and np.isfinite(sigma_cut):
            norm_resid = orders * residuals * np.sqrt(weights)
            r = norm_resid[good]
            robust_sigma = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r)))
            if robust_sigma > 0:
                keep &= np.abs(norm_resid) < sigma_cut * robust_sigma
        if max_vel_cut is not None and np.isfinite(max_vel_cut):
            keep &= np.abs(residuals_vel) < max_vel_cut
        good_new = np.flatnonzero(keep)
        if np.array_equal(good_new, good):
            break
        good = good_new

    return dict(
        coeffs=coeffs, good=good, iterations=i+1,
        pixels=pixels, wave=waves, model=model_flat, residuals=residuals, residuals_vel=residuals_vel,
        order_bounds=order_bounds, pixel_bounds=pixel_bounds
    )

def _build_wavesol_poly2d_unstructured(coeffs, pixels, orders, pixel_bounds, order_bounds):
    pixels_norm = _normalize_val(pixels, *pixel_bounds)
    orders_norm = _normalize_val(orders, *order_bounds)
    return np.polynomial.chebyshev.chebval2d(pixels_norm, orders_norm, coeffs) / orders

def _build_wavesol_poly2d_structured(coeffs, pixels, orders, pixel_bounds, order_bounds):
    nx, n_orders = len(pixels), len(orders)
    out = np.full((n_orders, nx), np.nan)
    for i in range(n_orders):
        out[i, :] = _build_wavesol_poly2d_unstructured(coeffs, pixels, np.full(nx, orders[i]), pixel_bounds, order_bounds)
    return out


# ---------------------------------------------------------------------------------------------
# Debug checks (debug=True)
# ---------------------------------------------------------------------------------------------
def _pix_err(wave_test, wave_ref):
    """(wave_test - wave_ref) in pixels of wave_ref, for (n_orders, nx) grids."""
    return (np.asarray(wave_test, float) - np.asarray(wave_ref, float)) / np.gradient(np.asarray(wave_ref, float), axis=1)


def _debug_check_wavesol(*, fiber, pixels, orders, waves, weights, fit_result, wavesol_full,
                         wave_input, order_vec, deg_intra_order, deg_inter_order):
    """Log a few checks of a 2D Chebyshev solution and return them as a dict.

    The "input" solution is the wave_fib the model arrived with. When it is (near) exact, as for
    a simulation or an LFC-calibrated reference, D and E measure the accuracy of the result.

      F  pulls     : chi2_r and std of (residual / error) of the kept lines. ~1 means the line
                     errors from the line fitting are trustworthy; >> 1 means they are too small.
      H  orders    : m*lambda at the central pixel should be ~constant for physical order
                     numbers; reports the label offset that makes it flattest.
      C  model     : the input solution sampled densely, refit with the same degrees. Large means
                     the polynomial cannot represent it (degrees too low, or order labels wrong).
      D  lines     : median (fitted - input) line position per order.
      E  solution  : per order mean / max of (final - input) in pixels, inside the line coverage.
      G  norm      : wave_fib rebuilt from the coefficients with `wavesol_norm_bounds` must match.
    """
    tag = f"[wavesol debug fib{fiber}]"
    out = {}
    good = np.asarray(fit_result['good'])
    pb, ob = fit_result['pixel_bounds'], fit_result['order_bounds']
    nx = wavesol_full.shape[1]
    xg = np.arange(nx, dtype=float)
    row_of = {float(m): i for i, m in enumerate(order_vec)}
    P, M, W = pixels[good], orders[good], waves[good]
    fit_orders = np.unique(M)

    # F: are the line errors consistent with the scatter?
    if weights is not None:
        pulls = np.asarray(fit_result['residuals'])[good] * M * np.sqrt(np.asarray(weights, float)[good])
        n_coef = fit_result['coeffs'].size
        chi2r = float(np.sum(pulls**2) / max(P.size - n_coef, 1))
        rms_v = float(np.sqrt(np.mean(np.asarray(fit_result['residuals_vel'])[good]**2)))
        out.update(chi2r=chi2r, pull_std=float(np.std(pulls)), rms_resid_vel=rms_v)
        logger.info(f"{tag} F: {P.size} lines, chi2_r {chi2r:.2f}, pull std {np.std(pulls):.2f}, "
                    f"rms residual {rms_v:.1f} m/s")

    # G: normalization bounds
    g = _build_wavesol_poly2d_structured(fit_result['coeffs'], xg, order_vec, *wavesol_norm_bounds(nx, order_vec))
    diff = float(np.nanmax(np.abs(_pix_err(g, wavesol_full))))
    out.update(norm_bounds_max_px=diff)
    (logger.warning if diff > 1e-3 else logger.info)(
        f"{tag} G: wave_fib rebuilt with wavesol_norm_bounds: max difference from wave_fib {diff:.4f} px")

    if wave_input is None or np.shape(wave_input) != wavesol_full.shape or not np.any(np.isfinite(wave_input)):
        logger.info(f"{tag} H, C, D, E skipped: no input wave_fib{fiber} to compare against")
        return out
    wave_input = np.asarray(wave_input, dtype=float)
    rows = [row_of[float(m)] for m in fit_orders if float(m) in row_of]

    # H: physical order numbers
    lam_mid = wave_input[rows, nx // 2]
    m_lab = np.array([float(m) for m in fit_orders if float(m) in row_of])
    ok = np.isfinite(lam_mid)
    if ok.sum() >= 3:
        spread = lambda k: float(np.std((m_lab[ok] + k) * lam_mid[ok]) / np.mean((m_lab[ok] + k) * lam_mid[ok]))
        ks = np.arange(-int(m_lab.min()) + 1, 500)
        k_best = int(ks[np.argmin([spread(k) for k in ks])])
        out.update(mlambda_rel_spread=spread(0), best_label_offset=k_best)
        (logger.warning if k_best != 0 else logger.info)(
            f"{tag} H: m*lambda at pixel {nx // 2} varies by {spread(0):.2%} across orders; "
            f"flattest with order label offset {k_best:+d} ({spread(k_best):.2%})"
            + ("  -> order labels are not physical diffraction orders" if k_best != 0 else ""))

    # C: can the polynomial represent the input solution?
    lo, hi = max(int(np.ceil(pb[0])), 0), min(int(np.floor(pb[1])), nx - 1)
    gx = np.linspace(lo, hi, 64)
    gp, gm = np.tile(gx, len(rows)), np.repeat(m_lab, gx.size)
    gw = np.concatenate([np.interp(gx, xg, wave_input[r]) for r in rows])
    rt = fit_wavesol_chebyshev2d(gp, gm, gw, deg_intra_order=deg_intra_order,
                                 deg_inter_order=deg_inter_order, pixel_bounds=pb, order_bounds=ob)
    rt_err = _pix_err(_build_wavesol_poly2d_structured(rt['coeffs'], xg, order_vec, pb, ob), wave_input)
    c = float(np.nanmax(np.abs(rt_err[rows][:, lo:hi + 1])))
    out['model_representation_max_px'] = c
    (logger.warning if c > 0.01 else logger.info)(
        f"{tag} C: input solution refit with deg ({deg_intra_order}, {deg_inter_order}): max error {c:.4f} px"
        + ("  -> raise the degrees (or check H)" if c > 0.01 else ""))

    # D and E, per order
    e = _pix_err(wavesol_full, wave_input)
    logger.info(f"{tag} D/E: order  n_lines  lines-input(median px)  final-input(mean, max px)")
    stats = {}
    for m in fit_orders:
        r = row_of.get(float(m))
        if r is None:
            continue
        sel = M == m
        w = wave_input[r]
        fin = np.isfinite(w)
        srt = np.argsort(w[fin])
        pix_in = np.interp(W[sel], w[fin][srt], xg[fin][srt], left=np.nan, right=np.nan)
        d_lines = float(np.nanmedian(P[sel] - pix_in))
        k1, k2 = int(np.floor(P[sel].min())), int(np.ceil(P[sel].max()))
        ein = e[r, k1:k2 + 1]
        stats[float(m)] = (int(sel.sum()), d_lines, float(np.nanmean(ein)), float(np.nanmax(np.abs(ein))))
        logger.info(f"{tag} D/E: {m:5.0f} {sel.sum():8d} {d_lines:+22.3f} {stats[float(m)][2]:+12.4f} {stats[float(m)][3]:8.4f}")
    out['per_order'] = stats
    return out
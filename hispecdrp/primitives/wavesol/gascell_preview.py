"""
Interactive Plotly preview of the gas cell wavelength solution fit (FitGasCellWavesolPrimitive).
"""
from ...utils.plotly_utils import FONT_FAMILY, MATHJAX_URL, show_figure
import logging

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

__all__ = ['create_gascell_preview']

# Runs in the preview HTML after the plot is drawn, and again after every redraw:
# - Plotly can only label slider ticks below the rail, one label per tick. Draw the order number
#   above the rail and the mean wavelength below each tick, both parsed from the step label
#   "<order> (<wavelength> µm)". The wavelengths are always tilted 45 degrees. Labels that would
#   still overlap are thinned.
# - The "Order" / "Full bandpass" buttons (method 'skip') are handled here, since their action
#   depends on the current fiber and slider position. Full bandpass shows every order of the
#   current fiber using the existing traces (no copies), hides the per-order chi2 and the pixel
#   axis, and zooms out. "Order" re-applies the slider's current step. Picking a fiber while in
#   full bandpass stays in full bandpass. Trace layout comes from layout.meta.
# - Plotly cannot recompute the pixel axis ticks on zoom (they are a fixed list placed at the
#   wavelengths of round pixels). Recompute them for the visible range of the shown order, with
#   about 8 ticks, the same rule as _pixel_axis_ticks. The pixel of a point is its index in the
#   spectrum trace's wavelength array.
# Plotly substitutes {plot_id} with a plain string replace, so other braces are literal.
_SLIDER_SCRIPT = """
(function() {
  var gd = document.getElementById('{plot_id}');
  var NS = 'http://www.w3.org/2000/svg';
  var STYLE = 'font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 10px; font-weight: bold; fill: #000000; pointer-events: none;';

  function makeText(parent, x, y, s) {
    var t = document.createElementNS(NS, 'text');
    t.setAttribute('class', 'order-tick-label');
    t.setAttribute('x', x);
    t.setAttribute('y', y);
    t.setAttribute('text-anchor', 'middle');
    t.setAttribute('style', STYLE);
    t.textContent = s;
    parent.appendChild(t);
    return t;
  }

  function maxWidth(texts) {
    return texts.reduce(function(w, t) { return Math.max(w, t.getComputedTextLength()); }, 0);
  }

  // Keep every k-th label so neighbours do not overlap
  function thin(texts, spacing, width) {
    var k = Math.ceil((width + 4) / spacing);
    if (k > 1) {
      texts.forEach(function(t, i) { if (i % k !== 0) { t.remove(); } });
    }
  }

  function draw() {
    var ticks = gd.querySelectorAll('.slider-container rect.slider-tick-rect');
    var slider = gd.layout.sliders && gd.layout.sliders[0];
    if (!ticks.length || !slider || !slider.steps) { return; }
    var parent = ticks[0].parentNode;
    parent.querySelectorAll('text.order-tick-label').forEach(function(t) { t.remove(); });

    var orders = [], waves = [], xs = [], tickY = 0, tickLen = 0;
    ticks.forEach(function(tick, i) {
      var m = tick.transform.baseVal.consolidate().matrix;
      var x = m.e + 0.5 * parseFloat(tick.getAttribute('width'));
      tickY = m.f;
      tickLen = parseFloat(tick.getAttribute('height'));
      var parts = /^(\\S+) \\((\\S+) /.exec(slider.steps[i].label);
      if (!parts) { return; }
      xs.push(x);
      // The rail and grip fill the 25 px above the ticks
      orders.push(makeText(parent, x, tickY - 29, parts[1]));
      waves.push(makeText(parent, x, tickY + tickLen + 11, parts[2]));
    });
    if (xs.length < 2) { return; }
    var spacing = (xs[xs.length - 1] - xs[0]) / (xs.length - 1);

    thin(orders, spacing, maxWidth(orders));

    // Wavelengths are always tilted 45 degrees. Neighbouring labels are then
    // spacing * sin(45) apart across their baselines.
    waves.forEach(function(t) {
      var x = t.getAttribute('x'), y = parseFloat(t.getAttribute('y')) - 4;
      t.setAttribute('y', y);
      t.setAttribute('text-anchor', 'end');
      t.setAttribute('transform', 'rotate(-45 ' + x + ' ' + y + ')');
    });
    thin(waves, spacing * Math.SQRT1_2, 10);
  }

  var lastPixelTicks = null;

  function updatePixelTicks() {
    var fl = gd._fullLayout;
    if (!fl.xaxis3 || !fl.xaxis || !fl.xaxis3.visible) { return; }
    // Wavelengths of the shown order: the first visible trace in the spectrum panel
    var x = null;
    for (var i = 0; i < gd._fullData.length; i++) {
      var d = gd._fullData[i];
      if (d.visible === true && d.xaxis === 'x' && d.x && d.x.length > 1) { x = d.x; break; }
    }
    if (!x) { return; }
    var lo = Math.min(fl.xaxis.range[0], fl.xaxis.range[1]);
    var hi = Math.max(fl.xaxis.range[0], fl.xaxis.range[1]);
    var first = -1, last = -1;
    for (var p = 0; p < x.length; p++) {
      if (x[p] >= lo && x[p] <= hi) { if (first < 0) { first = p; } last = p; }
    }
    if (first < 0) { return; }
    var raw = Math.max((last - first) / 8, 1);
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var step = [1, 2, 5, 10].map(function(m) { return m * mag; }).reduce(function(a, b) {
      return Math.abs(b - raw) < Math.abs(a - raw) ? b : a;
    });
    step = Math.max(1, Math.round(step));
    var vals = [], text = [];
    for (var q = Math.ceil(first / step) * step; q <= last; q += step) {
      if (isFinite(x[q])) { vals.push(x[q]); text.push(String(q)); }
    }
    // Skip when nothing changed, so this relayout does not trigger itself again
    var key = vals.join(',');
    if (key === lastPixelTicks) { return; }
    lastPixelTicks = key;
    Plotly.relayout(gd, {'xaxis3.tickvals': vals, 'xaxis3.ticktext': text});
  }

  function menuIndex(name) {
    var menus = gd.layout.updatemenus || [];
    for (var i = 0; i < menus.length; i++) { if (menus[i].name === name) { return i; } }
    return -1;
  }

  function menuActive(name) {
    var i = menuIndex(name);
    return i < 0 ? 0 : (gd.layout.updatemenus[i].active || 0);
  }

  function showFullBandpass() {
    var meta = gd.layout.meta, f = meta.fibers[menuActive('fiber')], tpo = meta.traces_per_order;
    var n = gd.data.length, vis = [], legend = [];
    for (var i = 0; i < n; i++) {
      var inFiber = i >= f.first_trace && i < f.first_trace + f.n_orders * tpo;
      vis.push(inFiber || i === n - 1);  // the last trace is the pixel axis placeholder
      legend.push(inFiber && i < f.first_trace + tpo);  // one legend entry per line type
    }
    Plotly.update(gd, {visible: vis, showlegend: legend}, {
      'title.text': f.title, 'title.subtitle.text': 'All orders',
      'annotations[0].text': f.chi2_global, 'annotations[1].text': '',
      'xaxis3.visible': false, 'xaxis.autorange': true, 'xaxis2.autorange': true,
      'yaxis.autorange': true, 'yaxis2.autorange': true
    });
  }

  function showOrder() {
    var slider = gd.layout.sliders[0], step = slider.steps[slider.active || 0];
    Plotly.update(gd, step.args[0], step.args[1]);
  }

  gd.on('plotly_buttonclicked', function(e) {
    var name = e.menu && e.menu.name;
    if (name === 'view') {
      if (e.active === 1) { showFullBandpass(); } else { showOrder(); }
    } else if (name === 'fiber' && menuActive('view') === 1) {
      // The fiber button shows that fiber's first order; switch back to full bandpass after it
      setTimeout(showFullBandpass, 50);
    }
  });

  function redraw() { setTimeout(function() { draw(); updatePixelTicks(); }, 0); }
  draw();
  updatePixelTicks();
  ['plotly_afterplot', 'plotly_sliderchange', 'plotly_buttonclicked', 'plotly_relayout'].forEach(function(ev) {
    gd.on(ev, redraw);
  });
})();
"""


def create_gascell_preview(primitive):
    """
    Interactive preview of the gas cell fit: the data and the best-fit forward model (top) and
    their residuals (bottom) against the fitted wavelength, one fit order at a time.

    A slider steps through the fitted orders in increasing wavelength (decreasing order number).
    Each tick shows the order number above the rail and the order's mean wavelength below it
    (from the starting wavelength grid where available, otherwise from the fit). With more than
    one fitted fiber, a dropdown left of the slider selects the fiber and rebuilds the slider
    for it. Order / Full bandpass buttons, between the dropdown and the slider, switch between
    one order and every order of the fiber. Each (fiber, order) combination has its own three
    traces, and the controls only toggle their visibility.

    Reads ``output`` and ``fit_results`` from ``primitive._output``, saves the HTML and PNG if
    ``primitive.save_preview``, shows the figure if ``primitive.show_preview``, and stores
    ``dict(html, png, fig)`` in ``primitive._output['preview']``.
    """
    output_model = primitive._output['output']
    fit_results = primitive._output['fit_results']

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.08,
    )

    if not fit_results:
        logger.warning("No gas cell fits available to preview.")
        fig.update_layout(title='No gas cell fits available to preview.')
        primitive._output['preview'] = dict(html=None, png=None, fig=fig)
        return

    order_vec = output_model.make_order_vec()
    traces_per_order = 3  # data, model, residuals

    # Mean wavelength of each fit order: from the starting wavelength grid where it has one,
    # otherwise from the fitted solution
    mean_waves_all = []
    for fit_result in fit_results:
        wave = getattr(output_model, f"wave_fib{fit_result['fiber']}")
        oi_fit = [np.flatnonzero(order_vec == order)[0] for order in fit_result['orders']]
        mean_fit = np.array([np.nanmean(wave[oi]) for oi in oi_fit])
        mean_start = fit_result.get('mean_wave_start')
        if mean_start is None:
            mean_start = np.full(len(oi_fit), np.nan)
        mean_waves_all.append(np.where(np.isfinite(mean_start), mean_start, mean_fit))
    wave_decimals = _order_wavelength_decimals(fit_results, mean_waves_all)

    # Hover shows the pixel (0-indexed): each trace holds a full order row, so the point index
    # is the pixel
    HOVER = 'Order {order}<br>Pixel %{{pointNumber}}<br>%{{x:.6f}} µm<br>%{{y:.2f}} e-/s'

    # One entry per fit fiber: its orders and the index of its first trace
    fiber_meta = []
    for fit_result, mean_waves in zip(fit_results, mean_waves_all):
        fiber = fit_result['fiber']
        wave = getattr(output_model, f'wave_fib{fiber}')
        spec = getattr(output_model, f'spec_fib{fiber}')
        first_trace = len(fig.data)
        # Plain-text title per fiber and subtitle per order. Plotly cannot mix LaTeX with plain
        # text in one string, so the reduced chi2 values are separate LaTeX annotations.
        title = f"Gas cell fit — Fiber {fiber}"
        chi2_global = _chi2_red_latex('global', fit_result['chi2_red'])
        subtitles = []
        chi2_orders = []
        pixel_ticks = []
        # Orders in increasing mean wavelength (decreasing order number). Every per-order list
        # below and the traces follow this sequence, so index 0 is the shortest wavelength.
        o_sorted = np.argsort(mean_waves, kind='stable')
        for k, o in enumerate(o_sorted):
            order = fit_result['orders'][o]
            oi = np.flatnonzero(order_vec == order)[0]
            w = wave[oi].astype(np.float32)
            visible = (len(fiber_meta) == 0 and k == 0)
            hover = HOVER.format(order=order)
            fig.add_trace(go.Scatter(
                x=w, y=spec[oi].astype(np.float32), mode='lines', name='Data',
                line=dict(color='black', width=2), legendgroup='data', showlegend=True, visible=visible,
                hovertemplate=hover,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=w, y=fit_result['model'][o].astype(np.float32), mode='lines', name='Model',
                line=dict(color='#e03d6b', width=2), legendgroup='model', showlegend=True, visible=visible,
                hovertemplate=hover,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=w, y=fit_result['residuals'][o].astype(np.float32), mode='lines', name='Residuals',
                line=dict(color='#0088CC', width=2), legendgroup='resid', showlegend=True, visible=visible,
                hovertemplate=hover,
            ), row=2, col=1)
            subtitles.append(f"Order {order} ({mean_waves[o]:.{wave_decimals}f} µm)")
            chi2_orders.append(_chi2_red_latex('order', fit_result['chi2_red_orders'][o]))
            pixel_ticks.append(_pixel_axis_ticks(wave[oi]))
        fiber_meta.append(dict(
            fiber=fiber, orders=np.asarray(fit_result['orders'])[o_sorted],
            mean_waves=np.asarray(mean_waves)[o_sorted], first_trace=first_trace,
            title=title, subtitles=subtitles, chi2_global=chi2_global, chi2_orders=chi2_orders,
            pixel_ticks=pixel_ticks,
        ))

    # Plotly only draws an axis that has a trace, so the pixel axis gets an empty one (last)
    fig.add_trace(go.Scatter(x=[], y=[], xaxis='x3', yaxis='y', showlegend=False, hoverinfo='skip'))
    n_traces = len(fig.data)

    def make_visibility(f_idx, o_idx):
        vis = [False] * n_traces
        base = fiber_meta[f_idx]['first_trace'] + o_idx * traces_per_order
        vis[base:base + traces_per_order] = [True] * traces_per_order
        vis[-1] = True  # pixel axis placeholder
        return vis

    # Order slider steps for one fiber, labeled "<order> (<mean wavelength> µm)". The script in
    # _SLIDER_SCRIPT parses this label to draw the order above and the wavelength below each tick.
    def build_order_slider_steps(f_idx):
        meta = fiber_meta[f_idx]
        return [
            dict(
                method='update',
                label=f"{order} ({meta['mean_waves'][o_idx]:.{wave_decimals}f} µm)",
                args=[
                    {'visible': make_visibility(f_idx, o_idx), 'showlegend': [True] * n_traces},
                    {
                        'title.text': meta['title'],
                        'title.subtitle.text': meta['subtitles'][o_idx],
                        'annotations[0].text': meta['chi2_global'],
                        'annotations[1].text': meta['chi2_orders'][o_idx],
                        'xaxis3.tickvals': meta['pixel_ticks'][o_idx][0],
                        'xaxis3.ticktext': meta['pixel_ticks'][o_idx][1],
                        **ORDER_VIEW,
                    },
                ],
            )
            for o_idx, order in enumerate(meta['orders'])
        ]

    # Layout settings that return to the single-order view (from full bandpass or a zoom)
    VIEW_MENU = 0  # index of the Order / Full bandpass buttons in layout.updatemenus
    ORDER_AXES = {
        'xaxis3.visible': True,
        'xaxis.autorange': True, 'xaxis2.autorange': True, 'yaxis.autorange': True, 'yaxis2.autorange': True,
    }
    # The fiber buttons leave the toggle alone, so the script can keep full bandpass on
    ORDER_VIEW = {f'updatemenus[{VIEW_MENU}].active': 0, **ORDER_AXES}

    SLIDER_PAD_TOP_PX = 60  # space between the x-axis title and the slider
    PIXEL_AXIS_PX = 48  # height of the pixel axis (ticks and title) above the spectrum panel
    SUBTITLE_PX = 26  # the subtitle hangs this far below the title's anchor

    # Fiber dropdown (only with more than one fiber): shows the fiber's first order and
    # rebuilds the order slider for that fiber
    fiber_buttons = [
        dict(
            label=f"Fiber {meta['fiber']}", method='update',
            args=[
                {'visible': make_visibility(f_idx, 0), 'showlegend': [True] * n_traces},
                {
                    **ORDER_AXES,
                    'title.text': meta['title'],
                    'title.subtitle.text': meta['subtitles'][0],
                    'annotations[0].text': meta['chi2_global'],
                    'annotations[1].text': meta['chi2_orders'][0],
                    'xaxis3.tickvals': meta['pixel_ticks'][0][0],
                    'xaxis3.ticktext': meta['pixel_ticks'][0][1],
                    'sliders[0].steps': build_order_slider_steps(f_idx),
                    'sliders[0].active': 0,
                },
            ],
        )
        for f_idx, meta in enumerate(fiber_meta)
    ]
    # Left of the order slider, level with its rail: the fiber dropdown (only with more than one
    # fiber), then the Order / Full bandpass toggle. The widths (px) are the rendered widths of
    # the controls plus a 30 px gap; the slider's left pad makes room for both.
    MENU_PAD_TOP_PX = SLIDER_PAD_TOP_PX + 38  # puts the controls level with the slider rail
    multi_fiber = len(fiber_meta) > 1
    dropdown_width = 110 if multi_fiber else 0
    view_menu_width = 194
    fiber_menu = [dict(
        name='fiber', buttons=fiber_buttons, direction='up', showactive=True,
        x=0, xanchor='left', y=0, yanchor='top', pad=dict(t=MENU_PAD_TOP_PX),
        bgcolor='#f0f0f0', bordercolor='#cccccc',
    )] if multi_fiber else []
    # method='skip': the preview script does the switching (see _SLIDER_SCRIPT)
    view_menu = dict(
        name='view', type='buttons', direction='right', showactive=True, active=0,
        buttons=[dict(label='Order', method='skip', args=[None]),
                 dict(label='Full bandpass', method='skip', args=[None])],
        x=0, xanchor='left', y=0, yanchor='top', pad=dict(t=MENU_PAD_TOP_PX, l=dropdown_width),
        bgcolor='#f0f0f0', bordercolor='#cccccc',
    )
    updatemenus = [view_menu] + fiber_menu  # view_menu at index VIEW_MENU

    fig.update_layout(
        # The title sits above the pixel axis, both anchored to the top of the spectrum panel
        title=dict(
            text=fiber_meta[0]['title'], subtitle=dict(text=fiber_meta[0]['subtitles'][0], font=dict(size=14)),
            xref='paper', x=0, xanchor='left',
            yref='paper', y=1, yanchor='bottom', pad=dict(b=PIXEL_AXIS_PX + SUBTITLE_PX), automargin=True,
        ),
        # Reduced chi2 of the fiber's fit [0] and of the shown order [1], right of the title.
        # The title starts at the plot's left edge and is ~170 px wide in the bold font.
        annotations=[
            dict(
                text=text, xref='paper', yref='paper', x=0, xanchor='left', xshift=200,
                y=1, yanchor='bottom', yshift=yshift, showarrow=False, font=dict(size=16),
            )
            for text, yshift in [
                (fiber_meta[0]['chi2_global'], PIXEL_AXIS_PX + SUBTITLE_PX),
                (fiber_meta[0]['chi2_orders'][0], PIXEL_AXIS_PX),
            ]
        ],
        # Pixel axis (0-indexed) on top of the spectrum panel. It shares the wavelength axis's
        # range, and its ticks sit at the wavelengths of round pixel numbers of the shown order.
        xaxis3=dict(
            matches='x2', overlaying='x', anchor='y', side='top', showgrid=False,
            tickmode='array', tickvals=fiber_meta[0]['pixel_ticks'][0][0],
            ticktext=fiber_meta[0]['pixel_ticks'][0][1], title=dict(text='Pixel', standoff=4),
        ),
        updatemenus=updatemenus,
        # Trace layout for the full bandpass view in _SLIDER_SCRIPT
        meta=dict(
            traces_per_order=traces_per_order,
            fibers=[
                dict(first_trace=meta['first_trace'], n_orders=len(meta['orders']),
                     title=meta['title'], chi2_global=meta['chi2_global'])
                for meta in fiber_meta
            ],
        ),
        sliders=[dict(
            active=0,
            # The offset leaves room for the order numbers drawn above the rail
            currentvalue=dict(prefix='Order: ', offset=26, font=dict(size=14, color='#000000', weight='bold')),
            # Plotly's own tick labels are hidden; _SLIDER_SCRIPT draws the order and wavelength
            font=dict(color='rgba(0,0,0,0)'),
            # Same length for every tick (Plotly shortens ticks it doesn't label)
            ticklen=7, minorticklen=7,
            pad=dict(t=SLIDER_PAD_TOP_PX, b=25, l=dropdown_width + view_menu_width),
            steps=build_order_slider_steps(0),
        )],
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        # Bold for all Plotly text (title, legend, axis titles, tick labels, slider, dropdown)
        font=dict(family=FONT_FAMILY, color='#000000', weight='bold'),
        legend=dict(bgcolor='#ffffff'),
        height=750,
    )
    grid = dict(showgrid=True, gridcolor='rgba(200,200,200,0.3)')
    fig.update_xaxes(**grid)
    fig.update_layout(xaxis3_showgrid=False)  # pixel ticks don't get their own grid lines
    fig.update_xaxes(title_text='Wavelength (micron)', row=2, col=1)
    fig.update_yaxes(**grid)
    fig.update_yaxes(title_text='Flux (e-/s)', row=1, col=1)
    fig.update_yaxes(title_text='Residuals (e-/s)', row=2, col=1)

    base_path = output_model._generate_output_path(output_dir=primitive.output_dir, suffix=primitive.suffix)
    output_path_html = base_path.replace('.fits', '_preview.html')
    output_path_png = base_path.replace('.fits', '_preview.png')

    if primitive.save_preview:
        fig.write_html(output_path_html, include_plotlyjs='cdn', include_mathjax=MATHJAX_URL, post_script=_SLIDER_SCRIPT)
        fig.write_image(output_path_png, scale=2)

    if primitive.show_preview:
        show_figure(fig, post_script=_SLIDER_SCRIPT)

    primitive._output['preview'] = {
        'html': output_path_html,
        'png': output_path_png,
        'fig': fig,
    }


def _pixel_axis_ticks(wave_row, target_nticks=8):
    """
    Ticks for the pixel axis of one order: round pixel numbers (0-indexed) about
    ``Ncols / target_nticks`` apart, as (tickvals, ticktext), with tickvals the order's wavelength
    at those pixels. Pixels with a non-finite wavelength are left out.
    """
    wave_row = np.asarray(wave_row, dtype=float)
    Ncols = wave_row.size
    raw = max(Ncols / target_nticks, 1)
    mag = 10 ** np.floor(np.log10(raw))
    step = int(min((m * mag for m in (1, 2, 5, 10)), key=lambda st: abs(st - raw)))
    pixels = np.arange(0, Ncols, step)
    pixels = pixels[np.isfinite(wave_row[pixels])]
    return wave_row[pixels].tolist(), [str(p) for p in pixels]


def _chi2_red_latex(label, value):
    """
    LaTeX for a reduced chi2, e.g. chi^2_{nu, global} = 1.10, all in bold (\\boldsymbol so the
    Greek letters are bold too; needs the full MathJax build, see utils.plotly_utils.MATHJAX_URL).
    """
    return rf"$\boldsymbol{{\chi^2_{{\nu,\,\mathrm{{{label}}}}} = {value:.2f}}}$"


def _order_wavelength_decimals(fit_results, mean_waves_all, frac=0.1, default=3):
    """
    Number of decimals (in microns) that resolve ``frac`` of the smallest wavelength step
    between adjacent echelle orders. The step between two listed orders is divided by their
    order-number difference, so subsampled orders give the per-order step. Returns ``default``
    when there are fewer than two orders with a finite mean wavelength.
    """
    steps = []
    for fit_result, mean_waves in zip(fit_results, mean_waves_all):
        orders = np.asarray(fit_result['orders'], dtype=float)
        mean_waves = np.asarray(mean_waves, dtype=float)
        good = np.isfinite(mean_waves)
        isort = np.argsort(orders[good])
        dm = np.abs(np.diff(orders[good][isort]))
        dw = np.abs(np.diff(mean_waves[good][isort]))
        ok = (dm > 0) & (dw > 0)
        steps.extend(dw[ok] / dm[ok])
    if len(steps) == 0:
        return default
    return max(0, int(np.ceil(-np.log10(frac * np.min(steps)))))

from ..base_primitive import BasePrimitive
from ... import datamodels

from .fit_lines import (fit_lines_gauss, LINE_DQ_FLAGS, flag_low_amplitude_lines,
                        flag_fwhm_outliers, flag_redchi2_outliers)
from ..utils.spectral_utils import delta_pix_to_delta_wave

import numpy as np
import os
from astropy.table import Table

import plotly.graph_objects as go

from pathlib import Path

import logging

logger = logging.getLogger(__name__)


__all__ = ['FitUraniumLinesPrimitive']

LINELIST_DIR = (Path(__file__).resolve().parents[2] / "data" / "wavesol").resolve()


class FitUraniumLinesPrimitive(BasePrimitive):
    """
    Fit the isolated uranium lamp lines in each order and flag suspect fits.

    For every order:

    1. Predict the pixel of every line-list entry from the input wavelength solution.
    2. Keep only isolated lines: no other listed line at least ``neighbor_flux_frac`` as bright
       within ``min_isolation_fwhm`` x FWHM. Everything else is treated as blended and not fit.
    3. Fit each isolated line with a single Gaussian, in a window that stops 1.5 FWHM short of
       its nearest significant neighbour (at most ``max_fit_window`` pixels wide).
    4. Flag fits that failed or look suspect (see LINE_DQ below).

    The reported ``MU_PIX`` / ``MU_ERR_PIX`` are the plain Gaussian fit and its formal error,
    with no rescaling. Downstream steps should use only lines with LINE_DQ == GOOD.

    LINE_DQ flags set here:
      FIT_FAILURE : the fit returned NaN (e.g. too few good pixels in the window)
      LOW_SNR     : line amplitude below ``min_snr`` times the local noise
      POOR_FIT    : any of
                      - the fitted center or FWHM stopped at the edge of its bounds
                      - the center is more than ``max_shift_resid`` px from the smooth
                        per-order offset (fitted - predicted) of the other lines
                      - two lines converged on the same peak
                      - reduced chi2 or FWHM is far above that of similar lines

    Parameters
    ----------
    input : str or L2Model
        Input level 2 model or path to a level 2 file, with an initial wavelength solution.
    suffix : str, optional
        Suffix added to the output filename. Default 'lamp_fit'.
    line_fwhm_pix : float, optional
        Typical line FWHM in pixels. Sets the isolation distance and fit windows.
    fwhm_bounds : tuple, optional
        Allowed (min, max) fitted FWHM in pixels.
    mu_bounds : tuple, optional
        Allowed (min, max) fitted center relative to the predicted pixel. Must exceed the error
        of the input wavelength solution.
    max_fit_window : int, optional
        Maximum full width of a fit window in pixels.
    neighbor_flux_frac : float, optional
        A listed line at least this bright (relative to the line being fit) counts as a neighbour.
    min_isolation_fwhm : float, optional
        Minimum distance, in FWHM, to the nearest neighbour for a line to be fit.
    max_shift_resid : float, optional
        Pixels; lines further than this from the per-order offset model are flagged POOR_FIT.
    min_snr : float, optional
        Lines with lower signal-to-noise are flagged LOW_SNR.
    shape_outlier_nsigma : float, optional
        Reduced chi2 and FWHM more than this many robust sigma above the trend of the other
        lines are flagged POOR_FIT. (3 flags many faint but accurate lines.)
    do_fibers, do_orders : list of int, optional
        Fibers / orders to process. None means all.

    Returns
    -------
    output : UraniumFitParamsModel
        The input model with the table ``uranium_fitparams_fib#`` populated.
    """

    PARAMETERS = {
        'suffix' : 'lamp_fit',
        'line_fwhm_pix' : 3.0,
        'fwhm_bounds' : (1, 8),
        'mu_bounds' : (-3, 3),
        'max_fit_window' : 15,
        'neighbor_flux_frac' : 0.01,
        'min_isolation_fwhm' : 2.5,
        'max_shift_resid' : 0.5,
        'min_snr' : 5.,
        'shape_outlier_nsigma' : 5.,
        'do_fibers' : None,
        'do_orders' : None,
    }

    def _perform(self):

        input_model = datamodels.open(self.dataset)
        output_model = datamodels.UraniumFitParamsModel.from_datamodels(
            input_model, include_data=True, primitive=self
        )
        do_fibers, do_orders = self.get_do_traces(input_model, self.do_fibers, self.do_orders)
        order_start = input_model.meta.order_start

        lamp_wvl, lamp_flux = _read_linelist(os.path.join(LINELIST_DIR, "UNe_lines_nir.txt"))
        logger.info(f'{len(lamp_wvl)} usable lines in the lamp line list')

        for fiber in do_fibers:
            wave = getattr(input_model, f'wave_fib{fiber}')
            spec = getattr(input_model, f'spec_fib{fiber}')
            spec_err = getattr(input_model, f'spec_err_fib{fiber}')
            spec_dq = getattr(input_model, f'spec_dq_fib{fiber}')
            xarr = np.arange(wave.shape[1])

            # Start from an empty table for the orders being fit (re-runs replace, not append)
            table_name = f'uranium_fitparams_fib{fiber}'
            table = getattr(output_model, table_name)
            if len(table) > 0:
                keep = ~np.isin(np.asarray(table['ORDER']), np.asarray(list(do_orders)))
                setattr(output_model, table_name, table[keep])
                table = getattr(output_model, table_name)

            for order in do_orders:
                oi = np.abs(order - order_start)
                rows = _fit_order_lines(
                    xarr, spec[oi], spec_err[oi], spec_dq[oi], wave[oi], lamp_wvl, lamp_flux,
                    line_fwhm_pix=self.line_fwhm_pix, fwhm_bounds=self.fwhm_bounds,
                    mu_bounds=self.mu_bounds, max_fit_window=self.max_fit_window,
                    neighbor_flux_frac=self.neighbor_flux_frac,
                    min_isolation_fwhm=self.min_isolation_fwhm,
                    max_shift_resid=self.max_shift_resid,
                )
                offs = [r['MU_PIX'] - r['PRED_PIX'] for r in rows if r['LINE_DQ'] == 0]
                n_good = len(offs)
                for r in rows:
                    r.pop('PRED_PIX')
                    r['ORDER'] = order
                    table.add_row(r)
                logger.info(f'fiber {fiber}, order {order}: fit {len(rows)} isolated lines, '
                            f'{n_good} unflagged so far; median offset from input solution '
                            f'{np.median(offs) if offs else np.nan:+.3f} px')

            table.sort(['ORDER', 'MU_PIX'])
            _flag_fiber_lines(table, spec, spec_dq, order_start, lamp_wvl, self.min_snr,
                              self.shape_outlier_nsigma)
            n_good = int(np.sum(np.asarray(table['LINE_DQ']) == 0))
            logger.info(f'fiber {fiber}: {n_good} of {len(table)} fitted lines are GOOD')

        return dict(output=output_model)


    def _create_preview(self):
        output_model = self._output['output']
        filename = output_model.meta.get('filename', 'Uranium lamp fit')

        do_fibers, do_orders = self.get_do_traces(output_model, self.do_fibers, self.do_orders)
        do_fibers = list(do_fibers)
        do_orders = list(do_orders)
        order_start = output_model.meta.order_start

        # Per (fiber, order): spectrum, good-line markers, bad-line markers
        traces_per_order = 3
        traces_per_fiber = len(do_orders) * traces_per_order
        total_traces = len(do_fibers) * traces_per_fiber

        hovertemplate = (
            'Line %{customdata[0]:d}<br>'
            'μ = %{x:.2f} px<br>'
            'λ = %{customdata[1]:.6f} µm<br>'
            'FWHM = %{customdata[2]:.2f} px<br>'
            'χ²ᵣ = %{customdata[3]:.2f}<br>'
            'LINE_DQ = %{customdata[4]:d}'
            '<extra></extra>'
        )

        def make_title(fiber, order):
            return f'Uranium line fits — Fiber {fiber}, Order {order}  |  {filename}'

        fig = go.Figure()

        for f_idx, fiber in enumerate(do_fibers):
            spec = getattr(output_model, f'spec_fib{fiber}')
            params_table = getattr(output_model, f'uranium_fitparams_fib{fiber}')
            xarr = np.arange(spec.shape[1])

            # Pull table columns out once, filling any masked values
            t_order = np.asarray(np.ma.filled(params_table['ORDER'], -1))
            t_mu    = _filled_float(params_table['MU_PIX'])
            t_wave  = _filled_float(params_table['WAVE'])
            t_fwhm  = _filled_float(params_table['FWHM_PIX'])
            t_chi2  = _filled_float(params_table['REDCHI2'])
            t_num   = np.asarray(np.ma.filled(params_table['LINE_NUM'], -1), dtype=int)
            t_dq    = np.asarray(np.ma.filled(params_table['LINE_DQ'], 0), dtype=int)

            for o_idx, order in enumerate(do_orders):
                visible = (f_idx == 0 and o_idx == 0)
                oi = np.abs(order - order_start)
                y = spec[oi, :]

                # Vertical extent of the line markers follows the spectrum
                finite = y[np.isfinite(y)]
                if finite.size:
                    y0, y1 = float(finite.min()), float(finite.max())
                else:
                    y0, y1 = 0.0, 1.0

                fig.add_trace(go.Scatter(
                    x           = xarr,
                    y           = y,
                    mode        = 'lines',
                    name        = f'Spectrum (fiber {fiber})',
                    line        = dict(color='black', width=1),
                    legendgroup = 'spec',
                    showlegend  = True,
                    visible     = visible,
                ))

                # Lines fitted in this order with a usable center
                in_order = (t_order == order) & np.isfinite(t_mu)
                good = in_order & (t_dq == 0)
                bad  = in_order & (t_dq != 0)

                for sel, color, label in ((good, 'royalblue', 'Good fit'),
                                          (bad,  'red',       'Flagged fit')):
                    xs, ys = _vline_segments(t_mu[sel], y0, y1)
                    customdata = np.repeat(
                        np.column_stack([t_num[sel], t_wave[sel], t_fwhm[sel],
                                         t_chi2[sel], t_dq[sel]]),
                        3, axis=0,
                    )
                    fig.add_trace(go.Scatter(
                        x             = xs,
                        y             = ys,
                        mode          = 'lines',
                        name          = f'{label} ({int(sel.sum())})',
                        line          = dict(color=color, width=1),
                        opacity       = 0.7,
                        customdata    = customdata,
                        hovertemplate = hovertemplate,
                        legendgroup   = label,
                        showlegend    = True,
                        visible       = visible,
                    ))

        def make_visibility(active_fiber: int, active_order: int) -> list:
            """Full visibility list with only the chosen fiber+order shown."""
            vis = [False] * total_traces
            base = active_fiber * traces_per_fiber + active_order * traces_per_order
            for t in range(traces_per_order):
                vis[base + t] = True
            return vis

        def make_slider_steps(f_idx: int) -> list:
            fiber = do_fibers[f_idx]
            return [
                dict(
                    method = 'update',
                    label  = str(order),
                    args   = [
                        {'visible': make_visibility(f_idx, o_idx)},
                        {'title': make_title(fiber, order)},
                    ],
                )
                for o_idx, order in enumerate(do_orders)
            ]

        def make_slider(f_idx: int) -> dict:
            return dict(
                active       = 0,
                currentvalue = dict(prefix='Order: ', font=dict(size=14)),
                pad          = dict(t=50),
                steps        = make_slider_steps(f_idx),
            )

        # Fiber dropdown: each button jumps to that fiber's first order and swaps
        # in a slider whose steps stay within that fiber.
        dropdown_buttons = [
            dict(
                label  = f'Fiber {fiber}',
                method = 'update',
                args   = [
                    {'visible': make_visibility(f_idx, 0)},
                    {
                        'title':   make_title(fiber, do_orders[0]),
                        'sliders': [make_slider(f_idx)],
                    },
                ],
            )
            for f_idx, fiber in enumerate(do_fibers)
        ]

        fig.update_layout(
            title = make_title(do_fibers[0], do_orders[0]),
            updatemenus = [dict(
                buttons     = dropdown_buttons,
                direction   = 'down',
                showactive  = True,
                x           = 1.0,
                xanchor     = 'right',
                y           = 1.15,
                yanchor     = 'top',
                bgcolor     = '#f0f0f0',
                bordercolor = '#cccccc',
            )] if len(do_fibers) > 1 else [],
            sliders       = [make_slider(0)],
            xaxis         = dict(title='Pixel',       showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
            yaxis         = dict(title='Flux (e-/s)', showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
            hovermode     = 'closest',
            plot_bgcolor  = '#ffffff',
            paper_bgcolor = '#ffffff',
            font          = dict(color='#000000'),
            legend        = dict(bgcolor='#ffffff'),
            height        = 550,
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


def _filled_float(col):
    """Table column as a float ndarray with masked entries set to NaN."""
    return np.asarray(np.ma.filled(np.ma.asarray(col, dtype=float), np.nan), dtype=float)


def _vline_segments(x, y0, y1):
    """
    Build x/y arrays that draw a vertical segment from y0 to y1 at each x,
    separated by NaN gaps, so many lines can live in a single Scatter trace.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    xs = np.full(3 * n, np.nan)
    ys = np.full(3 * n, np.nan)
    xs[0::3] = x
    xs[1::3] = x
    ys[0::3] = y0
    ys[1::3] = y1
    return xs, ys


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------
def _read_linelist(path):
    """Line list wavelengths (um, vacuum) and relative fluxes; drops blank or invalid rows."""
    linelist = Table.read(path, format="ascii.cds")
    wvl = np.ma.filled(np.ma.asarray(linelist['WaveLen'], dtype=float), np.nan) / 1e3  # nm -> um
    flux = np.ma.filled(np.ma.asarray(linelist['RFlux'], dtype=float), np.nan)
    ok = np.isfinite(wvl) & (wvl > 0) & np.isfinite(flux) & (flux > 0)
    if not np.all(ok):
        logger.warning(f'Dropping {np.sum(~ok)} line-list rows with missing or invalid wavelength/flux')
    s = np.argsort(wvl[ok])
    return wvl[ok][s], flux[ok][s]


def _wave_to_pixel(wvl, wave_row, xarr):
    """Predicted pixel of each wavelength from one order's wavelength solution (NaN outside it)."""
    ok = np.isfinite(wave_row)
    if ok.sum() < 2:
        return np.full(len(wvl), np.nan)
    w, x = np.asarray(wave_row, dtype=float)[ok], np.asarray(xarr, dtype=float)[ok]
    s = np.argsort(w)
    w, x = w[s], x[s]
    if np.any(np.diff(w) <= 0):
        logger.warning("Input wavelength solution is not monotonic in this order; skipping it.")
        return np.full(len(wvl), np.nan)
    return np.interp(wvl, w, x, left=np.nan, right=np.nan)


def _nearest_neighbor_dist(pix, flux, neighbor_flux_frac):
    """Distance (px) from each line to the nearest OTHER line at least neighbor_flux_frac as bright."""
    d = np.abs(pix[:, None] - pix[None, :])
    significant = flux[None, :] >= neighbor_flux_frac * flux[:, None]
    d = np.where(significant, d, np.inf)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1) if len(pix) else np.array([])


def _smooth_offset(x, y, deg=2, nsigma=3.0, n_iter=5):
    """Robust low-order polynomial y(x); returns a callable. Falls back to the median."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0:
        return lambda t: np.zeros_like(np.asarray(t, dtype=float))
    deg = min(deg, max(x.size // 5 - 1, 0))
    keep = np.ones(x.size, dtype=bool)
    coeffs = np.array([np.median(y)])
    for _ in range(n_iter):
        coeffs = np.polyfit(x[keep], y[keep], deg) if deg > 0 else np.array([np.median(y[keep])])
        r = y - np.polyval(coeffs, x)
        sig = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        if not sig > 0:
            break
        new_keep = np.abs(r) < nsigma * sig
        if np.array_equal(new_keep, keep) or new_keep.sum() <= deg + 1:
            break
        keep = new_keep
    return lambda t: np.polyval(coeffs, np.asarray(t, dtype=float))


def _fit_order_lines(xarr, spec, spec_err, spec_dq, wave, lamp_wvl, lamp_flux, *,
                     line_fwhm_pix, fwhm_bounds, mu_bounds, max_fit_window,
                     neighbor_flux_frac, min_isolation_fwhm, max_shift_resid):
    """Fit the isolated lines of one order. Returns one row dict per fitted line (no 'ORDER')."""
    pred_all = _wave_to_pixel(lamp_wvl, wave, xarr)
    in_order = np.flatnonzero(np.isfinite(pred_all))
    if in_order.size == 0:
        return []
    pred, flux = pred_all[in_order], lamp_flux[in_order]

    # 1) Isolation, judged against the FULL list (unfit lines are still in the spectrum)
    d_nn = _nearest_neighbor_dist(pred, flux, neighbor_flux_frac)
    isolated = d_nn >= min_isolation_fwhm * line_fwhm_pix
    idx, pred, d_nn = in_order[isolated], pred[isolated], d_nn[isolated]
    if idx.size == 0:
        return []

    # 2) Fit; each window stops 1.5 FWHM short of the nearest neighbour
    half = np.minimum(d_nn - 1.5 * line_fwhm_pix, max_fit_window / 2)
    results = fit_lines_gauss(xarr, spec, spec_err, spec_dq, pred, fwhm_bounds=fwhm_bounds,
                              mu_bounds=mu_bounds, max_fit_window=max_fit_window,
                              half_windows=half, absorption=False)
    mu = np.array([r.mu for r in results])
    fwhm = np.array([r.fwhm for r in results])

    # 3) Suspect centers (and fits whose error estimate is unreliable)
    dmu = mu - pred
    at_bound = (np.abs(dmu - mu_bounds[0]) < 0.05) | (np.abs(dmu - mu_bounds[1]) < 0.05)
    # A parameter pinned at a bound makes the covariance, and so MU_ERR, meaningless
    fwhm_at_bound = (np.abs(fwhm - fwhm_bounds[0]) < 0.01) | (np.abs(fwhm - fwhm_bounds[1]) < 0.01)
    offset = _smooth_offset(pred[~at_bound], dmu[~at_bound])
    off_model = np.abs(dmu - offset(pred)) > max_shift_resid
    same_peak = np.zeros(mu.size, dtype=bool)
    fin = np.flatnonzero(np.isfinite(mu))
    if fin.size > 1:
        o = fin[np.argsort(mu[fin])]
        close = np.diff(mu[o]) < 0.5 * line_fwhm_pix
        same_peak[o[:-1][close]] = same_peak[o[1:][close]] = True
    suspect = at_bound | fwhm_at_bound | off_model | same_peak

    rows = []
    for i, r in enumerate(results):
        rows.append({
            'AMPLITUDE' : r.amplitude, 'AMPLITUDE_ERR' : r.amplitude_err,
            'MU_PIX' : r.mu, 'MU_ERR_PIX' : r.mu_err,
            'FWHM_PIX' : r.fwhm, 'FWHM_ERR_PIX' : r.fwhm_err,
            'REDCHI2' : r.redchi2,
            'LINE_NUM' : idx[i], 'WAVE' : lamp_wvl[idx[i]],
            'FWHM_WAVE' : delta_pix_to_delta_wave(r.fwhm, r.mu, wave),
            'FWHM_ERR_WAVE' : delta_pix_to_delta_wave(r.fwhm_err, r.mu, wave),
            'MU_ERR_WAVE' : delta_pix_to_delta_wave(r.mu_err, r.mu, wave),
            'LINE_DQ' : LINE_DQ_FLAGS['POOR_FIT'] if suspect[i] else np.uint8(0),
            'PRED_PIX' : pred[i],   # for the per-order log; removed before the row is stored
        })
    return rows


def _flag_fiber_lines(table, spec, spec_dq, order_start, lamp_wvl, min_snr, shape_nsigma):
    """Flags that compare lines with each other across the whole fiber."""
    num_cols = [c for c in table.itercols() if c.dtype.kind in 'fc']
    nan_mask = np.logical_or.reduce([np.isnan(np.asarray(c, dtype=float)) for c in num_cols])
    table['LINE_DQ'][nan_mask] |= LINE_DQ_FLAGS['FIT_FAILURE']

    low_snr = flag_low_amplitude_lines(
        table['ORDER'], table['MU_PIX'], table['AMPLITUDE'], table['FWHM_PIX'],
        spec, spec_dq, order_start, snr_threshold=min_snr,
    )
    table['LINE_DQ'][low_snr] |= LINE_DQ_FLAGS['LOW_SNR']

    # Shape outliers, judged among the lines that are otherwise fine
    ok = np.flatnonzero(~nan_mask & ~low_snr)
    sub = table[ok]
    shape_bad = np.zeros(len(table), dtype=bool)
    shape_bad[ok] = (flag_redchi2_outliers(sub['REDCHI2'], sub['AMPLITUDE'], nsigma=shape_nsigma)
                     | flag_fwhm_outliers(sub['FWHM_PIX'], lamp_wvl[sub['LINE_NUM']], nsigma=shape_nsigma))
    table['LINE_DQ'][shape_bad] |= LINE_DQ_FLAGS['POOR_FIT']
from ..base_primitive import BasePrimitive
from ... import datamodels

from .fit_lines import fit_lines_gauss, LINE_DQ_FLAGS
from .find_lines import find_emission_lines
from ..utils.spectral_utils import delta_pix_to_delta_wave

import numpy as np
import scipy.constants
from scipy.stats import median_abs_deviation
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
  
    ----------
    input : str or L2Model
        Input level 2 model or path to a level 2 file. Must have some initial wavelength solution included.
    fwhm_bounds : tuple, optional
        Bounds on the full width at half maximum (FWHM) of the lamp lines in pixels, given as a tuple (min, max).
    suffix : str, optional
        Suffix to add to the output filename when saving the result. Default is 'etalon_fit'.
    mu_bounds : tuple, optional
        Bounds on the mean (mu) of the etalon lines in pixels relative to the centroid, given as a tuple (min, max) in units of pixels.
    max_fit_window : int, optional
        Maximum number of pixels to include in the fit for each line.
    do_fibers : list of int, optional
        List of fiber numbers to process. If None, all fibers will be processed.
    do_orders : list of int, optional
        List of echelle orders to process. If None, all orders will be processed.
    min_snr : int or float, optional
        The minimum individual line signal-to-noise ratio that a line must have to not be flagged as low SNR in returned fitparams
    
    Returns
    -------
    output_model
        The input model with the Table attribute ``lamp_fitparams_fib#`` populated.
    """

    PARAMETERS = {
        'suffix' : 'lamp_fit',
        'fwhm_bounds' : (1,8),
        'mu_bounds' : (-3, 3),
        'max_fit_window' : None,
        'do_fibers' : None,
        'do_orders' : None,
        'min_snr' : 5.,
    }

    def _perform(self):

        input_model = datamodels.open(self.dataset)
        output_model = datamodels.UraniumFitParamsModel.from_datamodels(
            input_model, include_data=True,
            primitive=self
        )

        # Which traces to process
        do_fibers, do_orders = self.get_do_traces(input_model, self.do_fibers, self.do_orders)
        
        fit_results = {fiber : [] for fiber in do_fibers}

        # read and parse line list
        linelist = Table.read(os.path.join(LINELIST_DIR,"UNe_lines_nir.txt"), format="ascii.cds")
        lamp_line_wvl = np.array(linelist['WaveLen']) / 1e3 # was in nm now in um
        lamp_line_rflux = np.array(linelist['RFlux'])

        # predict blended lines from line list
        blended = _flag_blended_lines(
            np.asarray(lamp_line_wvl),
            np.asarray(lamp_line_rflux),
            tol=3e-5, # ~approximatelly twice the FWHM
            min_flux_frac=0.05, # neighbor >= 5% of this line's flux
        )

        clean_lamp_wvl = lamp_line_wvl[~blended]
        clean_lamp_rflux = lamp_line_rflux[~blended]
        line_nums = np.arange(len(lamp_line_wvl))
        clean_line_nums = line_nums[~blended]
        logger.info(f'Total lines in lamp linelist: {len(lamp_line_wvl)}')
        logger.info(f'Lines not known to be blended in lamp linelist: {len(clean_lamp_wvl)}')

        for fiber in do_fibers:
            # parse input data
            wave = getattr(input_model, f'wave_fib{fiber}')
            spec = getattr(input_model, f'spec_fib{fiber}')
            spec_err = getattr(input_model, f'spec_err_fib{fiber}')
            spec_dq = getattr(input_model, f'spec_dq_fib{fiber}')
            xarr = np.arange(wave.shape[1])

            for order in do_orders:

                # Relative order index
                oi = np.abs(order - input_model.meta.order_start)

                # find lines in order
                order_lines_idx = (clean_lamp_wvl > np.nanmin(wave[oi,:])) & (clean_lamp_wvl < np.nanmax(wave[oi,:]))
                lines_wvl = clean_lamp_wvl[order_lines_idx]
                order_line_nums = clean_line_nums[order_lines_idx]
                lines_pixel = np.interp(lines_wvl, wave[oi,:], xarr)

                logger.info(f'fiber {fiber}, order {order}, attempting to fit {len(order_line_nums)} lines')
                # Gaussian fit to lines
                _fit_results = fit_lines_gauss(
                    xarr, spec[oi, :], spec_err[oi, :], spec_dq[oi, :], lines_pixel,
                    fwhm_bounds=self.fwhm_bounds, mu_bounds=self.mu_bounds,
                    max_fit_window=self.max_fit_window,
                    absorption=False,
                )
                fit_results[fiber].append(_fit_results)

                # Store results in the model
                params_table = getattr(output_model, f'uranium_fitparams_fib{fiber}')
                if order in params_table['ORDER']:
                    setattr(input_model, f'uranium_fitparams_fib{fiber}', params_table[params_table['ORDER'] != order])
                n_lines = len(_fit_results)
                for i in range(n_lines):
                    r = {
                        'ORDER' : order,
                        'AMPLITUDE' : _fit_results[i].amplitude,
                        'AMPLITUDE_ERR' : _fit_results[i].amplitude_err,
                        'MU_PIX' : _fit_results[i].mu,
                        'MU_ERR_PIX' : _fit_results[i].mu_err,
                        'FWHM_PIX' : _fit_results[i].fwhm,
                        'FWHM_ERR_PIX' : _fit_results[i].fwhm_err,
                        'REDCHI2' : _fit_results[i].redchi2,
                        'LINE_NUM' : order_line_nums[i],
                    }
                    r['WAVE'] = np.interp(r['MU_PIX'], xarr, wave[oi, :], left=np.nan, right=np.nan)
                    r['FWHM_WAVE'] = delta_pix_to_delta_wave(r['FWHM_PIX'], r['MU_PIX'], wave[oi, :])
                    r['FWHM_ERR_WAVE'] = delta_pix_to_delta_wave(r['FWHM_ERR_PIX'], r['MU_PIX'], wave[oi, :])
                    r['MU_ERR_WAVE'] = delta_pix_to_delta_wave(r['MU_ERR_PIX'], r['MU_PIX'], wave[oi, :])
                    params_table.add_row(r)
                params_table.sort(['ORDER', 'MU_PIX'])

            # flag all fits that produced nans as fit failures
            nan_mask = np.logical_or.reduce([
                np.isnan(col) for col in params_table.itercols() if col.dtype.kind in 'fc'
            ])

            params_table['LINE_DQ'][nan_mask] = LINE_DQ_FLAGS['FIT_FAILURE']

            # flag low S/N lines
            low_amp_mask = _flag_low_amplitude_lines(
                params_table['ORDER'], params_table['MU_PIX'],
                params_table['AMPLITUDE'], params_table['FWHM_PIX'],
                spec, spec_dq, input_model.meta.order_start,
                snr_threshold=self.min_snr,
            )
            params_table['LINE_DQ'][low_amp_mask] |= LINE_DQ_FLAGS['LOW_SNR']

            # shape outlier checks on the remaining good lines only ---
            good = ~nan_mask & ~low_amp_mask
            good_idx = np.flatnonzero(good)
            sub = params_table[good_idx]

            redchi2_outlier_mask = np.zeros(len(params_table), dtype=bool)
            redchi2_outlier_mask[good_idx] = _flag_redchi2_outliers(
                sub['REDCHI2'], sub['AMPLITUDE']
            )
            params_table['LINE_DQ'][redchi2_outlier_mask] |= LINE_DQ_FLAGS['POOR_FIT']

            fwhm_outlier_mask = np.zeros(len(params_table), dtype=bool)
            fwhm_outlier_mask[good_idx] = _flag_fwhm_outliers(
                sub['FWHM_PIX'], lamp_line_wvl[sub['LINE_NUM']]
            )
            params_table['LINE_DQ'][fwhm_outlier_mask] |= LINE_DQ_FLAGS['POOR_FIT']

        return dict(output=output_model, fit_results=fit_results)


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



def _flag_blended_lines(wvl, flux, tol=3e-5, min_flux_frac=0.05):
    """
    Flag lines that have a neighbor within `tol` whose flux is significant
    relative to the line's own flux.

    Parameters
    ----------
    wvl : array-like
        Line wavelengths (same units as `tol`, e.g. um).
    flux : array-like
        Line relative fluxes (same length as `wvl`).
    tol : float
        Maximum wavelength separation for two lines to be considered blended.
    min_flux_frac : float
        Neighbor must have flux >= min_flux_frac * (this line's flux) to count
        as a blend. E.g. 0.05 = neighbor is at least 5% as bright as the line.

    Returns
    -------
    blended : ndarray of bool
        True where the line is likely blended. Same order as the input.
    """
    wvl = np.asarray(wvl, dtype=float)
    flux = np.asarray(flux, dtype=float)
    n = len(wvl)
    blended = np.zeros(n, dtype=bool)

    # Sort by wavelength so neighbors within tol are a contiguous slice
    order = np.argsort(wvl)
    w = wvl[order]
    f = flux[order]

    # Window of candidate neighbors for each line: [lo, hi)
    lo = np.searchsorted(w, w - tol, side="left")
    hi = np.searchsorted(w, w + tol, side="right")

    flagged_sorted = np.zeros(n, dtype=bool)
    for i in range(n):
        if hi[i] - lo[i] <= 1:      # only the line itself
            continue
        nbr_f = f[lo[i]:hi[i]]
        # Max flux among neighbors, excluding the line itself
        # (set own flux to -inf temporarily via mask)
        mask = np.ones(len(nbr_f), dtype=bool)
        mask[i - lo[i]] = False
        if nbr_f[mask].max() >= min_flux_frac * f[i]:
            flagged_sorted[i] = True

    # Undo the sort
    blended[order] = flagged_sorted
    return blended

import matplotlib.pyplot as plt

def _flag_low_amplitude_lines(
    orders, mu_pix, amplitude, fwhm_pix, spec, spec_dq, order_start,
    snr_threshold=3.0, window_fwhm=15.0, min_window_pix=20,
    exclude_fwhm=3.0, min_pixels=10,
):
    """
    Flag fitted lines whose amplitude is low relative to the local noise.

    The noise for each line is estimated as the robust scatter (1.4826 * MAD)
    of good pixels in a window around the line, excluding the line core. Using
    MAD makes the estimate insensitive to neighboring lines in the window.

    Parameters
    ----------
    orders, mu_pix, amplitude, fwhm_pix : array-like
        Columns from the fit-params table (ORDER, MU_PIX, AMPLITUDE, FWHM_PIX).
    spec : ndarray (n_orders, n_pix)
        Extracted spectrum for this fiber.
    spec_dq : ndarray (n_orders, n_pix)
        DQ array for this fiber. Pixels with nonzero DQ are ignored.
    order_start : int
        ``input_model.meta.order_start``, used to map order number -> row index.
    snr_threshold : float
        Lines with amplitude / local noise below this are flagged.
    window_fwhm : float
        Half-width of the noise window, in units of the line FWHM.
    min_window_pix : int
        Minimum half-width of the noise window in pixels.
    exclude_fwhm : float
        Half-width, in FWHM, of the region around the line excluded from the
        noise estimate.
    min_pixels : int
        Minimum number of good pixels required to estimate the noise. Lines
        with fewer are left unflagged (noise can't be assessed).

    Returns
    -------
    mask : ndarray of bool
        True where the line has low amplitude relative to the surrounding noise.
    """
    orders = np.asarray(orders)
    mu_pix = np.asarray(mu_pix, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    fwhm_pix = np.asarray(fwhm_pix, dtype=float)

    mask = np.zeros(len(orders), dtype=bool)
    xarr = np.arange(spec.shape[1])

    for k in range(len(orders)):
        mu, amp, fwhm = mu_pix[k], amplitude[k], fwhm_pix[k]

        # failed fits are handled by the FIT_FAILURE flag
        if not (np.isfinite(mu) and np.isfinite(amp) and np.isfinite(fwhm)):
            continue

        oi = np.abs(int(orders[k]) - order_start)
        dist = np.abs(xarr - mu)
        half_window = max(window_fwhm * fwhm, min_window_pix)

        sel = (
            (dist > exclude_fwhm * fwhm)
            & (dist < half_window)
            & (spec_dq[oi, :] == 0)
            & np.isfinite(spec[oi, :])
        )
        if sel.sum() < min_pixels:
            continue

        vals = spec[oi, sel]
        noise = 1.4826 * np.median(np.abs(vals - np.median(vals)))
        if not np.isfinite(noise) or noise <= 0:
            continue

        mask[k] = (amp / noise) < snr_threshold

    return mask


def _flag_redchi2_outliers(redchi2, amplitude, nsigma=3.0, deg=1, max_iter=5):
    """Flag lines whose redchi2 is high relative to the redchi2-amplitude trend."""
    redchi2 = np.asarray(redchi2, dtype=float)
    amp = np.asarray(amplitude, dtype=float)
    outlier = np.zeros(len(redchi2), dtype=bool)

    ok = np.isfinite(redchi2) & np.isfinite(amp) & (redchi2 > 0) & (amp > 0)
    if ok.sum() < deg + 5:
        return outlier

    x, y = np.log10(amp[ok]), np.log10(redchi2[ok])
    keep = np.ones(len(x), dtype=bool)
    for _ in range(max_iter):
        coeffs = np.polyfit(x[keep], y[keep], deg)
        resid = y - np.polyval(coeffs, x)
        sigma = median_abs_deviation(resid[keep], scale='normal')
        new_keep = np.abs(resid) < nsigma * sigma
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    outlier[ok] = resid > nsigma * sigma   # one-sided: only worse-than-trend fits
    return outlier

def _flag_fwhm_outliers(fwhm, line_wavelength, nsigma=3.0, deg=1, max_iter=5):
    """Flag lines whose fwhm is an outlier relative to the fwhm-wavelength trend."""
    fwhm = np.asarray(fwhm, dtype=float)
    outlier = np.zeros(len(fwhm), dtype=bool)

    ok = np.isfinite(fwhm) & np.isfinite(line_wavelength) & (fwhm > 0) & (line_wavelength > 0)
    if ok.sum() < deg + 5:
        return outlier

    x, y = line_wavelength[ok], fwhm[ok]
    keep = np.ones(len(x), dtype=bool)
    for _ in range(max_iter):
        coeffs = np.polyfit(x[keep], y[keep], deg)
        resid = y - np.polyval(coeffs, x)
        sigma = median_abs_deviation(resid[keep], scale='normal')
        new_keep = np.abs(resid) < nsigma * sigma
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    outlier[ok] = resid > nsigma * sigma   # one-sided: only worse-than-trend fits
    return outlier
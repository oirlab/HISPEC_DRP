"""
Masking-based cross-correlation and RV combination utilities.

Used by CrossCorrelationPrimitive to compute radial velocity per spectral chunk
and combine with median/MAD outlier rejection and inverse-variance weighting.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from astropy.stats import mad_std

try:
    from numba import njit, prange, set_num_threads
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        def dec(f):
            return f
        return dec
    prange = range  # noqa: A001
    def set_num_threads(n):  # noqa: ARG001
        pass

# Speed of light in m/s for RV from log-wavelength shift (delta_ln = v / c)
C_M_S = 299792458.0

# Default CCF velocity search (m/s)
_V_MIN = -500000.0
_V_MAX = 500000.0
_V_STEP = 500.0


@njit(cache=True)
def _ccf_chunk_rv_numba(
    ln_lam: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    good: np.ndarray,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    v_min: float,
    v_max: float,
    v_step: float,
) -> tuple[float, float]:
    """
    Compute RV and uncertainty for a single wavelength chunk (Numba-jitted).

    Correlates the chunk spectrum with the template over a velocity grid;
    uses linear interpolation (np.interp) to shift the template in log(lambda).
    Returns (nan, nan) if fewer than 10 good pixels or no valid CCF peak.

    Parameters
    ----------
    ln_lam : np.ndarray
        Log(wavelength) in log(micron) for the chunk (1D).
    spec : np.ndarray
        Spectrum flux for the chunk (1D).
    spec_err : np.ndarray
        Spectrum uncertainty (1D).
    good : np.ndarray, bool
        True where pixel is valid for CCF.
    template_ln : np.ndarray
        Template log(wavelength) (log micron).
    template_flux : np.ndarray
        Template flux.
    v_min, v_max, v_step : float
        Velocity search range and step (m/s).

    Returns
    -------
    rv : float
        Radial velocity in m/s (nan if invalid).
    rv_err : float
        Approximate uncertainty in m/s.
    """
    n_good = 0
    for i in range(len(good)):
        if good[i]:
            n_good += 1
    if n_good < 10:
        return np.nan, np.nan

    n_vel = int((v_max - v_min) / v_step) + 1
    ccf = np.empty(n_vel)
    ccf[:] = np.nan
    v_grid = np.empty(n_vel)
    for i in range(n_vel):
        v_grid[i] = v_min + i * v_step

    for iv in range(n_vel):
        v = v_grid[iv]
        delta_ln = v / C_M_S
        t_ln_shifted = template_ln + delta_ln
        t_at_data = np.interp(ln_lam, t_ln_shifted, template_flux)
        sum_w = 0.0
        sum_d = 0.0
        sum_t = 0.0
        n_use = 0
        for i in range(len(ln_lam)):
            if good[i] and np.isfinite(t_at_data[i]) and t_at_data[i] != 0 and spec_err[i] > 0:
                w = 1.0 / max(spec_err[i] ** 2, 1e-30)
                sum_w += w
                sum_d += spec[i] * w
                sum_t += t_at_data[i] * w
                n_use += 1
        if n_use < 5:
            continue
        d_mean = sum_d / sum_w
        t_mean = sum_t / sum_w
        cov = 0.0
        for i in range(len(ln_lam)):
            if good[i] and np.isfinite(t_at_data[i]) and t_at_data[i] != 0 and spec_err[i] > 0:
                w = 1.0 / max(spec_err[i] ** 2, 1e-30)
                cov += (spec[i] - d_mean) * (t_at_data[i] - t_mean) * w
        ccf[iv] = cov / sum_w

    ccf_min = np.nan
    ccf_max = np.nan
    for i in range(n_vel):
        if np.isfinite(ccf[i]):
            if np.isnan(ccf_min) or ccf[i] < ccf_min:
                ccf_min = ccf[i]
            if np.isnan(ccf_max) or ccf[i] > ccf_max:
                ccf_max = ccf[i]
    if not np.isfinite(ccf_max):
        return np.nan, np.nan

    i_max = 0
    for i in range(n_vel):
        if np.isfinite(ccf[i]) and ccf[i] >= ccf_max:
            i_max = i
            break
    rv = v_grid[i_max]
    if ccf_max <= ccf_min or not np.isfinite(ccf_min):
        rv_err = v_step * 2.0
        return rv, rv_err
    half = (ccf_max + ccf_min) / 2.0
    left = 0
    for i in range(i_max - 1, -1, -1):
        if ccf[i] <= half:
            left = i
            break
    right = n_vel - 1
    for i in range(i_max + 1, n_vel):
        if ccf[i] <= half:
            right = i
            break
    hwhm = (right - left) * v_step * 0.5
    rv_err = max(hwhm / np.sqrt(n_good / 50.0), v_step)
    return rv, rv_err


@njit(parallel=True, cache=True)
def process_all_orders_rvs_numba(
    wave_2d: np.ndarray,
    spec_2d: np.ndarray,
    spec_err_2d: np.ndarray,
    dq_2d: np.ndarray,
    do_not_use: int,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    window_size_um: float,
    order_indices: np.ndarray,
    n_chunks_per_order: np.ndarray,
    chunk_start: np.ndarray,
    v_min: float,
    v_max: float,
    v_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process multiple orders in parallel (Numba prange over orders).

    For each order index in order_indices, clips the template to the order
    wavelength range, splits the order into chunks of window_size_um, and
    computes RV and uncertainty per chunk via _ccf_chunk_rv_numba. Writes
    results into preallocated output arrays at chunk_start offsets.

    Parameters
    ----------
    wave_2d : np.ndarray
        Wavelength per order (n_orders x n_pix), micron.
    spec_2d : np.ndarray
        Spectrum flux (n_orders x n_pix).
    spec_err_2d : np.ndarray
        Spectrum uncertainty (n_orders x n_pix).
    dq_2d : np.ndarray, uint32
        Data quality (n_orders x n_pix); do_not_use bits mark bad pixels.
    do_not_use : int
        DQ flag value for bad pixels.
    template_ln : np.ndarray
        Full template log(wavelength) in log(micron).
    template_flux : np.ndarray
        Full template flux.
    window_size_um : float
        Chunk size in micron.
    order_indices : np.ndarray, int
        Indices of orders to process (length n_processed).
    n_chunks_per_order : np.ndarray, int
        Number of chunks per processed order (length n_processed).
    chunk_start : np.ndarray, int
        Start index in output arrays for each processed order (length n_processed).
    v_min, v_max, v_step : float
        CCF velocity search range and step (m/s).

    Returns
    -------
    all_rvs : np.ndarray
        Per-chunk RV in m/s (length total_chunks); nan where invalid.
    all_rv_errs : np.ndarray
        Per-chunk RV uncertainty in m/s.
    all_wl : np.ndarray
        Per-chunk center wavelength in micron.
    """
    n_processed = len(order_indices)
    total_chunks = chunk_start[n_processed - 1] + n_chunks_per_order[n_processed - 1]
    all_rvs = np.empty(total_chunks)
    all_rvs[:] = np.nan
    all_rv_errs = np.empty(total_chunks)
    all_rv_errs[:] = np.nan
    all_wl = np.empty(total_chunks)
    all_wl[:] = np.nan

    for idx in prange(n_processed):
        oi = order_indices[idx]
        w = wave_2d[oi]
        s = spec_2d[oi]
        e = spec_err_2d[oi]
        d = dq_2d[oi]
        n_pix = len(w)
        mask = np.empty(n_pix, dtype=np.bool_)
        for i in range(n_pix):
            mask[i] = (d[i] & do_not_use) != 0
        good = np.empty(n_pix, dtype=np.bool_)
        for i in range(n_pix):
            good[i] = not mask[i] and np.isfinite(s[i]) and np.isfinite(e[i]) and e[i] > 0

        w_min = np.nanmin(w)
        w_max = np.nanmax(w)
        ln_min = np.log(w_min) - 0.01
        ln_max = np.log(w_max) + 0.01
        n_tpl = 0
        for i in range(len(template_ln)):
            if template_ln[i] >= ln_min and template_ln[i] <= ln_max:
                n_tpl += 1
        if n_tpl < 20:
            continue
        t_ln = np.empty(n_tpl)
        t_flux = np.empty(n_tpl)
        j = 0
        for i in range(len(template_ln)):
            if template_ln[i] >= ln_min and template_ln[i] <= ln_max:
                t_ln[j] = template_ln[i]
                t_flux[j] = template_flux[i]
                j += 1

        n_chunks = n_chunks_per_order[idx]
        start = chunk_start[idx]
        for k in range(n_chunks):
            lam_lo = w_min + k * (w_max - w_min) / n_chunks
            lam_hi = w_min + (k + 1) * (w_max - w_min) / n_chunks
            in_chunk = np.empty(n_pix, dtype=np.bool_)
            chunk_len = 0
            for i in range(n_pix):
                in_chunk[i] = w[i] >= lam_lo and w[i] < lam_hi
                if in_chunk[i]:
                    chunk_len += 1
            if chunk_len < 20:
                all_wl[start + k] = 0.5 * (lam_lo + lam_hi)
                continue
            ln_lam_c = np.empty(chunk_len)
            spec_c = np.empty(chunk_len)
            spec_err_c = np.empty(chunk_len)
            good_c = np.empty(chunk_len, dtype=np.bool_)
            j = 0
            for i in range(n_pix):
                if in_chunk[i]:
                    ln_lam_c[j] = np.log(w[i])
                    spec_c[j] = s[i]
                    spec_err_c[j] = e[i]
                    good_c[j] = good[i]
                    j += 1
            rv, rv_err = _ccf_chunk_rv_numba(
                ln_lam_c, spec_c, spec_err_c, good_c,
                t_ln, t_flux, v_min, v_max, v_step,
            )
            all_rvs[start + k] = rv
            all_rv_errs[start + k] = rv_err
            all_wl[start + k] = 0.5 * (lam_lo + lam_hi)

    return all_rvs, all_rv_errs, all_wl


def process_orders_rvs_numba_parallel(
    wave_2d: np.ndarray,
    spec_2d: np.ndarray,
    spec_err_2d: np.ndarray,
    dq_2d: np.ndarray,
    do_not_use: int,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    window_size_um: float,
    order_tasks: list,
    max_cores: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process multiple orders in parallel using Numba prange.

    Builds order_indices, n_chunks_per_order, and chunk_start from order_tasks,
    then calls process_all_orders_rvs_numba. When max_cores > 1, sets Numba's
    thread count to max_cores for the call.

    Parameters
    ----------
    wave_2d : np.ndarray
        Wavelength per order (n_orders x n_pix), micron.
    spec_2d : np.ndarray
        Spectrum flux (n_orders x n_pix).
    spec_err_2d : np.ndarray
        Spectrum uncertainty (n_orders x n_pix).
    dq_2d : np.ndarray, uint32
        Data quality (n_orders x n_pix).
    do_not_use : int
        DQ flag value for bad pixels.
    template_ln : np.ndarray
        Template log(wavelength) in log(micron).
    template_flux : np.ndarray
        Template flux.
    window_size_um : float
        Chunk size in micron.
    order_tasks : list of tuple (int, int)
        (order_index, order_number) for each order to process.
    max_cores : int, optional
        Maximum number of threads for Numba parallel (default 1).

    Returns
    -------
    all_rvs : np.ndarray
        Per-chunk RV in m/s (1D).
    all_rv_errs : np.ndarray
        Per-chunk RV uncertainty in m/s (1D).
    all_wl_centers : np.ndarray
        Per-chunk center wavelength in micron (1D).
    """
    if not order_tasks:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
    wave_2d = np.asarray(wave_2d, dtype=np.float64)
    spec_2d = np.asarray(spec_2d, dtype=np.float64)
    spec_err_2d = np.asarray(spec_err_2d, dtype=np.float64)
    dq_2d = np.asarray(dq_2d, dtype=np.uint32)
    template_ln = np.asarray(template_ln, dtype=np.float64)
    template_flux = np.asarray(template_flux, dtype=np.float64)

    order_indices = np.array([oi for oi, _ in order_tasks], dtype=np.int64)
    n_chunks_per_order = np.array([
        max(1, int((np.nanmax(wave_2d[oi]) - np.nanmin(wave_2d[oi])) / window_size_um))
        for oi, _ in order_tasks
    ], dtype=np.int64)
    chunk_start = np.zeros(len(order_indices), dtype=np.int64)
    for i in range(1, len(order_indices)):
        chunk_start[i] = chunk_start[i - 1] + n_chunks_per_order[i - 1]

    if _HAS_NUMBA:
        set_num_threads(max_cores)
    all_rvs, all_rv_errs, all_wl = process_all_orders_rvs_numba(
        wave_2d, spec_2d, spec_err_2d, dq_2d, do_not_use,
        template_ln, template_flux, window_size_um,
        order_indices, n_chunks_per_order, chunk_start,
        _V_MIN, _V_MAX, _V_STEP,
    )
    return all_rvs, all_rv_errs, all_wl


def continuum_normalize(
    flux: np.ndarray,
    wavelength: np.ndarray | None = None,
    lowpass_window: int | None = None,
    poly_order: int | None = None,
) -> np.ndarray:
    """
    Continuum-normalize flux by dividing by a smooth continuum (low-pass and/or polynomial).

    Continuum is estimated as: (optional) median filter then (optional) polynomial fit.
    If both are set, the median-filtered flux is used as the continuum (low-pass only),
    or a polynomial is fitted to the median-filtered flux for a smoother result.

    Parameters
    ----------
    flux : np.ndarray
        Flux array (1D).
    wavelength : np.ndarray or None
        Wavelength array (same length as flux). Used for polynomial fit if poly_order is set.
    lowpass_window : int or None
        Window size for median filter (must be odd; if even, 1 is added). None to skip.
    poly_order : int or None
        Polynomial order for continuum fit. Fit is in (wavelength or index) vs flux.
        None to skip.

    Returns
    -------
    np.ndarray
        Normalized flux (flux / continuum). Non-finite where continuum <= 0.
    """
    flux = np.asarray(flux, dtype=np.float64)
    n = len(flux)
    if n == 0 or (lowpass_window is None and poly_order is None):
        return flux
    x = np.asarray(wavelength, dtype=np.float64) if wavelength is not None else np.arange(n, dtype=np.float64)
    if len(x) != n:
        x = np.arange(n, dtype=np.float64)

    continuum = np.full_like(flux, np.nan)
    if lowpass_window is not None and lowpass_window > 0:
        w = int(lowpass_window)
        if w % 2 == 0:
            w += 1
        w = min(w, n)
        if w >= 3:
            continuum = median_filter(flux, size=w, mode="nearest")
        else:
            continuum = np.where(np.isfinite(flux), flux, np.nan)
    else:
        continuum = np.where(np.isfinite(flux), flux, np.nan)

    if poly_order is not None and poly_order >= 0:
        ok = np.isfinite(x) & np.isfinite(continuum) & (continuum > 0)
        if np.sum(ok) > max(poly_order + 1, 5):
            try:
                p = np.polyfit(x[ok], continuum[ok], poly_order)
                continuum = np.polyval(p, x)
            except Exception:
                pass

    continuum = np.maximum(continuum, 1e-30)
    return np.where(np.isfinite(flux) & np.isfinite(continuum), flux / continuum, flux)


def _ln_lam(wave: np.ndarray) -> np.ndarray:
    """
    Natural log of wavelength.

    Parameters
    ----------
    wave : np.ndarray
        Wavelength in micron.

    Returns
    -------
    np.ndarray
        log(wave) in log(micron), float64.
    """
    return np.log(np.asarray(wave, dtype=np.float64))


def masked_ccf_rv(
    wave: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    mask: np.ndarray,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    v_min: float = -500000.0,
    v_max: float = 500000.0,
    v_step: float = 500.0,
) -> tuple[float, float]:
    """
    Compute radial velocity (m/s) and uncertainty from masking-based cross correlation.

    Data and template are in wavelength space; correlation is done in log(lambda).
    Bad pixels (mask True) are excluded. Returns RV and approximate uncertainty (from
    curvature of CCF peak or from step size).

    Parameters
    ----------
    wave : np.ndarray
        Wavelength in micron (length n).
    spec : np.ndarray
        Spectrum flux (length n).
    spec_err : np.ndarray
        Spectrum uncertainty (length n).
    mask : np.ndarray, bool
        True = bad pixel, do not use.
    template_ln : np.ndarray
        Template log(wavelength) in same units as wave (log micron).
    template_flux : np.ndarray
        Template flux on template_ln grid.
    v_min, v_max, v_step : float
        Velocity search range in m/s and step (m/s).

    Returns
    -------
    rv : float
        Radial velocity in m/s.
    rv_err : float
        Approximate uncertainty in m/s.
    """
    good = ~mask & np.isfinite(spec) & np.isfinite(spec_err) & (spec_err > 0)
    if np.sum(good) < 10:
        return np.nan, np.nan

    ln_lam = _ln_lam(wave)
    v_grid = np.arange(v_min, v_max + v_step * 0.5, v_step, dtype=np.float64)
    ccf = np.zeros_like(v_grid)

    for i, v in enumerate(v_grid):
        # Shift template in log space: ln_lam_shifted = ln_lam + v/c (v in m/s)
        delta_ln = v / C_M_S
        template_ln_shifted = template_ln + delta_ln
        # Interpolate template onto data log wavelength (linear; 0 outside bounds)
        t_at_data = np.interp(
            ln_lam, template_ln_shifted, template_flux, left=0.0, right=0.0
        )
        # Masked correlation: sum over good pixels only, normalize
        use = good & np.isfinite(t_at_data) & (t_at_data != 0)
        if np.sum(use) < 5:
            ccf[i] = np.nan
            continue
        d = spec[use]
        t = t_at_data[use]
        w = 1.0 / np.maximum(spec_err[use] ** 2, 1e-30)
        d_norm = d - np.average(d, weights=w)
        t_norm = t - np.average(t, weights=w)
        ccf[i] = np.average(d_norm * t_norm, weights=w)

    valid = np.isfinite(ccf)
    if not np.any(valid):
        return np.nan, np.nan
    i_max = np.nanargmax(ccf)
    rv = float(v_grid[i_max])
    # Rough uncertainty from half-width at half max or from step
    ccf_max = ccf[i_max]
    if ccf_max > np.nanmin(ccf):
        half = (ccf_max + np.nanmin(ccf)) / 2.0
        below = np.where(ccf[:i_max] <= half)[0]
        above = np.where(ccf[i_max:] <= half)[0]
        left = int(below[-1]) if len(below) else 0
        right = i_max + int(above[0]) if len(above) else len(v_grid) - 1
        hwhm = (right - left) * v_step * 0.5
        rv_err = float(max(hwhm / np.sqrt(np.sum(good) / 50.0), v_step))
    else:
        rv_err = float(v_step * 2)
    return rv, rv_err


def masked_ccf_rv_full(
    wave: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    mask: np.ndarray,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    v_min: float = -500000.0,
    v_max: float = 500000.0,
    v_step: float = 500.0,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Same as masked_ccf_rv but also return the velocity grid and full CCF array.

    All velocities are in m/s.

    Returns
    -------
    rv : float
        Radial velocity in m/s.
    rv_err : float
        Uncertainty in m/s.
    v_grid : np.ndarray
        Velocity grid in m/s (length n_vel).
    ccf : np.ndarray
        CCF values (length n_vel).
    """
    good = ~mask & np.isfinite(spec) & np.isfinite(spec_err) & (spec_err > 0)
    if np.sum(good) < 10:
        v_grid = np.arange(v_min, v_max + v_step * 0.5, v_step, dtype=np.float64)
        return np.nan, np.nan, v_grid, np.full_like(v_grid, np.nan)

    ln_lam = _ln_lam(wave)
    v_grid = np.arange(v_min, v_max + v_step * 0.5, v_step, dtype=np.float64)
    ccf = np.zeros_like(v_grid)

    for i, v in enumerate(v_grid):
        delta_ln = v / C_M_S
        template_ln_shifted = template_ln + delta_ln
        t_at_data = np.interp(
            ln_lam, template_ln_shifted, template_flux, left=0.0, right=0.0
        )
        use = good & np.isfinite(t_at_data) & (t_at_data != 0)
        if np.sum(use) < 5:
            ccf[i] = np.nan
            continue
        d = spec[use]
        t = t_at_data[use]
        w = 1.0 / np.maximum(spec_err[use] ** 2, 1e-30)
        d_norm = d - np.average(d, weights=w)
        t_norm = t - np.average(t, weights=w)
        ccf[i] = np.average(d_norm * t_norm, weights=w)

    valid = np.isfinite(ccf)
    if not np.any(valid):
        return np.nan, np.nan, v_grid, ccf
    i_max = np.nanargmax(ccf)
    rv = float(v_grid[i_max])
    ccf_max = ccf[i_max]
    if ccf_max > np.nanmin(ccf):
        half = (ccf_max + np.nanmin(ccf)) / 2.0
        below = np.where(ccf[:i_max] <= half)[0]
        above = np.where(ccf[i_max:] <= half)[0]
        left = int(below[-1]) if len(below) else 0
        right = i_max + int(above[0]) if len(above) else len(v_grid) - 1
        hwhm = (right - left) * v_step * 0.5
        rv_err = float(max(hwhm / np.sqrt(np.sum(good) / 50.0), v_step))
    else:
        rv_err = float(v_step * 2)
    return rv, rv_err, v_grid, ccf


def masked_ccf_rv_diagnostics(
    wave: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    mask: np.ndarray,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    v_min: float = -500000.0,
    v_max: float = 500000.0,
    v_step: float = 500.0,
) -> tuple[float, float, dict]:
    """
    Same as masked_ccf_rv but also return a diagnostics dict for debugging.

    Parameters
    ----------
    wave : np.ndarray
        Wavelength in micron (length n).
    spec : np.ndarray
        Spectrum flux (length n).
    spec_err : np.ndarray
        Spectrum uncertainty (length n).
    mask : np.ndarray, bool
        True = bad pixel, do not use.
    template_ln : np.ndarray
        Template log(wavelength) in log(micron).
    template_flux : np.ndarray
        Template flux.
    v_min, v_max, v_step : float, optional
        Velocity search range and step in m/s.

    Returns
    -------
    rv : float
        Radial velocity in m/s.
    rv_err : float
        Uncertainty in m/s.
    diagnostics : dict
        Keys: ccf_max, ccf_min, n_good, n_use_at_peak, spec_mean, spec_std,
        template_mean_at_peak, template_std_at_peak, wl_min, wl_max.
    """
    good = ~mask & np.isfinite(spec) & np.isfinite(spec_err) & (spec_err > 0)
    n_good = int(np.sum(good))
    wl_min = float(np.nanmin(wave))
    wl_max = float(np.nanmax(wave))
    spec_mean = float(np.nanmean(spec[good])) if n_good > 0 else np.nan
    spec_std = float(np.nanstd(spec[good])) if n_good > 1 else 0.0

    if n_good < 10:
        return np.nan, np.nan, {
            "ccf_max": np.nan, "ccf_min": np.nan, "n_good": n_good,
            "n_use_at_peak": 0, "spec_mean": spec_mean, "spec_std": spec_std,
            "template_mean_at_peak": np.nan, "template_std_at_peak": np.nan,
            "wl_min": wl_min, "wl_max": wl_max,
        }

    ln_lam = _ln_lam(wave)
    v_grid = np.arange(v_min, v_max + v_step * 0.5, v_step, dtype=np.float64)
    ccf = np.zeros_like(v_grid)

    for i, v in enumerate(v_grid):
        delta_ln = v / C_M_S
        template_ln_shifted = template_ln + delta_ln
        t_at_data = np.interp(
            ln_lam, template_ln_shifted, template_flux, left=0.0, right=0.0
        )
        use = good & np.isfinite(t_at_data) & (t_at_data != 0)
        if np.sum(use) < 5:
            ccf[i] = np.nan
            continue
        d = spec[use]
        t = t_at_data[use]
        w = 1.0 / np.maximum(spec_err[use] ** 2, 1e-30)
        d_norm = d - np.average(d, weights=w)
        t_norm = t - np.average(t, weights=w)
        ccf[i] = np.average(d_norm * t_norm, weights=w)

    valid = np.isfinite(ccf)
    if not np.any(valid):
        return np.nan, np.nan, {
            "ccf_max": np.nan, "ccf_min": np.nan, "n_good": n_good,
            "n_use_at_peak": 0, "spec_mean": spec_mean, "spec_std": spec_std,
            "template_mean_at_peak": np.nan, "template_std_at_peak": np.nan,
            "wl_min": wl_min, "wl_max": wl_max,
        }
    i_max = np.nanargmax(ccf)
    rv = float(v_grid[i_max])
    ccf_max = float(ccf[i_max])
    ccf_min = float(np.nanmin(ccf))
    if ccf_max > ccf_min:
        half = (ccf_max + ccf_min) / 2.0
        below = np.where(ccf[:i_max] <= half)[0]
        above = np.where(ccf[i_max:] <= half)[0]
        left = int(below[-1]) if len(below) else 0
        right = i_max + int(above[0]) if len(above) else len(v_grid) - 1
        hwhm = (right - left) * v_step * 0.5
        rv_err = float(max(hwhm / np.sqrt(n_good / 50.0), v_step))
    else:
        rv_err = float(v_step * 2)

    delta_ln_peak = rv / C_M_S
    template_ln_shifted = template_ln + delta_ln_peak
    t_at_data = np.interp(
        ln_lam, template_ln_shifted, template_flux, left=0.0, right=0.0
    )
    use_peak = good & np.isfinite(t_at_data) & (t_at_data != 0)
    n_use_at_peak = int(np.sum(use_peak))
    t_mean = float(np.mean(t_at_data[use_peak])) if n_use_at_peak > 0 else np.nan
    t_std = float(np.std(t_at_data[use_peak])) if n_use_at_peak > 1 else 0.0

    diagnostics = {
        "ccf_max": ccf_max,
        "ccf_min": ccf_min,
        "n_good": n_good,
        "n_use_at_peak": n_use_at_peak,
        "spec_mean": spec_mean,
        "spec_std": spec_std,
        "template_mean_at_peak": t_mean,
        "template_std_at_peak": t_std,
        "wl_min": wl_min,
        "wl_max": wl_max,
    }
    return rv, rv_err, diagnostics


def rv_per_chunks(
    wave: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    mask: np.ndarray,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    window_size_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute RV and uncertainty in each wavelength chunk.

    Parameters
    ----------
    wave, spec, spec_err, mask : np.ndarray
        Per-pixel arrays (same length).
    template_ln, template_flux : np.ndarray
        Template in log wavelength and flux.
    window_size_um : float
        Chunk size in micron.

    Returns
    -------
    rvs : np.ndarray
        RV in m/s per chunk.
    rv_errs : np.ndarray
        Uncertainty in m/s per chunk.
    """
    w_min, w_max = np.nanmin(wave), np.nanmax(wave)
    n_chunk = max(1, int((w_max - w_min) / window_size_um))
    rvs = np.full(n_chunk, np.nan, dtype=np.float64)
    rv_errs = np.full(n_chunk, np.nan, dtype=np.float64)
    for k in range(n_chunk):
        lam_lo = w_min + k * (w_max - w_min) / n_chunk
        lam_hi = w_min + (k + 1) * (w_max - w_min) / n_chunk
        in_chunk = (wave >= lam_lo) & (wave < lam_hi)
        if np.sum(in_chunk) < 20:
            continue
        rv, rv_err = masked_ccf_rv(
            wave[in_chunk],
            spec[in_chunk],
            spec_err[in_chunk],
            mask[in_chunk],
            template_ln,
            template_flux,
        )
        rvs[k] = rv
        rv_errs[k] = rv_err
    return rvs, rv_errs


def process_one_order_rvs(
    wave: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    dq: np.ndarray,
    do_not_use: int,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    window_size_um: float,
    do_continuum_normalize: bool = False,
    continuum_lowpass_window: int | None = None,
    continuum_poly_order: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a single order: optionally continuum-normalize, clip template to order
    wavelength range, compute RV and uncertainty per chunk. Standalone for parallel
    execution (no logger, no non-picklable refs).

    Parameters
    ----------
    wave, spec, spec_err : np.ndarray
        Wavelength (micron), flux, and error for one order (1D, same length).
    dq : np.ndarray
        Data quality (uint32); bad pixels flagged with do_not_use are masked.
    do_not_use : int
        DQ flag value to treat as bad (e.g. 1).
    template_ln, template_flux : np.ndarray
        Full template in log(wavelength) and flux.
    window_size_um : float
        Chunk size in micron.
    do_continuum_normalize : bool
        If True, continuum-normalize spec before CCF.
    continuum_lowpass_window, continuum_poly_order : int or None
        Passed to continuum_normalize if do_continuum_normalize is True.

    Returns
    -------
    rvs : np.ndarray
        RV in m/s per chunk (may be empty if order skipped).
    rv_errs : np.ndarray
        Uncertainty in m/s per chunk.
    chunk_centers : np.ndarray
        Chunk center wavelength (micron) per chunk.
    """
    w = np.asarray(wave, dtype=np.float64)
    s = np.asarray(spec, dtype=np.float64).copy()
    e = np.asarray(spec_err, dtype=np.float64)
    mask = (np.asarray(dq, dtype=np.uint32) & do_not_use) != 0

    if do_continuum_normalize and (continuum_lowpass_window is not None or continuum_poly_order is not None):
        s = continuum_normalize(
            s,
            wavelength=w,
            lowpass_window=continuum_lowpass_window,
            poly_order=continuum_poly_order,
        )

    w_min, w_max = np.nanmin(w), np.nanmax(w)
    ln_min, ln_max = np.log(w_min), np.log(w_max)
    in_range = (template_ln >= ln_min - 0.01) & (template_ln <= ln_max + 0.01)
    if np.sum(in_range) < 20:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    t_ln = template_ln[in_range]
    t_flux = template_flux[in_range]

    rvs, rv_errs = rv_per_chunks(w, s, e, mask, t_ln, t_flux, window_size_um)
    n_chunk = len(rvs)
    chunk_centers = np.linspace(w_min, w_max, n_chunk + 1)
    chunk_centers = 0.5 * (chunk_centers[:-1] + chunk_centers[1:])
    return rvs, rv_errs, chunk_centers


def rv_per_chunks_with_ccf(
    wave: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    mask: np.ndarray,
    template_ln: np.ndarray,
    template_flux: np.ndarray,
    window_size_um: float,
    v_min: float = -500000.0,
    v_max: float = 500000.0,
    v_step: float = 500.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute RV and full CCF in each wavelength chunk.

    Uses the same velocity grid for all chunks. All velocities are in m/s.

    Parameters
    ----------
    wave, spec, spec_err, mask : np.ndarray
        Per-pixel arrays (same length), wavelength in micron.
    template_ln, template_flux : np.ndarray
        Template in log(wavelength) and flux.
    window_size_um : float
        Chunk size in micron.
    v_min, v_max, v_step : float, optional
        Velocity search range and step in m/s.

    Returns
    -------
    rvs : np.ndarray
        RV per chunk (m/s).
    rv_errs : np.ndarray
        RV uncertainty per chunk (m/s).
    chunk_wavelengths : np.ndarray
        Chunk center wavelength (micron) per chunk.
    v_grid : np.ndarray
        Velocity grid in m/s (same for all chunks).
    ccf_2d : np.ndarray
        Shape (n_chunks, n_vel); CCF values per chunk (nan where chunk invalid).
    """
    w_min, w_max = np.nanmin(wave), np.nanmax(wave)
    n_chunk = max(1, int((w_max - w_min) / window_size_um))
    v_grid = np.arange(v_min, v_max + v_step * 0.5, v_step, dtype=np.float64)
    rvs = np.full(n_chunk, np.nan, dtype=np.float64)
    rv_errs = np.full(n_chunk, np.nan, dtype=np.float64)
    ccf_2d = np.full((n_chunk, len(v_grid)), np.nan, dtype=np.float64)
    chunk_centers = np.full(n_chunk, np.nan, dtype=np.float64)

    for k in range(n_chunk):
        lam_lo = w_min + k * (w_max - w_min) / n_chunk
        lam_hi = w_min + (k + 1) * (w_max - w_min) / n_chunk
        in_chunk = (wave >= lam_lo) & (wave < lam_hi)
        if np.sum(in_chunk) < 20:
            chunk_centers[k] = 0.5 * (lam_lo + lam_hi)
            continue
        rv, rv_err, vg, ccf = masked_ccf_rv_full(
            wave[in_chunk],
            spec[in_chunk],
            spec_err[in_chunk],
            mask[in_chunk],
            template_ln,
            template_flux,
            v_min=v_min,
            v_max=v_max,
            v_step=v_step,
        )
        rvs[k] = rv
        rv_errs[k] = rv_err
        chunk_centers[k] = 0.5 * (lam_lo + lam_hi)
        ccf_2d[k, :] = ccf
    return rvs, rv_errs, chunk_centers, v_grid, ccf_2d


def combine_rvs_median_mad(
    rvs: np.ndarray,
    rv_errs: np.ndarray,
    outlier_sigma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flag outliers using median and MAD, then return mask of inliers.

    Parameters
    ----------
    rvs, rv_errs : np.ndarray
        Per-chunk RV and uncertainty (nan for invalid).
    outlier_sigma : float
        Reject if |rv - median| > outlier_sigma * MAD.

    Returns
    -------
    inlier : np.ndarray, bool
        True where chunk is kept.
    median_rv : float
        Median of RVs (for reference).
    mad_rv : float
        MAD of RVs.
    """
    valid = np.isfinite(rvs) & np.isfinite(rv_errs) & (rv_errs > 0)
    if not np.any(valid):
        return np.zeros_like(rvs, dtype=bool), np.nan, np.nan
    rv_valid = rvs[valid]
    median_rv = np.nanmedian(rv_valid)
    mad_rv = mad_std(rv_valid, ignore_nan=True)
    if mad_rv <= 0:
        mad_rv = np.nanstd(rv_valid)
    if not np.isfinite(mad_rv) or mad_rv <= 0:
        mad_rv = 1.0
    threshold = outlier_sigma * mad_rv
    inlier = valid & (np.abs(rvs - median_rv) <= threshold)
    return inlier, float(median_rv), float(mad_rv)


def weighted_mean_rv(
    rvs: np.ndarray,
    rv_errs: np.ndarray,
    inlier: np.ndarray,
) -> tuple[float, float]:
    """
    Inverse-variance weighted mean and standard deviation of RV.

    Parameters
    ----------
    rvs, rv_errs : np.ndarray
        Per-chunk RV and uncertainty.
    inlier : np.ndarray, bool
        Which chunks to include.

    Returns
    -------
    rv_mean : float
        Weighted mean RV (m/s).
    rv_std : float
        Weighted standard deviation (m/s).
    """
    use = inlier & np.isfinite(rvs) & np.isfinite(rv_errs) & (rv_errs > 0)
    if not np.any(use):
        return np.nan, np.nan
    w = 1.0 / (np.asarray(rv_errs, dtype=np.float64) ** 2)
    w = np.where(use, w, 0.0)
    if np.sum(w) <= 0:
        return np.nan, np.nan
    rv_mean = np.average(rvs, weights=w)
    # Weighted variance of the mean: 1 / sum(w)
    var_mean = 1.0 / np.sum(w)
    rv_std = np.sqrt(var_mean)
    return float(rv_mean), float(rv_std)

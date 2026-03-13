"""
Masking-based cross-correlation and RV combination utilities.

Used by StellarRVPrimitive to compute radial velocity per spectral chunk
and combine with median/MAD outlier rejection and inverse-variance weighting.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
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
_V_STEP = 50.0

# Typical velocity scale per pixel for L2 (BSPEC/RSPEC) from wavelength sampling:
#   Δv = c * (Δλ/λ).  With λ_center = 147/order µm and pixel size Δλ = (147/order²)/4096 µm,
#   velocity per pixel = c / (order * 4096) m/s.
#   BSPEC (order 110–148): ~494–665 m/s per pixel.  RSPEC (order 60–97): ~755–1220 m/s per pixel.

logger = logging.getLogger(__name__)


def _gaussian_ccf_model(v: np.ndarray, amplitude: float, v0: float, sigma: float, baseline: float) -> np.ndarray:
    """Gaussian CCF model: amplitude * exp(-(v - v0)^2 / (2*sigma^2)) + baseline."""
    return amplitude * np.exp(-0.5 * ((v - v0) / np.maximum(sigma, 1e-10)) ** 2) + baseline


# Sigma from HWHM for a Gaussian: sigma = HWHM / sqrt(2*ln(2))
_HWHM_TO_SIGMA = 1.0 / np.sqrt(2.0 * np.log(2.0))


def _choose_ccf_fit_window(
    v: np.ndarray,
    y: np.ndarray,
    v0_peak: float,
    v_step: float,
    n_sigma_window: float = 4.0,
    min_points: int = 15,
    min_half_width_vel: float | None = None,
) -> tuple[float, float]:
    """
    Choose a velocity window around the CCF peak for Gaussian fitting so the fit
    is optimized to the signal and minimizes influence of structured background.

    Uses the half-width at half maximum (HWHM) of the CCF around the peak to
    estimate the Gaussian width, then sets the window to v0 ± n_sigma_window * sigma_est.
    This keeps the window as large as the signal (maximizing use of peak data) while
    excluding distant background.

    Parameters
    ----------
    v, y : np.ndarray
        Velocity and CCF values (1D, same length).
    v0_peak : float
        Approximate peak position (m/s).
    v_step : float
        Velocity step in m/s (used for minimum window size).
    n_sigma_window : float
        Window half-width in units of estimated sigma (default 4).
    min_points : int
        Require at least this many points in the window; window is expanded if needed.
    min_half_width_vel : float or None
        Minimum half-width in m/s. If None, uses max(5*v_step, 2*estimated sigma).

    Returns
    -------
    v_lo, v_hi : float
        Velocity bounds [v_lo, v_hi] for the fit window.
    """
    v_min, v_max = float(np.nanmin(v)), float(np.nanmax(v))
    ccf_max = float(y[np.nanargmax(y)])
    baseline_est = float(np.nanmin(y))
    amp_est = ccf_max - baseline_est
    if amp_est <= 0:
        return v_min, v_max
    half_max = baseline_est + 0.5 * amp_est
    i_peak = int(np.argmin(np.abs(v - v0_peak)))
    # Left HWHM: first point left of peak where y <= half_max
    left_hw = np.nan
    for i in range(i_peak - 1, -1, -1):
        if y[i] <= half_max:
            if i + 1 < len(v):
                # Linear interpolate for finer HWHM
                t = (half_max - y[i]) / (y[i + 1] - y[i]) if y[i + 1] != y[i] else 0.0
                left_hw = v[i] + t * (v[i + 1] - v[i]) - v0_peak
            else:
                left_hw = v[i] - v0_peak
            break
    if np.isnan(left_hw):
        left_hw = v0_peak - v_min
    # Right HWHM
    right_hw = np.nan
    for i in range(i_peak + 1, len(v)):
        if y[i] <= half_max:
            if i - 1 >= 0:
                t = (half_max - y[i - 1]) / (y[i] - y[i - 1]) if y[i] != y[i - 1] else 0.0
                right_hw = v[i - 1] + t * (v[i] - v[i - 1]) - v0_peak
            else:
                right_hw = v[i] - v0_peak
            break
    if np.isnan(right_hw):
        right_hw = v_max - v0_peak
    hwhm = max(abs(left_hw), abs(right_hw), 1e-10)
    # sigma = HWHM / sqrt(2*ln(2))
    sigma_est = hwhm * _HWHM_TO_SIGMA
    half_width = n_sigma_window * sigma_est
    if min_half_width_vel is not None:
        half_width = max(half_width, min_half_width_vel)
    else:
        half_width = max(half_width, 5.0 * v_step, 2.0 * sigma_est)
    # Ensure enough points in window
    n_pts = np.sum((v >= v0_peak - half_width) & (v <= v0_peak + half_width))
    while n_pts < min_points and half_width <= (v_max - v_min) / 2.0:
        half_width *= 1.2
        n_pts = np.sum((v >= v0_peak - half_width) & (v <= v0_peak + half_width))
    v_lo = max(v_min, v0_peak - half_width)
    v_hi = min(v_max, v0_peak + half_width)
    return v_lo, v_hi


def fit_gaussian_to_ccf(
    v_grid: np.ndarray,
    ccf: np.ndarray,
    v_step: float,
    n_good: int | None = None,
    return_params: bool = False,
    use_dynamic_window: bool = True,
    n_sigma_window: float = 4.0,
):
    """
    Fit a Gaussian to CCF(velocity) and return best-fit RV and uncertainty.

    Model: CCF(v) = A * exp(-(v - v0)^2 / (2*sigma^2)) + B.
    The radial velocity is v0; the uncertainty is taken from the fit covariance
    (error on v0); no minimum floor, so uncertainty can be arbitrarily small.

    When use_dynamic_window is True (default), the fit is performed in a window
    around the CCF peak. The routine tries several window sizes (including narrow
    ones, 2–6 sigma_est from HWHM) and keeps the fit that minimizes the sum of
    squared residuals (RSS) between the CCF data and the Gaussian, so the chosen
    fit best matches the data and typically has smaller residuals. The reported
    RV uncertainty is from that fit. Diagnostics include fit_v_lo and fit_v_hi
    so callers can plot the Gaussian only over the fitted range.

    Parameters
    ----------
    v_grid : np.ndarray
        Velocity grid in m/s (1D).
    ccf : np.ndarray
        CCF values (1D, same length as v_grid).
    v_step : float
        Velocity step in m/s (used as minimum uncertainty).
    n_good : int or None
        Number of good pixels; if provided, used in sigma/sqrt(n) fallback when
        covariance is not used (no minimum floor on reported uncertainty).
    return_params : bool, optional
        If True, also return the full fit parameters and fit diagnostics as
        (rv, rv_err, params, diagnostics). params has keys 'amplitude', 'v0',
        'sigma', 'baseline'. diagnostics has 'fit_ok', 'used_covariance_for_error',
        'used_sigma_fallback', 'message' (default False).
    use_dynamic_window : bool, optional
        If True (default), restrict the fit to a dynamic window around the peak
        to reduce influence of structured background. If False, use the full
        velocity range.
    n_sigma_window : float, optional
        When use_dynamic_window is True, fit window half-width in units of
        estimated sigma (default 4.0).

    Returns
    -------
    rv : float
        Best-fit radial velocity in m/s (nan if fit fails).
    rv_err : float
        Uncertainty in m/s (nan if fit fails).
    params : dict, optional
        If return_params is True and fit succeeded: amplitude, v0, sigma, baseline.
    diagnostics : dict, optional
        If return_params is True: fit_ok, used_covariance_for_error,
        used_sigma_fallback, message (for logging and preview).
    """
    def _fail_diagnostics(msg: str):
        out = {
            "fit_ok": False,
            "used_covariance_for_error": False,
            "used_sigma_fallback": False,
            "message": msg,
        }
        logger.warning("CCF Gaussian fit: %s", msg)
        return out

    v_grid = np.asarray(v_grid, dtype=np.float64).ravel()
    ccf = np.asarray(ccf, dtype=np.float64).ravel()
    valid = np.isfinite(v_grid) & np.isfinite(ccf)
    if np.sum(valid) < 10:
        diag = _fail_diagnostics("insufficient valid points for Gaussian fit")
        return (np.nan, np.nan, {}, diag) if return_params else (np.nan, np.nan)
    v = v_grid[valid]
    y = ccf[valid]
    i_max = np.nanargmax(y)
    ccf_max = float(y[i_max])
    ccf_min = float(np.nanmin(y))
    if ccf_max <= ccf_min:
        diag = _fail_diagnostics("no CCF peak (ccf_max <= ccf_min)")
        return (np.nan, np.nan, {}, diag) if return_params else (np.nan, np.nan)
    v0_init = float(v[i_max])
    amplitude_init = ccf_max - ccf_min
    baseline_init = ccf_min
    v_full, y_full = v.copy(), y.copy()

    # Candidate window sizes (n_sigma * sigma_est). Include narrow windows (2–3 sigma)
    # so the fit focuses on the peak and minimizes residuals when CCF has structured wings.
    _N_SIGMA_CANDIDATES = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)

    # Select the fit that minimizes sum of squared residuals (RSS), not rv_err, so we
    # get a Gaussian that best matches the data and report its uncertainty (often smaller
    # when the fit is good and the window is appropriate).
    best_rss = np.inf
    best_result = None

    if use_dynamic_window:
        for n_sig in _N_SIGMA_CANDIDATES:
            v_lo, v_hi = _choose_ccf_fit_window(
                v_full, y_full, v0_init, v_step,
                n_sigma_window=n_sig,
                min_points=15,
            )
            in_window = (v_full >= v_lo) & (v_full <= v_hi)
            n_in = np.sum(in_window)
            if n_in < 10:
                v_lo, v_hi = float(np.nanmin(v_full)), float(np.nanmax(v_full))
                v_win, y_win = v_full, y_full
            else:
                v_win = v_full[in_window]
                y_win = y_full[in_window]
            window_half = (v_hi - v_lo) * 0.5
            sigma_init = max(window_half / n_sig, v_step * 10.0)
            # Tighter sigma_max so the Gaussian stays near the peak (reduces fitting to wings)
            sigma_max = window_half * 0.4
            p0 = (amplitude_init, v0_init, sigma_init, baseline_init)
            try:
                popt, pcov = curve_fit(
                    _gaussian_ccf_model,
                    v_win,
                    y_win,
                    p0=p0,
                    maxfev=200,
                    bounds=(
                        (-np.inf, v_lo, v_step * 2, -np.inf),
                        (np.inf, v_hi, sigma_max, np.inf),
                    ),
                )
                y_pred = _gaussian_ccf_model(v_win, popt[0], popt[1], popt[2], popt[3])
                rss = float(np.sum((y_win - y_pred) ** 2))
                if rss < best_rss:
                    best_rss = rss
                    best_result = (popt, pcov, v_win, v_lo, v_hi, n_sig)
            except (RuntimeError, ValueError):
                continue
        if best_result is None:
            v_lo, v_hi = float(np.nanmin(v_full)), float(np.nanmax(v_full))
            v_win, y_win = v_full, y_full
        else:
            popt, pcov, v_win, v_lo, v_hi, _ = best_result
            v, y = v_win, y_full[(v_full >= v_lo) & (v_full <= v_hi)]
    else:
        v_lo, v_hi = float(np.nanmin(v)), float(np.nanmax(v))
        v_win, y_win = v, y

    if best_result is None and use_dynamic_window:
        v_lo, v_hi = float(np.nanmin(v_full)), float(np.nanmax(v_full))
        v_win, y_win = v_full, y_full
        window_half = (v_hi - v_lo) * 0.5
        sigma_init = max(window_half / n_sigma_window, v_step * 10.0)
        amplitude_init = ccf_max - ccf_min
        baseline_init = ccf_min
        p0 = (amplitude_init, v0_init, sigma_init, baseline_init)
        sigma_max = window_half * 0.4
        try:
            popt, pcov = curve_fit(
                _gaussian_ccf_model,
                v_win,
                y_win,
                p0=p0,
                maxfev=200,
                bounds=(
                    (-np.inf, v_lo, v_step * 2, -np.inf),
                    (np.inf, v_hi, sigma_max, np.inf),
                ),
            )
            best_result = (popt, pcov, v_win, v_lo, v_hi, n_sigma_window)
        except (RuntimeError, ValueError):
            pass

    if best_result is not None:
        popt, pcov, v_win, v_lo, v_hi, _ = best_result
    else:
        v_lo, v_hi = float(np.nanmin(v_full)), float(np.nanmax(v_full))
        window_half = (v_hi - v_lo) * 0.5
        sigma_init = max(window_half / n_sigma_window, v_step * 10.0)
        amplitude_init = ccf_max - ccf_min
        baseline_init = ccf_min
        p0 = (amplitude_init, v0_init, sigma_init, baseline_init)
        sigma_max = window_half * 0.4
        try:
            popt, pcov = curve_fit(
                _gaussian_ccf_model,
                v_full,
                y_full,
                p0=p0,
                maxfev=200,
                bounds=(
                    (-np.inf, v_lo, v_step * 2, -np.inf),
                    (np.inf, v_hi, sigma_max, np.inf),
                ),
            )
            v_win = v_full
        except (RuntimeError, ValueError) as e:
            diag = _fail_diagnostics(f"curve_fit failed: {e!s}")
            if return_params:
                return np.nan, np.nan, {}, diag
            return np.nan, np.nan

    rv = float(popt[1])
    sigma_v0 = float(np.sqrt(pcov[1, 1])) if np.all(np.isfinite(pcov)) else np.nan
    used_covariance = np.isfinite(sigma_v0) and sigma_v0 > 0
    n_fit = len(v_win)
    if used_covariance:
        rv_err = sigma_v0
        used_sigma_fallback = False
        diag_msg = "RV uncertainty from fit covariance (sigma_v0)"
    else:
        rv_err = float(np.abs(popt[2])) / np.sqrt(n_fit / 50.0)
        if n_good is not None and n_good > 50:
            rv_err = max(rv_err, float(np.abs(popt[2])) / np.sqrt(n_good / 50.0))
        used_sigma_fallback = True
        reason = "sigma_v0 non-finite or non-positive from curve_fit"
        diag_msg = f"RV uncertainty from sigma/sqrt(n) fallback ({reason})"
        logger.warning("CCF Gaussian fit: %s", diag_msg)
    if return_params:
        params = {
            "amplitude": float(popt[0]),
            "v0": float(popt[1]),
            "sigma": float(popt[2]),
            "baseline": float(popt[3]),
        }
        diagnostics = {
            "fit_ok": True,
            "used_covariance_for_error": used_covariance,
            "used_sigma_fallback": used_sigma_fallback,
            "message": diag_msg,
            "fit_v_lo": v_lo,
            "fit_v_hi": v_hi,
        }
        return rv, rv_err, params, diagnostics
    return rv, rv_err


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
    The best-fit RV and uncertainty are derived from a parabolic (Gaussian)
    fit to the CCF peak: sub-pixel center from parabola and uncertainty from
    curvature; falls back to peak index and HWHM if the fit is invalid.
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

    # Gaussian fit via parabolic refinement: CCF(v) ≈ a*v^2 + b*v + c near peak; peak at v0 = -b/(2*a).
    # Three points (v1,y1), (v2,y2), (v3,y3) with v2-v1 = v3-v2 = v_step.
    if i_max >= 1 and i_max < n_vel - 1:
        y1 = ccf[i_max - 1]
        y2 = ccf[i_max]
        y3 = ccf[i_max + 1]
        if np.isfinite(y1) and np.isfinite(y2) and np.isfinite(y3):
            # a = (y1 + y3 - 2*y2) / (2*v_step^2); for maximum a < 0.
            a = (y1 + y3 - 2.0 * y2) / (2.0 * v_step * v_step)
            if a < 0:
                v2 = v_grid[i_max]
                v0_parab = v2 + (y1 - y3) / (4.0 * a * v_step)
                peak_parab = y2
                sigma_parab = np.sqrt(-peak_parab / (2.0 * a))
                if np.isfinite(v0_parab) and np.isfinite(sigma_parab) and sigma_parab > 0 and v_min <= v0_parab <= v_max:
                    rv = v0_parab
                    rv_err = max(sigma_parab / np.sqrt(n_good / 50.0), v_step)
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
        # Masked correlation: weighted covariance (not normalized; peak is not scaled to 1).
        # CCF = weighted mean of (data - mean) * (template_shifted - mean); units of flux².
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
    rv, rv_err = fit_gaussian_to_ccf(v_grid, ccf, v_step, n_good=int(np.sum(good)))
    if np.isfinite(rv) and np.isfinite(rv_err):
        return rv, rv_err
    # Fallback: peak index and HWHM
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
        # Weighted covariance (not normalized; peak not scaled to 1).
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
    rv, rv_err = fit_gaussian_to_ccf(v_grid, ccf, v_step, n_good=int(np.sum(good)))
    if np.isfinite(rv) and np.isfinite(rv_err):
        return rv, rv_err, v_grid, ccf
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
        # Weighted covariance (not normalized; peak not scaled to 1).
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
    rv, rv_err = fit_gaussian_to_ccf(v_grid, ccf, v_step, n_good=n_good)
    if not (np.isfinite(rv) and np.isfinite(rv_err)):
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
    else:
        ccf_max = float(np.nanmax(ccf))
        ccf_min = float(np.nanmin(ccf))

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

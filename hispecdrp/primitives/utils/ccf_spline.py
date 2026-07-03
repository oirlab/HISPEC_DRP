from __future__ import annotations

import math
import numpy as np
from numba import njit

from hispecdrp.primitives.utils.cubic_spline import cubic_spline_build, cubic_spline_eval_scalar
from hispecdrp.primitives.utils.ccf_utils2 import _make_velocity_grid, doppler_shift_from_barycenter

__all__ = [
    "compute_ccf_spline",
]

@njit(cache=True, nogil=True)
def _interp_var_linear(wave: np.ndarray, var: np.ndarray, xq: float) -> float:
    """
    Linearly interpolate the variance at xq.
    """
    n = wave.shape[0]
    if xq <= wave[0]:
        v = var[0]
        return v if v > 0.0 else np.inf
    if xq >= wave[n - 1]:
        v = var[n - 1]
        return v if v > 0.0 else np.inf
    i = np.searchsorted(wave, xq, side="left")
    # wave[i-1] <= xq < wave[i]
    if i <= 0:
        i = 1
    x0 = wave[i - 1]
    x1 = wave[i]
    v0 = var[i - 1]
    v1 = var[i]
    if v0 <= 0.0 or v1 <= 0.0:
        return np.inf
    t = (xq - x0) / (x1 - x0)
    return v0 * (1.0 - t) + v1 * t


@njit(cache=True, nogil=True)
def _compute_ccf_spline(
    wave, spec, spec_err, spec_dq,
    spec_spline_cache_xk,
    spec_spline_cache_a,
    spec_spline_cache_b,
    spec_spline_cache_c,
    spec_spline_cache_d,
    line_centers, line_weights,
    vel_grid, zb,
    use_variance,
    ccf_out, ccf_err_out,
):
    """
    Numba-compiled function to compute the CCF using spline interpolation.
    See `compute_ccf_spline` for parameter descriptions.
    """
    n_pix = wave.shape[0]
    n_vels = vel_grid.shape[0]
    n_lines = line_centers.shape[0]

    spline_cache = (
        spec_spline_cache_xk,
        spec_spline_cache_a,
        spec_spline_cache_b,
        spec_spline_cache_c,
        spec_spline_cache_d
    )

    spec_var = spec_err * spec_err

    wave_lo = wave[0]
    wave_hi = wave[n_pix - 1]

    for idx_vel in range(n_vels):

        vel = vel_grid[idx_vel]

        num = 0.0       # sum of w * flux
        denom = 0.0     # sum of w           (w = line_weight / var)
        var_acc = 0.0   # propagated variance accumulator

        for idx_line in range(n_lines):

            line_center = line_centers[idx_line]
            line_weight = line_weights[idx_line]

            # Doppler shift the line center for this trial velocity
            line_center_shifted = doppler_shift_from_barycenter(line_center, vel, zb)

            # Skip lines that fall outside the good-pixel span (no extrapolation
            # into regions with no data support).
            if line_center_shifted < wave_lo or line_center_shifted > wave_hi:
                continue

            # Spline-interpolated flux sample
            flux_spl = cubic_spline_eval_scalar(spline_cache, line_center_shifted)

            if use_variance:
                # Variance at the evaluation point (linear interp; +inf -> skip)
                var = _interp_var_linear(wave, spec_var, line_center_shifted)
                if not (var > 0.0) or not np.isfinite(var):
                    continue
                # Inverse-variance weight, modulated by the line weight
                w = line_weight / var
                num += flux_spl * w
                denom += w
                # Variance of the weighted mean: with w_i = lw_i / v_i and
                # estimator sum(w_i f_i)/sum(w_i), Var = sum(w_i^2 v_i)/denom^2.
                var_acc += (w * w) * var
            else:
                # No inverse-variance weighting; line weight only.
                w = line_weight
                num += flux_spl * w
                denom += w

        if denom > 0.0:
            ccf_out[idx_vel] = num / denom
            if use_variance:
                ccf_err_out[idx_vel] = math.sqrt(var_acc) / denom
            else:
                ccf_val = num / denom
                ccf_err_out[idx_vel] = math.sqrt(ccf_val) if ccf_val > 0.0 else np.nan
        else:
            ccf_out[idx_vel] = np.nan


def compute_ccf_spline(
    wave: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    spec_dq: np.ndarray,
    line_centers: np.ndarray,
    line_weights: np.ndarray,
    vel_window: float,
    vel_step: float,
    zb: float = 0.0,
    vsys: float = 0.0,
    use_variance: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the CCF between a spectrum and a set of lines.
    Each line represents a delta function with a weight to normalize for absorption depth.
    The delta function is shifted by a trial velocity, and sampled on the detector with a cubic spline interpolation fo the spectrum.

    Parameters
    ----------
    wave, spec, spec_err : ndarray
        Wavelength centers, flux, and flux error per pixel.
    spec_dq : ndarray
        Data-quality flags. Only pixels with dq=0 are used.
    line_centers, line_weights : ndarray
        Line list. ``line_window_widths`` is accepted for signature parity with
        the tophat version; the spline sampler uses point evaluation, so only a
        scalar summary (the median) is forwarded as a span guard.
    vel_grid : ndarray
        Trial velocities (m/s).
    zb : float
        Barycentric/again redshift term applied in the Doppler factor.
    vsys : float
        Systemic velocity; added to the trial grid so the CCF is centered in the
        stellar rest frame while nominal velocities still include vsys.
    use_variance : bool
        If True (default), weight each line by inverse variance (line_weight /
        var) and propagate the variance into ccf_err. If False, weight by
        line_weight only and return ccf_err = sqrt(ccf).

    Returns
    -------
    vel_grid:
        Trial velocities (m/s).
    ccf_out:
        CCF values at each trial velocity.
    ccf_err_out:
        CCF uncertainties at each trial velocity.
    """

    good = np.where(spec_dq == 0)[0]

    spline_cache = cubic_spline_build(wave[good], spec[good])
    xk, a, b, c, d = spline_cache

    # Effective vel grid includes systemic velocity so
    # the CCF is centered on the stellar rest frame,
    # but the nominal velocity still includes vsys.
    # I.e., this is just to get the windowing correct
    vel_grid = _make_velocity_grid(vel_window, vel_step, vsys)
    n_vels = len(vel_grid)

    # Init outputs
    ccf_out = np.full(n_vels, np.nan, dtype=spec.dtype)
    ccf_err_out = np.full(n_vels, np.nan, dtype=spec.dtype)

    _compute_ccf_spline(
        wave, spec, spec_err, spec_dq,
        xk, a, b, c, d,
        line_centers, line_weights,
        vel_grid,
        float(zb),
        use_variance,
        ccf_out, ccf_err_out,
    )

    return vel_grid, ccf_out, ccf_err_out
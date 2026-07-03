"""Cubic-spline mask CCF utilities (point-sampled delta lines on the spectrum)."""
from __future__ import annotations

import math

import numpy as np

from hispecdrp.utils.physical_constants import C_M_S
from ....utils.parallelization_utils import njit

from .ccf_tophat import _velocity_grid_from_window

__all__ = [
    "cubic_spline_build",
    "cubic_spline_eval_scalar",
    "compute_ccf_spline",
]


@njit(cache=True, nogil=True)
def cubic_spline_build(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Natural cubic spline on sorted knots ``x`` with values ``y``.

    Returns segment coefficients ``(xk, a, b, c, d)`` so that on
    ``[xk[i], xk[i + 1]]`` the spline is
    ``a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3`` with ``dx = xq - xk[i]``.
    """
    n = x.shape[0]
    xk = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    h = np.empty(n - 1, dtype=np.float64)
    for i in range(n - 1):
        h[i] = xk[i + 1] - xk[i]

    alpha = np.zeros(n, dtype=np.float64)
    for i in range(1, n - 1):
        alpha[i] = (
            3.0 / h[i] * (y[i + 1] - y[i])
            - 3.0 / h[i - 1] * (y[i] - y[i - 1])
        )

    l = np.ones(n, dtype=np.float64)
    mu = np.zeros(n, dtype=np.float64)
    z = np.zeros(n, dtype=np.float64)
    for i in range(1, n - 1):
        l[i] = 2.0 * (xk[i + 1] - xk[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    c_full = np.zeros(n, dtype=np.float64)
    b = np.zeros(n - 1, dtype=np.float64)
    d = np.zeros(n - 1, dtype=np.float64)
    a = np.zeros(n - 1, dtype=np.float64)

    for j in range(n - 2, -1, -1):
        c_full[j] = z[j] - mu[j] * c_full[j + 1]
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c_full[j + 1] + 2.0 * c_full[j]) / 3.0
        d[j] = (c_full[j + 1] - c_full[j]) / (3.0 * h[j])
        a[j] = y[j]

    c = c_full[: n - 1]
    return xk, a, b, c, d


@njit(cache=True, nogil=True)
def cubic_spline_eval_scalar(
    xk: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    xq: float,
) -> float:
    """Evaluate a spline built by :func:`cubic_spline_build` at scalar ``xq``."""
    n = xk.shape[0]
    if n < 2:
        return np.nan

    if xq <= xk[0]:
        i = 0
    elif xq >= xk[n - 1]:
        i = n - 2
    else:
        i = int(np.searchsorted(xk, xq, side="right")) - 1
        if i < 0:
            i = 0
        if i > n - 2:
            i = n - 2

    dx = xq - xk[i]
    return a[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx


@njit(cache=True, nogil=True)
def _effective_zb(zb: float) -> float:
    zb_eff = zb
    zb1 = 1.0 + zb_eff
    if not np.isfinite(zb1) or zb1 == 0.0:
        zb_eff = 0.0
    return zb_eff


@njit(cache=True, nogil=True)
def _doppler_shift_scalar(lam: float, vel: float, zb: float) -> float:
    zb_eff = _effective_zb(zb)
    return lam * (1.0 + vel / C_M_S) / (1.0 + zb_eff)


@njit(cache=True, nogil=True)
def _interp_var_linear(wave: np.ndarray, var: np.ndarray, xq: float) -> float:
    """Linearly interpolate variance at ``xq``; non-positive values yield ``+inf``."""
    n = wave.shape[0]
    if xq <= wave[0]:
        v = var[0]
        return v if v > 0.0 else np.inf
    if xq >= wave[n - 1]:
        v = var[n - 1]
        return v if v > 0.0 else np.inf
    i = int(np.searchsorted(wave, xq, side="left"))
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
    wave: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    spec_spline_cache_xk: np.ndarray,
    spec_spline_cache_a: np.ndarray,
    spec_spline_cache_b: np.ndarray,
    spec_spline_cache_c: np.ndarray,
    spec_spline_cache_d: np.ndarray,
    line_centers: np.ndarray,
    line_weights: np.ndarray,
    vel_grid: np.ndarray,
    zb: float,
    use_variance: bool,
    ccf_out: np.ndarray,
    ccf_err_out: np.ndarray,
) -> None:
    """
    Numba core for :func:`compute_ccf_spline`.

    Each mask line is a delta function sampled on the spectrum via natural cubic spline
    interpolation at the Doppler-shifted line center.
    """
    n_pix = wave.shape[0]
    n_vels = vel_grid.shape[0]
    n_lines = line_centers.shape[0]

    spec_var = spec_err * spec_err
    wave_lo = wave[0]
    wave_hi = wave[n_pix - 1]
    spline_lo = spec_spline_cache_xk[0]
    spline_hi = spec_spline_cache_xk[spec_spline_cache_xk.shape[0] - 1]

    for idx_vel in range(n_vels):
        vel = vel_grid[idx_vel]

        num = 0.0
        denom = 0.0
        var_acc = 0.0

        for idx_line in range(n_lines):
            line_center = line_centers[idx_line]
            line_weight = line_weights[idx_line]

            if not np.isfinite(line_center) or line_center <= 0.0:
                continue
            if not np.isfinite(line_weight) or line_weight <= 0.0:
                continue

            line_center_shifted = _doppler_shift_scalar(line_center, vel, zb)

            if (
                line_center_shifted < wave_lo
                or line_center_shifted > wave_hi
                or line_center_shifted < spline_lo
                or line_center_shifted > spline_hi
            ):
                continue

            flux_spl = cubic_spline_eval_scalar(
                spec_spline_cache_xk,
                spec_spline_cache_a,
                spec_spline_cache_b,
                spec_spline_cache_c,
                spec_spline_cache_d,
                line_center_shifted,
            )
            if not np.isfinite(flux_spl):
                continue

            if use_variance:
                var = _interp_var_linear(wave, spec_var, line_center_shifted)
                if not (var > 0.0) or not np.isfinite(var):
                    continue
                w = line_weight / var
                num += flux_spl * w
                denom += w
                var_acc += (w * w) * var
            else:
                w = line_weight
                num += flux_spl * w
                denom += w

        if denom > 0.0:
            ccf_val = num / denom
            ccf_out[idx_vel] = ccf_val
            if use_variance:
                ccf_err_out[idx_vel] = math.sqrt(var_acc) / denom
            else:
                ccf_err_out[idx_vel] = (
                    math.sqrt(ccf_val) if ccf_val > 0.0 else np.nan
                )
        else:
            ccf_out[idx_vel] = np.nan
            ccf_err_out[idx_vel] = np.nan


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
    Cross-correlation of a spectrum with delta-function mask lines via cubic spline sampling.

    Each mask line is a rest-frame delta with weight ``line_weights[j]``. At trial
    velocity ``v`` the line center is Doppler-mapped as
    :math:`\\lambda_{\\rm obs} = \\lambda_{\\rm rest}(1+v/c)/(1+z_b)` and the spectrum
    flux is read by **natural cubic spline** interpolation on good pixels (``dq=0``).
    The CCF at each velocity is a weighted mean over contributing lines.

    Parameters
    ----------
    wave, spec, spec_err
        Wavelength centers, flux, and 1-sigma uncertainty per pixel.
    spec_dq
        Data-quality flags (0 = good, nonzero = bad).
    line_centers, line_weights
        Rest-frame line centers and weights.
    vel_window
        Total velocity span of the output grid (m/s).
    vel_step
        Uniform velocity step (m/s).
    zb
        Barycentric factor :math:`z_b` (dimensionless).
    vsys
        Systemic velocity (m/s) at the center of ``vel_window``.
    use_variance
        If True (default), weight each line by inverse variance
        (``line_weight / var``) and propagate uncertainty into ``ccf_err``. If False,
        weight by ``line_weight`` only and set ``ccf_err = sqrt(ccf)``.

    Returns
    -------
    vel_grid, ccf, ccf_err
        Trial velocities (m/s), CCF value, and formal uncertainty per velocity sample.
    """
    wave = np.asarray(wave, dtype=np.float64).ravel()
    spec = np.asarray(spec, dtype=np.float64).ravel()
    spec_err = np.asarray(spec_err, dtype=np.float64).ravel()
    spec_dq = np.asarray(spec_dq).ravel()
    line_centers = np.asarray(line_centers, dtype=np.float64).ravel()
    line_weights = np.asarray(line_weights, dtype=np.float64).ravel()

    good = spec_dq == 0
    if int(np.count_nonzero(good)) < 2:
        vel_grid = _velocity_grid_from_window(vel_window, vel_step, vsys)
        n_vels = vel_grid.size
        return (
            vel_grid,
            np.full(n_vels, np.nan, dtype=np.float64),
            np.full(n_vels, np.nan, dtype=np.float64),
        )

    xk, a, b, c, d = cubic_spline_build(wave[good], spec[good])
    vel_grid = _velocity_grid_from_window(vel_window, vel_step, vsys)
    n_vels = vel_grid.size
    ccf_out = np.full(n_vels, np.nan, dtype=np.float64)
    ccf_err_out = np.full(n_vels, np.nan, dtype=np.float64)

    _compute_ccf_spline(
        wave,
        spec,
        spec_err,
        xk,
        a,
        b,
        c,
        d,
        line_centers,
        line_weights,
        vel_grid,
        float(zb),
        use_variance,
        ccf_out,
        ccf_err_out,
    )

    return vel_grid, ccf_out, ccf_err_out

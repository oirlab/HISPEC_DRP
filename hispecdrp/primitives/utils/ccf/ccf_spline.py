"""Cubic-spline mask CCF utilities (point-sampled delta lines on the spectrum)."""
from __future__ import annotations

import math

import numpy as np

from hispecdrp.utils.physical_constants import C_M_S
from ....utils.parallelization_utils import njit

from .ccf_common import _velocity_grid_from_window
from .cubic_spline import cubic_spline_build, cubic_spline_eval_scalar_arrays

__all__ = [
    "compute_ccf_spline",
]


@njit(cache=True, nogil=True)
def _doppler_shift_scalar(lam: float, vel: float, zb: float) -> float:
    return lam * (1.0 + vel / C_M_S) / (1.0 + zb)


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
def _line_lands_in_spectrum(
    line_center: float,
    line_weight: float,
    vel: float,
    zb: float,
    wave: np.ndarray,
    spec_var: np.ndarray,
    wave_lo: float,
    wave_hi: float,
    spline_lo: float,
    spline_hi: float,
    spec_spline_cache_xk: np.ndarray,
    spec_spline_cache_a: np.ndarray,
    spec_spline_cache_b: np.ndarray,
    spec_spline_cache_c: np.ndarray,
    spec_spline_cache_d: np.ndarray,
    use_variance: bool,
) -> bool:
    """True if the line is a valid CCF contributor at ``vel``."""
    if not np.isfinite(line_center) or line_center <= 0.0:
        return False
    if not np.isfinite(line_weight) or line_weight <= 0.0:
        return False

    line_center_shifted = _doppler_shift_scalar(line_center, vel, zb)
    if (
        line_center_shifted < wave_lo
        or line_center_shifted > wave_hi
        or line_center_shifted < spline_lo
        or line_center_shifted > spline_hi
    ):
        return False

    flux_spl = cubic_spline_eval_scalar_arrays(
        spec_spline_cache_xk,
        spec_spline_cache_a,
        spec_spline_cache_b,
        spec_spline_cache_c,
        spec_spline_cache_d,
        line_center_shifted,
    )
    if not np.isfinite(flux_spl):
        return False

    if use_variance:
        var = _interp_var_linear(wave, spec_var, line_center_shifted)
        if not (var > 0.0) or not np.isfinite(var):
            return False
    return True


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
    line_set_velocity: float,
    freeze_line_set: bool,
    ccf_out: np.ndarray,
    ccf_err_out: np.ndarray,
) -> None:
    """
    Numba core for :func:`compute_ccf_spline`.

    Each mask line is a delta function sampled on the spectrum via not-a-knot cubic
    spline interpolation at the Doppler-shifted line center.

    When ``freeze_line_set`` is True, the contributing line set is frozen at
    ``line_set_velocity`` and the normalizer :math:`W=\\sum_j w_j` over that set is
    fixed for the whole velocity grid. Members of the set that fall outside the
    spectrum at another trial velocity contribute :math:`F=0` (and no variance).

    When ``freeze_line_set`` is False, membership and :math:`\\sum w` are recomputed
    at each trial velocity (used for broad scout searches without an input RV).

    Catalog ``w_j`` only — not :math:`w/\\sigma^2`.
    """
    n_pix = wave.shape[0]
    n_vels = vel_grid.shape[0]
    n_lines = line_centers.shape[0]

    spec_var = spec_err * spec_err
    wave_lo = wave[0]
    wave_hi = wave[n_pix - 1]
    spline_lo = spec_spline_cache_xk[0]
    spline_hi = spec_spline_cache_xk[spec_spline_cache_xk.shape[0] - 1]

    active = np.zeros(n_lines, dtype=np.uint8)
    denom_fixed = 0.0
    if freeze_line_set:
        for idx_line in range(n_lines):
            if _line_lands_in_spectrum(
                line_centers[idx_line],
                line_weights[idx_line],
                line_set_velocity,
                zb,
                wave,
                spec_var,
                wave_lo,
                wave_hi,
                spline_lo,
                spline_hi,
                spec_spline_cache_xk,
                spec_spline_cache_a,
                spec_spline_cache_b,
                spec_spline_cache_c,
                spec_spline_cache_d,
                use_variance,
            ):
                active[idx_line] = 1
                denom_fixed += line_weights[idx_line]
        if not (denom_fixed > 0.0):
            for idx_vel in range(n_vels):
                ccf_out[idx_vel] = np.nan
                ccf_err_out[idx_vel] = np.nan
            return

    for idx_vel in range(n_vels):
        vel = vel_grid[idx_vel]

        num = 0.0
        denom = 0.0
        var_acc = 0.0
        n_present = 0

        for idx_line in range(n_lines):
            if freeze_line_set and active[idx_line] == 0:
                continue

            line_center = line_centers[idx_line]
            line_weight = line_weights[idx_line]

            if not freeze_line_set:
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
                # Frozen set: out-of-spectrum lines contribute F=0 (W unchanged).
                continue

            flux_spl = cubic_spline_eval_scalar_arrays(
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
                w = line_weight
                num += flux_spl * w
                var_acc += (w * w) * var
                n_present += 1
                if not freeze_line_set:
                    denom += w
            else:
                w = line_weight
                num += flux_spl * w
                n_present += 1
                if not freeze_line_set:
                    denom += w

        if freeze_line_set:
            denom = denom_fixed

        # No in-spectrum contributors: do not emit CCF=0 / σ=0 (breaks weighted fits
        # on single-line windows when the frozen line leaves the chunk).
        if denom > 0.0 and n_present > 0:
            ccf_val = num / denom
            ccf_out[idx_vel] = ccf_val
            if use_variance:
                if var_acc > 0.0:
                    ccf_err_out[idx_vel] = math.sqrt(var_acc) / denom
                else:
                    ccf_err_out[idx_vel] = np.nan
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
    vel_window: float | None = None,
    vel_step: float | None = None,
    zb: float = 0.0,
    vsys: float = 0.0,
    use_variance: bool = True,
    vel_grid: np.ndarray | None = None,
    line_set_velocity: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cross-correlation of a spectrum with delta-function mask lines via cubic spline sampling.

    Each mask line is a rest-frame delta with weight ``line_weights[j]``. At trial
    velocity ``v`` the line center is Doppler-mapped as
    :math:`\\lambda_{\\rm obs} = \\lambda_{\\rm rest}(1+v/c)/(1+z_b)` and the spectrum
    flux is read by **not-a-knot cubic spline** interpolation on good pixels (``dq=0``).

    When ``line_set_velocity`` is finite (the input / prior RV; use ``0`` when the
    caller wants rest-frame membership), the contributing line set is frozen there
    and :math:`W=\\sum_j w_j` over that set is fixed for the whole velocity grid.
    Members of the set that fall outside at another trial ``v`` contribute
    :math:`F=0` rather than being dropped from ``W``. When ``line_set_velocity`` is
    ``None``, membership and ``W`` are recomputed at each trial velocity (needed for
    broad scout searches before a prior RV exists).

    The CCF is :math:`\\sum_j w_j F_j\\,/\\,W` (catalog weights; not inverse-variance
    reweighted), with
    :math:`\\sigma_{\\mathrm{CCF}}=\\sqrt{\\sum_j w_j^2\\sigma_j^2}/W` when
    ``use_variance`` is True.
    """
    wave = np.asarray(wave, dtype=np.float64).ravel()
    spec = np.asarray(spec, dtype=np.float64).ravel()
    spec_err = np.asarray(spec_err, dtype=np.float64).ravel()
    spec_dq = np.asarray(spec_dq).ravel()
    line_centers = np.asarray(line_centers, dtype=np.float64).ravel()
    line_weights = np.asarray(line_weights, dtype=np.float64).ravel()

    if vel_grid is not None:
        vel_grid_out = np.asarray(vel_grid, dtype=np.float64).ravel()
    elif vel_window is not None and vel_step is not None:
        vel_grid_out = _velocity_grid_from_window(vel_window, vel_step, vsys)
    else:
        raise ValueError(
            "compute_ccf_spline: provide vel_grid or both vel_window and vel_step"
        )

    freeze_line_set = (
        line_set_velocity is not None and np.isfinite(float(line_set_velocity))
    )
    v_line_set = float(line_set_velocity) if freeze_line_set else 0.0

    good = spec_dq == 0
    if int(np.count_nonzero(good)) < 2:
        n_vels = vel_grid_out.size
        return (
            vel_grid_out,
            np.full(n_vels, np.nan, dtype=np.float64),
            np.full(n_vels, np.nan, dtype=np.float64),
        )

    xk, a, b, c, d = cubic_spline_build(wave[good], spec[good])
    vel_grid = vel_grid_out
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
        v_line_set,
        freeze_line_set,
        ccf_out,
        ccf_err_out,
    )

    return vel_grid, ccf_out, ccf_err_out

"""Top-hat (finite-width) mask CCF utilities."""
from __future__ import annotations

import math

import numpy as np

from hispecdrp.utils.physical_constants import C_M_S
from ....utils.parallelization_utils import njit

__all__ = [
    "TOPHAT_MASK_CCF_SEGMENT_SAMPLES",
    "compute_ccf_tophat",
]

# Legacy chunk-parallel mask CCF in :func:`~hispecdrp.primitives.utils.ccf.ccf_utils._calc_mask_ccf_chunks_numba`
# approximates finite segments by uniform point samples; retained until that path uses binning.
TOPHAT_MASK_CCF_SEGMENT_SAMPLES = 8


def _velocity_grid_from_window(
    vel_window: float,
    vel_step: float,
    vsys: float = 0.0,
) -> np.ndarray:
    """Uniform RV trial grid spanning ``vel_window`` (m/s) centered on ``vsys``."""
    v_min = -float(vel_window) / 2.0 + float(vsys)
    v_max = float(vel_window) / 2.0 + float(vsys)
    return np.arange(v_min, v_max + float(vel_step) * 0.5, float(vel_step), dtype=np.float64)


@njit(cache=True, nogil=True)
def _pixel_edges_from_wavelength_numba(wave: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Left/right wavelength edges for each pixel from monotonic centers.

    Interior edges are midpoints between adjacent centers; outer edges mirror the
    first / last pixel spacing outward.
    """
    n = wave.shape[0]
    lo = np.empty(n, dtype=np.float64)
    hi = np.empty(n, dtype=np.float64)
    lo[0] = wave[0] - 0.5 * (wave[1] - wave[0])
    hi[0] = wave[0] + 0.5 * (wave[1] - wave[0])
    for i in range(1, n):
        half = 0.5 * (wave[i] - wave[i - 1])
        lo[i] = wave[i] - half
        hi[i] = wave[i] + half
    return lo, hi


@njit(cache=True, nogil=True)
def _bin_tophat(
    pixel_lo: np.ndarray,
    pixel_hi: np.ndarray,
    tophat_lo: float,
    tophat_hi: float,
    tophat_height: float,
    out: np.ndarray,
) -> tuple[int, int]:
    """
    Bin a top hat onto discrete pixel edges with area normalization.

    The binned values sum to ``tophat_height`` (the line weight).

    Returns
    -------
    i_start, i_stop
        Inclusive pixel index range written, or ``(0, -1)`` when there is no overlap.
    """
    if tophat_hi <= pixel_lo[0] or tophat_lo >= pixel_hi[-1]:
        return 0, -1

    i_start = int(np.searchsorted(pixel_hi, tophat_lo, side="right"))
    i_stop = int(np.searchsorted(pixel_lo, tophat_hi, side="left")) - 1
    if i_stop < i_start:
        return 0, -1

    total = 0.0
    for i in range(i_start, i_stop + 1):
        lo = pixel_lo[i] if pixel_lo[i] > tophat_lo else tophat_lo
        hi = pixel_hi[i] if pixel_hi[i] < tophat_hi else tophat_hi
        overlap = hi - lo
        if overlap > 0.0:
            out[i] = overlap
            total += overlap

    if total > 0.0:
        scale = tophat_height / total
        for i in range(i_start, i_stop + 1):
            out[i] *= scale

    return i_start, i_stop


@njit(cache=True, nogil=True)
def _effective_zb(zb: float) -> float:
    zb_eff = zb
    zb1 = 1.0 + zb_eff
    if not np.isfinite(zb1) or zb1 == 0.0:
        zb_eff = 0.0
    return zb_eff


@njit(cache=True, nogil=True)
def _compute_ccf_tophat(
    wave: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    spec_dq: np.ndarray,
    line_centers: np.ndarray,
    line_weights: np.ndarray,
    tophat_width_m_s: float,
    vel_grid: np.ndarray,
    zb: float,
    ccf_out: np.ndarray,
    ccf_err_out: np.ndarray,
) -> None:
    """
    Numba core for :func:`compute_ccf_tophat`.

    Each line is a rest-frame top hat with constant velocity width ``tophat_width_m_s``,
    Doppler-shifted per trial velocity and binned onto pixel edges.
    """
    n_pix = spec.shape[0]
    n_vels = vel_grid.shape[0]
    n_lines = line_centers.shape[0]

    tophat_weights = np.zeros(n_pix, dtype=np.float64)
    half_width_frac = 0.5 * tophat_width_m_s / C_M_S
    pixel_lo, pixel_hi = _pixel_edges_from_wavelength_numba(wave)
    zb_eff = _effective_zb(zb)

    for idx_vel in range(n_vels):
        vel = vel_grid[idx_vel]
        sh = (1.0 + vel / C_M_S) / (1.0 + zb_eff)
        ccf = 0.0
        ccf_var = 0.0
        w_total = 0.0

        for idx_line in range(n_lines):
            line_center = line_centers[idx_line]
            line_weight = line_weights[idx_line]

            if not np.isfinite(line_center) or line_center <= 0.0:
                continue
            if not np.isfinite(line_weight) or line_weight <= 0.0:
                continue

            tophat_lo = line_center * (1.0 - half_width_frac) * sh
            tophat_hi = line_center * (1.0 + half_width_frac) * sh

            i_start, i_stop = _bin_tophat(
                pixel_lo,
                pixel_hi,
                tophat_lo,
                tophat_hi,
                line_weight,
                tophat_weights,
            )
            if i_stop < i_start:
                continue

            bad = False
            for i in range(i_start, i_stop + 1):
                if spec_dq[i] != 0:
                    bad = True
                    break
            if bad:
                for i in range(i_start, i_stop + 1):
                    tophat_weights[i] = 0.0
                continue

            for i in range(i_start, i_stop + 1):
                t_weight = tophat_weights[i]
                if t_weight == 0.0:
                    continue
                flux = spec[i]
                var = spec_err[i] * spec_err[i]
                if not np.isfinite(flux) or not np.isfinite(var) or var <= 0.0:
                    continue
                inv_var = t_weight / var
                ccf += flux * inv_var
                ccf_var += inv_var * inv_var * var
                w_total += inv_var

            for i in range(i_start, i_stop + 1):
                tophat_weights[i] = 0.0

        if w_total > 0.0:
            ccf_out[idx_vel] = ccf / w_total
            ccf_err_out[idx_vel] = math.sqrt(ccf_var) / w_total
        else:
            ccf_out[idx_vel] = np.nan
            ccf_err_out[idx_vel] = np.nan


def compute_ccf_tophat(
    wave: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    spec_dq: np.ndarray,
    line_centers: np.ndarray,
    line_weights: np.ndarray,
    tophat_width: float,
    vel_window: float,
    vel_step: float,
    zb: float = 0.0,
    vsys: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cross-correlation of a spectrum with a velocity-width top-hat line mask.

    Each mask line is a rest-frame segment with constant velocity width
    ``tophat_width`` (m/s full width). At trial velocity ``v`` the segment is
    Doppler-mapped as :math:`\\lambda_{\\rm obs} = \\lambda_{\\rm rest}(1+v/c)/(1+z_b)`
    and binned onto detector pixel edges (area-preserving). The CCF at each
    velocity is an inverse-variance weighted mean flux under the combined mask.

    Parameters
    ----------
    wave
        Monotonic wavelength centers (µm or any consistent unit).
    spec, spec_err
        Flux and 1-sigma uncertainty per pixel.
    spec_dq
        Data-quality flags (0 = good, nonzero = bad).
    line_centers, line_weights
        Rest-frame line centers and weights.
    tophat_width
        Full width of each top hat in m/s (rest frame).
    vel_window
        Total velocity span of the output grid (m/s).
    vel_step
        Uniform velocity step (m/s).
    zb
        Barycentric factor :math:`z_b` (dimensionless).
    vsys
        Systemic velocity (m/s) at the center of ``vel_window``.

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

    vel_grid = _velocity_grid_from_window(vel_window, vel_step, vsys)
    n_vels = vel_grid.size
    ccf_out = np.full(n_vels, np.nan, dtype=np.float64)
    ccf_err_out = np.full(n_vels, np.nan, dtype=np.float64)

    _compute_ccf_tophat(
        wave,
        spec,
        spec_err,
        spec_dq,
        line_centers,
        line_weights,
        float(tophat_width),
        vel_grid,
        float(zb),
        ccf_out,
        ccf_err_out,
    )

    return vel_grid, ccf_out, ccf_err_out

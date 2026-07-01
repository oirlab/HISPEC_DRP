from __future__ import annotations

import math
import numpy as np
from numba import njit

from hispecdrp.utils.physical_constants import C_M_S
from hispecdrp.primitives.utils.ccf_utils2 import _make_velocity_grid, doppler_shift_from_barycenter

__all__ = [
    "compute_ccf_tophat",
]

@njit(cache=True, nogil=True)
def _bin_tophat(
    pixel_lo : np.ndarray,
    pixel_hi : np.ndarray,
    tophat_lo : float,
    tophat_hi : float,
    tophat_height : float,
    out : np.ndarray,
) -> tuple[int, int]:
    """
    Bins a top hat onto discrete pixels.
    The top hat has effective area = tophat_weight,
    so the binned values are normalized such that they sum to tophat_weight.

    Parameters
    ----------
    pixel_lo : ndarray
        Left edges of the pixels in wavelength units.
    pixel_hi : ndarray
        Right edges of the pixels in wavelength units.
    tophat_lo : float
        Left edge of the top hat in wavelength units.
    tophat_hi : float
        Right edge of the top hat in wavelength units.
    tophat_height : float
        Height of the top hat.
    """

    # returns (i_start, i_stop) of the range written; (0,0) if nothing.
    if tophat_hi <= pixel_lo[0] or tophat_lo >= pixel_hi[-1]:
        return 0, 0

    i_start = np.searchsorted(pixel_hi, tophat_lo, side="right")
    i_stop = np.searchsorted(pixel_lo, tophat_hi, side="left") - 1

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
def _get_pixel_edges(wave: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute left/right wavelength edges for each pixel from wavelength centers.

    Interior edges are midpoints between adjacent centers.  The outermost edges
    are extrapolated by mirroring the first / last inter-pixel spacing outward.

    Parameters
    ----------
    wave : ndarray
        Monotonic wavelength centers (µm, or any consistent unit).

    Returns
    -------
    lo, hi : np.ndarray
        Left and right pixel edges in the same units as wave.
    """
    lo = np.empty(wave.shape, dtype=np.float64)
    hi = np.empty(wave.shape, dtype=np.float64)
    n = len(wave)
    lo[0] = wave[0] - 0.5 * (wave[1] - wave[0])
    hi[0] = wave[0] + 0.5 * (wave[1] - wave[0])
    for i in range(1, n):
        lo[i] = wave[i] - 0.5 * (wave[i] - wave[i-1])
        hi[i] = wave[i] + 0.5 * (wave[i] - wave[i-1])
    return lo, hi


@njit(cache=True, nogil=True)
def _compute_ccf_tophat(
    wave, spec, spec_err, spec_dq,
    line_centers, line_weights,
    tophat_width, vel_grid, zb,
    ccf_out, ccf_err_out,
):
    """
    Compute the CCF for a tophat window.
    See `compute_ccf_tophat` for parameter descriptions.
    The velocity grid is assumed to be centered on the systemic velocity.
    """
    n_pix = len(spec)
    n_vels = len(vel_grid)
    n_lines = len(line_centers)

    tophat_weights = np.zeros(n_pix, dtype=np.float32)
    tophat_width_half_m_s = tophat_width / 2.0
    pixel_lo, pixel_hi = _get_pixel_edges(wave)
    spec_var = spec_err * spec_err

    for idx_vel in range(n_vels):
        vel = vel_grid[idx_vel]
        ccf = 0.0
        ccf_var = 0.0
        w_total = 0.0

        for idx_line in range(n_lines):
            line_center = line_centers[idx_line]
            line_weight = line_weights[idx_line]

            if line_weight <= 0.0:
                continue

            # Line has constant width in velocity in rest frame, so we shift the end points
            tophat_lo = line_center * (1 - tophat_width_half_m_s / C_M_S)
            tophat_hi = line_center * (1 + tophat_width_half_m_s / C_M_S)
            tophat_lo = doppler_shift_from_barycenter(tophat_lo, vel, zb)
            tophat_hi = doppler_shift_from_barycenter(tophat_hi, vel, zb)

            # Bin the top hat onto the detector pixels.
            i_start, i_stop = _bin_tophat(
                pixel_lo, pixel_hi,
                tophat_lo, tophat_hi,
                line_weight, tophat_weights,
            )

            # Make sure all pixels in the range are good, if not skip this line
            if np.any(spec_dq[i_start:i_stop+1] > 0):
                for i in range(i_start, i_stop+1):
                    tophat_weights[i] = 0.0
                continue

            # Compute the actual CCF here
            for i in range(i_start, i_stop+1):
                T = tophat_weights[i]
                F = spec[i]
                V = spec_var[i]
                ccf += F * T / V
                ccf_var += (T / V)**2 * V
                w_total += T / V

            # Reset top hat array for next line
            for i in range(i_start, i_stop+1):
                tophat_weights[i] = 0.0

        if w_total > 0:
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
    Compute the cross-correlation function (CCF) of a spectrum with a tophat window.
    In this implementation, the CCF expresses the average line shape.

    Parameters
    ----------
    wave : np.ndarray
        Wavelength array of the spectrum.
    spec : np.ndarray
        Flux array of the spectrum.
    spec_err : np.ndarray
        Uncertainty array of the spectrum.
    spec_dq : np.ndarray
        Data quality flags of the spectrum.
    line_centers : np.ndarray
        Wavelengths of the lines in the mask.
    line_weights : np.ndarray
        Weights of the lines in the mask.
    tophat_width : float
        Width of the tophat window in m/s.
    vel_window : float
        Total velocity range to compute the CCF over in m/s.
    vel_step : float
        Velocity step size for the CCF in m/s.
    zb : float, optional
        Barycentric correction to apply to the wavelengths as * 1 / (1 + zb). Default is 0.
    vsys : float, optional
        Systemic velocity to center the CCF on in m/s. Default is 0.
    """

    # Effective vel grid includes systemic velocity so
    # the CCF is centered on the stellar rest frame,
    # but the nominal velocity still includes vsys.
    # I.e., this is just to get the windowing correct
    vel_grid = _make_velocity_grid(vel_window, vel_step, vsys)
    n_vels = len(vel_grid)

    # Init outputs
    ccf_out = np.full(n_vels, np.nan, dtype=spec.dtype)
    ccf_err_out = np.full(n_vels, np.nan, dtype=spec.dtype)

    # Run numba CCF
    _compute_ccf_tophat(
        wave, spec, spec_err, spec_dq,
        line_centers, line_weights,
        tophat_width,
        vel_grid,
        float(zb),
        ccf_out, ccf_err_out,
    )

    return vel_grid, ccf_out, ccf_err_out
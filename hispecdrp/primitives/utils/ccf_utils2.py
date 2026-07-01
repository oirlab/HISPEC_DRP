
import numpy as np
from numba import njit
from hispecdrp.utils.physical_constants import C_M_S

__all__ = ['compute_rv_content', 'doppler_shift_from_barycenter']

@njit(cache=True, nogil=True)
def _gradient(y, x, edge_order=1):
    """
    Fast equivalent of numpy.gradient(y, x, edge_order=edge_order) for 1D arrays
    on a (possibly) non-uniform grid.
 
    Interior: second-order non-uniform central differences.
    Boundaries:
        edge_order=1 -> first-order one-sided (numpy's DEFAULT)
        edge_order=2 -> second-order one-sided (more accurate)
 
    Matches numpy.gradient to floating-point round-off for the corresponding
    edge_order.
    """
    n = y.shape[0]
    out = np.empty(n, dtype=np.float64)
 
    if n == 1:
        out[0] = 0.0
        return out
 
    if n == 2:
        slope = (y[1] - y[0]) / (x[1] - x[0])
        out[0] = slope
        out[1] = slope
        return out
 
    for i in range(1, n - 1):
        hs = x[i] - x[i - 1]
        hd = x[i + 1] - x[i]
        a = -hd / (hs * (hs + hd))
        b = (hd - hs) / (hs * hd)
        c = hs / (hd * (hs + hd))
        out[i] = a * y[i - 1] + b * y[i] + c * y[i + 1]
 
    if edge_order == 1:
        out[0] = (y[1] - y[0]) / (x[1] - x[0])
        out[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])
    else:
        h0 = x[1] - x[0]
        h1 = x[2] - x[1]
        out[0] = (
            -(2.0 * h0 + h1) / (h0 * (h0 + h1)) * y[0]
            + (h0 + h1) / (h0 * h1) * y[1]
            - h0 / (h1 * (h0 + h1)) * y[2]
        )
        hm1 = x[n - 1] - x[n - 2]
        hm2 = x[n - 2] - x[n - 3]
        out[n - 1] = (
            hm1 / (hm2 * (hm2 + hm1)) * y[n - 3]
            - (hm2 + hm1) / (hm2 * hm1) * y[n - 2]
            + (2.0 * hm1 + hm2) / (hm1 * (hm2 + hm1)) * y[n - 1]
        )
 
    return out

@njit(cache=True, nogil=True)
def _make_velocity_grid(vel_window : float, vel_step : float, vsys : float = 0.0):
    """Uniform RV trial grid in m/s (matches Numba chunk CCF grid spacing)."""
    v_min = -vel_window / 2 + vsys
    v_max = vel_window / 2 + vsys
    return np.arange(v_min, v_max + vel_step * 0.5, vel_step, dtype=np.float64)

@njit(cache=True, nogil=True)
def doppler_shift_from_barycenter(wave : np.ndarray | float, vel : float, zb : float) -> np.ndarray | float:
    """
    Apply a Doppler shift to the input wavelength(s) for a given velocity.
    The barycentric correction (1 + zb) is divided from the input wavelength(s).
    """
    return wave * (1.0 + vel / C_M_S) / (1.0 + zb)


@njit(cache=True, nogil=True)
def compute_rv_content(
    wave : np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    spec_dq: np.ndarray
):
    """
    Photon-limited RV content per pixel and combined, following
    Bouchy, Pepe & Queloz (2001).

    The RV uncertainty contributed by a single pixel is

        dV_i = c * sigma_i / ( lambda_i * |dA_i/dlambda| )

    where sigma_i is the flux uncertainty at that pixel.
    
    The combined RV content adds in inverse quadrature over independent pixels:

        1 / dV_tot^2 = sum_i 1 / dV_i^2

    Parameters
    ----------
    wave : ndarray
        Wavelength solution (any units).
        result is in the units of C_M_S, i.e. m/s).
    spec : ndarray
        Flux spectrum.
    spec_err : ndarray
        1-sigma flux uncertainty per pixel.
    spec_dq : ndarray
        Data quality mask (0=good, nonzero=bad).
        Bad pixels are ignored in the RV content calculation.

    Returns
    -------
    rvc_per_pix : ndarray
        Per-pixel RV content dV_i (m/s).
    rvc_tot : float
        Combined photon-limited RV uncertainty (m/s).
    """
    # Derivative of the stellar spectrum w.r.t. wavelength
    grad = _gradient(spec, wave)

    rvc_per_pix = np.full_like(spec, np.nan, dtype=np.float64)

    for i in range(len(spec)):

        if spec_dq[i] != 0:
            continue

        Ai = spec[i]
        sigma_i = spec_err[i]
        dAi_dw_star = grad[i]

        # Need a finite, usable slope and a positive error
        if not np.isfinite(dAi_dw_star) or not np.isfinite(Ai):
            continue
        if not np.isfinite(sigma_i) or sigma_i <= 0.0:
            continue
        if dAi_dw_star == 0.0:
            continue

        # Per-pixel RV content: noise (sigma_i) over signal slope
        rvc_per_pix[i] = C_M_S * sigma_i / (wave[i] * abs(dAi_dw_star))

    # Add in quadrature
    rvc_tot = np.nansum(1.0 / rvc_per_pix**2) ** (-0.5)

    return rvc_per_pix, rvc_tot
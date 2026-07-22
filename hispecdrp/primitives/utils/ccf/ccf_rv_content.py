"""Photon-limited RV content and Doppler wavelength mapping for spectra."""
from __future__ import annotations

import numpy as np

from hispecdrp.utils.physical_constants import C_M_S
from ....utils.parallelization_utils import njit, prange

__all__ = [
    "compute_rv_content",
    "doppler_shift_from_barycenter",
]


@njit(cache=True, nogil=True)
def _gradient(y: np.ndarray, x: np.ndarray, edge_order: int = 1) -> np.ndarray:
    """
    Fast equivalent of ``numpy.gradient(y, x, edge_order=edge_order)`` for 1D arrays
    on a (possibly) non-uniform grid.

    Interior: second-order non-uniform central differences.
    Boundaries:

    * ``edge_order=1`` — first-order one-sided (NumPy default)
    * ``edge_order=2`` — second-order one-sided (more accurate)

    Matches ``numpy.gradient`` to floating-point round-off for the corresponding
    ``edge_order``.
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


@njit(cache=True, nogil=True, parallel=True)
def doppler_shift_from_barycenter(
    wave: np.ndarray,
    vel: float,
    zb: float,
) -> np.ndarray:
    """
    Map rest-frame wavelength(s) to the observed frame at trial velocity ``vel``.

    Applies :math:`\\lambda_{\\rm obs} = \\lambda_{\\rm rest} (1 + v/c) / (1 + z_b)`.

    ``wave`` must be a 1-D ``float64`` array. Non-finite ``zb`` or ``1 + z_b = 0`` are
    treated as ``z_b = 0``. The element loop uses ``prange`` so callers can compose this
    inside other parallel Numba kernels.
    """
    zb_eff = zb
    zb1 = 1.0 + zb_eff
    if not np.isfinite(zb1) or zb1 == 0.0:
        zb_eff = 0.0
    factor = (1.0 + vel / C_M_S) / (1.0 + zb_eff)
    n = wave.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = wave[i] * factor
    return out


@njit(cache=True, nogil=True)
def compute_rv_content(
    wave: np.ndarray,
    spec: np.ndarray,
    spec_err: np.ndarray,
    spec_dq: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Photon-limited RV content per pixel and combined, following Bouchy, Pepe & Queloz (2001).

    The RV uncertainty contributed by a single pixel is

    .. math::

        dV_i = c \\, \\sigma_i / (\\lambda_i \\, |dA_i/d\\lambda|)

    where :math:`\\sigma_i` is the flux uncertainty at that pixel.

    The combined RV content adds in inverse quadrature over independent pixels:

    .. math::

        1 / dV_{\\rm tot}^2 = \\sum_i 1 / dV_i^2

    Parameters
    ----------
    wave
        Wavelength solution (any units; RV content is returned in m/s).
    spec
        Flux spectrum.
    spec_err
        1-sigma flux uncertainty per pixel.
    spec_dq
        Data quality mask (0 = good, nonzero = bad). Bad pixels are excluded.

    Returns
    -------
    rvc_per_pix
        Per-pixel RV content :math:`dV_i` (m/s); NaN where the pixel is unusable.
    rvc_tot
        Combined photon-limited RV uncertainty (m/s).
    """
    grad = _gradient(spec, wave)
    rvc_per_pix = np.full_like(spec, np.nan, dtype=np.float64)

    for i in range(len(spec)):
        if spec_dq[i] != 0:
            continue

        sigma_i = spec_err[i]
        d_ai_dw = grad[i]

        if not np.isfinite(d_ai_dw) or not np.isfinite(spec[i]):
            continue
        if not np.isfinite(sigma_i) or sigma_i <= 0.0:
            continue
        if d_ai_dw == 0.0:
            continue

        rvc_per_pix[i] = C_M_S * sigma_i / (wave[i] * abs(d_ai_dw))

    rvc_tot = np.nansum(1.0 / rvc_per_pix**2) ** (-0.5)
    return rvc_per_pix, float(rvc_tot)

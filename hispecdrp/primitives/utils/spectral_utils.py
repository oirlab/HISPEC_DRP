"""
Wavelength conversions: air ↔ vacuum, Doppler shifts, and wavelength / velocity / pixel deltas.

Template files (paths, BT-Settl loading, 2MASS scaling) live in
:mod:`hispecdrp.utils.spectral_template_utils`.

Notes
-----
- **Air–vacuum** conversions use a dry-air refractivity polynomial in
  :func:`_edlen_refractivity_ppm_nm` (SPIROU / Morton-style Cauchy form); refractivity
  is evaluated at the **air** wavelength, so the inverse uses fixed-point iteration.

References
----------
* spirou-ccf (``spectrumlib.nrefrac``): https://github.com/edermartioli/spirou-ccf

See Also
--------
:func:`air_wavelength_nm_to_vacuum_nm` : Air → vacuum (nm).
:func:`vacuum_wavelength_nm_to_air_nm` : Vacuum → air (nm), inverse pairing.
"""
from __future__ import annotations

import numpy as np
import scipy.constants

__all__ = [
    "air_wavelength_nm_to_vacuum_nm",
    "vacuum_wavelength_nm_to_air_nm",
    "delta_pix_to_delta_wave",
    "delta_vel_to_delta_wave",
    "delta_wave_to_delta_vel",
    "doppler_shift_wave",
]


# -----------------------------------------------------------------------------
# Air ↔ vacuum
# -----------------------------------------------------------------------------

def _edlen_refractivity_ppm_nm(wavelength_nm: np.ndarray, *, air_density: float = 1.0) -> np.ndarray:
    """
    Dry-air refractivity :math:`(n - 1)` in parts per million (ppm).

    Uses the same Cauchy-style expansion as SPIROU ``spectrumlib.nrefrac`` /
    Morton (1991)-class dispersion: wavenumber :math:`\\sigma = 1/\\lambda` with
    :math:`\\lambda` in µm enters a quadratic polynomial in :math:`\\sigma^2`.

    Parameters
    ----------
    wavelength_nm : np.ndarray
        Wavelength in **nanometers** (air), at which :math:`(n-1)` is evaluated.
        Non-positive values produce inf/nan in wavenumber terms; callers should
        filter inputs.

    air_density : float, optional
        Multiplicative scale for refractivity (1.0 = STP dry air in this parameterization).

    Returns
    -------
    np.ndarray
        :math:`(n - 1)` in ppm (i.e. multiply by :math:`10^{-6}` for the fractional index
        excess used in :func:`air_wavelength_nm_to_vacuum_nm`).

    Notes
    -----
    This is an internal helper; public conversions should use
    :func:`air_wavelength_nm_to_vacuum_nm` / :func:`vacuum_wavelength_nm_to_air_nm`.
    """

    wl = np.asarray(wavelength_nm, dtype=np.float64)
    wl2inv = (1.0e3 / wl) ** 2
    refracstp = 272.643 + 1.2288 * wl2inv + 3.555e-2 * wl2inv**2
    return float(air_density) * refracstp


def air_wavelength_nm_to_vacuum_nm(
    air_wavelength_nm: np.ndarray | float, *, air_density: float = 1.0
) -> np.ndarray:
    """
    Convert **air** wavelength to **vacuum** wavelength (nanometers).

    Uses the standard mapping

    .. math::

        \\lambda_\\mathrm{vac} = \\lambda_\\mathrm{air}\\,
            \\bigl(1 + 10^{-6} N(\\lambda_\\mathrm{air})\\bigr),

    where :math:`N` is :func:`_edlen_refractivity_ppm_nm` evaluated at
    :math:`\\lambda_\\mathrm{air}`.

    Parameters
    ----------
    air_wavelength_nm : array_like
        Wavelength(s) in nanometers (air).
    air_density : float, optional
        Passed to :func:`_edlen_refractivity_ppm_nm`.

    Returns
    -------
    np.ndarray
        Vacuum wavelength(s) in nanometers (same shape as broadcast input).

    See Also
    --------
    vacuum_wavelength_nm_to_air_nm : Inverse conversion (fixed-point iteration).
    """

    air = np.asarray(air_wavelength_nm, dtype=np.float64)
    nppm = _edlen_refractivity_ppm_nm(air, air_density=air_density)
    return air * (1.0 + 1.0e-6 * nppm)


def vacuum_wavelength_nm_to_air_nm(
    vacuum_wavelength_nm: np.ndarray | float,
    *,
    air_density: float = 1.0,
    max_iter: int = 12,
    rtol: float = 1e-14,
) -> np.ndarray:
    """
    Convert **vacuum** wavelength to **air** wavelength (nanometers).

    The forward relation :func:`air_wavelength_nm_to_vacuum_nm` depends on refractivity
    evaluated at **air** wavelength, so the inverse is not a single closed-form division
    by a factor evaluated at vacuum λ. This function solves

    .. math::

        \\lambda_\\mathrm{vac} = \\lambda_\\mathrm{air}\\,
            \\bigl(1 + 10^{-6} N(\\lambda_\\mathrm{air})\\bigr)

    by fixed-point iteration

    .. math::

        \\lambda_\\mathrm{air}^{(k+1)} = \\frac{\\lambda_\\mathrm{vac}}
            {1 + 10^{-6} N\\bigl(\\lambda_\\mathrm{air}^{(k)}\\bigr)},

    starting from :math:`\\lambda_\\mathrm{air}^{(0)} = \\lambda_\\mathrm{vac}`.
    Convergence is rapid because :math:`10^{-6}N` is :math:`\\mathcal{O}(10^{-4})`.

    Parameters
    ----------
    vacuum_wavelength_nm : array_like
        Wavelength(s) in nanometers (vacuum).
    air_density : float, optional
        Passed to :func:`_edlen_refractivity_ppm_nm`.
    max_iter : int, optional
        Maximum iterations (default 12 is ample for double precision).
    rtol : float, optional
        Stop when all finite elements satisfy
        ``abs(air_new - air) <= rtol * max(abs(air), 1e-30)``.

    Returns
    -------
    np.ndarray
        Air wavelength(s) in nanometers. Non-finite or non-positive vacuum inputs
        are mapped to NaN.

    Raises
    ------
    ValueError
        If ``max_iter`` < 1.

    See Also
    --------
    air_wavelength_nm_to_vacuum_nm : Forward air → vacuum.

    Notes
    -----
    Round-trip ``air → vac → air`` matches the original air grid within ~machine
    precision for typical NIR wavelengths when using the same ``air_density``.
    """

    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")

    vac = np.asarray(vacuum_wavelength_nm, dtype=np.float64)
    orig_shape = vac.shape
    vac_flat = vac.reshape(-1)
    air = vac_flat.copy()
    bad = ~np.isfinite(air) | (air <= 0)
    air[bad] = np.nan

    for _ in range(max_iter):
        nppm = _edlen_refractivity_ppm_nm(air, air_density=air_density)
        factor = 1.0 + 1.0e-6 * nppm
        air_new = vac_flat / factor
        air_new[bad] = np.nan
        shift = np.abs(air_new - air)
        scale = np.maximum(np.abs(air), 1.0e-30)
        finite_ok = np.isfinite(air) & np.isfinite(air_new) & np.isfinite(shift)
        if np.all(~finite_ok | (shift <= rtol * scale)):
            return air_new.reshape(orig_shape)
        air = air_new

    return air.reshape(orig_shape)


# -----------------------------------------------------------------------------
# Wavelength / velocity / pixel conversions
# -----------------------------------------------------------------------------

def delta_wave_to_delta_vel(delta_wave : np.ndarray | float, wave : np.ndarray | float) -> np.ndarray | float:
    """
    Convert a change in wavelength to a change in velocity.

    Args:
        delta_wave (np.ndarray | float): Change in wavelength (any units).
        wave (np.ndarray | float): Wavelength (same units as delta_wave).

    Returns:
        np.ndarray | float: Change in velocity in m/s.
    """
    return delta_wave / wave * scipy.constants.c

def delta_vel_to_delta_wave(delta_vel : np.ndarray | float, wave : np.ndarray | float) -> np.ndarray | float:
    """
    Convert a change in velocity to a change in wavelength.

    Args:
        delta_vel (np.ndarray | float): Change in velocity in m/s.
        wave (np.ndarray | float): Wavelength (any units).

    Returns:
        np.ndarray | float: Change in wavelength in the same units as wave.
    """
    return delta_vel / scipy.constants.c * wave

def doppler_shift_wave(wave, vel_star : float = 0.0, vel_bc : float = 0.0):
    """
    Doppler-shift wavelengths by a stellar and a barycentric velocity.

    Returns ``wave * (1 + vel_star / c) * (1 + vel_bc / c)``. This is the Doppler equation used to model data.

    Args:
        wave (np.ndarray | float | Quantity): Wavelength(s), any units (returned in the same units).
        vel_star (float): Stellar radial velocity in m/s.
        vel_bc (float): Barycentric velocity in m/s.

    Returns:
        Shifted wavelength(s), same type and units as ``wave``.
    """
    c = scipy.constants.c
    return wave * (1 + vel_star / c) * (1 + vel_bc / c)

def delta_pix_to_delta_wave(
    delta_pix : float,
    x : float,
    wavesol : np.ndarray | None = None,
    dispersion : np.ndarray | float = None
) -> np.ndarray | float:
    """
    Convert a change in pixels to a change in wavelength.

    The sign follows ``dλ/dpix``. On a decreasing WAVE row the result is
    negative; callers that need a width (FWHM, uncertainty) should take
    ``abs(...)``.

    Args:
        delta_pix (float): Change in pixels.
        x (float): Pixel position.
        wavesol (np.ndarray | None): Wavelength solution used to calculate the dispersion. If None, the dispersion must be provided.
        dispersion (np.ndarray | float): Dispersion in wavelength per pixel. If None, it will be calculated from the wavelength solution.

    Returns:
        np.ndarray | float: Change in wavelength in the same units as the wavelength solution.
    """
    if dispersion is None:
        dispersion = abs(np.gradient(wavesol))
    disp = np.interp(x, np.arange(len(dispersion)), dispersion)
    dw = disp * delta_pix
    return dw

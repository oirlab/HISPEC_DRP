"""
Utilities for loading spectral templates (e.g. BT-Settl) from HDF5 and resolving paths.

Used by simulations, the cross-correlation primitive, and scripts that need
wavelength and flux arrays from BT-Settl or similar .h5 files.
"""
from __future__ import annotations

# Provenance: BT-Settl .h5 files are from the BT-Settl atmosphere grid (Allard et al.);
# cite the BT-Settl publication for science. fetch_svo_2mass_filter uses SVO Filter Profile
# Service (see SVO_FILTER_URL below).

import logging
import os
import urllib.request

import numpy as np

logger = logging.getLogger(__name__)

# SVO Filter Profile Service (https://svo2.cab.inta-csic.es/theory/fps3/)
SVO_FILTER_URL = "https://svo2.cab.inta-csic.es/theory/fps3/getdata.php?format=ascii&id=2MASS/2MASS.{band}"

# Default directory for bundled spectrum data (e.g. BT-Settl .h5 in simulations/data)
_DEFAULT_SPECTRUM_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "simulations", "data"
)

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment]

__all__ = [
    "air_wavelength_nm_to_vacuum_nm",
    "get_spectrum_path",
    "load_btsettl_spectrum",
    "fetch_svo_2mass_filter",
]


def _edlen_refractivity_ppm_nm(wavelength_nm: np.ndarray, *, air_density: float = 1.0) -> np.ndarray:
    """
    (n - 1) in ppm for dry air at STP; wavelength in **nanometers**.

    Same Cauchy form as SPIROU ``spectrumlib.nrefrac`` / Morton (1991) style dispersion.
    """

    wl = np.asarray(wavelength_nm, dtype=np.float64)
    wl2inv = (1.0e3 / wl) ** 2
    refracstp = 272.643 + 1.2288 * wl2inv + 3.555e-2 * wl2inv**2
    return float(air_density) * refracstp


def air_wavelength_nm_to_vacuum_nm(
    air_wavelength_nm: np.ndarray | float, *, air_density: float = 1.0
) -> np.ndarray:
    """
    Convert **air** to **vacuum** wavelength (same input units as output: nm).

    λ_vac = λ_air * (1 + 1e-6 * N) with N from :func:`_edlen_refractivity_ppm_nm`.
    """

    air = np.asarray(air_wavelength_nm, dtype=np.float64)
    nppm = _edlen_refractivity_ppm_nm(air, air_density=air_density)
    return air * (1.0 + 1.0e-6 * nppm)


def get_spectrum_path(path: str, base_dir: str | None = None) -> str:
    """
    Resolve a spectrum file path relative to a data directory.

    If ``path`` is absolute, it is returned unchanged. Otherwise the path is
    resolved relative to ``base_dir`` (default: hispecdrp/simulations/data/),
    e.g. for bundled BT-Settl .h5 files.

    Parameters
    ----------
    path : str
        Filename (e.g. lte025.0-5.5-0.0a+0.0.BT-Settl.spec.7.h5) or absolute path.
    base_dir : str or None, optional
        Directory for relative paths. If None, uses the default simulations data dir.

    Returns
    -------
    str
        Absolute path to the spectrum file.
    """
    if os.path.isabs(path):
        return path
    if base_dir is None:
        base_dir = _DEFAULT_SPECTRUM_DATA_DIR
    return os.path.join(base_dir, path)


def fetch_svo_2mass_filter(band: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Fetch 2MASS filter profile (wavelength in Angstrom, transmission) from the
    SVO Filter Profile Service.

    Parameters
    ----------
    band : str
        One of 'J', 'H', 'K'.

    Returns
    -------
    wavelength_angstrom : np.ndarray
        Wavelength in Angstrom.
    transmission : np.ndarray
        Dimensionless transmission (0--1).
    """
    band = band.upper()
    if band not in ("J", "H", "K"):
        raise ValueError("band must be one of 'J', 'H', 'K'")
    url = SVO_FILTER_URL.format(band=band)
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = np.loadtxt(resp, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Unexpected SVO filter data shape for 2MASS.{band}")
    wavelength_angstrom = data[:, 0]
    transmission = data[:, 1]
    return wavelength_angstrom, transmission


def _ab_mag_through_filter(
    wavelength_um: np.ndarray,
    flux_erg_scm2_angstrom: np.ndarray,
    filter_wl_angstrom: np.ndarray,
    filter_transmission: np.ndarray,
) -> float:
    """
    Compute AB magnitude of a spectrum through a filter.

    Spectrum: wavelength in micron, flux in erg/s/cm^2/Angstrom (same units as
    BT-Settl file). Filter: wavelength in Angstrom, dimensionless transmission.
    Uses the bandpass-weighted mean f_nu and m_AB = -2.5*log10(f_nu) - 48.60
    (f_nu in erg/s/cm^2/Hz).

    Parameters
    ----------
    wavelength_um : np.ndarray
        Spectrum wavelength in micron.
    flux_erg_scm2_angstrom : np.ndarray
        Spectrum flux in erg/s/cm^2/Angstrom (no units object).
    filter_wl_angstrom : np.ndarray
        Filter wavelength in Angstrom.
    filter_transmission : np.ndarray
        Filter transmission (dimensionless).

    Returns
    -------
    float
        AB magnitude in the band.
    """
    wl_A = np.asarray(wavelength_um, dtype=np.float64) * 1.0e4
    flux = np.asarray(flux_erg_scm2_angstrom, dtype=np.float64)
    trans = np.interp(wl_A, filter_wl_angstrom, filter_transmission)
    valid = trans > 1.0e-6
    if not np.any(valid):
        raise ValueError("Spectrum and filter wavelength ranges do not overlap")
    wl = wl_A[valid]
    fl = flux[valid]
    tr = trans[valid]
    c_angstrom_per_s = 2.99792458e18
    f_nu = fl * (wl**2) / c_angstrom_per_s
    num = np.trapz(f_nu * tr, wl)
    den = np.trapz(tr, wl)
    if den <= 0:
        raise ValueError("Filter transmission integral is non-positive")
    f_nu_band = num / den
    return float(-2.5 * np.log10(f_nu_band) - 48.60)


def load_btsettl_spectrum(
    filepath: str,
    wl_min_um: float = 0.8,
    wl_max_um: float = 3.0,
    mag_ab: float | None = None,
    filter_band: str | None = None,
    *,
    wavelengths_are_air: bool = True,
):
    """
    Load wavelength and flux from a BT-Settl spectrum HDF5 file.

    The file is expected to have 'wavelength' (in Angstrom) and 'flux'
    (in 1e-23 erg/cm^2/s/Angstrom) datasets. Values are converted to micron
    and astropy units, and clipped to the given wavelength range.

    By default (``wavelengths_are_air=True``) the HDF5 wavelength grid is treated as
    **air** Angstrom and converted to **vacuum** microns before return, so the template
    matches vacuum wavelength axes (e.g. HISPEC L2 ``wave_*``, SPIROU-style masks in
    vacuum nm). Set ``wavelengths_are_air=False`` if the file is already on a vacuum grid.

    If ``mag_ab`` and ``filter_band`` are set, the spectrum is normalized to
    the given AB magnitude in the corresponding 2MASS filter (J, H, or K).
    The 2MASS profile is fetched from the SVO Filter Profile Service.

    Parameters
    ----------
    filepath : str
        Path to the .h5 file (e.g. lte025.0-5.5-0.0a+0.0.BT-Settl.spec.7.h5).
    wl_min_um : float, optional
        Minimum wavelength in microns. Default is 0.8.
    wl_max_um : float, optional
        Maximum wavelength in microns. Default is 3.0.
    mag_ab : float or None, optional
        If set with ``filter_band``, normalize the spectrum to this AB magnitude
        in that band. Default is None.
    filter_band : str or None, optional
        One of 'J', 'H', 'K'. Used with ``mag_ab`` to normalize. Default is None.
    wavelengths_are_air : bool, optional
        If True (default), convert stored wavelengths from air to vacuum before applying
        ``wl_min_um`` / ``wl_max_um`` (interpreted as vacuum microns).

    Returns
    -------
    wavelength : astropy.units.Quantity
        Wavelength array in micron (**vacuum** when ``wavelengths_are_air`` is True).
    flux : astropy.units.Quantity
        Flux density in erg/s/cm^2/Angstrom (same as file).
    """
    if h5py is None:
        raise ImportError(
            "h5py is required to load BT-Settl spectrum; install with pip install h5py"
        )
    import astropy.units as u

    logger.debug(
        "Loading BT-Settl spectrum from %s (wl=%.2f--%.2f um)",
        filepath,
        wl_min_um,
        wl_max_um,
    )
    with h5py.File(filepath, "r") as f:
        wl_air_angstrom = np.asarray(f["wavelength"][:], dtype=np.float64)
        if wavelengths_are_air:
            wl_nm_air = wl_air_angstrom / 10.0
            wl_vac_nm = np.full_like(wl_nm_air, np.nan, dtype=np.float64)
            ok = np.isfinite(wl_nm_air) & (wl_nm_air > 0)
            wl_vac_nm[ok] = air_wavelength_nm_to_vacuum_nm(wl_nm_air[ok])
            wl_vac_um = wl_vac_nm * 1.0e-3
        else:
            wl_vac_um = wl_air_angstrom * 1.0e-4
        wavelength = wl_vac_um * u.micron
        flux = f["flux"][:] * 1.0e-23 * (u.erg / u.cm**2 / u.s / u.Angstrom)

    mask = np.isfinite(wavelength.value) & (wavelength >= wl_min_um * u.micron) & (
        wavelength <= wl_max_um * u.micron
    )
    wavelength = wavelength[mask]
    flux = flux[mask]
    logger.debug("Loaded %d wavelength points from %s", len(wavelength), filepath)

    if mag_ab is not None and filter_band is not None:
        filter_wl_A, filter_trans = fetch_svo_2mass_filter(filter_band)
        wl_um = wavelength.to(u.micron).value
        flux_val = flux.to(u.erg / u.cm**2 / u.s / u.Angstrom).value
        m_current = _ab_mag_through_filter(wl_um, flux_val, filter_wl_A, filter_trans)
        scale = 10.0 ** (-0.4 * (mag_ab - m_current))
        flux = flux * scale
        logger.info(
            "Normalized spectrum to %s=%.2f (AB mag); scale=%.4e (was %.2f mag)",
            filter_band,
            mag_ab,
            scale,
            m_current,
        )

    return wavelength, flux

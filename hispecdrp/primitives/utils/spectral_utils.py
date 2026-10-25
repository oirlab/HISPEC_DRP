"""
Spectral utilities: template I/O and wavelength / velocity / pixel conversions.

Includes BT-Settl template loading for stellar CCF, plus helpers that convert
between wavelength, velocity, and pixel deltas.

Loads **BT-Settl**-style HDF5 spectra (``lte*.h5``), resolves paths, converts
**air ↔ vacuum** wavelengths (see :func:`air_wavelength_nm_to_vacuum_nm` and
:func:`vacuum_wavelength_nm_to_air_nm`), normalizes to 2MASS AB magnitudes using
profiles from the **SVO Filter Profile Service**, and builds
:math:`(\\ln\\lambda, \\mathrm{flux})` grids for
:class:`~hispecdrp.primitives.stellar_rv.stellar_rv_primitive.StellarRVPrimitive`.

Submodules and data
-------------------
Bundled templates and format notes live under ``hispecdrp/data/templates/`` (see that
directory's ``README.md`` for BT-Settl CIFIST, air wavelengths, and HDF5 layout).

Notes
-----
- **BT-Settl** model grids: cite Allard et al. / BT-Settl papers for the atmosphere
  data, not HISPEC DRP.
- **2MASS filter curves** prefer bundled Cohen/IPAC RSR files under
  ``hispecdrp/data/filters/``; SVO (``SVO_FILTER_URL``) is only tried if those are missing.
- **Air–vacuum** conversions use a dry-air refractivity polynomial in
  :func:`_edlen_refractivity_ppm_nm` (SPIROU / Morton-style Cauchy form); refractivity
  is evaluated at the **air** wavelength, so the inverse uses fixed-point iteration.
- Wavelength convention for loaded templates matches vacuum L2 axes when
  ``wavelengths_are_air=True`` in :func:`load_btsettl_spectrum`.

References
----------
* BT-Settl spectral grid (project home / papers; cite per your file version):
  http://perso.ens-lyon.fr/france.allard/
* SVO Filter Profile Service: https://svo2.cab.inta-csic.es/theory/fps3/
* spirou-ccf (spectral order / CCF context): https://github.com/edermartioli/spirou-ccf/blob/main/spiroulib.py

See Also
--------
:func:`air_wavelength_nm_to_vacuum_nm` : Air → vacuum (nm).
:func:`vacuum_wavelength_nm_to_air_nm` : Vacuum → air (nm), inverse pairing.
"""
from __future__ import annotations

import logging
import os
import urllib.request

import numpy as np
import scipy.constants

logger = logging.getLogger(__name__)

# SVO Filter Profile Service (https://svo2.cab.inta-csic.es/theory/fps3/)
SVO_FILTER_URL = "https://svo2.cab.inta-csic.es/theory/fps3/getdata.php?format=ascii&id=2MASS/2MASS.{band}"

# Bundled Cohen/IPAC 2MASS RSR curves (same source SVO cites); used offline first.
_DEFAULT_FILTER_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "filters"
)

# Default directory for bundled spectrum templates (BT-Settl .h5 under hispecdrp/data/templates)
_DEFAULT_SPECTRUM_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "templates"
)

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment]

# Broad wavelength span when loading from disk before per-order/fiber clipping in CCF.
DEFAULT_STELLAR_TEMPLATE_LOAD_WL_MIN_UM = 0.8
DEFAULT_STELLAR_TEMPLATE_LOAD_WL_MAX_UM = 3.0

__all__ = [
    "DEFAULT_STELLAR_TEMPLATE_LOAD_WL_MAX_UM",
    "DEFAULT_STELLAR_TEMPLATE_LOAD_WL_MIN_UM",
    "TWOMASS_VEGA_TO_AB",
    "air_wavelength_nm_to_vacuum_nm",
    "vacuum_wavelength_nm_to_air_nm",
    "delta_pix_to_delta_wave",
    "delta_vel_to_delta_wave",
    "delta_wave_to_delta_vel",
    "fetch_svo_2mass_filter",
    "get_spectrum_path",
    "load_btsettl_spectrum",
    "load_template_from_path",
    "normalize_2mass_band",
    "resolve_stellar_template_to_log_micron",
    "resolve_twomass_mag_ab",
    "twomass_ab_to_vega",
    "twomass_vega_to_ab",
]

# 2MASS catalog magnitudes are Vega; template normalization uses AB.
# m_AB = m_Vega + offset (Blanton & Roweis 2007).
TWOMASS_VEGA_TO_AB = {"J": 0.910, "H": 1.390, "K": 1.850, "KS": 1.850}


def normalize_2mass_band(band: str) -> str:
    """Return canonical 2MASS band ``'J'``, ``'H'``, or ``'K'`` (``'Ks'`` → ``'K'``)."""
    b = str(band).strip().upper()
    if b == "KS":
        return "K"
    if b not in ("J", "H", "K"):
        raise ValueError("band must be one of 'J', 'H', 'K', or 'Ks'")
    return b


def twomass_vega_to_ab(mag_vega: float, band: str) -> float:
    """Convert a 2MASS Vega magnitude to AB using a fixed per-band offset."""
    b = normalize_2mass_band(band)
    return float(mag_vega) + float(TWOMASS_VEGA_TO_AB[b])


def twomass_ab_to_vega(mag_ab: float, band: str) -> float:
    """Convert a 2MASS AB magnitude to Vega using a fixed per-band offset."""
    b = normalize_2mass_band(band)
    return float(mag_ab) - float(TWOMASS_VEGA_TO_AB[b])


def resolve_twomass_mag_ab(
    *,
    mag_ab: float | None = None,
    mag_vega: float | None = None,
    filter_band: str | None = None,
) -> tuple[float | None, str | None]:
    """
    Resolve optional AB/Vega magnitudes to ``(mag_ab, band)`` for template scaling.

    ``mag_vega`` overrides ``mag_ab`` when both are set. A magnitude requires
    ``filter_band`` (2MASS ``J`` / ``H`` / ``K`` / ``Ks``).
    """
    if mag_vega is not None:
        if filter_band is None:
            raise ValueError("filter_band is required when mag_vega is set")
        band = normalize_2mass_band(filter_band)
        return twomass_vega_to_ab(float(mag_vega), band), band
    if mag_ab is not None:
        if filter_band is None:
            raise ValueError("filter_band is required when mag_ab is set")
        band = normalize_2mass_band(filter_band)
        return float(mag_ab), band
    if filter_band is not None:
        raise ValueError(
            "filter_band was set without mag_ab or mag_vega; "
            "provide a magnitude or set filter_band=None"
        )
    return None, None


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


def _coalesce_existing_spectrum_path(resolved: str) -> str:
    """Prefer ``resolved`` if it exists; else a Dropbox conflicted-copy sibling."""
    if os.path.isfile(resolved):
        return resolved
    parent = os.path.dirname(resolved)
    name = os.path.basename(resolved)
    stem, ext = os.path.splitext(name)
    if not os.path.isdir(parent) or ext.lower() != ".h5":
        return resolved
    candidates = [
        os.path.join(parent, fn)
        for fn in os.listdir(parent)
        if fn != name and fn.startswith(stem) and fn.lower().endswith(".h5")
    ]
    if not candidates:
        return resolved
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def get_spectrum_path(path: str, base_dir: str | None = None) -> str:
    """
    Resolve a spectrum file path, optionally relative to a package data directory.

    Parameters
    ----------
    path : str
        Absolute path, or a filename / relative path joined to ``base_dir``.
    base_dir : str or None, optional
        Root directory for relative ``path``. If ``None``, uses
        ``hispecdrp/data/templates/`` (see :data:`_DEFAULT_SPECTRUM_DATA_DIR`).

    Returns
    -------
    str
        Absolute filesystem path (``os.path.join(base_dir, path)`` when ``path`` is
        relative). If that file is missing, a same-stem ``*.h5`` sibling is used
        when present (Dropbox ``conflicted copy`` names).

    See Also
    --------
    load_btsettl_spectrum : Open resolved BT-Settl HDF5 files.
    """
    if os.path.isabs(path):
        return _coalesce_existing_spectrum_path(path)
    if base_dir is None:
        base_dir = _DEFAULT_SPECTRUM_DATA_DIR
    return _coalesce_existing_spectrum_path(os.path.join(base_dir, path))


def _load_2mass_filter_table(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a 2-column wavelength_Angstrom, transmission table from ``path``."""
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Unexpected 2MASS filter data shape in {path}")
    return data[:, 0], data[:, 1]


def fetch_svo_2mass_filter(band: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a 2MASS filter bandpass (wavelength Å, transmission).

    Prefers the bundled Cohen/IPAC RSR file under ``hispecdrp/data/filters/``.
    If that file is missing, falls back to downloading from the SVO Filter
    Profile Service (:data:`SVO_FILTER_URL`).

    Parameters
    ----------
    band : str
        One of ``'J'``, ``'H'``, or ``'K'`` (case-insensitive).

    Returns
    -------
    wavelength_angstrom : np.ndarray
        Wavelength sample points in Angstrom.
    transmission : np.ndarray
        Dimensionless transmission, same length as ``wavelength_angstrom``.

    Raises
    ------
    ValueError
        If ``band`` is not one of J, H, K, or if the table is not 2-column.
    urllib.error.URLError
        On network failure or timeout (30 s) when the bundled file is absent.

    References
    ----------
    * Cohen et al. 2003, AJ 126, 1090 (2MASS absolute calibration / RSR).
    * Spanish Virtual Observatory Filter Profile Service (FPS3):
      https://svo2.cab.inta-csic.es/theory/fps3/
    """
    band = normalize_2mass_band(band)
    bundled = os.path.join(_DEFAULT_FILTER_DATA_DIR, f"2MASS.{band}.dat")
    if os.path.isfile(bundled):
        return _load_2mass_filter_table(bundled)

    url = SVO_FILTER_URL.format(band=band)
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = np.loadtxt(resp, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Unexpected SVO filter data shape for 2MASS.{band}")
    return data[:, 0], data[:, 1]


def _ab_mag_through_filter(
    wavelength_um: np.ndarray,
    flux_erg_scm2_angstrom: np.ndarray,
    filter_wl_angstrom: np.ndarray,
    filter_transmission: np.ndarray,
) -> float:
    """
    Bandpass AB magnitude from :math:`f_\\nu`-weighted integration.

    Converts :math:`F_\\lambda` to :math:`f_\\nu \\propto \\lambda^2 F_\\lambda`,
    applies the filter transmission via linear interpolation onto the spectrum grid,
    and evaluates :math:`m_\\mathrm{AB} = -2.5\\log_{10}\\langle f_\\nu\\rangle - 48.60`
    (``f_nu`` in erg s\\ :sup:`−1` cm\\ :sup:`−2` Hz\\ :sup:`−1`).

    Parameters
    ----------
    wavelength_um : np.ndarray
        Spectrum wavelength in **micron** (vacuum if the spectrum came from
        :func:`load_btsettl_spectrum` with default air handling).
    flux_erg_scm2_angstrom : np.ndarray
        :math:`F_\\lambda` in erg s\\ :sup:`−1` cm\\ :sup:`−2` Å\\ :sup:`−1`.
    filter_wl_angstrom : np.ndarray
        Filter wavelength in Angstrom.
    filter_transmission : np.ndarray
        Filter transmission in [0, 1].

    Returns
    -------
    float
        In-band AB magnitude.

    Raises
    ------
    ValueError
        If spectrum and filter do not overlap in wavelength, or if the transmission
        integral is non-positive.

    Notes
    -----
    Private helper for :func:`load_btsettl_spectrum` magnitude normalization.
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
    num = np.trapezoid(f_nu * tr, wl)
    den = np.trapezoid(tr, wl)
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

    The file is expected to have ``wavelength`` (in Angstrom) and ``flux``
    (in 1e-23 erg/cm^2/s/Angstrom) datasets. SVO BT-Settl (CIFIST) HDF5 files also
    store model parameters on the **root** attributes ``temperature`` (K),
    ``surface_gravity`` (log g cgs), and ``metallicity``; read them with ``h5py``
    if needed (see ``hispecdrp/data/templates/README.md``).

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

    Raises
    ------
    ImportError
        If ``h5py`` is not installed.

    References
    ----------
    BT-Settl atmosphere models; cite the grid publication appropriate to your file.
    See ``hispecdrp/data/templates/README.md`` for CIFIST / SVO air-wavelength notes.

    See Also
    --------
    load_template_from_path : Same data as plain ``numpy`` arrays.
    vacuum_wavelength_nm_to_air_nm : If you must write vacuum-calibrated grids back
        to air nm for comparison with archive tables.
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


def load_template_from_path(
    filepath: str,
    wl_min_um: float = DEFAULT_STELLAR_TEMPLATE_LOAD_WL_MIN_UM,
    wl_max_um: float = DEFAULT_STELLAR_TEMPLATE_LOAD_WL_MAX_UM,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load wavelength and flux as plain numpy arrays from a stellar template file.

    Currently supports BT-Settl-style HDF5 (e.g. ``lte*.BT-Settl.spec.*.h5``) via
    :func:`load_btsettl_spectrum`.

    Parameters
    ----------
    filepath : str
        Path to the template spectrum file.
    wl_min_um, wl_max_um : float
        Wavelength clip in **vacuum** microns (defaults match
        :data:`DEFAULT_STELLAR_TEMPLATE_LOAD_WL_MIN_UM` /
        :data:`DEFAULT_STELLAR_TEMPLATE_LOAD_WL_MAX_UM`).

    Returns
    -------
    wave_um : np.ndarray
        Wavelength in **vacuum** microns (same convention as ``load_btsettl_spectrum``).
    flux : np.ndarray
        Flux density in erg/s/cm²/Å (numeric values from the loaded quantity).

    Notes
    -----
    Equivalent to :func:`load_btsettl_spectrum` with default ``wavelengths_are_air=True``
    and astropy quantities stripped to ``float64``.

    See Also
    --------
    load_btsettl_spectrum : Full interface with units and optional photometric norm.
    resolve_stellar_template_to_log_micron : Natural log wavelength for CCF.
    """
    wq, fq = load_btsettl_spectrum(filepath, wl_min_um=wl_min_um, wl_max_um=wl_max_um)
    return np.asarray(wq.to_value("micron"), dtype=np.float64), np.asarray(
        fq.value, dtype=np.float64
    )


def resolve_stellar_template_to_log_micron(
    template_path: str | None,
    template_wavelength,
    template_flux,
    *,
    wl_min_um: float = DEFAULT_STELLAR_TEMPLATE_LOAD_WL_MIN_UM,
    wl_max_um: float = DEFAULT_STELLAR_TEMPLATE_LOAD_WL_MAX_UM,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Resolve a stellar template to ``(ln λ, flux)`` with λ in microns, for CCF.

    If ``template_path`` is set and either wavelength or flux array is missing, loads
    from disk using :func:`load_template_from_path` over ``[wl_min_um, wl_max_um]``
    (a broad range; finer clipping may be done per fiber/order downstream).

    If both ``template_wavelength`` and ``template_flux`` are provided (and not
    overridden by a path-only load), they are validated, sorted in wavelength, and
    converted to natural log wavelength.

    Parameters
    ----------
    template_path : str or None
        Path to BT-Settl (or supported) file.
    template_wavelength, template_flux
        Optional wavelength (microns, linear) and flux arrays.

    Returns
    -------
    template_ln : np.ndarray or None
        :math:`\\ln(\\lambda)` with :math:`\\lambda` in microns, or ``None``.
    template_flux : np.ndarray or None
        Flux aligned with ``template_ln``, or ``None``.

    Notes
    -----
    On load failure from ``template_path``, logs an exception and returns
    ``(None, None)`` without re-raising.

    See Also
    --------
    load_template_from_path : Disk load used when arrays are missing.
    """
    tw = template_wavelength
    tf = template_flux
    if template_path is not None and (tw is None or tf is None):
        try:
            tw, tf = load_template_from_path(
                str(template_path), wl_min_um=wl_min_um, wl_max_um=wl_max_um
            )
        except Exception:
            logger.exception("Failed to load template from %s", template_path)
            return None, None
    if tw is None or tf is None:
        return None, None
    tw = np.asarray(tw, dtype=np.float64)
    tf = np.asarray(tf, dtype=np.float64)
    ok = np.isfinite(tw) & np.isfinite(tf) & (tw > 0)
    if not np.any(ok):
        return None, None
    tw = tw[ok]
    tf = tf[ok]
    idx = np.argsort(tw)
    tw = tw[idx]
    tf = tf[idx]
    ln_lam = np.log(tw)
    return ln_lam, tf

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
        dispersion = np.gradient(wavesol)
    disp = np.interp(x, np.arange(len(dispersion)), dispersion)
    dw = disp * delta_pix
    return dw


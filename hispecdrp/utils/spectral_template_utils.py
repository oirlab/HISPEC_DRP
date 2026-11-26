"""
Spectral template files: paths, loading, and flux scaling.

Resolves template paths, loads **BT-Settl**-style HDF5 spectra (``lte*.h5``) on a
vacuum wavelength grid, scales them to 2MASS AB magnitudes, and builds
:math:`(\\ln\\lambda, \\mathrm{flux})` grids for
:class:`~hispecdrp.primitives.stellar_rv.stellar_rv_primitive.StellarRVPrimitive`.
Also loads sky background spectra and generates etalon line lists.

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
- **2MASS magnitudes** are Vega in the catalog; template normalization uses AB, converted
  with the fixed offsets in :data:`TWOMASS_VEGA_TO_AB`.
- Wavelength convention for loaded templates matches vacuum L2 axes when
  ``wavelengths_are_air=True`` in :func:`load_btsettl_spectrum`. The air → vacuum
  conversion is :func:`~hispecdrp.primitives.utils.spectral_utils.air_wavelength_nm_to_vacuum_nm`.

References
----------
* BT-Settl spectral grid (project home / papers; cite per your file version):
  http://perso.ens-lyon.fr/france.allard/
* SVO Filter Profile Service: https://svo2.cab.inta-csic.es/theory/fps3/
* spirou-ccf (spectral order / CCF context): https://github.com/edermartioli/spirou-ccf/blob/main/spiroulib.py
"""
from __future__ import annotations

import logging
import os
import urllib.request

import numpy as np

from ..primitives.utils.spectral_utils import air_wavelength_nm_to_vacuum_nm

logger = logging.getLogger(__name__)

# SVO Filter Profile Service (https://svo2.cab.inta-csic.es/theory/fps3/)
SVO_FILTER_URL = "https://svo2.cab.inta-csic.es/theory/fps3/getdata.php?format=ascii&id=2MASS/2MASS.{band}"

# Bundled Cohen/IPAC 2MASS RSR curves (same source SVO cites); used offline first.
_DEFAULT_FILTER_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "filters"
)

# Default directory for bundled spectrum templates (BT-Settl .h5 under hispecdrp/data/templates)
_DEFAULT_SPECTRUM_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "templates"
)
_DEFAULT_SKYBKGND_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "atmosphere"
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
    "etalon_line_list",
    "fetch_svo_2mass_filter",
    "get_skybackground_path",
    "get_spectral_template_path",
    "load_btsettl_spectrum",
    "load_sky_spectrum",
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


def get_spectral_template_path(path: str, base_dir: str | None = None) -> str:
    """
    Resolve a spectral template file path, optionally relative to a package data directory.

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


def get_skybackground_path(path: str, base_dir: str | None = None) -> str:
    if os.path.isabs(path):
        return path
    if base_dir is None:
        base_dir = _DEFAULT_SKYBKGND_DATA_DIR
    return os.path.join(base_dir, path)


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
    hispecdrp.primitives.utils.spectral_utils.vacuum_wavelength_nm_to_air_nm : If you
        must write vacuum-calibrated grids back to air nm for comparison with archive tables.
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


def load_sky_spectrum(filepath: str, wl_min_um: float = 0.8, wl_max_um: float = 3.0):
    import astropy.units as u

    sky_wl, sky_flux_ph = np.loadtxt(filepath).T
    sky_wl *= u.nm
    sky_flux_ph *= u.ph / u.s / u.m**2 / u.arcsecond**2 / u.nm

    tele_diameter = 10*u.m

    # calculation assuming fiber is size of diffraction limit
    # diffraction_limit = (sky_wl/tele_diameter.to(u.micron)*u.radian).to(u.arcsec)
    # solidangle = diffraction_limit**2 * 1.13

    # calculation based on HK fiber core size, telescope focal length
    fiber_core = 5.5 * u.um
    focal_length = 17.5*u.m
    solidangle = ((fiber_core/focal_length).to(u.dimensionless_unscaled) * u.radian)**2

    sky_flux_ph = (sky_flux_ph*solidangle).to(
        u.photon / (u.s * u.nm * u.m**2)
    )

    sky_flux = sky_flux_ph.to(
        u.erg / (u.s * u.cm**2 * u.AA), equivalencies=u.spectral_density(sky_wl)
    )

    mask = np.isfinite(sky_wl.value) & (sky_wl >= wl_min_um * u.micron) & (
        sky_wl <= wl_max_um * u.micron
    )
    sky_wl = sky_wl[mask]
    sky_flux = sky_flux[mask]

    return sky_wl.to(u.micron), sky_flux.to(u.erg / (u.s * u.cm**2 * u.AA)) 

def etalon_line_list(
    fsr: u.Quantity,
    wavelength_range: tuple[u.Quantity, u.Quantity],
    n_cavity: float,
    wavelength_meas: u.Quantity,
    finesse_meas: np.ndarray,
    peak_trans_meas: np.ndarray,
):
    """Generate an etalon resonance line list over a wavelength range.

    The resonance wavelengths are calculated using the ideal Fabry–Pérot
    etalon resonance condition,

        lambda_m = 2 * n * d / m,

    where ``m`` is the integer resonance order and ``d`` is the cavity
    length. The cavity length is inferred from the supplied free spectral
    range (FSR) using

        FSR = c / (2 * n * d).

    The measured finesse and peak transmission are linearly interpolated
    onto the calculated resonance wavelengths.

    This function calculates only the resonance positions; it does not
    evaluate the full Airy/Fabry–Pérot transmission profile between
    resonances.

    Parameters
    ----------
    fsr : astropy.units.Quantity
        Free spectral range of the etalon. Converted internally to Hz.

    wavelength_range : tuple[astropy.units.Quantity, astropy.units.Quantity]
        Minimum and maximum wavelengths to include. Converted internally
        to nm for calculating the resonance orders.

    n_cavity : float
        Refractive index inside the etalon cavity.

    wavelength_meas : astropy.units.Quantity
        Wavelengths at which the finesse and peak transmission were measured.
        These values should be sorted in increasing wavelength for use with
        ``numpy.interp``.

    finesse_meas : numpy.ndarray
        Measured finesse values corresponding to ``wavelength_meas``.

    peak_trans_meas : numpy.ndarray
        Measured peak transmission values corresponding to
        ``wavelength_meas``.

    Returns
    -------
    order : numpy.ndarray
        Integer Fabry–Pérot resonance orders corresponding to each line.

    wavelength : astropy.units.Quantity
        Resonance wavelengths, returned in microns.

    finesse : numpy.ndarray
        Finesse interpolated from ``finesse_meas`` at each resonance
        wavelength.

    transmission : numpy.ndarray
        Peak transmission interpolated from ``peak_trans_meas`` at each
        resonance wavelength.

    Notes
    -----
    The resonance positions assume an ideal, plane-parallel Fabry–Pérot
    etalon with normal incidence. Dispersion in the cavity refractive index
    is not included because ``n_cavity`` is treated as a constant.

    ``numpy.interp`` uses the endpoint values when a calculated resonance
    lies outside the range of ``wavelength_meas``. Therefore, the measured
    wavelength range should generally cover the requested resonance range
    if extrapolation by constant endpoint values is not desired.
    """

    from astropy import units as u
    from astropy.constants import c

    fsr = fsr.to(u.Hz)
    wl_min, wl_max = [x.to(u.nm) for x in wavelength_range]
    wavelength_meas = wavelength_meas.to(u.nm)

    finesse_meas = np.asarray(finesse_meas, dtype=float)
    peak_trans_meas = np.asarray(peak_trans_meas, dtype=float)

    # FSR = c / (2 n d)
    d = c / (2 * n_cavity * fsr)

    # Resonance condition: lambda_m = 2 n d / m
    m_max = int(np.floor(
        (2 * n_cavity * d / wl_min).decompose().value
    ))
    m_min = int(np.ceil(
        (2 * n_cavity * d / wl_max).decompose().value
    ))

    order = np.arange(m_min, m_max + 1)

    wavelength = (2 * n_cavity * d / order).to(u.nm)

    # Keep only requested wavelength range
    mask = (wavelength >= wl_min) & (wavelength <= wl_max)

    order = order[mask]
    wavelength = wavelength[mask]

    finesse = np.interp(
        wavelength.value,
        wavelength_meas.value,
        finesse_meas,
    )

    transmission = np.interp(
        wavelength.value,
        wavelength_meas.value,
        peak_trans_meas,
    )

    return order, wavelength.to(u.micron), finesse, transmission
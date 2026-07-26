import logging
import math
import os
import re

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from hispecdrp import datamodels

logger = logging.getLogger(__name__)

__all__ = [
    "get_stellar_data_gaia_by_coordinates",
    "get_stellar_data_gaia_from_source_id",
    "_convert_epoch_gaia",
    "get_2mass_from_gaia",
    "get_2mass_data_vizier",
    "query_vizier",
    "vizier_mirrors",
    "vizier_timeout",
]

_GAIA_EPOCHS = {
    "DR3" : 2016.0,
    "DR4" : 2017.5,
}

# VizieR mirror servers (hostnames only); tried in order when a query fails.
VIZIER_MIRRORS = [
    "vizier.cds.unistra.fr",   # CDS / Strasbourg, France
    "vizier.nao.ac.jp",        # ADAC / Tokyo, Japan
    "vizier.cfa.harvard.edu",  # CfA / Harvard, USA
    "vizier.inasan.ru",        # INASAN / Moscow, Russia
    "vizier.iucaa.in",         # IUCAA / Pune, India
    "vizier.idia.ac.za",       # IDIA / South Africa
]

# Per-request socket timeout (seconds). A finite timeout is what makes the mirror
# fallback effective: astroquery's default can leave a stalled mirror blocking
# indefinitely, so the remaining servers are never tried.
VIZIER_TIMEOUT_DEFAULT = 60.0

# Number of full passes over the mirror list before giving up.
VIZIER_MAX_ROUNDS_DEFAULT = 4

# VizieR mirror of the Gaia DR3 main source catalog.
GAIA_DR3_VIZIER_ID = "I/355/gaiadr3"

# VizieR id of the 2MASS All-Sky Point Source Catalog (JHKs photometry, Vega system).
TWOMASS_PSC_VIZIER_ID = "II/246"


def vizier_mirrors() -> list[str]:
    """Return the VizieR servers to try, in order.

    ``HISPEC_DRP_VIZIER_MIRRORS`` (comma-separated hostnames) overrides the built-in
    list, which lets a site with a local VizieR/Gaia DR3 mirror (e.g. Keck's offline
    DR3) redirect every query in the DRP without code changes.
    """
    raw = os.environ.get("HISPEC_DRP_VIZIER_MIRRORS", "").strip()
    if not raw:
        return list(VIZIER_MIRRORS)
    servers = [s.strip() for s in raw.split(",") if s.strip()]
    return servers or list(VIZIER_MIRRORS)


def vizier_timeout() -> float:
    """Per-request VizieR timeout in seconds (``HISPEC_DRP_VIZIER_TIMEOUT`` overrides)."""
    raw = os.environ.get("HISPEC_DRP_VIZIER_TIMEOUT", "").strip()
    if raw:
        try:
            val = float(raw)
            if val > 0:
                return val
        except ValueError:
            logger.warning(
                "Ignoring invalid HISPEC_DRP_VIZIER_TIMEOUT=%r; using %.0fs",
                raw, VIZIER_TIMEOUT_DEFAULT,
            )
    return VIZIER_TIMEOUT_DEFAULT


def _vizier_class():
    """Import :class:`astroquery.vizier.Vizier` lazily.

    ``astroquery`` is imported on first query rather than at module import: importing it
    can stall on network/config access, and ``hispecdrp/__init__`` pulls in every
    primitive, so a top-level import would make ``import hispecdrp`` hang (see issue
    #201).
    """
    from astroquery.vizier import Vizier

    return Vizier


def query_vizier(
    *,
    catalog: str,
    columns: list[str] | None = None,
    center: SkyCoord | None = None,
    radius_deg: float | None = None,
    constraints: dict | None = None,
    column_filters: dict | None = None,
    row_limit: int = 50,
    max_rounds: int | None = None,
    description: str | None = None,
):
    """Query one VizieR catalog, trying mirror servers in order with retry rounds.

    Shared fallback layer for every VizieR query in the DRP, following the POTPyRI
    ``query_vizier_region`` / ``query_gaia_dr3_region`` pattern: clear the astroquery
    cache, walk :func:`vizier_mirrors` catching per-server errors, and repeat the whole
    list up to ``max_rounds`` times for transient failures. Each request uses a finite
    timeout (:func:`vizier_timeout`) so one unresponsive mirror cannot block the rest.

    Exactly one of ``center`` (with ``radius_deg``) or ``constraints`` must be given,
    selecting a cone search or a constraint query respectively.

    Parameters
    ----------
    catalog : str
        VizieR catalog identifier (e.g. ``'I/355/gaiadr3'``).
    columns : list of str, optional
        Columns to request; defaults to all columns (``['**']``).
    center : astropy.coordinates.SkyCoord, optional
        Cone-search center.
    radius_deg : float, optional
        Cone-search radius in degrees. Required with ``center``.
    constraints : dict, optional
        Column constraints for ``query_constraints`` (e.g. ``{'Source': '==12345'}``).
    column_filters : dict, optional
        VizieR server-side column filters (e.g. ``{'Gmag': '<8'}``).
    row_limit : int
        Maximum rows to return (``-1`` for unlimited).
    max_rounds : int, optional
        Full passes over the mirror list. Defaults to
        :data:`VIZIER_MAX_ROUNDS_DEFAULT`.
    description : str, optional
        Short label for log messages; defaults to ``catalog``.

    Returns
    -------
    astropy.table.Table or None
        First table of the response, or ``None`` if every mirror failed or no row
        matched.
    """
    if (center is None) == (constraints is None):
        raise ValueError("Provide exactly one of center (with radius_deg) or constraints.")
    if center is not None and radius_deg is None:
        raise ValueError("radius_deg is required with center.")

    vizier_cls = _vizier_class()
    label = description or catalog
    rounds = VIZIER_MAX_ROUNDS_DEFAULT if max_rounds is None else int(max_rounds)
    timeout = vizier_timeout()
    cols = list(columns) if columns else ["**"]

    # A stale/corrupt astroquery cache entry otherwise reproduces the same failure on
    # every mirror and every round (POTPyRI clears it for the same reason).
    try:
        vizier_cls.clear_cache()
    except Exception:  # noqa: BLE001 - cache clearing is best-effort
        logger.debug("VizieR cache clear failed (continuing).", exc_info=True)

    last_error: Exception | None = None
    for round_idx in range(rounds):
        for server in vizier_mirrors():
            try:
                vizier = vizier_cls(
                    columns=cols,
                    column_filters=dict(column_filters or {}),
                    vizier_server=server,
                    timeout=timeout,
                )
                vizier.ROW_LIMIT = row_limit
                if center is not None:
                    result = vizier.query_region(
                        center, radius=radius_deg * u.deg, catalog=catalog
                    )
                else:
                    result = vizier.query_constraints(catalog=catalog, **constraints)
                if result is not None and len(result) > 0 and len(result[0]) > 0:
                    logger.info("VizieR query succeeded via %s (%s)", server, label)
                    return result[0]
            except Exception as e:  # noqa: BLE001 - mirror fallback is intentional
                last_error = e
                logger.warning("VizieR mirror %s failed for %s: %s", server, label, e)
        if last_error is not None:
            logger.error(
                "All VizieR mirrors failed for %s (round %d/%d); last error: %s",
                label, round_idx + 1, rounds, last_error,
            )
    return None


# Map of the Gaia TAP ``gaiadr3.gaia_source`` column names (the keys returned by the
# original ``Gaia.launch_job`` queries) to candidate column names in the VizieR
# ``I/355/gaiadr3`` table. Matching is punctuation-insensitive but case-sensitive
# (see :func:`_norm`), which preserves VizieR's lower/upper bound prefixes (``b_``/``B_``)
# while tolerating VizieR's underscore/bracket mangling of astropy column names.
# Columns with no VizieR equivalent are filled as described in :func:`_vizier_row_to_gaia_dict`.
_TAP_TO_VIZIER_CANDIDATES: dict[str, list[str]] = {
    "source_id": ["Source"],
    "ra": ["RA_ICRS"],
    "ra_error": ["e_RA_ICRS"],
    "dec": ["DE_ICRS"],
    "dec_error": ["e_DE_ICRS"],
    "pmra": ["pmRA"],
    "pmra_error": ["e_pmRA"],
    "pmdec": ["pmDE"],
    "pmdec_error": ["e_pmDE"],
    "parallax": ["Plx"],
    "parallax_error": ["e_Plx"],
    # ``ref_epoch`` has no VizieR column; it is the fixed DR3 reference epoch (see below).
    "ruwe": ["RUWE"],
    "radial_velocity": ["RV"],
    "radial_velocity_error": ["e_RV"],
    "rv_nb_transits": ["o_RV"],
    "vbroad": ["Vbroad"],
    "vbroad_error": ["e_Vbroad"],
    "teff_gspphot": ["Teff"],
    "teff_gspphot_lower": ["b_Teff"],
    "teff_gspphot_upper": ["B_Teff"],
    "logg_gspphot": ["logg"],
    "logg_gspphot_lower": ["b_logg"],
    "logg_gspphot_upper": ["B_logg"],
    "mh_gspphot": ["__Fe_H_", "[Fe/H]"],
    "mh_gspphot_lower": ["b_[Fe/H]", "b__Fe_H_"],
    "mh_gspphot_upper": ["B_[Fe/H]", "B__Fe_H_"],
    "distance_gspphot": ["Dist"],
    "ebpminrp_gspphot": ["E_BP-RP_", "EBP-RP"],
    "phot_g_mean_mag": ["Gmag"],
    "phot_bp_mean_mag": ["BPmag"],
    "phot_rp_mean_mag": ["RPmag"],
    "bp_rp": ["BP-RP"],
    # ``phot_g_mean_flux_over_error`` is derived from VizieR FG / e_FG (see below).
}


def _norm(name: str) -> str:
    """Strip non-alphanumeric characters (keep case) for robust column-name matching.

    VizieR/astropy mangle special characters (``[Fe/H]`` -> ``__Fe_H_``, ``e_RA_ICRS``);
    stripping punctuation while preserving case keeps the lower/upper bound prefixes
    (``b_`` vs ``B_``) distinct.
    """
    return re.sub(r"[^0-9A-Za-z]", "", str(name))


def _cell(row, colname):
    """Extract a plain Python scalar from an astropy table cell, or ``None`` if masked/NaN."""
    try:
        val = row[colname]
    except (KeyError, IndexError):
        return None
    if val is np.ma.masked:
        return None
    mask = getattr(val, "mask", None)
    if mask is not None and bool(np.all(mask)):
        return None
    if hasattr(val, "item"):
        try:
            val = val.item()
        except (ValueError, TypeError):
            pass
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


def _vizier_row_to_gaia_dict(row, colnames, release: str = "DR3") -> dict:
    """Convert a VizieR ``I/355/gaiadr3`` row to the Gaia TAP output dict format.

    The returned dict has exactly the keys in :data:`_GAIA_COLUMNS` (same as the original
    ``Gaia.launch_job`` results), with values as plain Python scalars (``None`` where the
    VizieR table does not provide the field).
    """
    # First match wins; punctuation-insensitive, case-sensitive.
    norm_map: dict[str, str] = {}
    for c in colnames:
        norm_map.setdefault(_norm(c), c)

    def _lookup(field: str):
        for cand in _TAP_TO_VIZIER_CANDIDATES.get(field, []):
            actual = norm_map.get(_norm(cand))
            if actual is not None:
                value = _cell(row, actual)
                if value is not None:
                    return value
        return None

    out: dict = {col: _lookup(col) for col in _GAIA_COLUMNS}

    # ref_epoch is not a VizieR column: positions are at the fixed DR3 reference epoch.
    out["ref_epoch"] = _GAIA_EPOCHS.get(release.upper(), 2016.0)

    # phot_g_mean_flux_over_error: derive from VizieR FG / e_FG when available.
    if out.get("phot_g_mean_flux_over_error") is None:
        fg = norm_map.get(_norm("FG"))
        e_fg = norm_map.get(_norm("e_FG"))
        fg_val = _cell(row, fg) if fg else None
        e_fg_val = _cell(row, e_fg) if e_fg else None
        if fg_val is not None and e_fg_val not in (None, 0, 0.0):
            out["phot_g_mean_flux_over_error"] = float(fg_val) / float(e_fg_val)

    return out


def _query_gaia_dr3_vizier(
    *,
    source_id: int | str | None = None,
    center: SkyCoord | None = None,
    radius_deg: float | None = None,
    row_limit: int = 1,
    max_rounds: int | None = None,
):
    """Query the VizieR Gaia DR3 mirror via the shared mirror-fallback layer.

    Exactly one of ``source_id`` or ``center`` (with ``radius_deg``) must be provided.

    Returns
    -------
    astropy.table.Table or None
        The result table (sorted by angular separation for cone searches), or ``None``
        if every mirror failed or no source matched.
    """
    if (source_id is None) == (center is None):
        raise ValueError("Provide exactly one of source_id or center.")

    if source_id is not None:
        return query_vizier(
            catalog=GAIA_DR3_VIZIER_ID,
            constraints={"Source": f"=={source_id}"},
            row_limit=row_limit,
            max_rounds=max_rounds,
            description=f"Gaia DR3 source_id={source_id}",
        )
    # ``+_r`` adds and sorts ascending by angular separation from center.
    return query_vizier(
        catalog=GAIA_DR3_VIZIER_ID,
        columns=["**", "+_r"],
        center=center,
        radius_deg=radius_deg,
        row_limit=row_limit,
        max_rounds=max_rounds,
        description="Gaia DR3 cone search",
    )

# Shared column list for both query functions
_GAIA_COLUMNS = [
    # Astrometry
    "source_id",
    "ra", "ra_error",
    "dec", "dec_error",
    "pmra", "pmra_error",
    "pmdec", "pmdec_error",
    "parallax", "parallax_error",
    "ref_epoch",
    "ruwe",

    # Radial velocity
    "radial_velocity", "radial_velocity_error",
    "rv_nb_transits",

    # Broadening / v sin i proxy
    "vbroad", "vbroad_error",

    # Astrophysical parameters — GSP-Phot (BP/RP spectra, ~470M stars)
    "teff_gspphot", "teff_gspphot_lower", "teff_gspphot_upper",
    "logg_gspphot", "logg_gspphot_lower", "logg_gspphot_upper",
    "mh_gspphot",   "mh_gspphot_lower",   "mh_gspphot_upper",
    "distance_gspphot",
    "ebpminrp_gspphot",

    # Photometry
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "bp_rp",
    "phot_g_mean_flux_over_error",
]


def _row_to_dict(row, colnames: list[str]) -> dict:
    """Convert an astropy table row to a plain Python dict."""
    return {
        col: (row[col].item() if hasattr(row[col], "item") else row[col])
        for col in colnames
    }


def get_stellar_data_gaia_by_coordinates(
    ra: float,
    dec: float,
    radius_arcsec: float = 5.0,
    epoch: float = 2000.0,
    pmra: float = 0.0,
    pmdec: float = 0.0,
    px: float = 0.0,
    rvsys: float = 0.0,
    release : str = "DR3",
) -> dict | None:
    """
    Query Gaia DR3 for basic stellar properties nearest to the given coordinates.

    Parameters
    ----------
    ra : float
        Right ascension in degrees.
    dec : float
        Declination in degrees.
    radius_arcsec : float
        Search cone radius in arcseconds (default 5).
    epoch : float
        Epoch of the input coordinates (default 2000.0).
    pmra : float, optional
        Proper motion in RA (mas/yr).
    pmdec : float, optional
        Proper motion in Dec (mas/yr).
    px : float, optional
        Parallax (mas).
    rvsys : float, optional
        Systemic Radial velocity (km/s).
    release : str, optional
        The Gaia data release to use (default "DR3").

    Returns
    -------
    dict or None
        Gaia DR3 properties for the closest match, or None if no source found.
    """

    if epoch != _GAIA_EPOCHS[release.upper()]:
        ra, dec = _convert_epoch_gaia(
            ra=ra,
            dec=dec,
            epoch=epoch,
            pmra=pmra,
            pmdec=pmdec,
            px=px,
            rvsys=rvsys,
            to_gaia=True,
            release=release,
        )

    radius_deg = (radius_arcsec * u.arcsec).to(u.degree).value

    center = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    table = _query_gaia_dr3_vizier(center=center, radius_deg=radius_deg, row_limit=50)

    if table is None or len(table) == 0:
        print(f"No Gaia {release.upper()} sources found within {radius_arcsec}\" of ({ra}, {dec})")
        return None

    # Cone search is sorted by angular separation (``+_r``); take the nearest source.
    results = _vizier_row_to_gaia_dict(table[0], table.colnames, release=release)

    # Convert ID to string
    results["source_id"] = str(results["source_id"])

    return results


def get_stellar_data_gaia_from_source_id(source_id: int | str) -> dict | None:
    """
    Query Gaia DR3 for basic stellar properties using a specific Gaia source_id.

    Parameters
    ----------
    source_id : int | str
        Gaia DR3 source_id of the star.

    Returns
    -------
    dict or None
        Gaia DR3 properties for the specified source, or None if not found.
    """
    table = _query_gaia_dr3_vizier(source_id=source_id, row_limit=1)

    if table is None or len(table) == 0:
        print(f"No Gaia DR3 source found with source_id = {source_id}")
        return None

    results = _vizier_row_to_gaia_dict(table[0], table.colnames, release="DR3")

    # Convert ID to string
    results["source_id"] = str(results["source_id"])

    return results


def _linear_pm_epoch_shift(
    ra: float,
    dec: float,
    pmra: float,
    pmdec: float,
    dt_yr: float,
) -> tuple[float, float]:
    """
    First-order proper-motion shift when parallax/distance are unknown.

    Uses Gaia ``pmRA`` / ``pmDE`` (mas/yr, ``pm_ra_cosdec`` convention). Accurate to
    sub-milliarcsecond over typical Gaia–input epoch baselines; avoids ERFA ``pmsafe``
    warnings from propagating proper motion without a well-defined distance.
    """
    cos_dec = math.cos(math.radians(dec))
    if cos_dec == 0.0:
        cos_dec = 1e-30
    mas_to_deg = 1.0 / (3600.0 * 1000.0)
    d_ra = (float(pmra) * dt_yr * mas_to_deg) / cos_dec
    d_dec = float(pmdec) * dt_yr * mas_to_deg
    return ra + d_ra, dec + d_dec


def _convert_epoch_gaia(
    ra: float,
    dec: float,
    epoch: float = 2000.0,
    pmra: float = 0.0,
    pmdec: float = 0.0,
    px: float = 0.0,
    rvsys: float = 0.0,
    to_gaia: bool = True,
    release : str = "DR3",
) -> tuple[float, float]:
    """
    Convert epoch to Gaia epoch.

    Parameters
    ----------
    ra : float
        Right ascension in degrees.
    dec : float
        Declination in degrees.
    epoch : float
        Epoch of the input coordinates (e.g., 2000.0).
    pmra : float, optional
        Proper motion in RA (mas/yr).
    pmdec : float, optional
        Proper motion in Dec (mas/yr).
    px : float, optional
        Parallax (mas).
    rvsys : float, optional
        Systemic Radial velocity (km/s).
    to_gaia : bool, optional
        If True, convert to Gaia epoch; otherwise, convert from Gaia to the specified epoch.
    release : str, optional
        The Gaia data release to use (default "DR3").

    Returns
    -------
    tuple[float, float]
        Converted right ascension and declination.
    """

    if to_gaia:
        epoch_in = epoch
        epoch_out = _GAIA_EPOCHS[release.upper()]
    else:
        epoch_in = _GAIA_EPOCHS[release.upper()]
        epoch_out = epoch

    dt_yr = float(epoch_out) - float(epoch_in)
    if dt_yr == 0.0:
        return ra, dec

    has_dist = px is not None and np.isfinite(px) and float(px) > 0.0
    has_pm = (
        pmra is not None
        and pmdec is not None
        and (abs(float(pmra)) > 0.0 or abs(float(pmdec)) > 0.0)
    )
    has_rv = rvsys is not None and np.isfinite(rvsys) and float(rvsys) != 0.0

    if not has_dist:
        if has_pm:
            return _linear_pm_epoch_shift(
                ra, dec, float(pmra), float(pmdec), dt_yr
            )
        return ra, dec

    t_in = Time(epoch_in, format="jyear")
    t_out = Time(epoch_out, format="jyear")

    kwargs: dict = {}
    if has_pm:
        kwargs["pm_ra_cosdec"] = float(pmra) * u.mas / u.yr
        kwargs["pm_dec"] = float(pmdec) * u.mas / u.yr
    kwargs["distance"] = (float(px) * u.mas).to(u.parsec, u.parallax())
    if has_rv:
        kwargs["radial_velocity"] = float(rvsys) * u.km / u.s

    c = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        frame="icrs",
        obstime=t_in,
        **kwargs,
    )

    c2 = c.apply_space_motion(new_obstime=t_out)

    return c2.ra.deg, c2.dec.deg


def get_2mass_from_gaia(source_id, radius_arcsec=1.0):
    """Return the 2MASS counterpart for a Gaia DR3 source, entirely via VizieR.

    Primary match: the 2MASS designation carried in the VizieR Gaia DR3 table
    (``I/355/gaiadr3``) is looked up directly in the 2MASS PSC (``II/246``), so it does
    not depend on a search radius. Fallback: a cone search around the Gaia position
    propagated back to the 2MASS epoch (J2000) to handle proper motion.

    The returned dict matches the previous Gaia-TAP implementation
    (``twomass_id``, ``tm_ra``, ``tm_dec``, ``j_m``, ``j_msigcom``, ``h_m``,
    ``h_msigcom``, ``ks_m``, ``ks_msigcom``); JHKs are in the Vega system.
    """
    table = _query_gaia_dr3_vizier(source_id=source_id, row_limit=1)
    if table is None or len(table) == 0:
        return None

    row = table[0]
    norm_map: dict[str, str] = {}
    for c in table.colnames:
        norm_map.setdefault(_norm(c), c)

    def _g(vizier_name: str):
        actual = norm_map.get(_norm(vizier_name))
        return _cell(row, actual) if actual is not None else None

    # --- primary: 2MASS designation from the Gaia catalog (radius-independent) ---
    designation = _g("2MASS")
    if designation:
        res = _query_2mass_by_designation(str(designation))
        if res is not None:
            return res

    # --- fallback: cone search at the proper-motion-propagated 2MASS-epoch position ---
    ra = _g("RA_ICRS")
    dec = _g("DE_ICRS")
    if ra is None or dec is None:
        return None
    try:
        ra, dec = _convert_epoch_gaia(
            ra=float(ra),
            dec=float(dec),
            epoch=2000.0,
            pmra=_g("pmRA"),
            pmdec=_g("pmDE"),
            px=_g("Plx"),
            rvsys=_g("RV"),
            to_gaia=False,
            release="DR3",
        )
    except Exception:
        logger.exception("2MASS fallback: epoch propagation failed; using Gaia position.")
        ra, dec = float(ra), float(dec)
    return get_2mass_data_vizier(float(ra), float(dec), radius_arcsec=float(radius_arcsec))


def get_2mass_data_vizier(ra: float, dec: float, radius_arcsec: float = 3.0) -> dict | None:
    """Cone-search the 2MASS PSC on VizieR (``II/246``) and return the nearest source.

    Uses the same VizieR mirror fallback as the Gaia DR3 queries. The returned dict uses
    the same key names as :func:`get_2mass_from_gaia` (``twomass_id``, ``tm_ra``, ``tm_dec``,
    ``j_m``, ``j_msigcom``, ``h_m``, ``h_msigcom``, ``ks_m``, ``ks_msigcom``). 2MASS JHKs
    magnitudes are in the **Vega** system.

    Parameters
    ----------
    ra, dec : float
        ICRS coordinates in degrees (e.g. the Gaia source position).
    radius_arcsec : float
        Cone search radius in arcseconds (default 3).

    Returns
    -------
    dict or None
        Photometry/astrometry for the nearest 2MASS source, or ``None`` if no match.
    """
    center = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    radius_deg = (radius_arcsec * u.arcsec).to(u.degree).value

    table = query_vizier(
        catalog=TWOMASS_PSC_VIZIER_ID,
        # ``+_r`` adds and sorts ascending by angular separation from center.
        columns=_TWOMASS_PSC_COLUMNS + ["+_r"],
        center=center,
        radius_deg=radius_deg,
        row_limit=50,
        description="2MASS PSC cone search",
    )
    if table is None or len(table) == 0:
        return None

    return _twomass_row_to_dict(table[0], table.colnames)


def _twomass_row_to_dict(row, colnames) -> dict:
    """Convert a 2MASS PSC (``II/246``) VizieR row to the ``get_2mass_from_gaia`` dict format."""
    norm_map: dict[str, str] = {}
    for c in colnames:
        norm_map.setdefault(_norm(c), c)

    def _val(vizier_name: str):
        actual = norm_map.get(_norm(vizier_name))
        return _cell(row, actual) if actual is not None else None

    twomass_id = _val("2MASS")
    return {
        "twomass_id": str(twomass_id) if twomass_id is not None else None,
        "tm_ra": _val("RAJ2000"),
        "tm_dec": _val("DEJ2000"),
        "j_m": _val("Jmag"),
        "j_msigcom": _val("e_Jmag"),
        "h_m": _val("Hmag"),
        "h_msigcom": _val("e_Hmag"),
        "ks_m": _val("Kmag"),
        "ks_msigcom": _val("e_Kmag"),
    }


_TWOMASS_PSC_COLUMNS = [
    "2MASS", "RAJ2000", "DEJ2000",
    "Jmag", "e_Jmag", "Hmag", "e_Hmag", "Kmag", "e_Kmag",
]


def _query_2mass_by_designation(designation: str) -> dict | None:
    """Look up a single 2MASS PSC source by its designation on VizieR (mirror fallback)."""
    table = query_vizier(
        catalog=TWOMASS_PSC_VIZIER_ID,
        columns=_TWOMASS_PSC_COLUMNS,
        constraints={"2MASS": f"=={designation}"},
        row_limit=5,
        description=f"2MASS PSC designation={designation}",
    )
    if table is None or len(table) == 0:
        return None
    return _twomass_row_to_dict(table[0], table.colnames)
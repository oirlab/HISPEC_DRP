import logging
import math
import re

import numpy as np
from astroquery.vizier import Vizier
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

# VizieR mirror of the Gaia DR3 main source catalog.
GAIA_DR3_VIZIER_ID = "I/355/gaiadr3"

# VizieR id of the 2MASS All-Sky Point Source Catalog (JHKs photometry, Vega system).
TWOMASS_PSC_VIZIER_ID = "II/246"

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
    max_rounds: int = 4,
):
    """Query the VizieR Gaia DR3 mirror, iterating servers (and retry rounds) on failure.

    Exactly one of ``source_id`` or ``center`` (with ``radius_deg``) must be provided.

    Returns
    -------
    astropy.table.Table or None
        The result table (sorted by angular separation for cone searches), or ``None``
        if every mirror failed or no source matched.
    """
    if (source_id is None) == (center is None):
        raise ValueError("Provide exactly one of source_id or center.")

    for round_idx in range(max_rounds):
        last_error: Exception | None = None
        for server in VIZIER_MIRRORS:
            try:
                if source_id is not None:
                    vizier = Vizier(columns=["**"], vizier_server=server)
                    vizier.ROW_LIMIT = row_limit
                    result = vizier.query_constraints(
                        catalog=GAIA_DR3_VIZIER_ID, Source=f"=={source_id}"
                    )
                else:
                    # ``+_r`` adds and sorts ascending by angular separation from center.
                    vizier = Vizier(columns=["**", "+_r"], vizier_server=server)
                    vizier.ROW_LIMIT = row_limit
                    result = vizier.query_region(
                        center, radius=radius_deg * u.deg, catalog=GAIA_DR3_VIZIER_ID
                    )
                if result is not None and len(result) > 0 and len(result[0]) > 0:
                    logger.info(
                        "Gaia DR3 VizieR query succeeded via %s (%s)",
                        server, GAIA_DR3_VIZIER_ID,
                    )
                    return result[0]
            except Exception as e:  # noqa: BLE001 - mirror fallback is intentional
                last_error = e
                logger.warning(
                    "VizieR mirror %s failed for %s: %s", server, GAIA_DR3_VIZIER_ID, e
                )
        if last_error is not None:
            logger.error(
                "All VizieR mirrors failed for %s (round %d/%d); last error: %s",
                GAIA_DR3_VIZIER_ID, round_idx + 1, max_rounds, last_error,
            )
    return None

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

    t_in = Time(epoch_in, format="jyear")
    t_out = Time(epoch_out, format="jyear")

    kwargs = {}

    if pmra is not None and pmdec is not None:
        kwargs["pm_ra_cosdec"] = pmra * u.mas / u.yr
        kwargs["pm_dec"] = pmdec * u.mas / u.yr

    if px is not None:
        kwargs["distance"] = (px * u.mas).to(u.parsec, u.parallax())

    if rvsys is not None:
        kwargs["radial_velocity"] = rvsys * u.km / u.s

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
    columns = [
        "2MASS", "RAJ2000", "DEJ2000",
        "Jmag", "e_Jmag", "Hmag", "e_Hmag", "Kmag", "e_Kmag",
        "+_r",  # add and sort ascending by angular separation from center
    ]

    table = None
    last_error: Exception | None = None
    for server in VIZIER_MIRRORS:
        try:
            vizier = Vizier(columns=columns, vizier_server=server)
            vizier.ROW_LIMIT = 50
            result = vizier.query_region(
                center, radius=radius_deg * u.deg, catalog=TWOMASS_PSC_VIZIER_ID
            )
            if result is not None and len(result) > 0 and len(result[0]) > 0:
                logger.info(
                    "2MASS VizieR query succeeded via %s (%s)",
                    server, TWOMASS_PSC_VIZIER_ID,
                )
                table = result[0]
                break
        except Exception as e:  # noqa: BLE001 - mirror fallback is intentional
            last_error = e
            logger.warning(
                "VizieR mirror %s failed for %s: %s", server, TWOMASS_PSC_VIZIER_ID, e
            )
    if table is None:
        if last_error is not None:
            logger.error(
                "All VizieR mirrors failed for %s; last error: %s",
                TWOMASS_PSC_VIZIER_ID, last_error,
            )
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
    last_error: Exception | None = None
    for server in VIZIER_MIRRORS:
        try:
            vizier = Vizier(columns=_TWOMASS_PSC_COLUMNS, vizier_server=server)
            vizier.ROW_LIMIT = 5
            result = vizier.query_constraints(
                catalog=TWOMASS_PSC_VIZIER_ID, **{"2MASS": f"=={designation}"}
            )
            if result is not None and len(result) > 0 and len(result[0]) > 0:
                logger.info(
                    "2MASS VizieR designation lookup succeeded via %s (%s)",
                    server, TWOMASS_PSC_VIZIER_ID,
                )
                return _twomass_row_to_dict(result[0][0], result[0].colnames)
        except Exception as e:  # noqa: BLE001 - mirror fallback is intentional
            last_error = e
            logger.warning(
                "VizieR mirror %s failed for %s: %s", server, TWOMASS_PSC_VIZIER_ID, e
            )
    if last_error is not None:
        logger.error(
            "All VizieR mirrors failed for %s designation lookup; last error: %s",
            TWOMASS_PSC_VIZIER_ID, last_error,
        )
    return None
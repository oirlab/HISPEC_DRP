from astroquery.gaia import Gaia
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from hispecdrp import datamodels

__all__ = [
    "get_stellar_data_gaia_by_coordinates",
    "get_stellar_data_gaia_from_source_id",
    "_convert_epoch_gaia",
    "get_2mass_from_gaia"
]

_GAIA_EPOCHS = {
    "DR3" : 2016.0,
    "DR4" : 2017.5,
}

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
    cols = ", ".join(_GAIA_COLUMNS)

    query = f"""
        SELECT TOP 50 {cols},
               DISTANCE(
                   POINT('ICRS', ra, dec),
                   POINT('ICRS', {ra}, {dec})
               ) AS ang_sep
        FROM gaiadr3.gaia_source
        WHERE CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
        ) = 1
        ORDER BY ang_sep ASC
    """

    job = Gaia.launch_job(query)
    results = job.get_results()

    if len(results) == 0:
        print(f"No Gaia {release.upper()} sources found within {radius_arcsec}\" of ({ra}, {dec})")
        return None

    return _row_to_dict(results[0], results.colnames)


def get_stellar_data_gaia_from_source_id(source_id: int) -> dict | None:
    """
    Query Gaia DR3 for basic stellar properties using a specific Gaia source_id.

    Parameters
    ----------
    source_id : int
        Gaia DR3 source_id of the star.

    Returns
    -------
    dict or None
        Gaia DR3 properties for the specified source, or None if not found.
    """
    cols = ", ".join(_GAIA_COLUMNS)

    query = f"""
        SELECT TOP 1 {cols}
        FROM gaiadr3.gaia_source
        WHERE source_id = {source_id}
    """

    job = Gaia.launch_job(query)
    results = job.get_results()

    if len(results) == 0:
        print(f"No Gaia DR3 source found with source_id = {source_id}")
        return None

    return _row_to_dict(results[0], results.colnames)


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


def set_gaia_meta(input_model : datamodels.DataModel, gaia_id : int | str | None = None):
    if gaia_id is None:
        gaia_id = input_model.meta.target_gaia_id
    assert gaia_id is not None, "No Gaia ID provided and none found in input model metadata"
    
    result = get_stellar_data_gaia_from_source_id(gaia_id)

    # Populate metadata
    input_model.meta.target_gaia_id = result["source_id"]
    input_model.meta.target_gaia_ra = result["ra"]
    input_model.meta.target_gaia_dec = result["dec"]
    input_model.meta.target_gaia_epoch = result["ref_epoch"]
    input_model.meta.target_gaia_rapm = result["pmra"]
    input_model.meta.target_gaia_decpm = result["pmdec"]
    input_model.meta.target_gaia_px = result["parallax"]
    input_model.meta.target_gaia_rvsys = result["radial_velocity"]
    
    input_model.meta.target_gaia_teff = result["teff_gspphot"]

    return input_model


def get_2mass_from_gaia(source_id, radius_arcsec=1.0):
    # --- primary: Gaia-provided crossmatch ---
    query_primary = f"""
    SELECT
        tm.designation AS twomass_id,
        tm.ra AS tm_ra,
        tm.dec AS tm_dec,
        DISTANCE(
            POINT('ICRS', g.ra, g.dec),
            POINT('ICRS', tm.ra, tm.dec)
        ) AS ang_sep
    FROM gaiadr3.gaia_source AS g
    JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xmatch
        ON g.source_id = xmatch.source_id
    JOIN gaiadr1.tmass_original_valid AS tm
        ON xmatch.original_ext_source_id = tm.designation
    WHERE g.source_id = {source_id}
    """

    job = Gaia.launch_job(query_primary)
    res = job.get_results()

    if len(res) > 0:
        return res

    # --- fallback: cone search ---
    query_pos = f"""
    SELECT ra, dec
    FROM gaiadr3.gaia_source
    WHERE source_id = {source_id}
    """

    job = Gaia.launch_job(query_pos)
    r = job.get_results()

    if len(r) == 0:
        return None

    ra, dec = r["ra"][0], r["dec"][0]

    query_fallback = f"""
    SELECT
        designation AS twomass_id,
        ra AS tm_ra,
        dec AS tm_dec,
        DISTANCE(
            POINT('ICRS', {ra}, {dec}),
            POINT('ICRS', ra, dec)
        ) AS ang_sep
    FROM gaiadr1.tmass_original_valid
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec}/3600.)
    )
    ORDER BY ang_sep ASC
    """

    job = Gaia.launch_job(query_fallback)
    res = job.get_results()

    if len(res) == 0:
        return None

    return res
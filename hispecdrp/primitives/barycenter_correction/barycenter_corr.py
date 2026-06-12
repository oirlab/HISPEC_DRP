from barycorrpy import JDUTC_to_HJDTDB, get_BC_vel, JDUTC_to_BJDTDB
import numpy as np
from astropy.coordinates import EarthLocation
from dataclasses import dataclass

from ..utils.gaia_query import _convert_epoch_gaia
from ... import datamodels

import logging
logger = logging.getLogger(__name__)

KECK_LOCATION = EarthLocation.of_site('Keck')
PARVI_LOCATION = EarthLocation.of_site('Palomar')


@dataclass
class BarycorrpyStarData:
    ra : float
    dec : float
    epoch : float = 2000.0
    pm_ra : float = 0.0
    pm_dec : float = 0.0
    px : float = 0.0
    rvsys : float = 0.0


def get_barycorrpy_star_data(input_model : datamodels.L2Model) -> BarycorrpyStarData:
    """
    Extract the necessary stellar data from the input model for barycorrpy calculations.

    Parameters
    ----------
    input_model : datamodels.L2Model
        The input data model containing the necessary metadata.

    Returns
    -------
    BarycorrpyStarData
        A dataclass containing the extracted stellar data.
    """

    ra = input_model.meta.target_gaia_ra
    dec = input_model.meta.target_gaia_dec
    epoch = input_model.meta.target_gaia_epoch or 2016.5
    pm_ra = input_model.meta.target_gaia_rapm or 0.0
    pm_dec = input_model.meta.target_gaia_decpm or 0.0
    px = input_model.meta.target_gaia_px or 0.0
    rvsys = input_model.meta.target_gaia_rvsys or 0.0

    return BarycorrpyStarData(
        ra=ra, dec=dec, epoch=epoch,
        pm_ra=pm_ra, pm_dec=pm_dec,
        px=px, rvsys=rvsys
    )

def _resolve_location(observatory : str | EarthLocation) -> EarthLocation:
    if isinstance(observatory, EarthLocation):
        return observatory
    elif isinstance(observatory, str):
        if 'keck' in observatory.lower():
            return KECK_LOCATION
        elif any(k in observatory.lower() for k in ("palomar", "hale")):
            return PARVI_LOCATION
        else:
            return EarthLocation.of_site(observatory)
    else:
        raise ValueError(f"Invalid observatory: {observatory}")


def calc_barycenter_corrections(
    mjd_utc : float | np.ndarray,
    star_data : BarycorrpyStarData,
    observatory : str | EarthLocation,
    zmeas : float = 0.0,
) -> dict:
    """
    Calculate the barycenter corrections for the input data using barycorrpy.

    Parameters
    ----------
    mjd_utc : float | Sequence[float]
        Modified Julian Date in UTC.
    star_data : BarycorrpyStarData
        Stellar data required for barycenter correction calculation.
    observatory : str | EarthLocation
        Observatory name or EarthLocation object for the observation.
    zmeas : float, optional
        Measured redshift of the target, relevant for relativistic corrections.
        Default is 0.

    Returns
    -------
    mbjd : float
        Modified Barycentric Julian Date.
    vel_bc : float
        Barycentric Earth Radial Velocity correction in km/s.
    """

    # if epoch != 2000.0:
    #     logger.info(f"Converting input coordinates from epoch {epoch} to Gaia epoch for barycenter correction calculation...")
    #     ra, dec = _convert_epoch_gaia(
    #         ra=ra, dec=dec, epoch=epoch,
    #         pmra=pm_ra, pmdec=pm_dec,
    #         px=px,
    #         rvsys=rvsys,
    #         to_gaia=False,
    #     )
    #     epoch = 2000.0

    # Get observatory location
    obs_loc = _resolve_location(observatory)
    lat = obs_loc.lat.to_value('deg')
    longi = obs_loc.lon.to_value('deg')
    alt = obs_loc.height.to_value('m')

    # Convert to JDUTC
    JDUTC = mjd_utc + 2400000.5

    # NOTE: Consider checking the status returned by barycorrpy
    # and log errors/warnings as needed.
    bjd, _, _ = JDUTC_to_BJDTDB(
        JDUTC,
        ra=star_data.ra, dec=star_data.dec, epoch=star_data.epoch, pmra=star_data.pm_ra, pmdec=star_data.pm_dec, px=star_data.px, rv=star_data.rvsys,
        lat=lat, longi=longi, alt=alt,
        ephemeris='de430', leap_update=True
    )

    vel_bc, _, _ = get_BC_vel(
        JDUTC,
        ra=star_data.ra, dec=star_data.dec, epoch=star_data.epoch, pmra=star_data.pm_ra, pmdec=star_data.pm_dec, px=star_data.px, rv=star_data.rvsys,
        lat=lat, longi=longi, alt=alt,
        ephemeris='de430', leap_update=True, predictive=False,
        zmeas=zmeas
    )

    # Convert BJD to MBJD
    mbjd = bjd - 2400000.5

    return mbjd, vel_bc


def calc_solar_barycenter_corrections(
    mjd_utc : float | np.ndarray,
    observatory : str | EarthLocation = None,
) -> dict:
    """
    Calculate the solar system barycenter corrections for the input data using barycorrpy.

    Parameters
    ----------
    mjd_utc : float
        Modified Julian Date in UTC.

    Returns
    -------
    mbjd : float
        Modified Barycentric Julian Date.
    vel_bc : float
        Barycentric Earth Radial Velocity correction in km/s.
    """

    obs_loc = _resolve_location(observatory)
    lat = obs_loc.lat.to_value('deg')
    longi = obs_loc.lon.to_value('deg')
    alt = obs_loc.height.to_value('m')

    JDUTC = mjd_utc + 2400000.5

    # NOTE: USE BJD or HJD for solar RVs?
    # - Difference is a few seconds, negligible for most applications.
    bjd, _, _ = JDUTC_to_HJDTDB(
        JDUTC=JDUTC,
        lat=lat, longi=longi, alt=alt,
        ephemeris='de430', leap_update=True
    )

    vel_bc, _, _ = get_BC_vel(
        JDUTC=JDUTC,
        lat=lat, longi=longi, alt=alt,
        ephemeris='de430', leap_update=True,
        SolSystemTarget='Sun',
        predictive=True,
        zmeas=0.0
    )

    # Convert BJD to MBJD
    mbjd = bjd - 2400000.5

    return mbjd, vel_bc
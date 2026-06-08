from hispecdrp.primitives.utils import (
    get_stellar_data_gaia_by_coordinates,
    get_stellar_data_gaia_from_source_id,
    get_2mass_from_gaia,
)
from hispecdrp import datamodels

# HD 189733
source_params = {
    "source_id": '1827242816201846144',
    "ra": 300.18213726402,  # degrees
    "dec": 22.71085364818,  # degrees
    "pm_ra": -3.208,  # mas/yr
    "pm_dec": -250.323,  # mas/yr
    "epoch": 2000.0,
    "radial_velocity" : -2.317,  # km/s
    "px" : 50.5668, # mas
    "source_id_2mass" : "20004370+2242391",
}

def test_gaia_query_auto():
    meta = {
        'target_gaia_id': source_params["source_id"]
    }
    model = datamodels.RampModel(meta=meta)

    assert model.meta.target_gaia_id == source_params["source_id"]

    # NOTE: The Gaia DR3 coordinates are at epoch J2016.0, so won't be exactly equal.
    assert abs(model.meta.target_gaia_ra - source_params["ra"]) < 1e-4
    assert abs(model.meta.target_gaia_dec - source_params["dec"]) < 1e-2

    # Check 2mass cross match
    assert model.meta.target_2mass_id == source_params["source_id_2mass"]

def test_gaia_query_by_coordinates():
    _id, ra, dec = source_params["source_id"], source_params["ra"], source_params["dec"]
    epoch = source_params["epoch"]
    radial_velocity = source_params["radial_velocity"]
    result = get_stellar_data_gaia_by_coordinates(
        ra, dec,
        pmra=source_params["pm_ra"],
        pmdec=source_params["pm_dec"],
        epoch=epoch,
        radius_arcsec=1.0,
        rvsys=radial_velocity
    )

    assert result is not None
    assert result["source_id"] == _id
    assert abs(result["parallax"] - source_params["px"]) < 0.5
    assert abs(result["ra"] - source_params["ra"]) < 1e-4
    assert abs(result["dec"] - source_params["dec"]) < 1e-2
    assert 4800 < result["teff_gspphot"] < 5200  # early K dwarf
    assert abs(result["radial_velocity"] - source_params["radial_velocity"]) < 0.5


def test_gaia_query_by_source_id():
    _id, ra, dec = source_params["source_id"], source_params["ra"], source_params["dec"]
    result = get_stellar_data_gaia_from_source_id(_id)

    assert result is not None
    assert result["source_id"] == _id
    assert abs(result["parallax"] - source_params["px"]) < 0.5
    assert abs(result["ra"] - source_params["ra"]) < 1e-4
    assert abs(result["dec"] - source_params["dec"]) < 1e-2
    assert 4800 < result["teff_gspphot"] < 5200  # early K dwarf
    assert abs(result["radial_velocity"] - source_params["radial_velocity"]) < 0.5


def test_gaia_to_2mass_crossmatch():
    _id = source_params["source_id"]

    result = get_2mass_from_gaia(_id)

    assert result is not None
    assert len(result) > 0
    twomass_id = result["twomass_id"]
    assert twomass_id is not None
    assert twomass_id == source_params["source_id_2mass"]


def test_gaia_to_2mass_fallback_path():
    # Use an artificially tiny radius to force fallback behavior to matter
    _id = source_params["source_id"]

    result = get_2mass_from_gaia(_id, radius_arcsec=0.01)

    assert result is not None
    assert len(result) > 0

    twomass_id = result["twomass_id"]
    assert twomass_id == source_params["source_id_2mass"]
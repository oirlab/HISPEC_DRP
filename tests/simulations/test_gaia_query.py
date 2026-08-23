import subprocess
import sys

import pytest
from astropy.table import Table

from hispecdrp.primitives.utils import (
    get_stellar_data_gaia_by_coordinates,
    get_stellar_data_gaia_from_source_id,
    get_2mass_from_gaia,
)
from hispecdrp.primitives.utils import gaia_query
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


# --- VizieR mirror-fallback layer (offline; no network) ----------------------------

def _fake_vizier_class(attempts, ok_servers, table=None):
    """Build a stand-in ``Vizier`` class recording each server it is constructed for.

    Servers in *ok_servers* return *table*; all others raise, so tests can pin down
    exactly which mirrors were tried and in what order.
    """
    result_table = Table({"Source": [1]}) if table is None else table

    class FakeVizier:
        def __init__(self, columns=None, column_filters=None, vizier_server=None,
                     timeout=None):
            self.server = vizier_server
            self.columns = columns
            self.column_filters = column_filters
            self.timeout = timeout
            self.ROW_LIMIT = None
            attempts.append({"server": vizier_server, "timeout": timeout,
                             "columns": columns, "column_filters": column_filters})

        @staticmethod
        def clear_cache():
            attempts.append({"server": "<clear_cache>", "timeout": None,
                             "columns": None, "column_filters": None})

        def _respond(self):
            if self.server not in ok_servers:
                raise RuntimeError(f"mirror {self.server} down")
            return [result_table]

        def query_region(self, center, radius=None, catalog=None):
            return self._respond()

        def query_constraints(self, catalog=None, **constraints):
            return self._respond()

    return FakeVizier


def _servers_tried(attempts):
    return [a["server"] for a in attempts if a["server"] != "<clear_cache>"]


def test_query_vizier_falls_through_failing_mirrors(monkeypatch):
    mirrors = ["bad1.example", "bad2.example", "good.example", "unused.example"]
    monkeypatch.setattr(gaia_query, "VIZIER_MIRRORS", mirrors)
    monkeypatch.delenv("HISPEC_DRP_VIZIER_MIRRORS", raising=False)
    attempts = []
    monkeypatch.setattr(
        gaia_query, "_vizier_class",
        lambda: _fake_vizier_class(attempts, {"good.example"}),
    )

    table = gaia_query.query_vizier(
        catalog="I/355/gaiadr3", constraints={"Source": "==1"}
    )

    assert table is not None
    # Stops at the first working mirror, without touching the ones after it.
    assert _servers_tried(attempts) == ["bad1.example", "bad2.example", "good.example"]


def test_query_vizier_retries_full_mirror_list(monkeypatch):
    monkeypatch.setattr(gaia_query, "VIZIER_MIRRORS", ["a.example", "b.example"])
    monkeypatch.delenv("HISPEC_DRP_VIZIER_MIRRORS", raising=False)
    attempts = []
    monkeypatch.setattr(
        gaia_query, "_vizier_class", lambda: _fake_vizier_class(attempts, set())
    )

    table = gaia_query.query_vizier(
        catalog="II/246", constraints={"2MASS": "==x"}, max_rounds=3
    )

    assert table is None
    assert _servers_tried(attempts) == ["a.example", "b.example"] * 3


def test_query_vizier_clears_cache_before_querying(monkeypatch):
    monkeypatch.setattr(gaia_query, "VIZIER_MIRRORS", ["a.example"])
    monkeypatch.delenv("HISPEC_DRP_VIZIER_MIRRORS", raising=False)
    attempts = []
    monkeypatch.setattr(
        gaia_query, "_vizier_class",
        lambda: _fake_vizier_class(attempts, {"a.example"}),
    )

    gaia_query.query_vizier(catalog="II/246", constraints={"2MASS": "==x"})

    assert attempts[0]["server"] == "<clear_cache>"


def test_query_vizier_applies_timeout(monkeypatch):
    monkeypatch.setattr(gaia_query, "VIZIER_MIRRORS", ["a.example"])
    monkeypatch.delenv("HISPEC_DRP_VIZIER_MIRRORS", raising=False)
    monkeypatch.setenv("HISPEC_DRP_VIZIER_TIMEOUT", "7.5")
    attempts = []
    monkeypatch.setattr(
        gaia_query, "_vizier_class",
        lambda: _fake_vizier_class(attempts, {"a.example"}),
    )

    gaia_query.query_vizier(catalog="II/246", constraints={"2MASS": "==x"})

    assert [a["timeout"] for a in attempts if a["server"] == "a.example"] == [7.5]


def test_vizier_mirrors_env_override(monkeypatch):
    monkeypatch.setenv("HISPEC_DRP_VIZIER_MIRRORS", "local.mirror, other.mirror ")
    assert gaia_query.vizier_mirrors() == ["local.mirror", "other.mirror"]

    monkeypatch.setenv("HISPEC_DRP_VIZIER_MIRRORS", "   ")
    assert gaia_query.vizier_mirrors() == list(gaia_query.VIZIER_MIRRORS)


def test_vizier_timeout_env_override(monkeypatch):
    monkeypatch.delenv("HISPEC_DRP_VIZIER_TIMEOUT", raising=False)
    assert gaia_query.vizier_timeout() == gaia_query.VIZIER_TIMEOUT_DEFAULT

    monkeypatch.setenv("HISPEC_DRP_VIZIER_TIMEOUT", "12")
    assert gaia_query.vizier_timeout() == 12.0

    # Invalid / non-positive values fall back to the default rather than disabling it.
    for bad in ("not-a-number", "-5", "0"):
        monkeypatch.setenv("HISPEC_DRP_VIZIER_TIMEOUT", bad)
        assert gaia_query.vizier_timeout() == gaia_query.VIZIER_TIMEOUT_DEFAULT


def test_query_vizier_rejects_ambiguous_arguments():
    with pytest.raises(ValueError):
        gaia_query.query_vizier(catalog="II/246")
    with pytest.raises(ValueError):
        gaia_query.query_vizier(
            catalog="II/246", center=object(), radius_deg=1.0,
            constraints={"2MASS": "==x"},
        )
    with pytest.raises(ValueError):
        gaia_query.query_vizier(catalog="II/246", center=object())


def test_importing_hispecdrp_does_not_import_astroquery_vizier():
    """``astroquery.vizier`` must stay lazy: importing it can hang (issue #201)."""
    code = (
        "import sys, hispecdrp; "
        "print('astroquery.vizier' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().endswith("False")

def test_linear_pm_epoch_shift_offline():
    """First-order PM shift is pure arithmetic (no network / ERFA)."""
    import math

    ra0, dec0 = 10.0, 20.0
    pmra, pmdec = 1000.0, -500.0  # mas/yr
    dt = 2.0  # yr
    ra1, dec1 = gaia_query._linear_pm_epoch_shift(ra0, dec0, pmra, pmdec, dt)
    expected_dra = (1000.0 * 2.0 / (3600.0 * 1000.0)) / math.cos(math.radians(dec0))
    expected_ddec = -500.0 * 2.0 / (3600.0 * 1000.0)
    assert ra1 == pytest.approx(ra0 + expected_dra)
    assert dec1 == pytest.approx(dec0 + expected_ddec)


def test_convert_epoch_gaia_linear_fallback_without_parallax():
    """px<=0 uses the linear PM path (avoids ERFA pmsafe without distance)."""
    ra, dec = gaia_query._convert_epoch_gaia(
        180.0,
        0.0,
        epoch=2000.0,
        pmra=10.0,
        pmdec=0.0,
        px=0.0,
        to_gaia=True,
        release="DR3",
    )
    # DR3 epoch 2016 → dt=16 yr; 10 mas/yr → 0.16" at dec=0.
    assert ra == pytest.approx(180.0 + 16.0 * 10.0 / (3600.0 * 1000.0))
    assert dec == pytest.approx(0.0)

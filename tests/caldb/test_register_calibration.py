"""Tests for HISPECCalibrationStore.register_calibration filename / origin sync."""

from __future__ import annotations

import os

import astropy.time
import pytest

from hispecdrp import datamodels
from hispecdrp.calibrations import HISPECCalibrationStore
from hispecdrp.simulations.simulate_up_the_ramp import (
    create_wavelength_solution_model_from_sim,
    register_sim_lfc_wavesol,
)


@pytest.fixture
def caldb_cache(tmp_path, monkeypatch):
    cache = tmp_path / "KOA_CALIBRATION_CACHE"
    cache.mkdir()
    monkeypatch.setenv("KOA_CALIBRATION_CACHE", str(cache))
    return cache


@pytest.fixture(scope="module")
def bspec_geometric_wavesol():
    """Build geometric BSPEC wavesol once per module (HDF load is expensive)."""
    return create_wavelength_solution_model_from_sim(
        "BSPEC", meta={"spectrograph": "BSPEC", "instrument_era": "0.0.1"}
    )


def test_register_calibration_stores_non_null_filename(caldb_cache):
    """DB row must record the saved FITS basename (koa_middleware used to store None)."""
    dark = datamodels.DarkModel(
        meta={"mjd_start": astropy.time.Time.now().mjd, "master_cal": True}
    )
    with HISPECCalibrationStore(connect_remote=False, sync_on_init=False) as store:
        path, rec = store.register_calibration(dark, origin="LOCAL", new_version=True)

    assert path is not None
    assert os.path.isfile(path)
    assert rec is not None
    assert rec.get("filename"), f"expected non-null filename, got {rec!r}"
    assert rec["filename"] == os.path.basename(path)
    assert rec.get("koa_id") == rec["filename"]
    assert rec.get("origin") == "LOCAL"
    assert rec.get("cal_version"), f"expected cal_version, got {rec!r}"

    with HISPECCalibrationStore(connect_remote=False, sync_on_init=False) as store:
        rows = list(store.query())
    assert len(rows) == 1
    assert rows[0]["filename"] == rec["filename"]
    assert rows[0]["filename"] is not None


def test_register_sim_lfc_wavesol_usable_by_select_and_get(caldb_cache):
    """Sim LFC wavesol registration must leave a selectable, on-disk calibration."""
    from hispecdrp.calibrations.selectors import WavelengthSolutionSelector

    meta = {
        "spectrograph": "BSPEC",
        "instrument_era": "0.0.1",
        "mjd_start": 61210.0,
    }
    path, rec = register_sim_lfc_wavesol("BSPEC", meta=meta, origin="LOCAL", new_version=True)

    assert path is not None and os.path.isfile(path)
    assert rec["filename"]
    assert rec["origin"] == "LOCAL"
    assert "lfc_wavesol" in rec["filename"]

    # Stage 1 set_wavesol resolves via select_and_get_calibration; null filename used to
    # crash in os.path.join(data_dir, None).
    query_meta = datamodels.RampModel(meta=dict(meta)).meta
    with HISPECCalibrationStore(connect_remote=False, sync_on_init=False) as store:
        resolved_path, resolved_rec = store.select_and_get_calibration(
            query_meta, WavelengthSolutionSelector()
        )
    assert resolved_path is not None
    assert os.path.isfile(resolved_path)
    assert resolved_rec["filename"] == rec["filename"]


def test_register_sim_lfc_wavesol_idempotent_version_family(caldb_cache):
    """Re-registering without new_version must reuse the version-family match (a list)."""
    meta = {
        "spectrograph": "BSPEC",
        "instrument_era": "0.0.1",
        "mjd_start": 61210.0,
    }
    path1, rec1 = register_sim_lfc_wavesol(
        "BSPEC", meta=meta, origin="LOCAL", new_version=True
    )
    path2, rec2 = register_sim_lfc_wavesol(
        "BSPEC", meta=meta, origin="LOCAL", new_version=False
    )
    assert path1 is not None and os.path.isfile(path1)
    assert path2 == path1
    assert rec2["filename"] == rec1["filename"]


def test_create_wavelength_solution_model_from_sim_has_expected_cal_type(
    caldb_cache, bspec_geometric_wavesol
):
    model = bspec_geometric_wavesol
    assert model.meta.cal_type == "lfc_wavesol"
    assert getattr(model, "wave_fib1", None) is not None


def test_filter_version_latest_handles_none_and_str(caldb_cache):
    """LATEST must not crash when the version family mixes None and string versions."""
    from hispecdrp.calibrations.selectors import WavelengthSolutionSelector

    sel = WavelengthSolutionSelector()
    rows = [
        {"cal_version": None, "filename": "geo.fits"},
        {"cal_version": "001", "filename": "local.fits"},
        {"cal_version": None, "filename": "geo2.fits"},
    ]
    chosen = sel._filter_version(rows, cal_version="LATEST")
    assert len(chosen) == 1
    assert chosen[0]["filename"] == "local.fits"


def test_packaged_line_zp_preferred_over_geometric_lfc_wavesol(
    caldb_cache, bspec_geometric_wavesol
):
    """Stage-1 selector must pick packaged line-ZP even if geometric is closer/first."""
    from hispecdrp.calibrations.selectors import WavelengthSolutionSelector
    from hispecdrp.simulations.sim_hdf_wavesol_packaging import (
        PACKAGED_SIM_HDF_LINE_ZP_NOTE,
        load_packaged_sim_hdf_line_zp_wavesol,
        packaged_sim_hdf_wavesol_path,
    )

    if not packaged_sim_hdf_wavesol_path("BSPEC").is_file():
        pytest.skip("packaged BSPEC line-ZP WAVE not present")

    meta = {
        "spectrograph": "BSPEC",
        "instrument_era": "0.0.1",
        "mjd_start": 60577.0,
        "datetime_obs": astropy.time.Time(60577.0, format="mjd", scale="utc").isot,
    }
    # Geometric competitor registered first (same night / version family).
    geometric = bspec_geometric_wavesol.copy()
    for k, v in meta.items():
        setattr(geometric.meta, k, v)
    with HISPECCalibrationStore(connect_remote=False, sync_on_init=False) as store:
        geo_path, geo_rec = store.register_calibration(
            geometric, origin="LOCAL", new_version=True
        )
    assert geo_path and geo_rec.get("filename")
    assert PACKAGED_SIM_HDF_LINE_ZP_NOTE not in str(geo_rec.get("note") or "")

    pack_path, pack_rec = register_sim_lfc_wavesol(
        "BSPEC", meta=meta, origin="LOCAL"
    )
    assert pack_path and pack_rec.get("filename")
    assert PACKAGED_SIM_HDF_LINE_ZP_NOTE in str(pack_rec.get("note") or "")
    assert pack_rec["filename"] != geo_rec["filename"]

    query_meta = datamodels.RampModel(meta=dict(meta)).meta
    with HISPECCalibrationStore(connect_remote=False, sync_on_init=False) as store:
        resolved_path, resolved_rec = store.select_and_get_calibration(
            query_meta, WavelengthSolutionSelector()
        )
    assert resolved_path is not None and os.path.isfile(resolved_path)
    assert resolved_rec["filename"] == pack_rec["filename"]
    assert PACKAGED_SIM_HDF_LINE_ZP_NOTE in str(resolved_rec.get("note") or "")

    # Content must match the packaged arrays, not the geometric deposit-inverse.
    packaged = load_packaged_sim_hdf_line_zp_wavesol("BSPEC")
    resolved = datamodels.open(resolved_path)
    import numpy as np

    pack_w = np.asarray(packaged.wave_fib2, float)
    got_w = np.asarray(resolved.wave_fib2, float)
    ok = np.isfinite(pack_w) & np.isfinite(got_w)
    dv = (got_w[ok] - pack_w[ok]) / pack_w[ok] * 2.99792458e8
    assert abs(float(np.median(dv))) < 1.0

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


def test_create_wavelength_solution_model_from_sim_has_expected_cal_type(caldb_cache):
    model = create_wavelength_solution_model_from_sim(
        "BSPEC", meta={"spectrograph": "BSPEC", "instrument_era": "0.0.1"}
    )
    assert model.meta.cal_type == "lfc_wavesol"
    assert getattr(model, "wave_fib1", None) is not None

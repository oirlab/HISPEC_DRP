"""Tests for E2E calibration provenance summary helpers."""

from __future__ import annotations

import json
import os

from hispecdrp import datamodels
from hispecdrp.simulations.e2e_calibrations import complete_stage0_calibrations
from hispecdrp.simulations.e2e_support import (
    DEFAULT_HISPEC_DETECTOR_SHAPE,
    E2E_STAGE2_PRIMITIVES,
    apply_e2e_save_and_preview_flags,
    build_calibration_dataset_summary,
    calibration_model_detector_shape,
    calibration_source_category,
    dummy_cal_path,
    get_default_stage2_prim_args,
    open_cached_stage0_calibration,
    write_calibration_manifest,
)


def test_e2e_stage2_stellar_rv_preview_write_png_disabled():
    prim_args = apply_e2e_save_and_preview_flags(
        get_default_stage2_prim_args(),
        E2E_STAGE2_PRIMITIVES,
    )
    assert prim_args["stellar_rv"]["preview_write_png"] is False


def test_calibration_source_category():
    assert calibration_source_category("koa") == "KOA"
    assert calibration_source_category("config:/path/to/bias.fits") == "User config"
    assert calibration_source_category("simulated") == "Simulated"
    assert calibration_source_category("fiberflat_pipeline") == "Generated (fiberflat pipeline)"


def test_build_and_write_calibration_manifest(tmp_path):
    bias = datamodels.BiasModel(meta={"origin": "LOCAL"})
    bias_path = tmp_path / "sim_bias.fits"
    bias.save(filepath=str(bias_path))

    sources = {
        "bias": f"simulated_cached:{bias_path}",
        "dark": "koa",
    }
    models = {"bias": bias, "dark": datamodels.DarkModel(meta={"origin": "KECK"})}

    summary = build_calibration_dataset_summary(
        output_dir=str(tmp_path),
        calibration_models=models,
        sources=sources,
        expected_products=["bias", "dark", "gain"],
    )
    assert summary.rows[0].category in ("Simulated (cached)", "Simulated")
    assert summary.missing_products == ["gain"]
    assert any("gain" in w for w in summary.validation_warnings)

    manifest = write_calibration_manifest(str(tmp_path), summary)
    assert os.path.isfile(manifest)
    with open(manifest, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["missing_products"] == ["gain"]
    assert len(data["products"]) >= 2


def test_open_cached_stage0_calibration_rejects_wrong_shape(tmp_path):
    bias = datamodels.BiasModel(
        BIAS=__import__("numpy").zeros((4096, 4096), dtype=__import__("numpy").float32),
    )
    path = dummy_cal_path(str(tmp_path), "bias")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bias.save(filepath=path)

    assert open_cached_stage0_calibration(path, DEFAULT_HISPEC_DETECTOR_SHAPE) is None
    assert calibration_model_detector_shape(bias) == (4096, 4096)


def test_complete_stage0_regenerates_stale_cached_shape(tmp_path):
    import numpy as np

    out = str(tmp_path)
    os.makedirs(os.path.join(out, "cals"), exist_ok=True)
    stale = datamodels.DarkModel(
        RATE=np.zeros((4096, 4096), dtype=np.float32),
    )
    stale.save(filepath=dummy_cal_path(out, "dark"))

    models: dict = {}
    sources: dict = {}
    complete_stage0_calibrations(models, None, out, sources)

    assert calibration_model_detector_shape(models["dark"]) == DEFAULT_HISPEC_DETECTOR_SHAPE
    assert sources["dark"] == "simulated"

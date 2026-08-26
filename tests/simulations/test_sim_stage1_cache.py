"""Reuse of simulated Stage-1 products under ``<output_dir>/cals``."""

from __future__ import annotations

import json
from pathlib import Path

from hispecdrp.simulations.simulate_up_the_ramp import (
    create_dummy_fiberflat_rate_files,
    get_expected_fiberflat_rate_paths,
    get_sim_stage1_master_paths,
    sim_stage1_masters_exist,
)


def test_create_dummy_fiberflat_reuses_existing_rates(tmp_path):
    paths = get_expected_fiberflat_rate_paths(str(tmp_path), "BSPEC", [2])
    Path(paths[0]).parent.mkdir(parents=True, exist_ok=True)
    Path(paths[0]).write_bytes(b"cached-rate")

    out = create_dummy_fiberflat_rate_files(
        [2],
        "BSPEC",
        str(tmp_path),
        calibration_models={},
        reuse_existing=True,
    )
    assert out == paths
    assert Path(paths[0]).read_bytes() == b"cached-rate"


def test_sim_stage1_masters_exist_checks_fibers_and_aperture(tmp_path):
    paths = get_sim_stage1_master_paths(str(tmp_path), "BSPEC")
    for key in ("trace_pos", "trace_prof", "fiberflat"):
        Path(paths[key]).write_bytes(b"x")
    Path(paths["meta"]).write_text(
        json.dumps(
            {
                "spectrograph": "BSPEC",
                "fibers": [2],
                "extraction_aperture": 10.0,
            }
        ),
        encoding="utf-8",
    )

    assert sim_stage1_masters_exist(str(tmp_path), "BSPEC", [2], extraction_aperture=10.0)
    assert not sim_stage1_masters_exist(
        str(tmp_path), "BSPEC", [1, 2], extraction_aperture=10.0
    )
    assert not sim_stage1_masters_exist(
        str(tmp_path), "BSPEC", [2], extraction_aperture=8.0
    )

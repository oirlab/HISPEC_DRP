"""Tests for E2E RV closure helpers (SP extract + LSF mask wiring)."""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from hispecdrp.simulations.e2e_pipeline import (
    E2EConfig,
    e2e_rv_closure_config,
    merge_infinite_snr_stage1_primitives,
    merge_sp_lsf_e2e_stage1_primitives,
    summarize_per_line_rv_mad,
)


def test_merge_sp_stage1_sets_spectroperfectionism():
    prim = merge_sp_lsf_e2e_stage1_primitives(None, extraction_aperture=8)
    assert prim["spec_extract"]["method"] == "spectroperfectionism"
    assert prim["spec_extract"]["extraction_aperture"] == 8


def test_merge_sheared_sp_method():
    prim = merge_infinite_snr_stage1_primitives(None, method="sheared_sp")
    assert prim["spec_extract"]["method"] == "sheared_sp"


def test_e2e_rv_closure_config_sets_stage1_extract_method(tmp_path):
    cfg = E2EConfig(output_dir=str(tmp_path), template_filename=None)
    out = e2e_rv_closure_config(cfg, extract_method="sheared_sp", use_lsf_mask=False)
    assert out.infinite_snr_rv_validation is True
    assert out.stage1_primitives is not None
    assert out.stage1_primitives["spec_extract"]["method"] == "sheared_sp"


def test_summarize_per_line_rv_mad():
    class _Dummy:
        pass

    m = _Dummy()
    inj = 100.0
    m.rvccfavg_fib2 = Table({"RV": [102.0], "RV_ERR": [0.5]})
    m.rvccf_fib2 = Table(
        {
            "RV": np.array([101.0, 99.0, 103.0, 100.0]),
            "INLIER": np.array([True, True, True, True]),
        }
    )
    s = summarize_per_line_rv_mad(m, injected_rv_m_s=inj, fiber=2)
    assert s["n_lines"] == 4
    assert abs(float(s["aggregate_bias_m_s"]) - 2.0) < 1e-12
    assert abs(float(s["per_line_mad_m_s"]) - 1.0) < 1e-12
    assert abs(float(s["formal_rv_err_m_s"]) - 0.5) < 1e-12

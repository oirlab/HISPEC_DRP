"""Smoke tests for E2E RV simulation helpers."""

from __future__ import annotations

from hispecdrp.simulations.e2e_pipeline import (
    DEFAULT_E2E_RV_SIM_EXTRACTION_APERTURE,
    stellar_rv_kwargs_for_e2e_sim,
)
from hispecdrp.utils.line_mask_utils import CCFOPT_SINGLE_LINE_STELLAR_RV_KW


def test_stellar_rv_kwargs_for_e2e_sim_defaults():
    mask = "/tmp/fake.mas"
    kw = stellar_rv_kwargs_for_e2e_sim(1.0, mask, fibers=[2])
    ccfopt = CCFOPT_SINGLE_LINE_STELLAR_RV_KW
    assert kw["algorithm"] == "ccf_line_mask"
    assert kw["mask_path"] == mask
    assert kw["ccf_input_velocity"] == 1000.0
    assert kw["do_fibers"] == ["2"]
    assert kw["window_line_by_line"] is True
    assert kw["window_size_um"] == float(ccfopt["window_size_um"])
    assert kw["v_half_window"] == float(ccfopt["v_half_window_m_s"])
    assert kw["v_step_coarse"] == float(ccfopt["v_step_m_s"])
    assert kw["ccf_zb"] == 0.0


def test_default_e2e_rv_sim_extraction_aperture_matches_notebook():
    assert DEFAULT_E2E_RV_SIM_EXTRACTION_APERTURE == 8.0

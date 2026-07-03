"""Tests for infinite-S/N RV validation helpers and optional full E2E run."""

from __future__ import annotations

import numpy as np
import pytest

from hispecdrp.simulations.e2e_infinite_snr_rv import (
    HIGH_PRECISION_STELLAR_RV_ARGS,
    apply_infinite_snr_rv_e2e_config,
    assert_infinite_snr_rv_recovery,
    evaluate_infinite_snr_rv_recovery,
    merge_infinite_snr_stage2_prim_args,
)
from hispecdrp.simulations.e2e_pipeline import (
    E2EConfig,
    STAGE1_TRACE_SOURCE_SIM_HDF,
    stage1_uses_sim_hdf_trace,
)
from hispecdrp.utils.line_mask_utils import paired_line_mask_path_for_template


def test_paired_line_mask_path_for_template_default():
    path = paired_line_mask_path_for_template(
        "lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5"
    )
    assert path.endswith("lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.delta.mas")


def test_apply_infinite_snr_rv_e2e_config():
    cfg = apply_infinite_snr_rv_e2e_config(
        E2EConfig(
            output_dir="/tmp/e2e",
            template_filename="lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5",
        )
    )
    assert cfg.infinite_snr_rv_validation is True
    assert cfg.poisson_noise is False
    assert cfg.convert_to_uint16 is False
    assert cfg.stage1_trace_source == STAGE1_TRACE_SOURCE_SIM_HDF
    assert cfg.stellar_rv_mask_path is not None
    assert stage1_uses_sim_hdf_trace(cfg)


def test_merge_infinite_snr_stage2_prim_args():
    merged = merge_infinite_snr_stage2_prim_args({})
    assert merged["stellar_rv"]["mask_ccf_pixel_interp_order"] == 1
    assert merged["stellar_rv"]["v_step_fine"] == HIGH_PRECISION_STELLAR_RV_ARGS["v_step_fine"]


def test_evaluate_infinite_snr_rv_recovery_mock():
    from types import SimpleNamespace

    injected_km_s = 34.42
    injected_m_s = injected_km_s * 1000.0
    model = SimpleNamespace(
        rvccfavg_fib1=np.array(
            [(injected_m_s + 50.0, 10.0, 1)],
            dtype=[("RV", "f8"), ("RV_ERR", "f8"), ("INLIER", "i4")],
        ),
        rvccf_fib1=np.array(
            [
                (injected_m_s + 40.0, True),
                (injected_m_s + 50.0, True),
                (injected_m_s + 60.0, True),
                (injected_m_s + 5000.0, False),
            ],
            dtype=[("RV", "f8"), ("INLIER", "?")],
        ),
    )
    ev = evaluate_infinite_snr_rv_recovery(model, injected_rv_km_s=injected_km_s, fibers=[1])
    assert ev.max_abs_bias_km_s == pytest.approx(0.05, abs=1e-6)
    assert ev.fibers[0].n_inlier_chunks == 3
    assert_infinite_snr_rv_recovery(ev)


@pytest.mark.slow
def test_infinite_snr_rv_e2e_recovery(tmp_path):
    from hispecdrp.simulations.e2e_gaia import DEFAULT_E2E_MJD_START
    from hispecdrp.simulations.e2e_pipeline import run_e2e_simulation
    from hispecdrp.simulations.run_e2e_sim import DEFAULT_GAIA_SOURCE_ID, DEFAULT_TEMPLATE

    cfg = apply_infinite_snr_rv_e2e_config(
        E2EConfig(
            output_dir=str(tmp_path / "infinite_snr"),
            n_reads=6,
            max_cores=1,
            spectrograph="BSPEC",
            mjd_start=DEFAULT_E2E_MJD_START,
            fibers=[1, 2, 3],
            fiberflat_fibers=[1, 2, 3],
            use_master_calibrations=True,
            template_filename=DEFAULT_TEMPLATE,
            gaia_source_id=DEFAULT_GAIA_SOURCE_ID,
            overwrite_ramp=True,
            infinite_snr_rv_validation=True,
            infinite_snr_rv_assert=False,
            make_preview_html=False,
        )
    )
    res = run_e2e_simulation(cfg)
    assert res.stage2_model is not None
    assert res.infinite_snr_rv_evaluation is not None
    assert res.infinite_snr_rv_evaluation.max_abs_bias_km_s is not None

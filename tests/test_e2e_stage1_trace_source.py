"""Tests for E2E Stage 1 trace / fiberflat source selection."""

from __future__ import annotations

from hispecdrp.simulations.e2e_calibrations import (
    E2ECalibrationBundle,
    STAGE1_TRACE_KEYS,
    stage1_masters_from_user_config,
)
from hispecdrp.simulations.e2e_pipeline import (
    E2EConfig,
    STAGE1_TRACE_SOURCE_KOA,
    STAGE1_TRACE_SOURCE_PIPELINE,
    effective_stage1_trace_source,
    stage1_uses_fiberflat_pipeline,
)


def test_effective_stage1_trace_source_explicit():
    cfg = E2EConfig(output_dir=".", stage1_trace_source="koa", derive_stage1_trace_from_sim=True)
    assert effective_stage1_trace_source(cfg) == STAGE1_TRACE_SOURCE_KOA
    assert not stage1_uses_fiberflat_pipeline(cfg)


def test_effective_stage1_trace_source_legacy_alias():
    cfg = E2EConfig(output_dir=".", derive_stage1_trace_from_sim=False)
    assert effective_stage1_trace_source(cfg) == STAGE1_TRACE_SOURCE_KOA

    cfg2 = E2EConfig(output_dir=".", derive_stage1_trace_from_sim=True)
    assert effective_stage1_trace_source(cfg2) == STAGE1_TRACE_SOURCE_PIPELINE
    assert stage1_uses_fiberflat_pipeline(cfg2)


def test_stage1_masters_from_user_config():
    bundle = E2ECalibrationBundle(
        trace_pos=object(),
        trace_prof=object(),
        fiberflat=object(),
        sources={
            "trace_pos": "config:/tmp/tracepos.fits",
            "trace_prof": "config:/tmp/traceprof.fits",
            "fiberflat": "config:/tmp/fiberflat.fits",
        },
    )
    assert stage1_masters_from_user_config(bundle)

    bundle.sources["fiberflat"] = "koa"
    assert not stage1_masters_from_user_config(bundle)

    bundle2 = E2ECalibrationBundle()
    assert not stage1_masters_from_user_config(bundle2)
    assert list(STAGE1_TRACE_KEYS) == ["trace_pos", "trace_prof", "fiberflat"]

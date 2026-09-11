"""
Ideal-spectrum RV bias regression: compare recovery configs vs injected bulk RV.

Exercises the same path as the **native-paint** ideal-spectrum RV recovery
example (packaged lte040 + ``*.ccfopt.delta.mas``, no HDF LSF).  This test
keeps that native grid so interpolation / window-alignment regressions stay
isolated.

All recovery comparisons use the full BSPEC echelle order range (39 orders).
Tests are marked ``slow`` because each bulk-RV point runs ``build_l2`` + line-mask
CCF over the full stack (~30–40 s per point after template cache is warm).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from hispecdrp.primitives import StellarRVPrimitive
from hispecdrp.simulations.spectra import (
    build_l2,
    paint_order_list,
    quiet_pipeline_logging,
)
from hispecdrp.utils.line_mask_utils import (
    CCFOPT_SINGLE_LINE_STELLAR_RV_KW,
    paired_line_mask_path_for_template,
)

LTE040 = "lte040.0-4.5-0.0a+0.0.BT-Settl.spec.7.h5"
SPECTROGRAPH = "BSPEC"
FIBER = 1
NOTEBOOK_BULK_RV_KM_S = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)
TARGET_MAX_BIAS_MM_S = 100.0


@dataclass(frozen=True)
class IdealRvRecoveryConfig:
    """Paint + recovery knobs under test."""

    name: str
    mask_ccf_pixel_interp_order: int
    window_size_um: float | None = None
    """``None`` → ``StellarRVPrimitive`` default (0.25 µm), matching the notebook omission."""


# Notebook-like: Lanczos-3 paint/recovery, wide default line window (legacy).
BASELINE_NOTEBOOK = IdealRvRecoveryConfig(
    "baseline_notebook",
    mask_ccf_pixel_interp_order=3,
    window_size_um=0.25,
)

# Recommended: primitive ccfopt line-by-line defaults (narrow window + velocity grid).
RECOMMENDED = IdealRvRecoveryConfig(
    "aligned_lanczos3",
    mask_ccf_pixel_interp_order=3,
    window_size_um=None,
)

ALIGNED_LANCZOS3 = RECOMMENDED

ALIGNED_LINEAR = IdealRvRecoveryConfig(
    "aligned_linear",
    mask_ccf_pixel_interp_order=1,
    window_size_um=float(CCFOPT_SINGLE_LINE_STELLAR_RV_KW["window_size_um"]),
)


def _mask_path() -> str:
    return paired_line_mask_path_for_template(LTE040, mask_style="ccfopt.delta")


def _orders() -> tuple[int, ...]:
    return paint_order_list(SPECTROGRAPH, FIBER, None, None)


def _recover_bias_m_s(bulk_km_s: float, config: IdealRvRecoveryConfig) -> float:
    inj_m_s = float(bulk_km_s) * 1000.0
    orders = _orders()
    l2 = build_l2(
        bulk_km_s,
        spectrograph=SPECTROGRAPH,
        template=LTE040,
        fiber=FIBER,
        mask_ccf_pixel_interp_order=int(config.mask_ccf_pixel_interp_order),
    )
    rv_kw: dict = {
        "algorithm": "ccf_line_mask",
        "mask_path": _mask_path(),
        "ccf_input_velocity": inj_m_s,
        "do_fibers": [str(FIBER)],
        "do_orders": list(orders),
        "window_line_by_line": True,
        "ccf_zb": 0.0,
        "mask_ccf_pixel_interp_order": int(config.mask_ccf_pixel_interp_order),
        "save_result": False,
        "save_preview": False,
        "show_preview": False,
    }
    if config is BASELINE_NOTEBOOK:
        rv_kw["ccfopt_line_by_line_defaults"] = False
        rv_kw["window_size_um"] = 0.25
        rv_kw["v_half_window"] = 15000.0
        rv_kw["v_step_coarse"] = 25.0
    elif config.window_size_um is not None:
        rv_kw["ccfopt_line_by_line_defaults"] = False
        rv_kw["window_size_um"] = float(config.window_size_um)
        rv_kw["v_half_window"] = float(CCFOPT_SINGLE_LINE_STELLAR_RV_KW["v_half_window_m_s"])
        rv_kw["v_step_coarse"] = float(CCFOPT_SINGLE_LINE_STELLAR_RV_KW["v_step_m_s"])
    # else: recommended path — primitive applies ccfopt defaults

    with quiet_pipeline_logging():
        out = StellarRVPrimitive.call(l2, **rv_kw)
    avg = getattr(out, f"rvccfavg_fib{FIBER}")
    rec_m_s = float(avg["RV"][0])
    return rec_m_s - inj_m_s


def _bias_table_mm_s(
    config: IdealRvRecoveryConfig,
    bulk_km_s: tuple[float, ...] = NOTEBOOK_BULK_RV_KM_S,
) -> dict[float, float]:
    return {
        float(bulk): _recover_bias_m_s(bulk, config) * 1e3 for bulk in bulk_km_s
    }


def _max_abs_bias_mm_s(table: dict[float, float]) -> float:
    return float(max(abs(v) for v in table.values()))


def _max_differential_bias_mm_s(table: dict[float, float]) -> float:
    biases = np.asarray(list(table.values()), dtype=float)
    if biases.size < 2:
        return 0.0
    return float(np.max(np.abs(biases - biases[0])))


def test_ideal_rv_bias_fixture_paths_exist() -> None:
    """Fast sanity check: template mask and BSPEC order list are present."""
    assert Path(_mask_path()).is_file()
    orders = _orders()
    assert orders[0] == 110
    assert orders[-1] == 148
    assert len(orders) == 39


@pytest.fixture(scope="module")
def full_bspec_bias_tables() -> dict[str, dict[float, float]]:
    """Bias (mm/s) per bulk RV for each config on the full BSPEC stack."""
    configs = (BASELINE_NOTEBOOK, ALIGNED_LANCZOS3, ALIGNED_LINEAR)
    return {cfg.name: _bias_table_mm_s(cfg) for cfg in configs}


@pytest.mark.slow
def test_ideal_rv_bias_aligned_lanczos_beats_baseline_full_bspec(
    full_bspec_bias_tables: dict[str, dict[float, float]],
) -> None:
    """Passing ``window_size_um`` from mask-build settings should cut peak |bias|."""
    base = _max_abs_bias_mm_s(full_bspec_bias_tables[BASELINE_NOTEBOOK.name])
    aligned = _max_abs_bias_mm_s(full_bspec_bias_tables[ALIGNED_LANCZOS3.name])
    assert aligned < base, (
        f"aligned Lanczos-3 max|bias|={aligned:.1f} mm/s should be < "
        f"notebook baseline {base:.1f} mm/s; tables={full_bspec_bias_tables}"
    )


@pytest.mark.slow
def test_ideal_rv_bias_recommended_below_target_full_bspec(
    full_bspec_bias_tables: dict[str, dict[float, float]],
) -> None:
    """Target: |bias| < 100 mm/s at each notebook bulk RV with recommended config."""
    table = full_bspec_bias_tables[RECOMMENDED.name]
    for bulk_km_s, bias_mm_s in table.items():
        assert abs(bias_mm_s) < TARGET_MAX_BIAS_MM_S, (
            f"{RECOMMENDED.name} bias at {bulk_km_s:g} km/s = {bias_mm_s:.1f} mm/s "
            f"(limit {TARGET_MAX_BIAS_MM_S:g} mm/s); full table={table}"
        )


@pytest.mark.slow
def test_ideal_rv_bias_recommended_differential_full_bspec(
    full_bspec_bias_tables: dict[str, dict[float, float]],
) -> None:
    """Differential bias vs RV=0 should stay below target with recommended config."""
    table = full_bspec_bias_tables[RECOMMENDED.name]
    d_max = _max_differential_bias_mm_s(table)
    assert d_max < TARGET_MAX_BIAS_MM_S, (
        f"{RECOMMENDED.name} max|Δbias| vs RV=0 = {d_max:.1f} mm/s "
        f"(limit {TARGET_MAX_BIAS_MM_S:g} mm/s); table={table}"
    )


@pytest.mark.slow
def test_ideal_rv_bias_linear_vs_lanczos_documented_full_bspec(
    full_bspec_bias_tables: dict[str, dict[float, float]],
) -> None:
    """
    Record linear vs Lanczos on the full ccfopt stack (linear is not always better).

    On a single echelle order, linear sampling can beat Lanczos-3; on the full
    lte040 ccfopt mask, aligned Lanczos-3 currently wins.  This test only requires
    finite biases so both configs stay in the regression suite.
    """
    lanczos = full_bspec_bias_tables[ALIGNED_LANCZOS3.name]
    linear = full_bspec_bias_tables[ALIGNED_LINEAR.name]
    for name, table in (("lanczos", lanczos), ("linear", linear)):
        for bulk, bias in table.items():
            assert np.isfinite(bias), f"{name} bias at {bulk} km/s is not finite: {bias}"
    # Guard against accidental regression to notebook-scale bias on the recommended path.
    assert _max_abs_bias_mm_s(lanczos) < 0.5 * _max_abs_bias_mm_s(
        full_bspec_bias_tables[BASELINE_NOTEBOOK.name]
    )

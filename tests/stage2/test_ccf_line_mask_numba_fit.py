"""Mask-line stored CCF fits: preview and pipeline must agree (Numba centroid)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hispecdrp.primitives.utils.ccf.ccf_utils import (
    CCF_PREVIEW_SLOT_FIT_CONFIG,
    LINE_MASK_CCF_STORED_NEGATED_FROM_RAW,
    effective_ccf_v_step_for_fit,
    fit_ccf_rv,
    fit_ccf_rv_rows_numba,
)

_FIXTURE = Path(__file__).resolve().parents[1] / "test_data" / "ccf_line_mask_chunk0.npz"


@pytest.mark.skipif(not _FIXTURE.is_file(), reason="chunk-0 CCF fixture missing")
def test_fit_ccf_rv_matches_numba_on_e2e_mask_line_chunk0():
    """
    Regression: Astropy ``fit_ccf_rv`` used to return ~2110 m/s with ``fit_ok=True`` on
    this shallow sloped mask-line CCF while the pipeline Numba fitter returned ~1049 m/s.
    """
    data = np.load(_FIXTURE)
    v = np.asarray(data["v"], dtype=np.float64)
    ccf = np.asarray(data["ccf"], dtype=np.float64)
    rv_table = float(data["rv_table"])
    inj = float(data["rv_true"])
    vs = effective_ccf_v_step_for_fit(v)
    cfg = CCF_PREVIEW_SLOT_FIT_CONFIG

    rv_n, _err_n, ok_n = fit_ccf_rv_rows_numba(
        v,
        ccf.reshape(1, -1),
        gaussian_seed_mean_m_s=inj,
        pass1_half_width_m_s=float(cfg.pass1_half_width_km_s) * 1000.0,
        pass2_half_width_m_s=float(cfg.pass2_half_width_km_s) * 1000.0,
        initial_stddev_m_s=float(cfg.initial_stddev_km_s) * 1000.0,
        levmar_maxiter=int(cfg.levmar_maxiter),
        min_points=int(cfg.min_points),
        max_cores=1,
    )
    assert int(ok_n[0]) == 1

    rv_a, err_a, params, diag = fit_ccf_rv(
        v,
        ccf,
        vs,
        return_params=True,
        config=cfg,
        ccf_line_mask_stored_negated=LINE_MASK_CCF_STORED_NEGATED_FROM_RAW,
        line_mask_line_window_by_line=True,
        gaussian_seed_mean_m_s=inj,
        max_velocity_samples=None,
    )
    assert bool(diag.get("fit_ok"))
    assert bool(diag.get("fit_via_numba_line_mask"))
    np.testing.assert_allclose(rv_a, rv_n[0], rtol=0, atol=2.0)
    np.testing.assert_allclose(rv_a, rv_table, rtol=0, atol=2.0)
    np.testing.assert_allclose(rv_a, inj, rtol=0, atol=60.0)
    assert abs(rv_a - inj) < abs(rv_a - 2110.0)
    assert params["sigma"] > 0.0
    assert np.isfinite(err_a)


def test_fit_ccf_rv_matches_numba_on_clean_gaussian_bump():
    true_rv = 250.0
    v = np.linspace(-3000.0, 3000.0, 601, dtype=np.float64)
    ccf = 0.8 * np.exp(-0.5 * ((v - true_rv) / 180.0) ** 2) - 1.0
    vs = effective_ccf_v_step_for_fit(v)
    cfg = CCF_PREVIEW_SLOT_FIT_CONFIG

    rv_n, _err_n, ok_n = fit_ccf_rv_rows_numba(
        v,
        ccf.reshape(1, -1),
        gaussian_seed_mean_m_s=true_rv,
        pass1_half_width_m_s=float(cfg.pass1_half_width_km_s) * 1000.0,
        pass2_half_width_m_s=float(cfg.pass2_half_width_km_s) * 1000.0,
        initial_stddev_m_s=float(cfg.initial_stddev_km_s) * 1000.0,
        levmar_maxiter=int(cfg.levmar_maxiter),
        min_points=int(cfg.min_points),
        max_cores=1,
    )
    assert int(ok_n[0]) == 1

    rv_a, _err_a, _params, diag = fit_ccf_rv(
        v,
        ccf,
        vs,
        return_params=True,
        config=cfg,
        ccf_line_mask_stored_negated=True,
        gaussian_seed_mean_m_s=true_rv,
        max_velocity_samples=None,
    )
    assert bool(diag.get("fit_via_numba_line_mask"))
    np.testing.assert_allclose(rv_a, rv_n[0], rtol=0, atol=2.0)
    np.testing.assert_allclose(rv_a, true_rv, rtol=0, atol=15.0)

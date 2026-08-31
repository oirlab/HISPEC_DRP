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
    rv_per_chunks_line_mask,
)
from hispecdrp.utils.line_mask_utils import MaskLines

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


def test_rv_per_chunks_line_mask_accepts_decreasing_wavelength():
    """HDF WAVE often decreases along detector rows; pixel-edge CCF needs increasing λ."""
    n = 128
    w_inc = np.linspace(1.200, 1.220, n, dtype=np.float64)
    lam0 = 1.210
    spec = 1.0 - 0.5 * np.exp(-0.5 * ((w_inc - lam0) / 1.5e-4) ** 2)
    err = np.full(n, 0.01, dtype=np.float64)
    dq = np.zeros(n, dtype=bool)
    mask = MaskLines.from_delta_centers_um(np.array([lam0]), np.array([1.0]))
    kw = dict(
        window_size_um=0.02,
        v_min_m_s=-20_000.0,
        v_max_m_s=20_000.0,
        v_step_m_s=100.0,
        window_line_by_line=True,
        line_mask_line_chunk_v_center_m_s=0.0,
        min_good_pixels=9,
        max_cores=1,
        return_ccf=False,
    )
    rvs_inc, _, _, _, _, _, _, _, _, _, _, _ = rv_per_chunks_line_mask(
        w_inc, spec, err, dq, mask, **kw
    )
    rvs_dec, _, _, _, _, _, _, _, _, _, _, _ = rv_per_chunks_line_mask(
        w_inc[::-1].copy(), spec[::-1].copy(), err[::-1].copy(), dq[::-1].copy(), mask, **kw
    )
    assert rvs_inc.size >= 1
    assert rvs_dec.size >= 1
    assert np.any(np.isfinite(rvs_inc))
    assert np.any(np.isfinite(rvs_dec))

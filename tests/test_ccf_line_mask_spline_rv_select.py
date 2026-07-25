"""Tests for spline mask CCF RV selection (bisector gate + peak fallback)."""
from __future__ import annotations

import numpy as np

from hispecdrp.primitives.utils.ccf.ccf_fitting import CCFMetrics
from hispecdrp.primitives.utils.ccf.ccf_line_mask_spline import (
    _BISECTOR_VS_GAUSS_MAX_ABS_M_S,
    _FIT_VS_PEAK_DISAGREE_MIN_M_S,
    _pipeline_rv_from_metrics,
)


def _metrics(
    *,
    rv: float,
    rv_err: float = 7.5,
    bis_rv: float | None,
    bis_rv_err: float | None = 7.5,
    fwhm: float = 5000.0,
) -> CCFMetrics:
    return CCFMetrics(
        contrast=1.0,
        contrast_err=0.1,
        rv=rv,
        rv_err=rv_err,
        fwhm=fwhm,
        fwhm_err=100.0,
        offset=0.0,
        offset_err=0.1,
        bis_rv=bis_rv,
        bis_rv_err=bis_rv_err,
    )


def test_pipeline_rv_rejects_bisector_far_from_gaussian():
    """Tight gate: |bis−gauss| above max(0.25×FWHM, 500 m/s) keeps the Gaussian."""
    gauss = -110_000.0
    fwhm = 4_000.0
    gate = max(_BISECTOR_VS_GAUSS_MAX_ABS_M_S, 0.25 * fwhm)
    bis = gauss + (gate + 200.0)
    rv, err = _pipeline_rv_from_metrics(
        _metrics(rv=gauss, bis_rv=bis, fwhm=fwhm)
    )
    assert abs(rv - gauss) < 1e-6
    assert err == 7.5


def test_pipeline_rv_accepts_bisector_near_gaussian():
    gauss = -110_000.0
    bis = gauss + 50.0
    rv, _ = _pipeline_rv_from_metrics(
        _metrics(rv=gauss, bis_rv=bis, bis_rv_err=3.0, fwhm=4_000.0)
    )
    assert abs(rv - bis) < 1e-6


def test_pipeline_rv_falls_back_to_peak_on_fit_disagreement():
    """Biased Gaussian/bisector yield to the local discrete/parabolic CCF peak."""
    inj = -110_000.0
    v = np.arange(inj - 5_000.0, inj + 5_000.0 + 25.0, 50.0)
    stored = np.exp(-0.5 * ((v - inj) / 800.0) ** 2)
    raw = -stored
    biased = inj + (_FIT_VS_PEAK_DISAGREE_MIN_M_S + 50.0)
    rv, err = _pipeline_rv_from_metrics(
        _metrics(rv=biased, bis_rv=biased + 10.0, fwhm=4_000.0),
        vel_grid=v,
        ccf_raw=raw,
    )
    assert abs(rv - inj) < 25.0
    assert err > 0.0


def test_pipeline_rv_ignores_distant_alias_peak():
    """Local peak window must not jump to a stronger distant alias on a broad grid."""
    inj = -110_000.0
    alias = inj - 8_000.0
    v = np.arange(inj - 20_000.0, inj + 20_000.0 + 25.0, 50.0)
    stored = np.exp(-0.5 * ((v - inj) / 800.0) ** 2)
    stored += 1.5 * np.exp(-0.5 * ((v - alias) / 800.0) ** 2)  # stronger alias
    raw = -stored
    rv, _ = _pipeline_rv_from_metrics(
        _metrics(rv=inj + 5.0, bis_rv=inj - 3.0, fwhm=4_000.0),
        vel_grid=v,
        ccf_raw=raw,
    )
    # Stay with Gaussian/bisector near the true core — do not fall back to alias.
    assert abs(rv - inj) < 50.0

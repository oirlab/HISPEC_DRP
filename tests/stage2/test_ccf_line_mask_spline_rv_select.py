"""Tests for spline mask CCF RV selection (Gaussian only; no discrete-peak snap-back)."""
from __future__ import annotations

import numpy as np

from hispecdrp.primitives.utils.ccf.ccf_fitting import CCFMetrics
from hispecdrp.primitives.utils.ccf.ccf_line_mask_spline import (
    _ALIAS_PRIOR_RECENTER_GATE_M_S,
    _pipeline_rv_from_metrics,
    _resolve_spline_mask_metrics_center_m_s,
)
from hispecdrp.primitives.utils.ccf.ccf_utils import LINE_MASK_CCF_STORED_NEGATED_FROM_RAW


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


def test_pipeline_rv_uses_gaussian_even_when_bisector_near():
    """Reported chunk RV is the Gaussian; bisector is diagnostic only."""
    gauss = -110_000.0
    bis = gauss + 50.0
    rv, err = _pipeline_rv_from_metrics(
        _metrics(rv=gauss, bis_rv=bis, bis_rv_err=3.0, fwhm=4_000.0)
    )
    assert abs(rv - gauss) < 1e-6
    assert err == 7.5


def test_pipeline_rv_ignores_bisector_far_from_gaussian():
    gauss = -110_000.0
    bis = gauss + 5_000.0
    rv, err = _pipeline_rv_from_metrics(
        _metrics(rv=gauss, bis_rv=bis, fwhm=4_000.0)
    )
    assert abs(rv - gauss) < 1e-6
    assert err == 7.5


def test_pipeline_rv_never_snaps_to_discrete_peak():
    """Biased Gaussian must not be replaced by a local discrete/parabolic peak."""
    inj = -110_000.0
    v = np.arange(inj - 5_000.0, inj + 5_000.0 + 25.0, 50.0)
    stored = np.exp(-0.5 * ((v - inj) / 800.0) ** 2)
    raw = -stored
    biased = inj + 100.0
    rv, err = _pipeline_rv_from_metrics(
        _metrics(rv=biased, bis_rv=biased + 10.0, fwhm=4_000.0, rv_err=3.0),
        vel_grid=v,
        ccf_raw=raw,
    )
    assert abs(rv - biased) < 1e-6
    assert abs(err - 3.0) < 1e-6


def test_pipeline_rv_keeps_gaussian_even_with_stronger_alias_peak():
    """A stronger distant alias on the CCF must not override the Gaussian RV."""
    inj = -110_000.0
    alias = inj - 8_000.0
    v = np.arange(inj - 20_000.0, inj + 20_000.0 + 25.0, 50.0)
    stored = np.exp(-0.5 * ((v - inj) / 800.0) ** 2)
    stored += 1.5 * np.exp(-0.5 * ((v - alias) / 800.0) ** 2)
    raw = -stored
    gauss = inj + 5.0
    rv, _ = _pipeline_rv_from_metrics(
        _metrics(rv=gauss, bis_rv=inj - 3.0, fwhm=4_000.0),
        vel_grid=v,
        ccf_raw=raw,
    )
    assert abs(rv - gauss) < 1e-6


def test_pipeline_rv_keeps_gaussian_near_prior_despite_far_alias():
    """Near-prior Gaussian is kept even if a far alias peak is deeper."""
    prior = 0.0
    shoulder = -150.0
    alias = 284.0
    v = np.arange(-5_000.0, 5_000.0 + 25.0, 25.0)
    stored = 0.3 * np.exp(-0.5 * ((v - prior) / 400.0) ** 2)
    stored += 1.2 * np.exp(-0.5 * ((v - alias) / 200.0) ** 2)
    raw = -stored
    rv, err = _pipeline_rv_from_metrics(
        _metrics(rv=shoulder, bis_rv=shoulder, fwhm=4_800.0, rv_err=0.8),
        vel_grid=v,
        ccf_raw=raw,
        peak_anchor_m_s=prior,
    )
    assert abs(rv - shoulder) < 1e-6
    assert abs(err - 0.8) < 1e-6


def test_pipeline_rv_keeps_alias_locked_gaussian_without_peak_snap():
    """Alias-locked Gaussian is reported as-is (no discrete-peak recovery snap)."""
    from hispecdrp.primitives.utils.ccf.ccf_line_mask_spline import (
        _ALIAS_PRIOR_SEARCH_HALF_WIDTH_M_S,
    )

    inj = 0.0
    alias = 480.0
    assert abs(alias) > _ALIAS_PRIOR_RECENTER_GATE_M_S
    assert abs(alias) > _ALIAS_PRIOR_SEARCH_HALF_WIDTH_M_S
    v = np.arange(-5_000.0, 5_000.0 + 25.0, 25.0)
    stored = np.exp(-0.5 * ((v - inj) / 120.0) ** 2)
    stored += 1.2 * np.exp(-0.5 * ((v - alias) / 120.0) ** 2)
    raw = -stored
    rv, err = _pipeline_rv_from_metrics(
        _metrics(rv=alias, bis_rv=alias, fwhm=1_500.0, rv_err=2.5),
        vel_grid=v,
        ccf_raw=raw,
        peak_anchor_m_s=inj,
    )
    assert abs(rv - alias) < 1e-6
    assert abs(err - 2.5) < 1e-6


def test_resolve_metrics_center_recenters_on_competitive_prior_peak():
    """Prominence on an alias yields to a competitive trough near the RV prior."""
    if not LINE_MASK_CCF_STORED_NEGATED_FROM_RAW:
        return
    from hispecdrp.primitives.utils.ccf.ccf_line_mask_spline import (
        _ALIAS_PRIOR_SEARCH_HALF_WIDTH_M_S,
    )

    prior = 0.0
    alias = 480.0
    assert abs(alias) > _ALIAS_PRIOR_SEARCH_HALF_WIDTH_M_S
    v = np.arange(-5_000.0, 5_000.0 + 25.0, 25.0)
    # Negated-from-raw stored: troughs are negative; deeper = more negative.
    stored = -np.exp(-0.5 * ((v - prior) / 120.0) ** 2)
    stored += -1.05 * np.exp(-0.5 * ((v - alias) / 120.0) ** 2)
    v_c = _resolve_spline_mask_metrics_center_m_s(
        v,
        stored,
        prominence_center_m_s=alias,
        prior_m_s=prior,
    )
    assert abs(v_c - prior) < 50.0

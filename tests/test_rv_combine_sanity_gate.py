"""Tests for robust stellar-RV chunk aggregation."""

from __future__ import annotations

import numpy as np

from hispecdrp.primitives.stellar_rv.rv_combine_utils import combine_stellar_rv_chunk_estimates


def test_combine_stellar_rv_rejects_alias_peaks_with_tiny_errors():
    """E2E-like failure mode: alias chunks with σ≈0 must not dominate the weighted mean."""
    true_ms = 34420.0
    good = np.full(12, true_ms) + np.linspace(-500, 500, 12)
    bad = np.array([468818.8, -489368.9, -361298.7, 387261.6])
    rvs = np.concatenate([good, bad])
    errs = np.concatenate([np.full(12, 8.0), np.array([0.0015, 0.0039, 0.0055, 0.0023])])
    rv_mean, rv_std, inlier, _, _ = combine_stellar_rv_chunk_estimates(
        rvs, errs, outlier_sigma=4.0, chunk_rv_max_abs_m_s=120_000.0
    )
    assert np.sum(inlier) == 12
    assert abs(rv_mean - true_ms) < 3000.0
    assert rv_std > 1.0


def test_combine_stellar_rv_median_fallback_when_iv_mean_biased():
    """When IV mean disagrees with inlier median, return the median."""
    med_ms = 34000.0
    rvs = np.array([med_ms, med_ms + 100.0, med_ms - 80.0, 450000.0])
    errs = np.array([5.0, 6.0, 7.0, 0.002])
    rv_mean, _, inlier, _, _ = combine_stellar_rv_chunk_estimates(
        rvs,
        errs,
        chunk_rv_max_abs_m_s=120_000.0,
    )
    assert np.sum(inlier) == 3
    assert abs(rv_mean - med_ms) < 200.0


def test_combine_stellar_rv_prior_gate_recovers_fiber2_like_bimodal_aliases():
    """Alias peaks within ±120 km/s must not dominate when a CCF center prior is known."""
    true_ms = 34420.0
    good = np.array([34322.0, 34116.9, 36168.2, 34952.3, 33076.7])
    bad_aliases = np.array(
        [-17177.5, -42367.8, -56709.4, -45553.2, 103587.7, -114377.9, -71378.5, -17359.5, -3479.6]
    )
    rvs = np.concatenate([good, bad_aliases])
    errs = np.full(rvs.size, 10.0)
    rv_mean, rv_std, inlier, _, _ = combine_stellar_rv_chunk_estimates(
        rvs,
        errs,
        chunk_rv_prior_m_s=true_ms,
        chunk_rv_prior_half_width_m_s=10_000.0,
    )
    assert np.sum(inlier) == good.size
    assert abs(rv_mean - true_ms) < 500.0
    assert rv_std > 1.0

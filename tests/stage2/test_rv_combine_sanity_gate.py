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


def test_error_aware_rejection_keeps_large_err_outside_mad():
    """A chunk outside the MAD envelope but consistent with its σ must be kept."""
    # Tight cluster → small MAD; offset point with large formal error covers the gap.
    rvs = np.concatenate([np.linspace(-0.2, 0.2, 12), [8.0]])
    errs = np.concatenate([np.full(12, 1.0), [5.0]])
    _, _, inlier, _, _ = combine_stellar_rv_chunk_estimates(
        rvs, errs, outlier_sigma=4.0, chunk_rv_max_abs_m_s=1.0e6
    )
    assert bool(inlier[-1])
    assert int(np.sum(inlier)) == rvs.size


def test_aggregate_rv_err_is_max_of_ivw_and_scatter_se():
    """Quoted fiber σ is max(photon IVW, MAD/√N), not IVW of oversized bars alone."""
    # Tight cluster with huge formal σ → IVW stays large, scatter_se is small.
    rvs = np.linspace(-1.0, 1.0, 16)
    errs_large = np.full(16, 1000.0)
    _, rv_std_large, _, _, scatter_se = combine_stellar_rv_chunk_estimates(
        rvs, errs_large, outlier_sigma=4.0, chunk_rv_max_abs_m_s=1.0e6
    )
    assert scatter_se < 10.0
    assert rv_std_large >= scatter_se
    assert rv_std_large > 100.0

    # Same RVs with tiny photon σ → IVW ≪ scatter; report scatter_se.
    errs_tiny = np.full(16, 0.01)
    _, rv_std_tiny, _, ivw, scatter_tiny = combine_stellar_rv_chunk_estimates(
        rvs, errs_tiny, outlier_sigma=4.0, chunk_rv_max_abs_m_s=1.0e6
    )
    assert scatter_tiny > 0.1
    assert ivw < scatter_tiny
    np.testing.assert_allclose(rv_std_tiny, scatter_tiny, rtol=1e-6, atol=1e-9)


def test_aggregate_rv_err_ignores_exploded_formal_sigma():
    """Infinite-S/N Hessian blow-up must not set the quoted bar; use MAD/√N."""
    rvs = np.linspace(-0.05, 0.05, 32)
    errs = np.full(32, 1.0e14)
    _, rv_std, _, _, scatter_se = combine_stellar_rv_chunk_estimates(
        rvs, errs, outlier_sigma=4.0, chunk_rv_max_abs_m_s=1.0e6
    )
    assert scatter_se < 1.0
    np.testing.assert_allclose(rv_std, scatter_se, rtol=1e-6, atol=1e-12)


def test_error_aware_rejection_drops_formal_outlier():
    """A chunk outside both the MAD envelope and its formal σ must be rejected."""
    rvs = np.concatenate([np.linspace(-0.2, 0.2, 12), [8.0]])
    errs = np.concatenate([np.full(12, 1.0), [0.5]])
    _, _, inlier, _, _ = combine_stellar_rv_chunk_estimates(
        rvs, errs, outlier_sigma=4.0, chunk_rv_max_abs_m_s=1.0e6
    )
    assert not bool(inlier[-1])
    assert int(np.sum(inlier)) == 12

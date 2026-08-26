"""Tests for Stage-0 rate pull / residual diagnostics."""

from __future__ import annotations

import numpy as np

from hispecdrp.simulations.plotting.stage0_rate_noise import (
    stage0_rate_pulls,
    stage0_truth_scaled_rate_err,
    summarize_stage0_rate_residuals,
)


def test_truth_scaled_err_removes_fit_dependent_pull_bias():
    rng = np.random.default_rng(0)
    n = 200_000
    truth = np.full(n, 5.0)
    # Unbiased rate scatter; Poisson-like err ∝ sqrt(|rate|)
    rate = truth + rng.normal(0.0, 1.0, size=n)
    alpha = 0.2
    vr = 0.05
    vp = alpha * np.abs(rate)
    rate_err = np.sqrt(vp + vr)

    fit_pulls = stage0_rate_pulls(
        rate, rate_err, truth, min_truth_rate=0.0, error_mode="fit"
    )
    truth_pulls = stage0_rate_pulls(
        rate,
        rate_err,
        truth,
        min_truth_rate=0.0,
        rate_var_poisson=vp,
        rate_var_rnoise=np.full(n, vr),
        error_mode="truth",
    )
    # Fit-based pulls are biased low; truth-scaled should be near 0.
    assert fit_pulls.mean() < -0.02
    assert abs(truth_pulls.mean()) < 0.01
    assert abs(truth_pulls.std(ddof=1) - 1.0) < 0.05


def test_summarize_prefers_mean_residual_z_and_truth_mode():
    rng = np.random.default_rng(1)
    ny, nx = 64, 64
    truth = np.full((ny, nx), 10.0)
    alpha, vr = 0.1, 0.05
    sig_truth = float(np.sqrt(alpha * 10.0 + vr))
    rate = truth + rng.normal(0.0, sig_truth, size=truth.shape)
    vp = alpha * np.abs(rate)
    err = np.sqrt(vp + vr)
    summary = summarize_stage0_rate_residuals(
        rate,
        err,
        truth,
        min_truth_rate=0.0,
        rate_var_poisson=vp,
        rate_var_rnoise=np.full(truth.shape, vr),
        error_mode="truth",
        print_stats=False,
    )
    assert summary["error_mode"] == "truth"
    assert abs(summary["mean_residual_z"]) < 3.0
    assert abs(summary["pull_mu_robust"]) < 0.05
    assert abs(summary["pull_sigma_robust"] - 1.0) < 0.1


def test_truth_scaled_rate_err_matches_alpha_truth_plus_rn():
    rate = np.array([2.0, 4.0, 8.0])
    truth = np.array([3.0, 3.0, 3.0])
    alpha = 0.5
    vr = 1.0
    vp = alpha * np.abs(rate)
    err = np.sqrt(vp + vr)
    out = stage0_truth_scaled_rate_err(rate, err, truth, vp, np.full(3, vr))
    expect = np.sqrt(alpha * truth + vr)
    np.testing.assert_allclose(out, expect, rtol=1e-6)


def test_full_frame_truth_pulls_include_empty_sky():
    """Empty + illuminated pixels with correctly reported var_rnoise."""
    rng = np.random.default_rng(2)
    ny, nx = 256, 256
    truth = np.zeros((ny, nx))
    truth[:, nx // 2 :] = 20.0
    alpha, vr = 0.05, 0.004  # vr matches true RN-limited variance
    rate = truth + rng.normal(0.0, np.sqrt(alpha * truth + vr))
    vp = alpha * np.abs(rate)
    err = np.sqrt(vp + vr)
    pulls = stage0_rate_pulls(
        rate,
        err,
        truth,
        min_truth_rate=0.0,
        rate_var_poisson=vp,
        rate_var_rnoise=np.full(truth.shape, vr),
        error_mode="truth",
    )
    assert pulls.size == truth.size
    assert abs(pulls.mean()) < 0.05
    assert abs(pulls.std(ddof=1) - 1.0) < 0.1

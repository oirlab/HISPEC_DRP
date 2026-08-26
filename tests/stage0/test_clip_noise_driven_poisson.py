"""Tests for post-fit Poisson variance reporting modes."""

from __future__ import annotations

import numpy as np
import pytest

from hispecdrp.primitives.ramp_fitting.ramp_fit_primitive import (
    clip_noise_driven_poisson_variance,
)
from hispecdrp.simulations.plotting.stage0_rate_noise import robust_gaussian_params


def test_detect_nsigma_restores_empty_sky_pulls():
    rng = np.random.default_rng(0)
    n = 200_000
    vr = 0.004194191045402118  # RN-limited OLS var for rn=1, n_reads=10, dt=1.7
    sig_rn = np.sqrt(vr)
    truth = np.zeros(n)
    rate = rng.normal(0.0, sig_rn, size=n)
    # Wald-style plug-in: Poisson term only for positive rate (stcal-like)
    alpha = 0.065
    vp = alpha * np.maximum(rate, 0.0)
    err_bad = np.sqrt(vp + vr)
    pulls_bad = (rate - truth) / err_bad
    mu_b, sig_b, _ = robust_gaussian_params(pulls_bad)
    assert sig_b < 0.95 or abs(mu_b) > 0.05

    vp2, err2, n_clip = clip_noise_driven_poisson_variance(
        rate, vp, np.full(n, vr), mode="detect_nsigma"
    )
    assert n_clip > 0
    assert np.all(vp2[rate <= 0] == 0.0)
    pulls = (rate - truth) / err2
    mu, sig, _ = robust_gaussian_params(pulls)
    assert abs(mu) < 0.02
    assert abs(sig - 1.0) < 0.05
    assert abs(pulls.mean()) < 0.02
    assert abs(pulls.std(ddof=1) - 1.0) < 0.05


def test_positive_only_keeps_subthreshold_poisson():
    """positive_only: clear Poisson only for rate <= 0, keep faint positive rates."""
    sig_rn = 0.065
    vr = np.full(4, sig_rn**2)
    rate = np.array([-0.1, 0.0, 0.1, 50.0])
    vp = np.array([0.01, 0.02, 0.0065, 3.0])
    vp2, err2, n_clip = clip_noise_driven_poisson_variance(
        rate, vp, vr, mode="positive_only"
    )
    assert n_clip == 2  # negative and zero
    np.testing.assert_allclose(vp2, [0.0, 0.0, 0.0065, 3.0])
    np.testing.assert_allclose(err2, np.sqrt(vp2 + vr))

    vp3, _, n_clip3 = clip_noise_driven_poisson_variance(
        rate, vp, vr, mode="detect_nsigma", n_sigma=3.0
    )
    assert n_clip3 == 3  # also clears rate=0.1 < 3σ
    np.testing.assert_allclose(vp3, [0.0, 0.0, 0.0, 3.0])


def test_smooth_detect_keeps_faint_illuminated_structure():
    """
    Isolated empty noise loses Poisson; a compact faint blob keeps it even when
    some pixels sit below a hard 3σ per-pixel gate.
    """
    rng = np.random.default_rng(1)
    ny, nx = 64, 64
    sig_rn = 0.065
    vr = np.full((ny, nx), sig_rn**2)
    alpha = 0.065
    rate = rng.normal(0.0, sig_rn, size=(ny, nx))
    # Compact illuminated patch at ~2σ mean (below old 3σ pixel gate).
    rate[28:36, 28:36] = 0.13
    vp = alpha * np.maximum(rate, 0.0)

    vp_sm, _, _ = clip_noise_driven_poisson_variance(
        rate, vp, vr, mode="smooth_detect", smooth_sigma=1.0, smooth_nsigma=1.0
    )
    vp_dn, _, _ = clip_noise_driven_poisson_variance(
        rate, vp, vr, mode="detect_nsigma", n_sigma=3.0
    )

    empty = np.ones((ny, nx), dtype=bool)
    empty[26:38, 26:38] = False
    # Empty: smooth_detect should clear almost all Poisson (like detect_nsigma).
    assert np.mean(vp_sm[empty] > 0) < 0.05
    # Blob interior: smooth_detect keeps Poisson; hard 3σ gate drops it.
    blob = np.zeros((ny, nx), dtype=bool)
    blob[30:34, 30:34] = True
    assert np.mean(vp_sm[blob] > 0) > 0.9
    assert np.mean(vp_dn[blob] > 0) < 0.1


def test_stcal_mode_leaves_poisson_unchanged():
    rate = np.array([-1.0, 0.05, 10.0])
    vr = np.array([0.004, 0.004, 0.004])
    vp = np.array([0.1, 0.003, 0.65])
    vp2, err2, n_clip = clip_noise_driven_poisson_variance(
        rate, vp, vr, mode="stcal"
    )
    assert n_clip == 0
    np.testing.assert_allclose(vp2, vp)
    np.testing.assert_allclose(err2, np.sqrt(vp + vr))


def test_clip_preserves_bright_poisson_term():
    rate = np.array([50.0, 100.0])
    vr = np.array([0.004, 0.004])
    vp = np.array([3.0, 6.0])
    vp2, err2, n_clip = clip_noise_driven_poisson_variance(
        rate, vp, vr, mode="positive_only"
    )
    assert n_clip == 0
    np.testing.assert_allclose(vp2, vp)
    np.testing.assert_allclose(err2, np.sqrt(vp + vr))


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="poisson variance mode"):
        clip_noise_driven_poisson_variance(
            np.array([1.0]), np.array([0.1]), np.array([0.01]), mode="nope"
        )


def test_clip_preserves_rate_err_when_variance_planes_nan():
    """Fitters without a Poisson/RN split must not lose their total rate_err."""
    rate = np.array([0.0, 1.0, 10.0])
    vp = np.array([np.nan, np.nan, np.nan])
    vr = np.array([np.nan, np.nan, np.nan])
    err = np.array([0.2, 0.3, 0.4])
    vp2, err2, n_clip = clip_noise_driven_poisson_variance(
        rate, vp, vr, mode="smooth_detect", rate_err=err
    )
    assert n_clip == 0
    np.testing.assert_allclose(err2, err)
    assert np.all(np.isnan(vp2))

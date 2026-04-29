"""Helpers for combining per-chunk RV estimates into aggregates."""

from __future__ import annotations

import math

import numpy as np
from astropy.stats import mad_std

__all__ = [
    "combine_rvs_median_mad",
    "weighted_mean_rv",
    "combine_rv_estimates_median_mad_weighted",
]


def combine_rvs_median_mad(
    rvs: np.ndarray,
    rv_errs: np.ndarray,
    outlier_sigma: float = 4.0,
) -> tuple[np.ndarray, float, float]:
    """
    Robust outlier mask on per-chunk RVs (median ± ``outlier_sigma`` × MAD).

    Only chunks with finite ``rvs`` and positive ``rv_errs`` are considered for the
    median/MAD; invalid chunks are marked not inlier.
    """
    valid = np.isfinite(rvs) & np.isfinite(rv_errs) & (rv_errs > 0)
    if not np.any(valid):
        return np.zeros_like(rvs, dtype=bool), np.nan, np.nan
    rv_valid = rvs[valid]
    median_rv = np.nanmedian(rv_valid)
    mad_rv = mad_std(rv_valid, ignore_nan=True)
    if mad_rv <= 0:
        mad_rv = np.nanstd(rv_valid)
    if not np.isfinite(mad_rv) or mad_rv <= 0:
        mad_rv = 1.0
    threshold = outlier_sigma * mad_rv
    inlier = valid & (np.abs(rvs - median_rv) <= threshold)
    return inlier, float(median_rv), float(mad_rv)


def weighted_mean_rv(
    rvs: np.ndarray,
    rv_errs: np.ndarray,
    inlier: np.ndarray,
) -> tuple[float, float]:
    """Inverse-variance weighted mean RV and error on the mean."""
    use = inlier & np.isfinite(rvs) & np.isfinite(rv_errs) & (rv_errs > 0)
    if not np.any(use):
        return np.nan, np.nan
    w = 1.0 / (np.asarray(rv_errs, dtype=np.float64) ** 2)
    w = np.where(use, w, 0.0)
    if np.sum(w) <= 0:
        return np.nan, np.nan
    rv_mean = np.average(rvs, weights=w)
    var_mean = 1.0 / np.sum(w)
    rv_std = np.sqrt(var_mean)
    return float(rv_mean), float(rv_std)


def combine_rv_estimates_median_mad_weighted(
    rvs: np.ndarray,
    rv_errs: np.ndarray,
    outlier_sigma: float = 4.0,
) -> tuple[float, float, np.ndarray, float, float]:
    """
    Robust rejection, inverse-variance mean, and conservative uncertainty.

    The reported ``rv_std`` is ``max(rv_std_internal, scatter_se)`` where
    ``rv_std_internal`` is the Gaussian standard error of the weighted mean
    (``1/sqrt(sum 1/σ²)``) and ``scatter_se`` is ``MAD/√N`` of inlier RVs.
    """
    rvs = np.asarray(rvs, dtype=np.float64).ravel()
    rv_errs = np.asarray(rv_errs, dtype=np.float64).ravel()
    if rvs.size != rv_errs.size:
        raise ValueError("combine_rv_estimates_median_mad_weighted: rvs and rv_errs length mismatch")
    inlier, _, _ = combine_rvs_median_mad(rvs, rv_errs, outlier_sigma)
    rv_mean, rv_std_internal = weighted_mean_rv(rvs, rv_errs, inlier)
    n_in = int(np.sum(inlier))
    scatter_se = float("nan")
    if n_in > 1:
        rv_in = rvs[inlier]
        mad_disp = float(mad_std(rv_in, ignore_nan=True))
        if math.isfinite(mad_disp) and mad_disp > 0:
            scatter_se = mad_disp / math.sqrt(float(n_in))
    if math.isfinite(rv_std_internal) and math.isfinite(scatter_se):
        rv_std = max(rv_std_internal, scatter_se)
    elif math.isfinite(rv_std_internal):
        rv_std = rv_std_internal
    else:
        rv_std = scatter_se
    return float(rv_mean), float(rv_std), inlier, float(rv_std_internal), float(scatter_se)


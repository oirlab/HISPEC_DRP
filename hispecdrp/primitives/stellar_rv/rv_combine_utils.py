"""Helpers for combining per-chunk RV estimates into aggregates."""

from __future__ import annotations

import math

import numpy as np
from astropy.stats import mad_std

# Reject coarse-grid CCF alias peaks before aggregation (typical stellar |v| < few × 10 km/s).
DEFAULT_CHUNK_RV_MAX_ABS_M_S = 120_000.0
# When ``ccf_input_velocity`` is set, keep chunks within this half-width of the prior (m/s).
DEFAULT_CHUNK_RV_PRIOR_HALF_WIDTH_M_S = 10_000.0

__all__ = [
    "DEFAULT_CHUNK_RV_MAX_ABS_M_S",
    "DEFAULT_CHUNK_RV_PRIOR_HALF_WIDTH_M_S",
    "combine_rvs_median_mad",
    "weighted_mean_rv",
    "combine_rv_estimates_median_mad_weighted",
    "combine_stellar_rv_chunk_estimates",
]


def combine_rvs_median_mad(
    rvs: np.ndarray,
    rv_errs: np.ndarray,
    outlier_sigma: float = 4.0,
    *,
    center_m_s: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """
    Robust outlier mask on per-chunk RVs (center ± ``outlier_sigma`` × MAD).

    Only chunks with finite ``rvs`` and positive ``rv_errs`` are considered for the
    center/MAD; invalid chunks are marked not inlier. When ``center_m_s`` is finite it
    replaces the sample median as the rejection anchor (prior-aware aggregation).
    """
    valid = np.isfinite(rvs) & np.isfinite(rv_errs) & (rv_errs > 0)
    if not np.any(valid):
        return np.zeros_like(rvs, dtype=bool), np.nan, np.nan
    rv_valid = rvs[valid]
    if center_m_s is not None and math.isfinite(float(center_m_s)):
        median_rv = float(center_m_s)
    else:
        median_rv = float(np.nanmedian(rv_valid))
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
    *,
    center_m_s: float | None = None,
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
    inlier, _, _ = combine_rvs_median_mad(
        rvs, rv_errs, outlier_sigma, center_m_s=center_m_s
    )
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


def combine_stellar_rv_chunk_estimates(
    rvs: np.ndarray,
    rv_errs: np.ndarray,
    outlier_sigma: float = 4.0,
    *,
    chunk_rv_max_abs_m_s: float = DEFAULT_CHUNK_RV_MAX_ABS_M_S,
    chunk_rv_prior_m_s: float | None = None,
    chunk_rv_prior_half_width_m_s: float | None = None,
    median_fallback_sigma: float = 3.0,
) -> tuple[float, float, np.ndarray, float, float]:
    """
    Aggregate per-chunk RVs for :class:`~hispecdrp.primitives.stellar_rv.stellar_rv_primitive.StellarRVPrimitive`.

    Applies a **sanity gate**, robust MAD rejection, and inverse-variance weighting with a
    scatter-based uncertainty floor (:func:`combine_rv_estimates_median_mad_weighted`).

    When ``chunk_rv_prior_m_s`` is finite, the gate keeps ``|RV − prior| <= half_width`` (default
    ``DEFAULT_CHUNK_RV_PRIOR_HALF_WIDTH_M_S``) and MAD rejection is anchored to the prior. Otherwise
    the gate uses ``|RV| <= chunk_rv_max_abs_m_s`` and the sample median anchors MAD rejection.

    When the weighted mean disagrees with the inlier median by more than
    ``median_fallback_sigma × MAD``, the median is used instead (guards against alias peaks with
    underestimated formal errors on coarse CCF grids).
    """
    rvs = np.asarray(rvs, dtype=np.float64).ravel()
    rv_errs = np.asarray(rv_errs, dtype=np.float64).ravel()
    if rvs.size != rv_errs.size:
        raise ValueError("combine_stellar_rv_chunk_estimates: rvs and rv_errs length mismatch")

    finite = np.isfinite(rvs) & np.isfinite(rv_errs) & (rv_errs > 0)
    use_prior_gate = chunk_rv_prior_m_s is not None and math.isfinite(float(chunk_rv_prior_m_s))
    mad_center: float | None = None
    if use_prior_gate:
        prior = float(chunk_rv_prior_m_s)
        half_w = chunk_rv_prior_half_width_m_s
        if half_w is None or not math.isfinite(float(half_w)) or float(half_w) <= 0:
            half_w = DEFAULT_CHUNK_RV_PRIOR_HALF_WIDTH_M_S
        else:
            half_w = float(half_w)
        sane = finite & (np.abs(rvs - prior) <= half_w)
        mad_center = prior
    else:
        sane = finite & (np.abs(rvs) <= float(chunk_rv_max_abs_m_s))
    if not np.any(sane):
        return np.nan, np.nan, np.zeros_like(rvs, dtype=bool), np.nan, np.nan

    rv_errs_gated = np.asarray(rv_errs, dtype=np.float64).copy()
    rv_errs_gated[~sane] = np.nan

    rv_mean, rv_std, inlier, std_iv, scatter_se = combine_rv_estimates_median_mad_weighted(
        rvs,
        rv_errs_gated,
        outlier_sigma,
        center_m_s=mad_center,
    )
    inlier = inlier & sane

    use = inlier & sane
    if not np.any(use):
        med = float(np.median(rvs[sane]))
        mad = float(mad_std(rvs[sane], ignore_nan=True))
        n_s = int(np.sum(sane))
        rv_std_med = mad / math.sqrt(float(n_s)) if n_s > 1 and mad > 0 else max(mad, 1.0)
        return med, float(rv_std_med), sane, np.nan, float(rv_std_med)

    med = float(np.median(rvs[use]))
    mad = float(mad_std(rvs[use], ignore_nan=True))
    if not math.isfinite(mad) or mad <= 0:
        mad = float(np.nanstd(rvs[use])) if np.sum(use) > 1 else 1.0
    if (
        math.isfinite(rv_mean)
        and math.isfinite(med)
        and math.isfinite(mad)
        and mad > 0
        and abs(rv_mean - med) > float(median_fallback_sigma) * mad
    ):
        rv_mean = med
        n_u = int(np.sum(use))
        rv_std = max(
            rv_std if math.isfinite(rv_std) else 0.0,
            mad / math.sqrt(float(n_u)) if n_u > 1 else mad,
        )
    elif use_prior_gate and math.isfinite(med):
        # Prior-gated pools are small and formal chunk errors often hit the bisector
        # floor; the weighted mean is biased by ~km/s outliers still inside the gate.
        rv_mean = med
        n_u = int(np.sum(use))
        rv_std = max(
            rv_std if math.isfinite(rv_std) else 0.0,
            mad / math.sqrt(float(n_u)) if n_u > 1 and math.isfinite(mad) and mad > 0 else (
                mad if math.isfinite(mad) and mad > 0 else 1.0
            ),
        )

    return float(rv_mean), float(rv_std), inlier, float(std_iv), float(scatter_se)


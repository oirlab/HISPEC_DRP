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
    Error-aware robust outlier mask on per-chunk RVs.

    Chunks with finite ``rvs`` and positive ``rv_errs`` set the rejection center
    (sample median, or ``center_m_s`` when finite) and the ensemble MAD. A chunk
    is kept when

    ``|RV − center| <= outlier_sigma * max(MAD, σ_i)``

    so points that agree within the MAD are never rejected, while points that are
    both outside the MAD envelope and inconsistent with their own formal ``σ_i``
    are rejected. Invalid chunks are marked not inlier.
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
    # Per-chunk threshold: MAD floor keeps mutually consistent points; larger
    # formal σ widens the gate so noisy chunks are not rejected on scatter alone.
    thresh = float(outlier_sigma) * np.maximum(float(mad_rv), np.asarray(rv_errs, dtype=np.float64))
    inlier = valid & (np.abs(rvs - median_rv) <= thresh)
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


# Formal per-chunk CCF σ is typically ~3–4× too small vs order-to-order scatter
# on e2e simulations (Fisher / Gaussian cov ignores mask systematics). Inflate
# before IVW *weighting* so the weighted mean is not dominated by underdispersed
# chunks. The reported aggregate σ is max(uninflated IVW SE, MAD/√N), which
# converges to MAD/√N as per-chunk σ → 0.
DEFAULT_CHUNK_RV_ERR_INFLATION = 3.5


def combine_rv_estimates_median_mad_weighted(
    rvs: np.ndarray,
    rv_errs: np.ndarray,
    outlier_sigma: float = 4.0,
    *,
    center_m_s: float | None = None,
    rv_err_inflation: float = DEFAULT_CHUNK_RV_ERR_INFLATION,
) -> tuple[float, float, np.ndarray, float, float]:
    """
    Error-aware robust rejection, inverse-variance mean, and IVW uncertainty.

    Outlier rejection uses the **uninflated** per-chunk ``rv_errs`` with
    :func:`combine_rvs_median_mad` (``|Δ| <= outlier_sigma * max(MAD, σ_i)``).
    Per-chunk ``rv_errs`` are scaled by ``rv_err_inflation`` only for IVW
    *weights* (default 3.5) so the mean is not dominated by underdispersed
    formal σ. The reported ``rv_std`` is
    :math:`\\max(\\sigma_{\\mathrm{IVW}},\\,\\mathrm{MAD}/\\sqrt{N})` of the
    inliers: photon-weighted IVW when it is honest, and the empirical line
    scatter when formal σ are optimistic or systematics dominate.
    As per-chunk σ → 0 (infinite S/N), this converges to ``MAD/√N``.
    If the median formal σ is >10⁶× the inlier MAD (broken Hessian), IVW is
    discarded and ``MAD/√N`` is reported.
    Inflated IVW SE and empirical ``MAD/√N`` are returned as ``rv_std_internal``
    and ``scatter_se`` for diagnostics.
    """
    rvs = np.asarray(rvs, dtype=np.float64).ravel()
    rv_errs = np.asarray(rv_errs, dtype=np.float64).ravel()
    if rvs.size != rv_errs.size:
        raise ValueError("combine_rv_estimates_median_mad_weighted: rvs and rv_errs length mismatch")
    infl = float(rv_err_inflation) if rv_err_inflation is not None else 1.0
    if not math.isfinite(infl) or infl <= 0:
        infl = 1.0
    rv_errs_w = rv_errs * infl
    # Rejection uses formal (uninflated) σ so large-error chunks can survive a
    # MAD cut when still consistent with their error bars.
    inlier, _, _ = combine_rvs_median_mad(
        rvs, rv_errs, outlier_sigma, center_m_s=center_m_s
    )
    rv_mean, rv_std_internal = weighted_mean_rv(rvs, rv_errs_w, inlier)
    _, rv_std_uninfl = weighted_mean_rv(rvs, rv_errs, inlier)
    n_in = int(np.sum(inlier))
    scatter_se = float("nan")
    if n_in > 1:
        rv_in = rvs[inlier]
        mad_disp = float(mad_std(rv_in, ignore_nan=True))
        if math.isfinite(mad_disp) and mad_disp > 0:
            scatter_se = mad_disp / math.sqrt(float(n_in))
    if math.isfinite(rv_std_uninfl) and rv_std_uninfl > 0:
        rv_std = float(rv_std_uninfl)
    elif math.isfinite(rv_std_internal) and rv_std_internal > 0:
        rv_std = float(rv_std_internal)
    else:
        rv_std = scatter_se
    if math.isfinite(scatter_se) and scatter_se > 0.0:
        if math.isfinite(rv_std) and rv_std > 0.0:
            rv_std = max(float(rv_std), float(scatter_se))
        else:
            rv_std = float(scatter_se)
        # Exploded formal σ (ill-conditioned Hessian at infinite S/N) make IVW
        # >> MAD. That is not a photon error; quote the empirical floor.
        if n_in >= 3:
            med_err = float(np.median(rv_errs[inlier]))
            mad_disp = scatter_se * math.sqrt(float(n_in))
            if (
                math.isfinite(med_err)
                and math.isfinite(mad_disp)
                and mad_disp > 0.0
                and med_err > 1.0e6 * mad_disp
            ):
                rv_std = float(scatter_se)
    # Prefer the inlier median when formal-IVW and median disagree — common when
    # a few chunks still have underestimated σ after inflation. Keep IVW σ.
    if n_in >= 3 and math.isfinite(rv_mean):
        med = float(np.median(rvs[inlier]))
        mad_disp = float(mad_std(rvs[inlier], ignore_nan=True))
        if (
            math.isfinite(med)
            and math.isfinite(mad_disp)
            and mad_disp > 0
            and abs(rv_mean - med) > 1.0 * mad_disp
        ):
            rv_mean = med
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

    Applies a **sanity gate**, error-aware MAD rejection
    (``|Δ| <= outlier_sigma * max(MAD, σ_i)``), and inverse-variance weighting
    (:func:`combine_rv_estimates_median_mad_weighted`). The reported aggregate
    ``rv_std`` is :math:`\\max(\\sigma_{\\mathrm{IVW}},\\,\\mathrm{MAD}/\\sqrt{N})`
    of the inliers.

    When ``chunk_rv_prior_m_s`` is finite, the gate keeps ``|RV − prior| <= half_width`` (default
    ``DEFAULT_CHUNK_RV_PRIOR_HALF_WIDTH_M_S``) and MAD rejection is anchored to the prior. Otherwise
    the gate uses ``|RV| <= chunk_rv_max_abs_m_s`` and the sample median anchors MAD rejection.

    When the weighted mean disagrees with the inlier median by more than
    ``median_fallback_sigma × MAD``, the median is used for the aggregate RV (guards against
    alias peaks with underestimated formal errors on coarse CCF grids); the reported
    uncertainty remains the IVW SE.
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
        _, rv_std_sane = weighted_mean_rv(rvs, rv_errs, sane)
        if not (math.isfinite(rv_std_sane) and rv_std_sane > 0):
            mad = float(mad_std(rvs[sane], ignore_nan=True))
            n_s = int(np.sum(sane))
            rv_std_sane = mad / math.sqrt(float(n_s)) if n_s > 1 and mad > 0 else max(mad, 1.0)
        return med, float(rv_std_sane), sane, np.nan, float(rv_std_sane)

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
    elif use_prior_gate and math.isfinite(med):
        # Prior-gated pools are small and formal chunk errors often hit the bisector
        # floor; the weighted mean is biased by ~km/s outliers still inside the gate.
        rv_mean = med

    return float(rv_mean), float(rv_std), inlier, float(std_iv), float(scatter_se)


"""Stage-0 rate pull histogram (noise self-consistency notebooks)."""

from __future__ import annotations

import io
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from ...datamodels.dqflags import DQ_FLAGS

# Default science / self-check exclusions for Stage-0 rate pulls.
# REFERENCE_PIXEL alone was previously included (bit 31) and biased the histogram.
DEFAULT_STAGE0_PULL_DQ_EXCLUDE = (
    int(DQ_FLAGS["DO_NOT_USE"])
    | int(DQ_FLAGS["SATURATED"])
    | int(DQ_FLAGS["JUMP_DET"])
    | int(DQ_FLAGS["NON_LINEAR"])
    | int(DQ_FLAGS["NON_SCIENCE"])
    | int(DQ_FLAGS["DEAD"])
    | int(DQ_FLAGS["HOT"])
    | int(DQ_FLAGS["UNRELIABLE_SLOPE"])
    | int(DQ_FLAGS["OTHER_BAD_PIXEL"])
    | int(DQ_FLAGS["REFERENCE_PIXEL"])
)

# Fit-based empty-sky pulls are near N(0,1) under default ``smooth_detect``
# Poisson reporting (gate on a smoothed rate). Prefer ``error_mode='truth'``
# for variance-model checks of illuminated pixels; ``positive_only`` inflates
# empty-sky σ, and legacy ``detect_nsigma`` underestimates faint-source σ.
DEFAULT_STAGE0_PULL_MIN_TRUTH = 0.0

# Minimum |fitted rate| used when estimating α = var_poisson / |rate| for
# truth-scaled pull denominators.
_TRUTH_SCALE_RATE_FLOOR = 1.0


def stage0_truth_scaled_rate_err(
    rate: np.ndarray,
    rate_err: np.ndarray,
    truth_rate: np.ndarray,
    rate_var_poisson: np.ndarray,
    rate_var_rnoise: np.ndarray,
    *,
    rate_floor: float = _TRUTH_SCALE_RATE_FLOOR,
) -> np.ndarray:
    """
    Rebuild ``σ`` with the Poisson term evaluated at ``truth_rate``, not ``rate``.

    Stage-0 stores ``rate_err² ≈ var_poisson(|rate|) + var_rnoise``. Dividing
    residuals by that fit-based σ makes the pull mean negative even when
    ``E[rate] = truth``: low excursions shrink the Poisson term (larger |pull|),
    high excursions inflate it (smaller |pull|). For validating the estimator
    against an injected map, rescale:

    ``α ≈ median(var_poisson / |rate|)`` on bright-enough pixels, then
    ``σ_truth = sqrt(α * truth + var_rnoise)``.

    This does **not** rescale ``var_rnoise``; if empty-sky pulls are wider than
    N(0,1), fix the ramp-fit RN convention (see ``fit_ramp_jwst``) rather than
    adjusting σ in the diagnostic.
    """
    rate = np.asarray(rate, dtype=float)
    rate_err = np.asarray(rate_err, dtype=float)
    truth_rate = np.asarray(truth_rate, dtype=float)
    vp = np.asarray(rate_var_poisson, dtype=float)
    vr = np.asarray(rate_var_rnoise, dtype=float)
    if not (rate.shape == rate_err.shape == truth_rate.shape == vp.shape == vr.shape):
        raise ValueError(
            "shape mismatch among rate / rate_err / truth / var_poisson / var_rnoise"
        )

    floor = float(rate_floor)
    usable = (
        np.isfinite(rate)
        & np.isfinite(vp)
        & (np.abs(rate) >= floor)
        & (vp >= 0.0)
    )
    if not np.any(usable):
        return np.asarray(rate_err, dtype=float)

    alpha = float(np.median(vp[usable] / np.abs(rate[usable])))
    if not np.isfinite(alpha) or alpha < 0.0:
        return np.asarray(rate_err, dtype=float)

    var = np.maximum(alpha * np.maximum(truth_rate, 0.0), 0.0) + np.maximum(vr, 0.0)
    out = np.sqrt(var)
    # Fall back to reported err where the rebuild is non-finite / non-positive.
    bad = ~(np.isfinite(out) & (out > 0.0))
    out = out.copy()
    out[bad] = rate_err[bad]
    return out


def stage0_rate_pulls(
    rate: np.ndarray,
    rate_err: np.ndarray,
    truth_rate: np.ndarray,
    dq: np.ndarray | None = None,
    *,
    dq_exclude_flags: int | None = None,
    min_truth_rate: float | None = DEFAULT_STAGE0_PULL_MIN_TRUTH,
    rate_var_poisson: np.ndarray | None = None,
    rate_var_rnoise: np.ndarray | None = None,
    error_mode: str = "fit",
) -> np.ndarray:
    """
    Return 1-D pull samples ``(rate - truth) / σ`` on good pixels.

    Pixels are kept when ``rate``, ``σ``, and ``truth_rate`` are finite,
    ``σ > 0``, ``truth_rate`` meets ``min_truth_rate`` (if set), and (if
    ``dq`` is given) none of ``dq_exclude_flags`` are set.

    Parameters
    ----------
    dq_exclude_flags
        Bitmask of DQ flags to reject. ``None`` uses
        :data:`DEFAULT_STAGE0_PULL_DQ_EXCLUDE` (``DO_NOT_USE``, ``SATURATED``,
        ``JUMP_DET``, ``NON_LINEAR``, ``NON_SCIENCE``, ``DEAD``, ``HOT``,
        ``UNRELIABLE_SLOPE``, ``OTHER_BAD_PIXEL``, ``REFERENCE_PIXEL``).
        Pass ``0`` to disable DQ filtering; pass ``DQ_FLAGS['DO_NOT_USE']`` for
        the historical DO_NOT_USE-only cut.
    min_truth_rate
        Require ``truth_rate >= min_truth_rate`` (e-/s). Default
        :data:`DEFAULT_STAGE0_PULL_MIN_TRUTH` (0). Pass a positive cut
        (e.g. 0.5) when using ``error_mode='fit'`` on empty sky.
    rate_var_poisson, rate_var_rnoise
        Optional Stage-0 variance planes. Required for ``error_mode='truth'``.
    error_mode
        ``'fit'`` (default): divide by reported ``rate_err``.
        ``'truth'``: divide by :func:`stage0_truth_scaled_rate_err` (optional
        diagnostic only; requires variance planes).
    """
    rate = np.asarray(rate, dtype=float)
    rate_err = np.asarray(rate_err, dtype=float)
    truth_rate = np.asarray(truth_rate, dtype=float)
    if rate.shape != rate_err.shape or rate.shape != truth_rate.shape:
        raise ValueError(
            f"shape mismatch: rate={rate.shape} rate_err={rate_err.shape} "
            f"truth_rate={truth_rate.shape}"
        )

    mode = str(error_mode).strip().lower()
    if mode == "truth":
        if rate_var_poisson is None or rate_var_rnoise is None:
            raise ValueError(
                "error_mode='truth' requires rate_var_poisson and rate_var_rnoise"
            )
        sigma = stage0_truth_scaled_rate_err(
            rate, rate_err, truth_rate, rate_var_poisson, rate_var_rnoise
        )
    elif mode == "fit":
        sigma = rate_err
    else:
        raise ValueError(f"error_mode must be 'fit' or 'truth', got {error_mode!r}")

    good = stage0_rate_good_mask(
        rate,
        sigma,
        truth_rate,
        dq=dq,
        dq_exclude_flags=dq_exclude_flags,
        min_truth_rate=min_truth_rate,
    )
    return ((rate - truth_rate) / sigma)[good]


def robust_gaussian_params(
    samples: np.ndarray,
    *,
    n_sigma: float = 3.0,
    max_iter: int = 12,
    min_keep: int = 50,
) -> tuple[float, float, int]:
    """
    Iterative median / MAD σ-clip, then mean and std of the surviving core.

    More reliable than ``scipy.stats.norm.fit`` on heavy-tailed Stage-0 pulls:
    outliers no longer inflate ``σ`` or shift ``μ`` into the left tail.
    """
    x = np.asarray(samples, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError(f"need ≥2 finite samples; got {x.size}")

    work = x
    for _ in range(int(max_iter)):
        med = float(np.median(work))
        mad = float(np.median(np.abs(work - med)))
        scale = 1.4826 * mad if mad > 0.0 else float(np.std(work))
        if not np.isfinite(scale) or scale <= 0.0:
            break
        keep = np.abs(work - med) <= float(n_sigma) * scale
        n_keep = int(np.count_nonzero(keep))
        if n_keep < int(min_keep) or n_keep == work.size:
            if n_keep >= int(min_keep):
                work = work[keep]
            break
        work = work[keep]

    mu = float(np.mean(work))
    sigma = float(np.std(work, ddof=1)) if work.size > 1 else 0.0
    return mu, sigma, int(work.size)


def stage0_rate_good_mask(
    rate: np.ndarray,
    rate_err: np.ndarray,
    truth_rate: np.ndarray,
    dq: np.ndarray | None = None,
    *,
    dq_exclude_flags: int | None = None,
    min_truth_rate: float | None = DEFAULT_STAGE0_PULL_MIN_TRUTH,
) -> np.ndarray:
    """Boolean mask of pixels used by :func:`stage0_rate_pulls`."""
    rate = np.asarray(rate, dtype=float)
    rate_err = np.asarray(rate_err, dtype=float)
    truth_rate = np.asarray(truth_rate, dtype=float)
    good = (
        np.isfinite(rate)
        & np.isfinite(rate_err)
        & np.isfinite(truth_rate)
        & (rate_err > 0.0)
    )
    if min_truth_rate is not None and float(min_truth_rate) > 0.0:
        good &= truth_rate >= float(min_truth_rate)
    if dq is not None:
        dq_arr = np.asarray(dq)
        if dq_arr.shape != rate.shape:
            raise ValueError(f"dq shape {dq_arr.shape} != rate shape {rate.shape}")
        exclude = (
            DEFAULT_STAGE0_PULL_DQ_EXCLUDE
            if dq_exclude_flags is None
            else int(dq_exclude_flags)
        )
        if exclude != 0:
            good &= ~(dq_arr.astype(np.uint32) & np.uint32(exclude)).astype(bool)
    return good


def summarize_stage0_rate_residuals(
    rate: np.ndarray,
    rate_err: np.ndarray,
    truth_rate: np.ndarray,
    dq: np.ndarray | None = None,
    *,
    dq_exclude_flags: int | None = None,
    min_truth_rate: float | None = DEFAULT_STAGE0_PULL_MIN_TRUTH,
    rate_var_poisson: np.ndarray | None = None,
    rate_var_rnoise: np.ndarray | None = None,
    error_mode: str = "fit",
    print_stats: bool = True,
) -> dict[str, Any]:
    """
    Absolute residual + pull judgement for Stage-0 rate self-checks.

    Default ``error_mode='fit'`` uses the pipeline's reported ``rate_err`` so
    mismatches in rate or uncertainty stay visible. Optional
    ``error_mode='truth'`` rebuilds the Poisson term at the injected rate
    (requires variance planes) for a secondary diagnostic only.
    """
    rate = np.asarray(rate, dtype=float)
    rate_err = np.asarray(rate_err, dtype=float)
    truth_rate = np.asarray(truth_rate, dtype=float)

    mode = str(error_mode).strip().lower()
    if mode == "truth":
        if rate_var_poisson is None or rate_var_rnoise is None:
            raise ValueError(
                "error_mode='truth' requires rate_var_poisson and rate_var_rnoise "
                "(do not omit them — silent fit fallback was removed because it "
                "reproduces a spurious pull mean ≈ -0.05)."
            )
        sigma = stage0_truth_scaled_rate_err(
            rate, rate_err, truth_rate, rate_var_poisson, rate_var_rnoise
        )
    elif mode == "fit":
        sigma = rate_err
    else:
        raise ValueError(f"error_mode must be 'fit' or 'truth', got {error_mode!r}")

    good = stage0_rate_good_mask(
        rate,
        sigma,
        truth_rate,
        dq=dq,
        dq_exclude_flags=dq_exclude_flags,
        min_truth_rate=min_truth_rate,
    )
    n = int(np.count_nonzero(good))
    if n < 10:
        raise ValueError(f"too few good pixels for residual summary: n={n}")

    residual = (rate - truth_rate)[good]
    pulls = residual / sigma[good]
    mu, sigma_pull, n_core = robust_gaussian_params(pulls)
    med_res = float(np.median(residual))
    mean_res = float(np.mean(residual))
    mad_res = float(np.median(np.abs(residual - med_res)))
    se_mean_res = float(np.std(residual, ddof=1) / np.sqrt(float(n)))
    med_err = float(np.median(sigma[good]))
    mean_pull = float(np.mean(pulls))
    sem_iid = 1.0 / np.sqrt(float(n))
    summary: dict[str, Any] = {
        "n_pix": n,
        "n_core": int(n_core),
        "error_mode": mode,
        "median_residual": med_res,
        "mean_residual": mean_res,
        "mean_residual_se": se_mean_res,
        "mean_residual_z": float(mean_res / se_mean_res) if se_mean_res > 0 else np.nan,
        "mad_residual": mad_res,
        "median_rate_err": med_err,
        "pull_mean": mean_pull,
        "pull_mu_robust": float(mu),
        "pull_sigma_robust": float(sigma_pull),
        "pull_sem_iid": float(sem_iid),
        "pull_mean_over_sem": float(mean_pull / sem_iid) if sem_iid > 0 else np.nan,
        "min_truth_rate": None if min_truth_rate is None else float(min_truth_rate),
    }
    if print_stats:
        print(
            f"judgement mask: n_pix={n}  min_truth={summary['min_truth_rate']}  "
            f"error_mode={mode}  1/sqrt(N)={sem_iid:.5f}"
        )
        print(
            f"absolute residual [e-/s]: mean={mean_res:+.5f}±{se_mean_res:.5f} "
            f"(z={summary['mean_residual_z']:+.2f})  median={med_res:+.5f}  "
            f"MAD={mad_res:.5f}  median(σ)={med_err:.5f}"
        )
        print(
            f"pull: mean={mean_pull:+.5f}  robust core mu={mu:+.5f} "
            f"sigma={sigma_pull:.5f}  mean/(1/sqrtN)={summary['pull_mean_over_sem']:.1f}"
        )
        if abs(float(summary["mean_residual_z"])) > 3.0:
            print(
                "note: |mean residual| exceeds 3σ of std/sqrt(N) → genuine "
                "additive estimator bias (not a median-skew artifact)."
            )
        elif abs(med_res) > 0.05 * max(med_err, 1e-6):
            print(
                "note: |median residual| can look large while mean≈0 under "
                "Poisson-skewed OLS; prefer mean±SE for bias validation."
            )
        if mode == "fit" and abs(mean_pull) > 5.0 * sem_iid:
            print(
                "note: pull mean differs from 0 by ≫1/sqrt(N). Check mean "
                "residual (rate bias) vs rate_err scale; a negative pull mean "
                "with mean residual≈0 can also come from rate_err using "
                "var_poisson(|rate_fit|)."
            )
        if abs(sigma_pull - 1.0) > 0.05:
            print(
                "note: robust pull sigma differs from 1 by >5% → σ scale "
                "mismatch (under/over-estimated variance) and/or non-Gaussian tails."
            )
    return summary


def plot_stage0_rate_pull_histogram(
    rate: np.ndarray,
    rate_err: np.ndarray,
    truth_rate: np.ndarray,
    dq: np.ndarray | None = None,
    *,
    bins: int = 80,
    x_lim: float = 5.0,
    title: str | None = None,
    display: bool = True,
    print_stats: bool = True,
    savefig: str | os.PathLike[str] | None = None,
    dq_exclude_flags: int | None = None,
    min_truth_rate: float | None = DEFAULT_STAGE0_PULL_MIN_TRUTH,
    robust_clip_sigma: float = 3.0,
    rate_var_poisson: np.ndarray | None = None,
    rate_var_rnoise: np.ndarray | None = None,
    error_mode: str = "fit",
) -> dict[str, Any]:
    """
    Histogram of Stage-0 rate pulls with an overlaid Gaussian fit.

    Default ``error_mode='fit'`` divides by the pipeline ``rate_err``.
    ``error_mode='truth'`` is optional (requires variance planes).

    Parameters
    ----------
    savefig
        Optional path to write a PNG (dpi=150). Independent of ``display``.
    """
    mode = str(error_mode).strip().lower()
    if mode == "truth" and (rate_var_poisson is None or rate_var_rnoise is None):
        raise ValueError(
            "error_mode='truth' requires rate_var_poisson and rate_var_rnoise"
        )

    pulls = stage0_rate_pulls(
        rate,
        rate_err,
        truth_rate,
        dq=dq,
        dq_exclude_flags=dq_exclude_flags,
        min_truth_rate=min_truth_rate,
        rate_var_poisson=rate_var_poisson,
        rate_var_rnoise=rate_var_rnoise,
        error_mode=mode,
    )
    if pulls.size < 10:
        raise ValueError(f"too few good pull pixels: n={pulls.size}")

    mu, sigma, n_core = robust_gaussian_params(
        pulls, n_sigma=float(robust_clip_sigma)
    )
    frac_1sig = float(np.mean(np.abs(pulls) < 1.0))
    mean_all = float(np.mean(pulls))
    std_all = float(np.std(pulls))
    med_all = float(np.median(pulls))
    mad_all = float(np.median(np.abs(pulls - med_all)))
    sigma_mad = 1.4826 * mad_all

    if title is None:
        title = (
            "Stage 0 rate (truth-scaled σ)"
            if mode == "truth"
            else "Stage 0 rate"
        )
    xlabel = (
        r"$({\rm rate} - {\rm truth}) / \sigma_{\rm truth}$"
        if mode == "truth"
        else r"$({\rm rate} - {\rm truth}) / {\rm rate\_err}$"
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(
        pulls,
        bins=bins,
        range=(-float(x_lim), float(x_lim)),
        density=True,
        histtype="stepfilled",
        alpha=0.55,
        color="C0",
        edgecolor="C0",
        label=f"pixels (n={pulls.size})",
    )
    x = np.linspace(-float(x_lim), float(x_lim), 400)
    ax.plot(
        x,
        norm.pdf(x, mu, sigma),
        color="C1",
        lw=2.0,
        label=f"Gaussian fit N({mu:.3f}, {sigma:.3f})",
    )
    ax.plot(
        x,
        norm.pdf(x, 0.0, 1.0),
        color="0.35",
        lw=1.5,
        ls="--",
        label="N(0, 1)",
    )
    ax.axvline(0.0, color="0.5", lw=0.8)
    ax.set_xlim(-float(x_lim), float(x_lim))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("probability density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if savefig is not None:
        out = os.fspath(savefig)
        parent = os.path.dirname(out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(out, format="png", dpi=150, bbox_inches="tight")

    if display:
        from IPython.display import Image, display as ipy_display

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        ipy_display(Image(data=buf.getvalue()))
    plt.close(fig)

    summary: dict[str, Any] = {
        "pulls": pulls,
        "n_pix": int(pulls.size),
        "n_core": int(n_core),
        "error_mode": mode,
        "mu": float(mu),
        "sigma": float(sigma),
        "frac_within_1sig": frac_1sig,
        "mean_pull": mean_all,
        "std_pull": std_all,
        "median_pull": med_all,
        "sigma_mad": float(sigma_mad),
    }

    if print_stats:
        print(f"n_pix={summary['n_pix']}  n_core={summary['n_core']}  error_mode={mode}")
        print(
            f"Gaussian fit: mu={mu:.4f}  sigma={sigma:.4f}  "
            f"(MAD-clip {robust_clip_sigma:g}σ)"
        )
        print(
            f"all-pixel moments: mean={mean_all:.4f}  std={std_all:.4f}  "
            f"median={med_all:.4f}  sigma_MAD={sigma_mad:.4f}"
        )
        print(
            f"fraction |z|<1: {frac_1sig:.3f}  "
            f"(N(0,1) ≈ {2.0 * norm.cdf(1.0) - 1.0:.3f})"
        )

    return summary


def diagnose_stage0_rate_noise(
    rate: np.ndarray,
    rate_err: np.ndarray,
    truth_rate: np.ndarray,
    rate_var_poisson: np.ndarray,
    rate_var_rnoise: np.ndarray,
    dq: np.ndarray | None = None,
    *,
    dq_exclude_flags: int | None = None,
    empty_sky_truth_max: float = 0.05,
    alpha_rate_floor: float = _TRUTH_SCALE_RATE_FLOOR,
    truth_bin_edges: tuple[float, ...] = (
        0.0, 0.05, 0.5, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, np.inf
    ),
    print_report: bool = True,
) -> dict[str, Any]:
    """
    Unfudged Stage-0 / simulator diagnostics (no σ rescaling to force N(0,1)).

    Reports:

    1. **Rate bias** — mean / median of ``rate - truth`` (no denominator).
    2. **Empty-sky RN scale** — ``std(residual) / sqrt(mean(var_rnoise))``.
    3. **Truth-binned variance ratio** — empirical ``Var(residual)`` vs
       ``mean(rate_err²)`` and ``mean(var_rnoise + α·truth)``.
       Primary ratios use **means**, not medians: ``emp_var`` estimates the
       mean per-pixel variance, so comparing to ``median(err²)`` in a wide
       flux bin (where ``σ²∝rate``) spuriously reports ~20–35%
       “underestimates”. ``emp/median`` is retained only as a secondary flag.
    4. **Truth-conditioned pull** — robust μ/σ of
       ``(rate - truth) / σ(truth)`` (secondary; Wald-free model check).
    5. **Fit-based pull** — robust μ/σ of ``(rate - truth) / rate_err``
       (gross check; not expected to be exact at ``1/√N``).

    None of these modify pipeline products or inflate / deflate uncertainties.
    """
    rate = np.asarray(rate, dtype=float)
    rate_err = np.asarray(rate_err, dtype=float)
    truth = np.asarray(truth_rate, dtype=float)
    vp = np.asarray(rate_var_poisson, dtype=float)
    vr = np.asarray(rate_var_rnoise, dtype=float)
    if not (rate.shape == rate_err.shape == truth.shape == vp.shape == vr.shape):
        raise ValueError("shape mismatch among rate / err / truth / variance planes")

    good = stage0_rate_good_mask(
        rate,
        rate_err,
        truth,
        dq=dq,
        dq_exclude_flags=dq_exclude_flags,
        min_truth_rate=0.0,
    )
    n = int(np.count_nonzero(good))
    if n < 10:
        raise ValueError(f"too few good pixels for diagnostics: n={n}")

    residual = (rate - truth)[good]
    mean_res = float(np.mean(residual))
    med_res = float(np.median(residual))
    std_res = float(np.std(residual, ddof=1))
    se_res = std_res / np.sqrt(float(n))
    bias_z = float(mean_res / se_res) if se_res > 0.0 else float("nan")

    # --- empty-sky RN ---
    empty = good & (truth <= float(empty_sky_truth_max))
    n_empty = int(np.count_nonzero(empty))
    empty_stats: dict[str, Any]
    if n_empty >= 50:
        res_e = (rate - truth)[empty]
        vr_e = np.maximum(vr[empty], 0.0)
        vr_e = vr_e[np.isfinite(vr_e)]
        mean_vr = float(np.mean(vr_e)) if vr_e.size else float("nan")
        med_rn = float(np.median(np.sqrt(vr_e))) if vr_e.size else float("nan")
        std_e = float(np.std(res_e, ddof=1))
        emp_var_e = float(np.var(res_e, ddof=1))
        mean_err2_e = float(np.mean(rate_err[empty] ** 2))
        rn_scale = float(std_e / np.sqrt(mean_vr)) if mean_vr > 0.0 else float("nan")
        vp_e = vp[empty]
        frac_pois_on = float(np.mean(np.isfinite(vp_e) & (vp_e > 0.0)))
        empty_stats = {
            "n_pix": n_empty,
            "std_residual": std_e,
            "emp_var": emp_var_e,
            "mean_var_rnoise": mean_vr,
            "mean_rate_err2": mean_err2_e,
            "median_sigma_rnoise": med_rn,
            "rn_scale": rn_scale,
            "emp_over_mean_err2": (
                float(emp_var_e / mean_err2_e) if mean_err2_e > 0.0 else float("nan")
            ),
            "frac_var_poisson_gt0": frac_pois_on,
            "mean_residual": float(np.mean(res_e)),
        }
    else:
        empty_stats = {
            "n_pix": n_empty,
            "std_residual": float("nan"),
            "emp_var": float("nan"),
            "mean_var_rnoise": float("nan"),
            "mean_rate_err2": float("nan"),
            "median_sigma_rnoise": float("nan"),
            "rn_scale": float("nan"),
            "emp_over_mean_err2": float("nan"),
            "frac_var_poisson_gt0": float("nan"),
            "mean_residual": float("nan"),
        }

    # --- α from bright fitted rates (pipeline Poisson coefficient) ---
    floor = float(alpha_rate_floor)
    alpha_ok = (
        good
        & np.isfinite(vp)
        & np.isfinite(rate)
        & (np.abs(rate) >= floor)
        & (vp >= 0.0)
    )
    if np.any(alpha_ok):
        alpha = float(np.median(vp[alpha_ok] / np.abs(rate[alpha_ok])))
    else:
        alpha = float("nan")

    # --- truth-binned variance ratio (means, not medians, are primary) ---
    edges = np.asarray(truth_bin_edges, dtype=float)
    bins: list[dict[str, Any]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = good & (truth >= lo) & (truth < hi)
        n_bin = int(np.count_nonzero(in_bin))
        nan_row = {
            "truth_lo": float(lo),
            "truth_hi": float(hi),
            "n_pix": n_bin,
            "emp_var": float("nan"),
            "mean_rate_err2": float("nan"),
            "median_rate_err2": float("nan"),
            "mean_model_var": float("nan"),
            "median_model_var": float("nan"),
            "emp_over_mean_err2": float("nan"),
            "emp_over_median_err2": float("nan"),
            "emp_over_mean_model": float("nan"),
            "var_ratio": float("nan"),  # alias: primary = emp/mean(err²)
            "model_var": float("nan"),  # alias: mean model var
            "mean_residual": float("nan"),
        }
        if n_bin < 50:
            bins.append(nan_row)
            continue
        res_b = (rate - truth)[in_bin]
        emp_var = float(np.var(res_b, ddof=1))
        err2 = rate_err[in_bin] ** 2
        mean_err2 = float(np.mean(err2))
        med_err2 = float(np.median(err2))
        model = np.maximum(vr[in_bin], 0.0)
        if np.isfinite(alpha) and alpha >= 0.0:
            model = model + alpha * np.maximum(truth[in_bin], 0.0)
        mean_model = float(np.mean(model))
        med_model = float(np.median(model))
        emp_over_mean_err2 = (
            float(emp_var / mean_err2) if mean_err2 > 0.0 else float("nan")
        )
        bins.append(
            {
                "truth_lo": float(lo),
                "truth_hi": float(hi),
                "n_pix": n_bin,
                "emp_var": emp_var,
                "mean_rate_err2": mean_err2,
                "median_rate_err2": med_err2,
                "mean_model_var": mean_model,
                "median_model_var": med_model,
                "emp_over_mean_err2": emp_over_mean_err2,
                "emp_over_median_err2": (
                    float(emp_var / med_err2) if med_err2 > 0.0 else float("nan")
                ),
                "emp_over_mean_model": (
                    float(emp_var / mean_model) if mean_model > 0.0 else float("nan")
                ),
                # Back-compat aliases used by older notebook prints
                "var_ratio": emp_over_mean_err2,
                "model_var": mean_model,
                "mean_residual": float(np.mean(res_b)),
            }
        )

    # --- pulls: fit-based and truth-conditioned (no product mutation) ---
    fit_pulls = stage0_rate_pulls(
        rate, rate_err, truth, dq=dq, dq_exclude_flags=dq_exclude_flags, error_mode="fit"
    )
    fit_mu, fit_sig, fit_ncore = robust_gaussian_params(fit_pulls)

    truth_pulls = stage0_rate_pulls(
        rate,
        rate_err,
        truth,
        dq=dq,
        dq_exclude_flags=dq_exclude_flags,
        rate_var_poisson=vp,
        rate_var_rnoise=vr,
        error_mode="truth",
    )
    truth_mu, truth_sig, truth_ncore = robust_gaussian_params(truth_pulls)

    out: dict[str, Any] = {
        "n_pix": n,
        "poisson_alpha": alpha,
        "n_alpha_pix": int(np.count_nonzero(alpha_ok)),
        "rate_bias": {
            "mean_residual": mean_res,
            "median_residual": med_res,
            "std_residual": std_res,
            "mean_residual_se": float(se_res),
            "mean_residual_z": bias_z,
        },
        "empty_sky_rn": empty_stats,
        "truth_bins": bins,
        "fit_pull": {
            "mu_robust": float(fit_mu),
            "sigma_robust": float(fit_sig),
            "n_core": int(fit_ncore),
            "n_pix": int(fit_pulls.size),
        },
        "truth_pull": {
            "mu_robust": float(truth_mu),
            "sigma_robust": float(truth_sig),
            "n_core": int(truth_ncore),
            "n_pix": int(truth_pulls.size),
        },
        "empty_sky_truth_max": float(empty_sky_truth_max),
    }

    if print_report:
        _print_stage0_rate_diagnosis(out)
    return out


def _print_stage0_rate_diagnosis(diag: dict[str, Any]) -> None:
    """Pretty-print :func:`diagnose_stage0_rate_noise` (no σ fudging)."""
    b = diag["rate_bias"]
    e = diag["empty_sky_rn"]
    fp = diag["fit_pull"]
    tp = diag["truth_pull"]
    print(f"n_pix={diag['n_pix']}  poisson α={diag['poisson_alpha']:.6g}  "
          f"(from {diag['n_alpha_pix']} bright pixels)")
    print(
        f"rate bias: mean={b['mean_residual']:+.5g}  med={b['median_residual']:+.5g}  "
        f"SE={b['mean_residual_se']:.5g}  z={b['mean_residual_z']:+.2f}  "
        f"(z≠0 ⇒ rate bias in sim or Stage 0)"
    )
    print(
        f"empty sky (truth≤{diag['empty_sky_truth_max']:g}): n={e['n_pix']}  "
        f"RN scale={e['rn_scale']:.4f}  "
        f"emp/mean(err²)={e.get('emp_over_mean_err2', float('nan')):.4f}  "
        f"frac(var_poisson>0)={e['frac_var_poisson_gt0']:.4f}"
    )
    print(
        f"pulls: fit-based N({fp['mu_robust']:.4f}, {fp['sigma_robust']:.4f})  |  "
        f"truth-σ N({tp['mu_robust']:.4f}, {tp['sigma_robust']:.4f})  "
        f"(truth-σ off 0/1 ⇒ variance model; fit-only off ⇒ Wald/plug-in)"
    )
    print(
        f"{'truth_lo':>10s} {'truth_hi':>10s} {'n':>9s} "
        f"{'emp_var':>11s} {'mean_err2':>11s} {'emp/mean':>9s} "
        f"{'emp/med':>8s} {'mean_res':>10s}"
    )
    for row in diag["truth_bins"]:
        hi = row["truth_hi"]
        hi_s = "inf" if np.isinf(hi) else f"{hi:g}"
        print(
            f"{row['truth_lo']:10.4g} {hi_s:>10s} {row['n_pix']:9d} "
            f"{row['emp_var']:11.5g} {row['mean_rate_err2']:11.5g} "
            f"{row['emp_over_mean_err2']:9.4f} "
            f"{row['emp_over_median_err2']:8.4f} "
            f"{row['mean_residual']:+10.4g}"
        )
    print(
        "Note: primary metric is emp/mean(err²). emp/med(err²) can look ~1.2–1.3 "
        "in wide flux bins even when rate_err is correct (mean vs median of σ²∝rate)."
    )

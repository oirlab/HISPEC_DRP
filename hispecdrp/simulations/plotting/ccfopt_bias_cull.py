"""Line-bias cull histograms for ``ccfopt_bias_cull_from_order_scan``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from ...utils.line_mask_utils import MaskLines
from ...utils.physical_constants import C_M_S


def compute_line_bias_table(
    bias_rv_results: Sequence[Mapping[str, Any]],
    rezero_mask_path: str | Path,
    *,
    match_half_m_s: float = 5000.0,
    bias_table_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Match per-inject chunk RVs to rezeroed-mask lines and build a bias table.

    Parameters
    ----------
    bias_rv_results
        Sequence of dicts with keys ``inject_m_s`` and ``rvccf`` (table with
        ``WAVELENGTH`` / ``RV`` columns), as produced by the bias-cull notebook.
    rezero_mask_path
        Interim rezeroed ``*.mas`` used as the stable rest-λ / weight reference.

    Returns
    -------
    result
        ``line_bias`` table, ``ref_lam``, ``ref_w``, ``mean_bias``, ``rms_bias``,
        ``bias_grid``, and optional written ``bias_table_path``.
    """
    ml_ref = MaskLines.load(str(rezero_mask_path))
    ref_lam = np.asarray(ml_ref.line_lambda_um, dtype=np.float64)
    ref_w = np.asarray(ml_ref.weight, dtype=np.float64)
    n_ref = int(ref_lam.size)

    rv_grid = np.full((n_ref, len(bias_rv_results)), np.nan, dtype=np.float64)
    for j, r in enumerate(bias_rv_results):
        tbl = r["rvccf"]
        chunk_wl = np.asarray(tbl["WAVELENGTH"], dtype=np.float64).ravel()
        chunk_rv = np.asarray(tbl["RV"], dtype=np.float64).ravel()
        used = np.zeros(chunk_wl.size, dtype=bool)
        for i, lam0 in enumerate(ref_lam):
            if not np.isfinite(lam0) or lam0 <= 0:
                continue
            dv = C_M_S * (chunk_wl - lam0) / lam0
            ok = (
                np.isfinite(dv)
                & np.isfinite(chunk_rv)
                & (np.abs(dv) <= float(match_half_m_s))
                & (~used)
            )
            if not np.any(ok):
                continue
            idx = np.flatnonzero(ok)
            k = int(idx[np.argmin(np.abs(dv[idx]))])
            used[k] = True
            rv_grid[i, j] = float(chunk_rv[k])

    injects = np.asarray([r["inject_m_s"] for r in bias_rv_results], dtype=np.float64)
    bias_grid = rv_grid - injects[None, :]
    n_finite = np.sum(np.isfinite(bias_grid), axis=1)
    mean_bias = np.nanmean(bias_grid, axis=1)
    rms_bias = np.nanstd(bias_grid, axis=1)

    rows = []
    for i in range(n_ref):
        rows.append(
            {
                "line_index": i,
                "wavelength_um": float(ref_lam[i]),
                "wavelength_nm": float(ref_lam[i] * 1000.0),
                "weight": float(ref_w[i]),
                "n_inject_finite": int(n_finite[i]),
                "mean_bias_m_s": (
                    float(mean_bias[i]) if np.isfinite(mean_bias[i]) else np.nan
                ),
                "rms_bias_m_s": (
                    float(rms_bias[i]) if np.isfinite(rms_bias[i]) else np.nan
                ),
                "abs_mean_bias_m_s": (
                    float(abs(mean_bias[i])) if np.isfinite(mean_bias[i]) else np.nan
                ),
            }
        )
    line_bias = Table(rows)
    written = None
    if bias_table_path is not None:
        written = Path(bias_table_path)
        line_bias.write(written, format="ascii.ecsv", overwrite=True)

    return {
        "line_bias": line_bias,
        "ref_lam": ref_lam,
        "ref_w": ref_w,
        "mean_bias": mean_bias,
        "rms_bias": rms_bias,
        "bias_grid": bias_grid,
        "injects": injects,
        "bias_table_path": str(written) if written is not None else None,
    }


def write_bias_cull_histograms(
    mean_bias: np.ndarray,
    bias_abs_cut_m_s: float,
    out_png: str | Path,
    *,
    show: bool = False,
) -> dict[str, Any]:
    """Write mean-bias and |mean-bias| histograms with the cull threshold marked."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    finite = np.isfinite(mean_bias)
    mb = np.asarray(mean_bias[finite], dtype=float)
    cut = float(bias_abs_cut_m_s)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(mb, bins=80, color="steelblue", edgecolor="k", linewidth=0.3)
    axes[0].axvline(0.0, color="k", lw=1)
    axes[0].axvline(+cut, color="crimson", ls="--", lw=1.2, label=f"+/-cut={cut}")
    axes[0].axvline(-cut, color="crimson", ls="--", lw=1.2)
    axes[0].set_xlabel(r"per-line mean bias $\langle RV - inject\rangle$ (m/s)")
    axes[0].set_ylabel("number of lines")
    axes[0].set_title("Histogram of average line bias")
    axes[0].legend(loc="best", fontsize=9)

    axes[1].hist(np.abs(mb), bins=80, color="darkorange", edgecolor="k", linewidth=0.3)
    axes[1].axvline(cut, color="crimson", ls="--", lw=1.2, label=f"cut={cut}")
    axes[1].set_xlabel(r"|mean bias| (m/s)")
    axes[1].set_ylabel("number of lines")
    axes[1].set_title("|mean bias| distribution")
    axes[1].legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    return {
        "png": str(out_png),
        "n_finite": int(mb.size),
        "median_m_s": float(np.median(mb)) if mb.size else float("nan"),
        "mad_m_s": (
            float(np.median(np.abs(mb - np.median(mb)))) if mb.size else float("nan")
        ),
        "p95_abs_m_s": float(np.percentile(np.abs(mb), 95)) if mb.size else float("nan"),
        "p99_abs_m_s": float(np.percentile(np.abs(mb), 99)) if mb.size else float("nan"),
        "max_abs_m_s": float(np.max(np.abs(mb))) if mb.size else float("nan"),
    }


def analyze_line_bias_and_plot(
    bias_rv_results: Sequence[Mapping[str, Any]],
    rezero_mask_path: str | Path,
    *,
    bias_table_path: str | Path,
    hist_png: str | Path,
    bias_abs_cut_m_s: float,
    match_half_m_s: float = 5000.0,
    show: bool = False,
) -> dict[str, Any]:
    """Compute the line-bias table, write it, and plot cull histograms."""
    computed = compute_line_bias_table(
        bias_rv_results,
        rezero_mask_path,
        match_half_m_s=match_half_m_s,
        bias_table_path=bias_table_path,
    )
    hist = write_bias_cull_histograms(
        computed["mean_bias"],
        bias_abs_cut_m_s,
        hist_png,
        show=show,
    )
    mean_bias = computed["mean_bias"]
    ref_w = computed["ref_w"]
    finite = np.isfinite(mean_bias)
    keep = finite & (np.abs(mean_bias) < float(bias_abs_cut_m_s)) & (ref_w > 0)
    return {
        **computed,
        **hist,
        "finite": finite,
        "keep": keep,
        "n_keep": int(np.sum(keep)),
        "n_drop": int(np.sum(finite) - np.sum(keep)),
        "bias_abs_cut_m_s": float(bias_abs_cut_m_s),
    }

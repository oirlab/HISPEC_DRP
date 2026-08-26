"""Deposit-inverse WAVE geometry validation plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_comb_wavesol_order_residual_png(
    out_path: str | Path,
    *,
    fiber: int,
    order: int,
    pix: np.ndarray,
    vel_cm_s: np.ndarray,
    lam_mid_um: float,
    mad_cm_s: float,
    rms_cm_s: float,
    target_cm_s: float = 10.0,
    legacy_alias_path: str | Path | None = None,
) -> None:
    """Per-order residual vs deposit spectral pixel."""
    out_path = Path(out_path)
    pix = np.asarray(pix, dtype=float)
    vel = np.asarray(vel_cm_s, dtype=float)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax = axes[0]
    ax.plot(pix, vel, ".", ms=2.5, alpha=0.5, color="C0", label="features")
    ax.axhline(0, color="C3", lw=1.5, label="WAVE = deposit inverse")
    ax.axhline(
        target_cm_s, color="k", ls="--", lw=0.8, label=f"±{target_cm_s:g} cm/s target"
    )
    ax.axhline(-target_cm_s, color="k", ls="--", lw=0.8)
    ax.set_ylabel(f"Δv (cm/s) at ⟨λ⟩={lam_mid_um:.4f} µm")
    ax.set_title(
        f"Fiber {fiber} order {order}: geometry inject/recover  |  "
        f"MAD={mad_cm_s:.3f} cm/s  RMS={rms_cm_s:.3f} cm/s  N={pix.size}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-30, 30)

    axr = axes[1]
    axr.plot(pix, vel, ".", ms=2.0, alpha=0.45, color="C0")
    axr.axhline(0, color="C3", lw=1.2)
    axr.axhline(mad_cm_s, color="k", ls="--", lw=0.8)
    axr.axhline(-mad_cm_s, color="k", ls="--", lw=0.8)
    axr.set_xlabel("deposit spectral pixel (exact λ→row)")
    axr.set_ylabel("residual (cm/s)")
    axr.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    if legacy_alias_path is not None:
        fig.savefig(legacy_alias_path, dpi=150)
    plt.close(fig)


def write_comb_wavesol_per_order_summary_png(
    out_path: str | Path,
    *,
    fiber: int,
    orders: Sequence[int],
    per_order: Mapping[int, Mapping[str, Any]],
    mads_cm_s: Sequence[float],
    target_cm_s: float = 10.0,
    legacy_alias_path: str | Path | None = None,
) -> None:
    """MAD/RMS vs echelle order for one fiber."""
    orders = list(orders)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(orders, [per_order[o]["mad_cm_s"] for o in orders], "o-", ms=4, label="MAD")
    ax.plot(orders, [per_order[o]["rms_cm_s"] for o in orders], "s-", ms=3, label="RMS")
    ax.axhline(target_cm_s, color="k", ls="--", lw=1, label=f"{target_cm_s:g} cm/s target")
    ax.set_xlabel("echelle order")
    ax.set_ylabel("geometry residual (cm/s)")
    ax.set_title(
        f"Fiber {fiber} deposit-inverse WAVE  |  "
        f"median MAD={float(np.median(mads_cm_s)):.3f} cm/s"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    if legacy_alias_path is not None:
        fig.savefig(legacy_alias_path, dpi=140)
    plt.close(fig)


def write_comb_wavesol_per_fiber_summary_png(
    out_path: str | Path,
    *,
    per_fiber: Mapping[int, Mapping[str, Any]],
    target_cm_s: float = 10.0,
) -> None:
    """Median / max MAD across fibers."""
    fibs = sorted(per_fiber)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        fibs,
        [per_fiber[f]["median_mad_cm_s"] for f in fibs],
        "o-",
        ms=6,
        label="median MAD",
    )
    ax.plot(
        fibs,
        [per_fiber[f]["max_mad_cm_s"] for f in fibs],
        "s-",
        ms=5,
        label="max MAD",
    )
    ax.axhline(target_cm_s, color="k", ls="--", lw=1, label=f"{target_cm_s:g} cm/s target")
    ax.set_xticks(fibs)
    ax.set_xlabel("fiber")
    ax.set_ylabel("geometry residual (cm/s)")
    ax.set_title("Deposit-inverse WAVE by fiber")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

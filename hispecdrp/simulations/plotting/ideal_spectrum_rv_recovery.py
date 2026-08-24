"""Ideal-spectrum injected-vs-recovered RV plots (lte040 validation notebook)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_ideal_spectrum_rv_recovery(
    injected_m_s: np.ndarray,
    recovered_m_s: np.ndarray,
    recovered_err_m_s: np.ndarray,
    *,
    title: str = "BT-Settl lte040: injected vs recovered RV",
    out_path: str | Path | None = None,
    display: bool = True,
) -> dict[str, Any]:
    """
    Two-panel ideal-spectrum RV diagnostic: 1:1 recovery and zoomed residuals.

    Parameters
    ----------
    injected_m_s, recovered_m_s, recovered_err_m_s
        Per-bulk-shift arrays in m/s (same length).
    title
        Title for the top panel.
    out_path
        When set, write a PNG to this path.
    display
        If True, embed a PNG in the notebook via ``IPython.display``.

    Returns
    -------
    summary
        Bias statistics in m/s and mm/s.
    """
    inj_m_s = np.asarray(injected_m_s, dtype=float).ravel()
    rec_m_s = np.asarray(recovered_m_s, dtype=float).ravel()
    err_m_s = np.asarray(recovered_err_m_s, dtype=float).ravel()
    if not (inj_m_s.size == rec_m_s.size == err_m_s.size):
        raise ValueError("injected, recovered, and err arrays must have the same length")

    inj_km_s = inj_m_s / 1000.0
    rec_km_s = rec_m_s / 1000.0
    err_km_s = err_m_s / 1000.0
    residual_m_s = rec_m_s - inj_m_s
    bias_m_s = residual_m_s.copy()
    max_abs = float(np.max(np.abs(bias_m_s)))
    rms = float(np.sqrt(np.mean(bias_m_s**2)))
    d_bias = bias_m_s - bias_m_s[0]
    max_d = float(np.max(np.abs(d_bias[1:]))) if bias_m_s.size > 1 else 0.0

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(6.5, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.05},
    )

    lo = float(min(inj_km_s.min(), rec_km_s.min()) - 2.0)
    hi = float(max(inj_km_s.max(), rec_km_s.max()) + 2.0)
    ax_top.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
    ax_top.errorbar(
        inj_km_s,
        rec_km_s,
        yerr=err_km_s,
        fmt="o",
        ms=6,
        capsize=3,
        label="recovered (RV_ERR)",
    )
    ax_top.set_xlim(lo, hi)
    ax_top.set_ylim(lo, hi)
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.set_ylabel("Recovered RV (km/s)")
    ax_top.set_title(title)
    ax_top.legend(loc="upper left")
    ax_top.grid(True, alpha=0.3)

    ax_bot.axhline(0.0, color="k", ls="--", lw=1, label="zero residual")
    ax_bot.errorbar(
        inj_km_s,
        residual_m_s,
        yerr=err_m_s,
        fmt="o",
        ms=6,
        capsize=3,
        color="C1",
        label="residual +/- RV_ERR",
    )
    ax_bot.set_xlabel("Injected RV (km/s)")
    ax_bot.set_ylabel("Residual (m/s)")
    med = float(np.median(residual_m_s))
    scatter_half = float(np.max(np.abs(residual_m_s - med))) * 1.5
    err_half = float(np.max(err_m_s)) * 2.0
    half = max(scatter_half, err_half, 0.05)
    ax_bot.set_ylim(med - half, med + half)
    ax_bot.legend(loc="best", fontsize=9)
    ax_bot.grid(True, alpha=0.3)

    fig.subplots_adjust(hspace=0.05)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if display:
        from IPython.display import Image, display as ipy_display

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        ipy_display(Image(data=buf.getvalue()))
    plt.close(fig)

    return {
        "injected_m_s": inj_m_s,
        "recovered_m_s": rec_m_s,
        "recovered_err_m_s": err_m_s,
        "bias_m_s": bias_m_s,
        "max_abs_bias_m_s": max_abs,
        "max_abs_bias_mm_s": max_abs * 1e3,
        "rms_bias_m_s": rms,
        "rms_bias_mm_s": rms * 1e3,
        "max_differential_bias_m_s": max_d,
        "max_differential_bias_mm_s": max_d * 1e3,
        "out_path": str(out_path) if out_path is not None else None,
    }

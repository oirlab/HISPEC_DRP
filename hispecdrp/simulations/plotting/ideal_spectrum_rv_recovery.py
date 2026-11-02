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
    Ideal-spectrum RV residual diagnostic (recovered minus injected).

    Parameters
    ----------
    injected_m_s, recovered_m_s, recovered_err_m_s
        Per-bulk-shift arrays in m/s (same length).
    title
        Base title; the mean and standard deviation of the residuals (in
        cm/s) are appended to it.
    out_path
        When set, write a PNG to this path.
    display
        If True, embed a PNG in the notebook via ``IPython.display``.

    Returns
    -------
    summary
        Bias statistics in m/s, mm/s, and cm/s.
    """
    inj_m_s = np.asarray(injected_m_s, dtype=float).ravel()
    rec_m_s = np.asarray(recovered_m_s, dtype=float).ravel()
    err_m_s = np.asarray(recovered_err_m_s, dtype=float).ravel()
    if not (inj_m_s.size == rec_m_s.size == err_m_s.size):
        raise ValueError("injected, recovered, and err arrays must have the same length")

    inj_km_s = inj_m_s / 1000.0
    residual_m_s = rec_m_s - inj_m_s
    bias_m_s = residual_m_s.copy()
    max_abs = float(np.max(np.abs(bias_m_s)))
    rms = float(np.sqrt(np.mean(bias_m_s**2)))
    d_bias = bias_m_s - bias_m_s[0]
    max_d = float(np.max(np.abs(d_bias[1:]))) if bias_m_s.size > 1 else 0.0

    residual_cm_s = residual_m_s * 100.0
    err_cm_s = err_m_s * 100.0
    mean_cm_s = float(np.mean(residual_cm_s))
    std_cm_s = float(np.std(residual_cm_s))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.axhline(0.0, color="k", ls="--", lw=1, label="zero residual")
    ax.errorbar(
        inj_km_s,
        residual_cm_s,
        yerr=err_cm_s,
        fmt="o",
        ms=6,
        capsize=3,
        color="C1",
        label="residual +/- RV_ERR",
    )
    ax.set_xlabel("Injected RV (km/s)")
    ax.set_ylabel("Residual (cm/s)")
    ax.set_title(f"{title}\nmean = {mean_cm_s:.3f} cm/s, std = {std_cm_s:.3f} cm/s")
    med = float(np.median(residual_cm_s))
    scatter_half = float(np.max(np.abs(residual_cm_s - med))) * 1.5
    err_half = float(np.max(err_cm_s)) * 2.0
    half = max(scatter_half, err_half, 5.0)
    ax.set_ylim(med - half, med + half)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
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
        "mean_bias_cm_s": mean_cm_s,
        "std_bias_cm_s": std_cm_s,
        "out_path": str(out_path) if out_path is not None else None,
    }

"""Comparison plots for step-by-step RV recovery diagnostics."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_rv_step_comparison(
    summaries: dict[str, dict[str, Any]],
    *,
    title: str = "RV recovery by pipeline step",
    out_path: str | Path | None = None,
    display: bool = True,
) -> dict[str, Any]:
    """Overlay residual and formal RV_ERR for each completed recovery step."""
    rows = [(name, s) for name, s in summaries.items() if s]
    if not rows:
        raise ValueError("summaries is empty")

    fig, (ax_bias, ax_err) = plt.subplots(2, 1, sharex=True, figsize=(7.0, 6.5))
    ax_bias.axhline(0.0, color="k", ls="--", lw=1)
    for i, (name, summary) in enumerate(rows):
        inj_km_s = np.asarray(summary["injected_m_s"], dtype=float) / 1000.0
        bias_m_s = np.asarray(summary["bias_m_s"], dtype=float)
        err_m_s = np.asarray(summary["recovered_err_m_s"], dtype=float)
        color = f"C{i}"
        ax_bias.errorbar(
            inj_km_s,
            bias_m_s,
            yerr=err_m_s,
            fmt="o",
            ms=6,
            capsize=3,
            color=color,
            label=name,
        )
        ax_err.plot(inj_km_s, err_m_s, "o-", ms=6, color=color, label=name)
    ax_bias.set_ylabel("Residual (m/s)")
    ax_bias.set_title(title)
    ax_bias.legend(loc="best", fontsize=8)
    ax_bias.grid(True, alpha=0.3)
    ax_err.set_xlabel("Injected RV (km/s)")
    ax_err.set_ylabel("RV_ERR (m/s)")
    ax_err.grid(True, alpha=0.3)
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
    return {"out_path": str(out_path) if out_path is not None else None, "n_steps": len(rows)}

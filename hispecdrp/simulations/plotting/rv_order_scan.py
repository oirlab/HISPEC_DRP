"""Per-order RV scan validation plots (``rv_order_scan_sim`` notebook)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.backends.backend_pdf import PdfPages


def write_per_order_rv_scan_report(
    out_dir: str | Path,
    *,
    order_table: Table | None = None,
    order_csv: str | Path | None = None,
    fiber: int = 2,
    spectrograph: str = "BSPEC",
    pdf_name: str = "per_order_rv_scan.pdf",
    png_name: str = "per_order_rv_scan.png",
    show: bool = False,
) -> dict[str, Any]:
    """
    Write injected-vs-recovered and bias-heatmap pages for a per-order RV scan.

    Provide either ``order_table`` or ``order_csv`` (``per_order_rv_scan.ecsv``).

    Returns
    -------
    summary
        Paths and bias statistics, including ``n_line_count_changes``.
    """
    out_dir = Path(out_dir)
    if order_table is None:
        if order_csv is None:
            order_csv = out_dir / "per_order_rv_scan.ecsv"
        order_csv = Path(order_csv)
        if not order_csv.is_file():
            raise FileNotFoundError(
                f"Missing {order_csv}. Run the stellar-RV section first."
            )
        order_table = Table.read(order_csv)

    n_col = "n_lines" if "n_lines" in order_table.colnames else "n_chunks"
    injects = np.array(sorted({float(v) for v in order_table["inject_m_s"]}))
    orders = np.array(sorted({int(o) for o in order_table["order"]}))

    rv_mat = np.full((len(orders), len(injects)), np.nan)
    bias_mat = np.full_like(rv_mat, np.nan)
    n_mat = np.full_like(rv_mat, np.nan)
    od_index = {int(o): i for i, o in enumerate(orders)}
    inj_index = {float(v): j for j, v in enumerate(injects)}
    for row in order_table:
        i = od_index[int(row["order"])]
        j = inj_index[float(row["inject_m_s"])]
        rv_mat[i, j] = float(row["rv_order_m_s"])
        bias_mat[i, j] = float(row["bias_order_m_s"])
        n_mat[i, j] = float(row[n_col])

    pdf_path = out_dir / pdf_name
    png_path = out_dir / png_name
    n_changes = 0

    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0.05, 0.95, len(orders)))

        ax = axes[0]
        lo = float(min(injects.min(), np.nanmin(rv_mat))) - 50.0
        hi = float(max(injects.max(), np.nanmax(rv_mat))) + 50.0
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="1:1", zorder=0)
        for i, od in enumerate(orders):
            ax.plot(
                injects,
                rv_mat[i],
                "o-",
                ms=4,
                lw=1.0,
                color=colors[i],
                alpha=0.85,
                label=str(int(od)) if i % 4 == 0 else None,
            )
        ax.set_xlabel("injected RV (m/s)")
        ax.set_ylabel("recovered order RV (m/s)")
        ax.set_title("Injected vs recovered (per order)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(title="order (sample)", fontsize=7, loc="upper left")

        ax = axes[1]
        for i, od in enumerate(orders):
            ax.plot(
                injects,
                bias_mat[i],
                "o-",
                ms=3.5,
                lw=0.9,
                color=colors[i],
                alpha=0.8,
            )
        ax.axhline(0.0, color="k", lw=1.0)
        ax.set_xlabel("injected RV (m/s)")
        ax.set_ylabel("order bias = recovered − injected (m/s)")
        ax.set_title("Per-order bias vs inject")
        ax.grid(True, alpha=0.3)
        fig.suptitle(
            f"{spectrograph} fiber {fiber}: line-by-line ccfopt  |  "
            f"{len(injects)} steps × {len(orders)} orders",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(png_path, dpi=150)
        pdf.savefig(fig)
        if show:
            plt.show()
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12.5, 7.5))
        vmax = float(np.nanpercentile(np.abs(bias_mat), 95))
        vmax = max(vmax, 1.0)
        im = ax.imshow(
            bias_mat,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        for i in range(len(orders)):
            for j in range(len(injects)):
                n = n_mat[i, j]
                if not np.isfinite(n):
                    continue
                b = bias_mat[i, j]
                txt_color = (
                    "white" if (np.isfinite(b) and abs(b) > 0.55 * vmax) else "black"
                )
                ax.text(
                    j,
                    i,
                    f"{int(round(n))}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=txt_color,
                    fontweight="semibold",
                    clip_on=True,
                )

        for i in range(len(orders)):
            for j in range(len(injects) - 1):
                a, b = n_mat[i, j], n_mat[i, j + 1]
                if not (np.isfinite(a) and np.isfinite(b)):
                    continue
                if int(round(a)) == int(round(b)):
                    continue
                n_changes += 1
                ax.plot(
                    [j + 0.5, j + 0.5],
                    [i - 0.5, i + 0.5],
                    color="gold",
                    lw=3.8,
                    solid_capstyle="butt",
                    zorder=5,
                )

        ax.set_xticks(np.arange(len(injects)))
        ax.set_xticklabels([f"{v:.0f}" for v in injects], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(orders)))
        ax.set_yticklabels([str(int(o)) for o in orders])
        ax.set_xlabel("injected RV (m/s)")
        ax.set_ylabel("echelle order")
        ax.set_title(
            "Per-order bias heatmap (recovered − injected)\n"
            f"cell numeral = {n_col}  |  gold border = {n_col} change ({n_changes} edges)"
        )
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label("bias (m/s)")
        fig.tight_layout()
        pdf.savefig(fig)
        if show:
            plt.show()
        plt.close(fig)

    med = float(np.nanmedian(bias_mat))
    return {
        "pdf": str(pdf_path),
        "png": str(png_path),
        "n_inject": int(len(injects)),
        "n_orders": int(len(orders)),
        "n_col": n_col,
        "n_line_count_changes": int(n_changes),
        "bias_median_m_s": med,
        "bias_mad_m_s": float(np.nanmedian(np.abs(bias_mat - med))),
        "bias_abs_max_m_s": float(np.nanmax(np.abs(bias_mat))),
        "order_table": order_table,
    }

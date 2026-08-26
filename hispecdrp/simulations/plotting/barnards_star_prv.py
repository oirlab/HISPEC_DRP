"""PRV injected-vs-recovered diagnostic plots (Barnard's star example notebook)."""

from __future__ import annotations

import io
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


def plot_injected_vs_recovered_rv(
    l2_results: Mapping[str, Mapping[str, Any]],
    *,
    systemic_rv_km_s: float,
    fiber: int = 1,
    title: str = "Barnard's star PRV: injected vs recovered",
    residual_ylim_m_s: float = 1.0,
    display: bool = True,
    print_table: bool = True,
) -> dict[str, Any]:
    """
    Three-panel PRV diagnostic: raw RVs, zoomed residual, and demeaned RV signal.

    Expects ``l2_results`` entries with ``mjd`` (Julian date), ``inject_m_s``,
    and ``l2`` (Stage-2 model with ``rvccfavg_fib{fiber}``). Julian dates are
    converted to MJD for the x-axis.

    Panels
    ------
    1. Raw injected and recovered RVs (absolute, m/s).
    2. Residual ``(rec - inj) - mean(rec - inj)`` with fixed y-limits of
       ``±residual_ylim_m_s`` (default ±1 m/s).
    3. RV signal: injected about the systemic RV, and recovered minus a
       zero-point (mean recovered).

    Parameters
    ----------
    l2_results
        Mapping of epoch label → result dict from a PRV segment loop.
    systemic_rv_km_s
        Systemic stellar RV in km/s (used for the injected signal panel).
    fiber
        Science fiber used for recovered RV (default ``1``).
    title
        Top-panel title.
    residual_ylim_m_s
        Half-span of the middle residual panel in m/s (default ``1.0``).
    display
        If True, embed a PNG in the notebook via ``IPython.display``.
    print_table
        If True, print a short epoch table and summary statistics.

    Returns
    -------
    summary
        Arrays and residual zero-point / RMS statistics.
    """
    rows: list[tuple[float, float, float, float]] = []
    for result in l2_results.values():
        jd = float(result["mjd"])
        mjd = jd - 2400000.5
        injected_m_s = float(result["inject_m_s"])
        avg = getattr(result["l2"], f"rvccfavg_fib{fiber}", None)
        if avg is not None and len(avg):
            recovered_m_s = float(avg["RV"][0])
            try:
                recovered_err_m_s = float(avg["RV_ERR"][0])
            except Exception:
                recovered_err_m_s = np.nan
        else:
            recovered_m_s = np.nan
            recovered_err_m_s = np.nan
        rows.append((mjd, injected_m_s, recovered_m_s, recovered_err_m_s))

    rows.sort(key=lambda r: r[0])
    mjd = np.asarray([r[0] for r in rows], dtype=float)
    injected_m_s = np.asarray([r[1] for r in rows], dtype=float)
    recovered_m_s = np.asarray([r[2] for r in rows], dtype=float)
    recovered_err_m_s = np.asarray([r[3] for r in rows], dtype=float)
    residual_m_s = recovered_m_s - injected_m_s

    systemic_m_s = float(systemic_rv_km_s) * 1000.0
    injected_signal_m_s = injected_m_s - systemic_m_s
    yerr = recovered_err_m_s if np.isfinite(recovered_err_m_s).any() else None

    residual_zp_m_s = (
        float(np.nanmean(residual_m_s)) if np.isfinite(residual_m_s).any() else np.nan
    )
    residual_demeaned_m_s = residual_m_s - residual_zp_m_s

    recovered_zp_m_s = (
        float(np.nanmean(recovered_m_s)) if np.isfinite(recovered_m_s).any() else np.nan
    )
    recovered_signal_m_s = recovered_m_s - recovered_zp_m_s

    fig, (ax0, ax1, ax2) = plt.subplots(
        3,
        1,
        figsize=(7.5, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1.0, 1.4], "hspace": 0.06},
    )

    # --- Top: raw absolute RVs ---
    ax0.plot(mjd, injected_m_s, "o-", label="injected", color="C0", zorder=3)
    if np.isfinite(recovered_m_s).any():
        ax0.errorbar(
            mjd,
            recovered_m_s,
            yerr=yerr,
            fmt="s--",
            label=f"recovered (fiber {fiber})",
            color="C1",
            capsize=3,
            zorder=2,
        )
    ax0.set_ylabel("RV [m/s]")
    ax0.set_title(title)
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    # --- Middle: demeaned residual on a fixed ±1 m/s scale ---
    half = float(residual_ylim_m_s)
    if np.isfinite(residual_demeaned_m_s).any():
        ax1.axhline(0.0, color="0.5", lw=0.8, zorder=1)
        ax1.errorbar(
            mjd,
            residual_demeaned_m_s,
            yerr=yerr,
            fmt="D-",
            color="C2",
            capsize=3,
            label="(rec - inj) - mean(rec - inj)",
        )
        ax1.set_ylim(-half, half)
        ax1.legend(fontsize=8)
    else:
        ax1.text(
            0.5,
            0.5,
            "no recovered RVs",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )
        ax1.set_ylim(-half, half)
    ax1.set_ylabel("residual [m/s]")
    ax1.grid(True, alpha=0.3)

    # --- Bottom: RV signal (injected about systemic; recovered minus ZP) ---
    ax2.plot(
        mjd,
        injected_signal_m_s,
        "o-",
        label="injected - RV0",
        color="C0",
        zorder=3,
    )
    if np.isfinite(recovered_signal_m_s).any():
        ax2.errorbar(
            mjd,
            recovered_signal_m_s,
            yerr=yerr,
            fmt="s--",
            label="recovered - ZP",
            color="C1",
            capsize=3,
            zorder=2,
        )
    ax2.axhline(0.0, color="0.5", lw=0.8, zorder=1)
    ax2.set_xlabel("MJD")
    ax2.set_ylabel("RV signal [m/s]")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if display:
        from IPython.display import Image, display as ipy_display

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        ipy_display(Image(data=buf.getvalue()))
    plt.close(fig)

    summary: dict[str, Any] = {
        "mjd": mjd,
        "injected_m_s": injected_m_s,
        "recovered_m_s": recovered_m_s,
        "recovered_err_m_s": recovered_err_m_s,
        "residual_m_s": residual_m_s,
        "residual_demeaned_m_s": residual_demeaned_m_s,
        "residual_zp_m_s": residual_zp_m_s,
        "recovered_zp_m_s": recovered_zp_m_s,
        "injected_signal_m_s": injected_signal_m_s,
        "recovered_signal_m_s": recovered_signal_m_s,
        "systemic_m_s": systemic_m_s,
        "fiber": int(fiber),
        "n_epochs": int(len(mjd)),
        "residual_ylim_m_s": half,
    }

    if print_table:
        print(f"n_epochs={len(mjd)}  systemic RV0={systemic_m_s:.3f} m/s")
        if len(mjd):
            print(f"MJD range: {mjd.min():.5f} -> {mjd.max():.5f}")
            print(
                f"injected signal range: {np.nanmin(injected_signal_m_s):.3f} -> "
                f"{np.nanmax(injected_signal_m_s):.3f} m/s"
            )
        if np.isfinite(residual_m_s).any():
            demeaned_rms = float(np.nanstd(residual_demeaned_m_s))
            summary["residual_demeaned_rms_m_s"] = demeaned_rms
            print(
                f"residual ZP (mean rec-inj)={residual_zp_m_s:.3f} m/s  "
                f"demeaned rms={demeaned_rms:.3f} m/s"
            )
            print(f"recovered ZP (mean recovered)={recovered_zp_m_s:.3f} m/s")
            print("MJD          injected   recovered   residual   residual-ZP  signal")
            for t, inj, rec, res, dem, sig in zip(
                mjd,
                injected_m_s,
                recovered_m_s,
                residual_m_s,
                residual_demeaned_m_s,
                recovered_signal_m_s,
            ):
                print(
                    f"{t:11.5f}  {inj:9.3f}  {rec:9.3f}  {res:9.3f}  "
                    f"{dem:9.3f}  {sig:9.3f}"
                )

    return summary

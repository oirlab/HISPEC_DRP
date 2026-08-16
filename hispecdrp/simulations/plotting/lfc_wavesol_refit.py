"""PDF / JSON validation plots for the LFC elliptical-Gaussian WAVE refit notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.backends.backend_pdf import PdfPages

from ...primitives.utils.spectral_utils import delta_wave_to_delta_vel


def _mad(x: Any) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.median(np.abs(x - np.median(x))))


def _symmetric_hist_xlim(
    x: np.ndarray,
    *,
    min_half_span: float,
    percentile: tuple[float, float] = (0.5, 99.5),
    pad: float = 1.15,
) -> tuple[float, float]:
    """
    Symmetric x-limits about zero with a *minimum* half-span.

    Stops matplotlib from auto-zooming into machine-precision noise when
    residuals are ~0 (common for noiseless injected LFC spots).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    half = float(min_half_span)
    if x.size:
        lo, hi = np.percentile(x, list(percentile))
        half = max(half, abs(float(lo)), abs(float(hi)), abs(float(np.median(x))))
        mad = _mad(x)
        if np.isfinite(mad) and mad > 0:
            half = max(half, 5.0 * mad)
    half = max(half * float(pad), float(min_half_span))
    return (-half, half)


def write_lfc_elliptical_wave_report(
    out_dir: str | Path,
    *,
    tbl: Table,
    success_mask: np.ndarray,
    per_order: Mapping[int, Mapping[str, Any]],
    a_pix: float,
    b_pix: float,
    pa_deg: float,
    spectrograph: str,
    fiber: int,
    fiber_object: str,
    lfc_meta: Mapping[str, Any],
    mode_stride: int,
    n_modes_total: int,
    n_modes_per_order: Sequence[int],
    rate_path: str | Path,
    fits_table_path: str | Path,
    pdf_name: str = "lfc_velocity_residuals.pdf",
    json_name: str = "lfc_velocity_residuals.json",
) -> dict[str, Any]:
    """
    Write the LFC elliptical-fit WAVE residual PDF and JSON summary.

    Parameters
    ----------
    tbl
        Per-spot elliptical-fit table (``lfc_elliptical_fits.ecsv`` columns).
    success_mask
        Boolean mask over ``tbl`` rows used for profile / WAVE diagnostics.
    per_order
        Output of the notebook WAVE residual loop: for each order, keys include
        ``lam_inj``, ``lam_rec``, ``dvel``, ``mad``, ``bias``, ``lam_mid``,
        ``n``, ``poly_vs_wave_mad_cm_s``.
    lfc_meta
        Dict from :func:`~hispecdrp.simulations.e2e_pipeline.build_lfc_line_list`
        (at least ``nu0``, ``delta_nu``).

    Returns
    -------
    summary
        The JSON-serializable summary dict that was written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / pdf_name
    json_path = out_dir / json_name

    ok = np.asarray(success_mask, dtype=bool)
    ods = sorted(int(o) for o in per_order)
    mads = np.asarray([per_order[o]["mad"] for o in ods], dtype=float)
    biases = np.asarray([per_order[o]["bias"] for o in ods], dtype=float)
    poly_mads = np.asarray(
        [per_order[o]["poly_vs_wave_mad_cm_s"] for o in ods], dtype=float
    )
    n_per = list(n_modes_per_order)
    med_modes = int(np.median(n_per)) if n_per else 0

    with PdfPages(pdf_path) as pdf:
        # --- Profile recovery page ---
        # Each panel is a histogram of (fit − truth) for one elliptical-Gaussian
        # parameter over successful spot fits.  N = number of spots in each bin.
        fig, axes = plt.subplots(2, 3, figsize=(11, 8.2))
        fig.suptitle(
            f"2D elliptical-Gaussian spot recovery (fit − truth)\n"
            f"injected a={a_pix:.2f} pix, b={b_pix:.2f} pix, PA={pa_deg:.0f}°  |  "
            f"N_ok={int(ok.sum())} spots",
            fontsize=11,
        )
        panels = [
            (
                axes[0, 0],
                tbl["drow_pix"][ok],
                "Δrow = row_fit − row_true (pix)",
                0.05,
                "Spectral-row centroid error",
            ),
            (
                axes[0, 1],
                tbl["dcol_pix"][ok],
                "Δcol = col_fit − col_true (pix)",
                0.05,
                "Cross-dispersion centroid error",
            ),
            (
                axes[0, 2],
                np.rad2deg(tbl["dpa_rad"][ok]),
                "ΔPA = PA_fit − PA_true (deg)",
                2.0,
                "Ellipse position-angle error",
            ),
            (
                axes[1, 0],
                tbl["da_pix"][ok],
                "Δa = a_fit − a_true (pix)",
                0.05,
                "Semimajor (1-σ) axis error",
            ),
            (
                axes[1, 1],
                tbl["db_pix"][ok],
                "Δb = b_fit − b_true (pix)",
                0.05,
                "Semiminor (1-σ) axis error",
            ),
            (
                axes[1, 2],
                tbl["residual_rms"][ok] / np.maximum(tbl["amp_fit"][ok], 1e-30),
                "fit residual RMS / peak amplitude",
                1.0e-3,
                "Relative stamp fit residual",
            ),
        ]
        for ax, x, xlabel, min_half, title in panels:
            x = np.asarray(x, dtype=float)
            x = x[np.isfinite(x)]
            xlim = _symmetric_hist_xlim(x, min_half_span=min_half)
            ax.hist(x, bins=40, range=xlim, color="C0", alpha=0.85)
            ax.axvline(0, color="k", lw=0.8, label="truth")
            ax.axvline(np.median(x), color="C3", ls="--", lw=1.2, label="median")
            ax.set_xlim(xlim)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel("N spots")
            ax.set_title(
                f"{title}\nmed={np.median(x):+.3g}  MAD={_mad(x):.3g}",
                fontsize=9,
            )
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.25)
        fig.text(
            0.5,
            0.01,
            "Truth = HDF deposit (row, col) and injected ellipse (a, b, PA).  "
            "Black = 0 (perfect recovery); red dashed = sample median.  "
            "Axes use a minimum span so near-zero residuals are not over-zoomed.",
            ha="center",
            va="bottom",
            fontsize=8,
            wrap=True,
        )
        fig.tight_layout(rect=[0, 0.04, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        # --- WAVE summary page ---
        fig = plt.figure(figsize=(11, 8.5))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.35, wspace=0.28)

        ax0 = fig.add_subplot(gs[0, :])
        ax0.bar(np.arange(len(ods)), mads, color="C0", alpha=0.85, label="WAVE@row_fit MAD")
        ax0.plot(
            np.arange(len(ods)), poly_mads, "o-", ms=3, color="C1", label="poly vs WAVE MAD"
        )
        ax0.axhline(
            np.median(mads),
            color="C3",
            ls="--",
            lw=1.5,
            label=f"median MAD = {np.median(mads):.3f} cm/s",
        )
        ax0.set_xticks(np.arange(len(ods)), [str(o) for o in ods], rotation=90, fontsize=7)
        ax0.set_ylabel("velocity MAD (cm/s)")
        ax0.set_xlabel("echelle order")
        ax0.set_title(
            f"{spectrograph} fiber {fiber} ({fiber_object}): WAVE(row_fit) vs λ_inj  |  "
            f"elliptical Gauss fit  |  ~{med_modes} modes/order"
        )
        ax0.legend(loc="upper right", fontsize=8)
        ax0.set_ylim(
            0, max(0.25, 1.3 * float(np.nanmax(np.concatenate([mads, poly_mads]))))
        )
        ax0.grid(True, axis="y", alpha=0.3)

        all_dv = np.concatenate(
            [np.asarray(per_order[o]["dvel"], float) - per_order[o]["bias"] for o in ods]
        )
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.hist(all_dv, bins=60, color="C0", alpha=0.85)
        ax1.axvline(0, color="k", lw=1)
        ax1.set_xlabel("ZP-subtracted residual (cm/s)")
        ax1.set_ylabel("N lines")
        ax1.set_title(f"All lines (N={all_dv.size})")
        ax1.text(
            0.98,
            0.95,
            f"median MAD/order = {np.median(mads):.3f} cm/s\n"
            f"global MAD = {np.median(np.abs(all_dv)):.3f} cm/s",
            transform=ax1.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        ax2 = fig.add_subplot(gs[1, 1])
        ax2.plot(ods, biases, "o-", ms=4, color="C1")
        ax2.axhline(0, color="k", lw=0.8)
        ax2.axhline(
            np.median(biases),
            color="C3",
            ls="--",
            label=f"median bias = {np.median(biases):.3f} cm/s",
        )
        ax2.set_xlabel("echelle order")
        ax2.set_ylabel("median Δv (cm/s)")
        ax2.set_title("Per-order zero-point (WAVE@row_fit)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        pdf.savefig(fig)
        plt.close(fig)

        # Per-order pages (3 orders each)
        for i0 in range(0, len(ods), 3):
            chunk = ods[i0 : i0 + 3]
            fig, axes = plt.subplots(
                len(chunk),
                2,
                figsize=(11, 3.4 * len(chunk) + 0.8),
                squeeze=False,
                gridspec_kw={"wspace": 0.35, "hspace": 0.65},
            )
            fig.suptitle(
                "LFC elliptical fit: WAVE(row_fit) vs λ_inj",
                fontsize=11,
                y=0.995,
            )
            for row, od in enumerate(chunk):
                d = per_order[od]
                dv = np.asarray(d["dvel"], float)
                dv0 = dv - d["bias"]
                lam = np.asarray(d["lam_inj"], float)

                axL = axes[row, 0]
                axL.axhline(0, color="0.5", lw=0.9)
                axL.plot(lam, dv0, ".", ms=5, color="C0", alpha=0.85, label="ZP-subtracted")
                axL.plot(lam, dv, ".", ms=3.5, color="C1", alpha=0.45, label="raw (incl. ZP)")
                axL.set_ylabel("Δv (cm/s)")
                axL.set_xlabel("λ_inj (µm)")
                axL.set_title(
                    f"Order {od}  ⟨λ⟩={d['lam_mid']:.4f} µm  "
                    f"MAD={d['mad']:.3f} cm/s  bias={d['bias']:.3f} cm/s  n={d['n']}\n"
                    f"poly vs WAVE MAD={d['poly_vs_wave_mad_cm_s']:.3f} cm/s",
                    fontsize=8.5,
                    pad=8,
                )
                axL.legend(fontsize=7)
                axL.grid(True, alpha=0.25)
                span = max(0.5, 5 * d["mad"], float(np.percentile(np.abs(dv0), 95)))
                axL.set_ylim(-span, span)

                axR = axes[row, 1]
                lam_ref = float(np.median(lam))
                v_inj = (
                    np.asarray(delta_wave_to_delta_vel(lam - lam_ref, lam_ref), float)
                    * 100.0
                )
                v_rec = (
                    np.asarray(
                        delta_wave_to_delta_vel(
                            np.asarray(d["lam_rec"], float) - lam_ref, lam_ref
                        ),
                        float,
                    )
                    * 100.0
                )
                s = np.argsort(v_inj)
                axR.plot(v_inj[s], v_inj[s], "-", color="0.4", lw=1.0, label="1:1")
                axR.plot(
                    v_inj, v_rec, ".", ms=5, color="C0", alpha=0.85, label="WAVE@row_fit"
                )
                axR.set_xlabel("v(λ_inj) − v(⟨λ⟩) (cm/s)")
                axR.set_ylabel("v(λ_WAVE) − v(⟨λ⟩) (cm/s)")
                axR.set_title(f"Order {od}: expected vs recovered velocity", fontsize=8.5, pad=8)
                axR.legend(fontsize=7)
                axR.grid(True, alpha=0.25)
                axR.set_aspect("equal", adjustable="datalim")

            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig)
            plt.close(fig)

        pdf.infodict()["Title"] = "LFC elliptical-Gaussian WAVE residuals"

    summary: dict[str, Any] = {
        "spectrograph": spectrograph,
        "fiber": int(fiber),
        "fiber_object": fiber_object,
        "lfc_nu0_hz": float(lfc_meta["nu0"]),
        "lfc_delta_nu_hz": float(lfc_meta["delta_nu"]),
        "mode_stride": int(mode_stride),
        "n_modes_total": int(n_modes_total),
        "n_modes_per_order_median": med_modes,
        "n_orders": len(ods),
        "a_pix": float(a_pix),
        "b_pix": float(b_pix),
        "pa_deg": float(pa_deg),
        "n_success": int(ok.sum()),
        "centroid_drow_mad_pix": _mad(tbl["drow_pix"][ok]),
        "centroid_dcol_mad_pix": _mad(tbl["dcol_pix"][ok]),
        "profile_da_mad_pix": _mad(tbl["da_pix"][ok]),
        "profile_db_mad_pix": _mad(tbl["db_pix"][ok]),
        "profile_dpa_mad_deg": _mad(np.rad2deg(tbl["dpa_rad"][ok])),
        "median_mad_cm_s": float(np.median(mads)),
        "median_bias_cm_s": float(np.median(biases)),
        "global_mad_cm_s": float(np.median(np.abs(all_dv))),
        "poly_vs_wave_median_mad_cm_s": float(np.nanmedian(poly_mads)),
        "pdf": str(pdf_path),
        "fits_table": str(fits_table_path),
        "rate_map": str(rate_path),
        "per_order_mad_cm_s": {str(o): float(per_order[o]["mad"]) for o in ods},
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary

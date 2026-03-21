"""Plotly preview for stellar RV (per-chunk RV + CCF panels)."""
from __future__ import annotations

# Provenance: HISPEC DRP original (Plotly graph_objects/subplots). CCF fits from ccf_utils.

import logging
import os
import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils.ccf_utils import (
    _V_MAX,
    _V_MIN,
    _gaussian_ccf_model,
    fit_ccf_rv,
)

logger = logging.getLogger(__name__)


def _iterative_sigma_clip_ylim(
    y: np.ndarray,
    *,
    sigma: float = 3.0,
    max_iter: int = 5,
    pad_frac: float = 0.05,
) -> tuple[float, float]:
    """
    Default y-axis limits for RV scatter: iterative sigma clipping on finite values,
    then symmetric-ish padding so the panel spans the \"good\" points.
    """

    v = np.asarray(y, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return -1.0, 1.0
    if v.size == 1:
        m = float(v[0])
        d = max(abs(m) * 1e-6, 1e-9)
        return m - d, m + d
    mask = np.ones(v.shape[0], dtype=bool)
    for _ in range(max_iter):
        if np.sum(mask) < 2:
            break
        mu = float(np.mean(v[mask]))
        sd = float(np.std(v[mask]))
        if not np.isfinite(sd) or sd <= 0:
            sd = max(abs(mu) * 1e-9, 1e-12)
        new_m = mask & (np.abs(v - mu) <= sigma * sd)
        if np.array_equal(new_m, mask):
            break
        mask = new_m
    vf = v[mask]
    if vf.size == 0:
        vf = v
    lo, hi = float(np.min(vf)), float(np.max(vf))
    span = hi - lo
    pad = pad_frac * span if span > 0 else max(abs(lo) * pad_frac, 1e-6)
    return lo - pad, hi + pad


def _preview_ccf_y_limits(ccf: np.ndarray) -> tuple[float, float]:
    """Y-limits for CCF panel: central ~99.9% of finite values plus pad."""
    c = np.asarray(ccf, dtype=np.float64).ravel()
    v = c[np.isfinite(c)]
    if v.size == 0:
        return -1.0, 1.0
    if v.size == 1:
        m = float(v[0])
        d = max(abs(m) * 1e-6, 1e-9)
        return m - d, m + d
    lo, hi = np.percentile(v, [0.05, 99.95])
    span = float(hi - lo)
    if not np.isfinite(span) or span <= 0:
        m = float(np.median(v))
        span = max(float(np.nanstd(v)), abs(m) * 1e-9, 1e-12)
        return m - span, m + span
    pad = 0.025 * span
    return float(lo - pad), float(hi + pad)


def _preview_abbrev_ccf_uncertainty(diag: dict) -> str:
    """Short legend text for how RV uncertainty was estimated (fit_ok only)."""
    if diag.get("used_kpf_ccf_error"):
        return "KPF ccf_error_calc"
    if diag.get("used_covariance_for_error"):
        return "LevMar cov. (mean)"
    if diag.get("used_sigma_fallback"):
        return "σ/√n (fallback)"
    return "—"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _preview_should_write_png(primitive) -> bool:
    """
    Whether to run Kaleido PNG export.

    PNG via Kaleido is often several seconds for large multi-panel figures; HTML
    is usually much faster.

    When env ``HISPEC_DRP_SHOW_CCF_PREVIEW`` is set (typical interactive preview
    runs), PNG export is disabled and only HTML is written when ``save_preview``
    is True.

    Otherwise, set ``preview_save_png=False`` on the primitive to skip PNG while
    still saving HTML.
    """
    if not getattr(primitive, "save_preview", False):
        return False
    if _env_flag("HISPEC_DRP_SHOW_CCF_PREVIEW"):
        return False
    return bool(getattr(primitive, "preview_save_png", True))


def create_stellar_rv_preview(primitive) -> None:
    """
    Plotly preview: for each fiber, three panels stacked vertically:
    1) RV vs wavelength (per-chunk with legend),
    2) CCF vs RV shift + Gaussian fit + metadata, with dropdown to select combined or per-order,
    3) Residual (CCF − Gaussian) vs RV shift (same dropdown selection).
    """
    t0 = time.perf_counter()
    out = primitive._output.get("output") if primitive._output else None
    if out is None:
        logger.warning("No output model for stellar RV preview.")
        return
    if isinstance(out, (list, tuple)):
        if not out:
            logger.warning("Empty output list for stellar RV preview.")
            return
        output_model = out[0]
    else:
        output_model = out

    fibers_with_ccf = []
    for f in ["1", "2", "3"]:
        tbl = getattr(output_model, f"rvccf_fib{f}", None)
        if tbl is None or not hasattr(tbl, "columns") or "RV" not in tbl.colnames:
            continue
        wl = np.asarray(tbl["WAVELENGTH"], dtype=np.float64)
        rv = np.asarray(tbl["RV"], dtype=np.float64)
        valid = np.isfinite(wl) & np.isfinite(rv)
        if not np.any(valid):
            continue
        fibers_with_ccf.append(f)

    if not fibers_with_ccf:
        logger.warning("No RVCCF tables found on output for stellar RV preview.")
        return

    # Three rows per fiber: RV vs λ, CCF (dropdown), Residual (dropdown)
    row_plan: list[tuple] = []
    for fiber in fibers_with_ccf:
        row_plan.append(("rv", fiber))
        row_plan.append(("ccf", fiber))
        row_plan.append(("resid", fiber))

    n_rows = len(row_plan)
    subplot_titles = []
    for entry in row_plan:
        kind, fiber = entry[0], entry[1]
        if kind == "rv":
            subplot_titles.append(f"Fiber {fiber} — per-chunk RV")
        elif kind == "ccf":
            subplot_titles.append(f"Fiber {fiber} — CCF + Gaussian fit (select order below)")
        else:
            subplot_titles.append(f"Fiber {fiber} — CCF residual (data − Gaussian)")

    # More spacing so RV plot x-axis sits clearly above the CCF panel
    v_spacing = max(0.06, min(0.12, 1.8 / max(n_rows, 8)))
    if n_rows > 1:
        v_spacing = min(v_spacing, 1.0 / (n_rows - 1))
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=v_spacing,
    )

    # Precompute Gaussian fits and metadata text per (fiber, slot)
    fit_slot_cache: dict[int, tuple] = {}
    slot_metadata: dict[str, list[str]] = {}
    for fiber in fibers_with_ccf:
        detail = primitive._ccf_preview_by_fiber.get(fiber)
        if not detail:
            slot_metadata[fiber] = []
            continue
        v_step = float(detail["v_step"])
        slots = detail["slots"]
        slot_metadata[fiber] = []
        for sl in slots:
            sid = id(sl)
            if sid not in fit_slot_cache:
                kpf = sl.get("kpf_fit")
                if kpf and isinstance(kpf, dict) and kpf.get("params"):
                    params = kpf.get("params") or {}
                    diag = kpf.get("diagnostics") or {}
                    rv_err = float(kpf.get("rv_err", np.nan))
                    fit_slot_cache[sid] = (params, diag, rv_err)
                else:
                    _rv, _re, params, diag = fit_ccf_rv(
                        sl["v"], sl["ccf"], v_step, return_params=True
                    )
                    rv_err = float(_re) if _re is not None and np.isfinite(_re) else np.nan
                    fit_slot_cache[sid] = (params, diag, rv_err)
            params, diag, rv_err = fit_slot_cache[sid]
            # Build dynamic CCF legend: compact; σ then line break then amp (small font)
            if sl.get("kind") == "mean":
                meta = "<b>Mean</b> · all orders"
            else:
                o = sl.get("order", "?")
                wl_lo, wl_hi = sl.get("wl_lo", np.nan), sl.get("wl_hi", np.nan)
                if np.isfinite(wl_lo) and np.isfinite(wl_hi):
                    meta = f"<b>Ord {o}</b> · λ {wl_lo:.4f}–{wl_hi:.4f} µm"
                else:
                    meta = f"<b>Ord {o}</b>"
            if params and diag.get("fit_ok"):
                v0 = float(params["v0"])
                sig = float(params["sigma"])
                amp = float(params["amplitude"])
                meta += (
                    f"<br><b>Fit</b>: RV = {v0/1000:.4f} km/s · "
                    f"σ = {sig/1000:.4f} km/s"
                    f"<br>amp = {amp:.4g}"
                )
                unc = "—"
                if np.isfinite(rv_err):
                    meta += f" · RV err = {rv_err:.1f} m/s"
                    unc = _preview_abbrev_ccf_uncertainty(diag)
                meta += f"<br><b>Unc.</b> {unc}"
            else:
                meta += "<br><i>Fit failed</i>"
                fail_msg = (diag.get("message") or "").strip()
                if fail_msg:
                    tail = fail_msg[:56] + "…" if len(fail_msg) > 56 else fail_msg
                    meta += f" · {tail}"
            slot_metadata[fiber].append(meta)

    rv_panel_y_title: dict[str, str] = {}
    rv_panel_y_range: dict[str, tuple[float, float]] = {}
    ccf_trace_indices: dict[str, list[tuple[int, int]]] = {}
    resid_trace_indices: dict[str, list[int]] = {}

    for ri, entry in enumerate(row_plan):
        row = ri + 1
        kind, fiber = entry[0], entry[1]

        if kind == "rv":
            tbl = getattr(output_model, f"rvccf_fib{fiber}", None)
            avg_tbl = getattr(output_model, f"rvccfavg_fib{fiber}", None)
            wl = np.asarray(tbl["WAVELENGTH"], dtype=np.float64)
            rv = np.asarray(tbl["RV"], dtype=np.float64)
            rv_err = np.asarray(tbl["RV_ERR"], dtype=np.float64)
            inlier = (
                np.asarray(tbl["INLIER"], dtype=bool)
                if "INLIER" in tbl.colnames
                else np.ones(len(rv), dtype=bool)
            )
            valid = np.isfinite(wl) & np.isfinite(rv)
            if not np.any(valid):
                continue
            wl, rv = wl[valid], rv[valid]
            rv_err = rv_err[valid]
            inlier = inlier[valid]
            if avg_tbl is not None and len(avg_tbl) > 0 and np.isfinite(float(avg_tbl["RV"][0])):
                rv_agg = float(avg_tbl["RV"][0])
                rv_agg_err = float(avg_tbl["RV_ERR"][0])
            else:
                rv_agg = float(np.nanmean(rv))
                rv_agg_err = float(np.nanmedian(rv_err[np.isfinite(rv_err) & (rv_err > 0)]))
                if not np.isfinite(rv_agg_err):
                    rv_agg_err = np.nan
            rv_mean_cm = rv_agg * 100.0
            rv_mean_cm_r = round(rv_mean_cm, 1)
            y_cm_off = rv * 100.0 - rv_mean_cm
            err_y_cm = rv_err * 100.0
            if rv_mean_cm_r < 0:
                y_title = f"RV (cm/s) + {abs(rv_mean_cm_r):.4e}"
            else:
                y_title = f"RV (cm/s) − {rv_mean_cm_r:.4e}"
            rv_panel_y_title[fiber] = y_title
            if "ORDER" in tbl.colnames:
                order_col = np.asarray(tbl["ORDER"], dtype=np.int64)
            else:
                order_col = np.full(len(wl), -1, dtype=np.int64)
            ord_lbl = np.where(order_col >= 0, order_col.astype(str), "—")
            in_ok = inlier & np.isfinite(rv_err) & (rv_err > 0)
            out_ok = ~inlier & np.isfinite(rv_err) & (rv_err > 0)
            if np.any(in_ok):
                y_clip_src = y_cm_off[in_ok]
            else:
                y_clip_src = y_cm_off[np.isfinite(y_cm_off)]
            if y_clip_src.size:
                rv_panel_y_range[fiber] = _iterative_sigma_clip_ylim(y_clip_src)
            if np.any(in_ok):
                fig.add_trace(
                    go.Scatter(
                        x=wl[in_ok],
                        y=y_cm_off[in_ok],
                        error_y=dict(type="data", array=err_y_cm[in_ok], visible=True),
                        mode="markers",
                        name=f"Fiber {fiber} (inlier)",
                        marker=dict(size=6, color="royalblue"),
                        customdata=ord_lbl[in_ok].reshape(-1, 1),
                        hovertemplate=(
                            "λ = %{x:.5f} µm<br>"
                            "ΔRV = %{y:.2f} ± %{error_y.array:.2f} cm/s<br>"
                            "Order %{customdata[0]}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=1,
                )
            if np.any(out_ok):
                fig.add_trace(
                    go.Scatter(
                        x=wl[out_ok],
                        y=y_cm_off[out_ok],
                        error_y=dict(type="data", array=err_y_cm[out_ok], visible=True),
                        mode="markers",
                        name=f"Fiber {fiber} (outlier)",
                        marker=dict(size=5, color="grey", symbol="circle-open"),
                        customdata=ord_lbl[out_ok].reshape(-1, 1),
                        hovertemplate=(
                            "λ = %{x:.5f} µm<br>"
                            "ΔRV = %{y:.2f} ± %{error_y.array:.2f} cm/s<br>"
                            "Order %{customdata[0]}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=1,
                )
            if not np.any(in_ok) and np.any(valid):
                fig.add_trace(
                    go.Scatter(
                        x=wl,
                        y=y_cm_off,
                        error_y=dict(type="data", array=err_y_cm, visible=True),
                        mode="markers",
                        name=f"Fiber {fiber} chunks",
                        marker=dict(size=6, color="royalblue"),
                        customdata=ord_lbl.reshape(-1, 1),
                        hovertemplate=(
                            "λ = %{x:.5f} µm<br>"
                            "ΔRV = %{y:.2f} ± %{error_y.array:.2f} cm/s<br>"
                            "Order %{customdata[0]}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=1,
                )
            x_range = [float(np.nanmin(wl)), float(np.nanmax(wl))] if len(wl) else [0.0, 1.0]
            rv_kms = rv_agg / 1000.0
            mean_line = f"⟨RV⟩ = {rv_kms:.6f} km/s"
            if np.isfinite(rv_agg_err):
                unc_cm_disp = round(float(rv_agg_err) * 100.0, 1)
                leg = f"{mean_line}<br>uncertainty = {unc_cm_disp:.1f} cm/s"
            else:
                leg = f"{mean_line}<br>uncertainty = —"
            # RMS bias from mean: sqrt(sum of squared biases over good points / n_good)
            n_good = np.sum(inlier)
            if n_good > 0:
                biases_m_s = rv[inlier] - rv_agg
                rms_bias_m_s = float(np.sqrt(np.sum(biases_m_s**2) / n_good))
                rms_bias_cm_s = rms_bias_m_s * 100.0
                leg += f"<br>RMS bias = {rms_bias_cm_s:.1f} cm/s"
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=[0.0, 0.0],
                    mode="lines",
                    line=dict(dash="dash", color="crimson", width=2),
                    name=leg,
                ),
                row=row,
                col=1,
            )
            if np.isfinite(rv_agg_err) and rv_agg_err > 0:
                sig_cm = float(rv_agg_err) * 100.0
                for y_val, label in [(sig_cm, "+1σ (cm/s)"), (-sig_cm, "-1σ (cm/s)")]:
                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=[y_val, y_val],
                            mode="lines",
                            line=dict(dash="dot", color="crimson", width=1),
                            name=label,
                        ),
                        row=row,
                        col=1,
                    )
            continue

        if kind == "ccf":
            detail = primitive._ccf_preview_by_fiber.get(fiber)
            if not detail:
                continue
            v_step = float(detail["v_step"])
            slots = detail["slots"]
            ccf_trace_indices[fiber] = []
            first_ccf_legend = True
            ccf_ylim_lo, ccf_ylim_hi = -1.0, 1.0
            for si, sl in enumerate(slots):
                params, diag = fit_slot_cache.get(id(sl), (None, {}, np.nan))[:2]
                vg = np.asarray(sl["v"], dtype=np.float64)
                vg_km = vg / 1000.0  # x-axis in km/s
                ccf = np.asarray(sl["ccf"], dtype=np.float64)
                visible = si == 0
                idx_data = len(fig.data)
                fig.add_trace(
                    go.Scatter(
                        x=vg_km,
                        y=ccf,
                        mode="markers",
                        name="CCF data",
                        marker=dict(size=3, color="darkgreen"),
                        showlegend=first_ccf_legend,
                        visible=visible,
                    ),
                    row=row,
                    col=1,
                )
                idx_fit = len(fig.data)
                if params and diag.get("fit_ok"):
                    v_lo = float(diag.get("fit_v_lo", np.nanmin(vg)))
                    v_hi = float(diag.get("fit_v_hi", np.nanmax(vg)))
                    vf = np.linspace(v_lo, v_hi, max(50, int((v_hi - v_lo) / max(v_step, 1.0))))
                    vf_km = vf / 1000.0  # x-axis in km/s
                    cf = _gaussian_ccf_model(
                        vf,
                        float(params["amplitude"]),
                        float(params["v0"]),
                        float(params["sigma"]),
                        float(params["baseline"]),
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=vf_km,
                            y=cf,
                            mode="lines",
                            name="Gaussian fit",
                            line=dict(width=2, color="crimson"),
                            showlegend=first_ccf_legend,
                            visible=visible,
                        ),
                        row=row,
                        col=1,
                    )
                else:
                    fig.add_trace(
                        go.Scatter(x=[], y=[], showlegend=False, visible=visible),
                        row=row,
                        col=1,
                    )
                ccf_trace_indices[fiber].append((idx_data, idx_fit))
                first_ccf_legend = False
                if si == 0:
                        ccf_ylim_lo, ccf_ylim_hi = _preview_ccf_y_limits(ccf)
            fig.update_yaxes(range=[ccf_ylim_lo, ccf_ylim_hi], autorange=False, row=row, col=1)
            continue

        # resid
        detail = primitive._ccf_preview_by_fiber.get(fiber)
        if not detail:
            continue
        slots = detail["slots"]
        resid_trace_indices[fiber] = []
        for si, sl in enumerate(slots):
            params, diag = fit_slot_cache.get(id(sl), (None, {}, np.nan))[:2]
            vg = np.asarray(sl["v"], dtype=np.float64)
            vg_km = vg / 1000.0  # x-axis in km/s
            ccf = np.asarray(sl["ccf"], dtype=np.float64)
            visible = si == 0
            if params and diag.get("fit_ok"):
                model_at = _gaussian_ccf_model(
                    vg,
                    float(params["amplitude"]),
                    float(params["v0"]),
                    float(params["sigma"]),
                    float(params["baseline"]),
                )
                residual = ccf - model_at
            else:
                residual = np.full_like(ccf, np.nan, dtype=np.float64)
            fig.add_trace(
                go.Scatter(
                    x=vg_km,
                    y=residual,
                    mode="lines",
                    name="CCF residual",
                    line=dict(width=1.0, color="darkslateblue"),
                    showlegend=False,
                    visible=visible,
                ),
                row=row,
                col=1,
            )
            resid_trace_indices[fiber].append(len(fig.data) - 1)
        fig.add_hline(
            y=0.0,
            line_dash="dot",
            line_color="gray",
            line_width=1,
            row=row,
            col=1,
        )
        # Share x-axis with CCF row above (velocity in km/s)
        fig.update_xaxes(
            range=[_V_MIN / 1000.0, _V_MAX / 1000.0],
            matches=f"x{row - 1}",
            row=row,
            col=1,
        )

    n_traces = len(fig.data)

    # Dropdowns: one per fiber to select combined (mean) or individual order
    # Build initial annotations (one per fiber, for CCF panel) — use domain refs so (x,y) are 0–1 in panel
    annotations: list[dict] = []
    for fi, fiber in enumerate(fibers_with_ccf):
        ccf_row = 3 * fi + 2
        xref = f"x{ccf_row} domain"
        yref = f"y{ccf_row} domain"
        text = slot_metadata.get(fiber, [""])[0] if slot_metadata.get(fiber) else ""
        annotations.append(
            dict(
                x=0.98,
                y=0.98,
                xref=xref,
                yref=yref,
                text=text,
                showarrow=False,
                xanchor="right",
                yanchor="top",
                font=dict(size=8),
                bgcolor="rgba(255,255,255,0.9)",
                borderpad=4,
            )
        )

    updatemenus = []
    for fi, fiber in enumerate(fibers_with_ccf):
        slots = primitive._ccf_preview_by_fiber.get(fiber, {}).get("slots", [])
        if not slots:
            continue
        ccf_indices = ccf_trace_indices.get(fiber, [])
        res_indices = resid_trace_indices.get(fiber, [])
        meta_list = slot_metadata.get(fiber, [])
        buttons = []
        for si in range(len(slots)):
            sl = slots[si]
            if sl.get("kind") == "mean":
                label = "Combined (mean)"
            else:
                o = sl.get("order", "?")
                wl_lo, wl_hi = sl.get("wl_lo", np.nan), sl.get("wl_hi", np.nan)
                if np.isfinite(wl_lo) and np.isfinite(wl_hi):
                    label = f"Order {o} ({wl_lo:.3f}-{wl_hi:.3f} μm)"
                else:
                    label = f"Order {o}"
            visible = [True] * n_traces
            for slot_idx in range(len(slots)):
                if slot_idx != si:
                    if slot_idx < len(ccf_indices):
                        visible[ccf_indices[slot_idx][0]] = False
                        visible[ccf_indices[slot_idx][1]] = False
                    if slot_idx < len(res_indices):
                        visible[res_indices[slot_idx]] = False
            new_annotations = [dict(a) for a in annotations]
            if fi < len(new_annotations) and si < len(meta_list):
                new_annotations[fi]["text"] = meta_list[si]
            buttons.append(
                dict(
                    label=label,
                    method="update",
                    args=[
                        {"visible": visible},
                        {"annotations": new_annotations},
                    ],
                )
            )
        if not buttons:
            continue
        # Position dropdown to the right of this fiber's CCF panel (paper coords)
        ccf_row_1based = 3 * fi + 2
        row_height = (1.0 - (n_rows - 1) * v_spacing) / n_rows
        from_top = (ccf_row_1based - 0.5) * row_height + (ccf_row_1based - 1) * v_spacing
        y_dropdown = 1.0 - from_top
        updatemenus.append(
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                showactive=True,
                x=1.035,
                xanchor="left",
                y=y_dropdown,
                yanchor="middle",
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.9)",
                borderwidth=1,
                bordercolor="gray",
            )
        )

    algo = getattr(primitive, "algorithm", None) or "ccf_median_mad"
    title = f"Stellar RV — algorithm: {algo}"
    if hasattr(output_model, "meta") and getattr(output_model.meta, "filename", None):
        title += f" — {output_model.meta.filename}"
    fig_h = int(min(20000, 340 + max(0, n_rows - 1) * 145))
    fig.update_layout(
        height=fig_h,
        width=920,
        title_text=title,
        showlegend=True,
        legend=dict(
            x=1.01,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            borderwidth=0,
            font=dict(size=9),
            tracegroupgap=6,
        ),
        margin=dict(l=70, r=300, t=80, b=50),
        annotations=annotations,
        updatemenus=updatemenus,
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    for ri, entry in enumerate(row_plan):
        row = ri + 1
        kind, fiber = entry[0], entry[1]
        if kind == "rv":
            fig.update_xaxes(title_text="Wavelength (µm)", row=row, col=1)
            y_kwargs: dict = {
                "title_text": rv_panel_y_title.get(fiber, "RV (cm/s) − ⟨RV⟩"),
            }
            if fiber in rv_panel_y_range:
                y_kwargs["range"] = list(rv_panel_y_range[fiber])
                y_kwargs["autorange"] = False
            fig.update_yaxes(row=row, col=1, **y_kwargs)
        elif kind == "ccf":
            fig.update_xaxes(
                range=[_V_MIN / 1000.0, _V_MAX / 1000.0],
                showticklabels=False,
                row=row,
                col=1,
            )
            fig.update_yaxes(
                title_text="CCF (flux²)",
                row=row,
                col=1,
            )
        else:
            fig.update_xaxes(
                title_text="Velocity shift (km/s)",
                range=[_V_MIN / 1000.0, _V_MAX / 1000.0],
                row=row,
                col=1,
            )
            fig.update_yaxes(title_text="CCF residual", row=row, col=1)

    t_after_figure = time.perf_counter()
    logger.info(
        "PREVIEW_TIMING create_stellar_rv_preview figure_build (traces+layout) wall_s=%.3f",
        t_after_figure - t0,
    )

    # get_output_path requires output_dir or filename; use cwd when only showing preview
    output_dir = getattr(primitive, "output_dir", None) or os.getcwd()
    base_path = output_model.get_output_path(
        output_dir=output_dir, suffix=primitive.suffix
    )
    output_path_html = base_path.replace(".fits", "_preview.html")
    output_path_png = base_path.replace(".fits", "_preview.png")
    wrote_png = False

    t_save0 = time.perf_counter()
    if primitive.save_preview:
        t_html = time.perf_counter()
        fig.write_html(output_path_html, include_plotlyjs="cdn")
        logger.info(
            "PREVIEW_TIMING write_html wall_s=%.3f",
            time.perf_counter() - t_html,
        )
        if _preview_should_write_png(primitive):
            t_png = time.perf_counter()
            try:
                # Use Kaleido v1 API directly so Plotly's write_image(engine=...) path is avoided.
                import kaleido

                scale = float(getattr(primitive, "preview_png_scale", 2.0))
                scale = max(0.25, min(scale, 8.0))
                kaleido.write_fig_sync(fig, path=output_path_png, opts={"scale": scale})
                wrote_png = True
                logger.info(
                    "PREVIEW_TIMING kaleido_png scale=%.2f wall_s=%.3f",
                    scale,
                    time.perf_counter() - t_png,
                )
            except Exception as e:
                logger.debug(
                    "Could not write preview PNG (kaleido/Chrome may be missing): %s", e
                )
        else:
            if _env_flag("HISPEC_DRP_SHOW_CCF_PREVIEW"):
                skip_reason = "HISPEC_DRP_SHOW_CCF_PREVIEW (HTML-only preview)"
            elif not getattr(primitive, "preview_save_png", True):
                skip_reason = "preview_save_png=False"
            else:
                skip_reason = "disabled"
            logger.info(
                "PREVIEW_TIMING kaleido_png skipped (%s) wall_s=0.000",
                skip_reason,
            )
    if primitive.save_preview:
        logger.info(
            "PREVIEW_TIMING save_preview_total (html +/- png) wall_s=%.3f",
            time.perf_counter() - t_save0,
        )
    t_show0 = time.perf_counter()
    if primitive.show_preview:
        logger.info("Showing stellar RV preview...")
        fig.show()
    if primitive.show_preview:
        logger.info(
            "PREVIEW_TIMING fig.show wall_s=%.3f", time.perf_counter() - t_show0
        )
    logger.info(
        "PREVIEW_TIMING create_stellar_rv_preview total wall_s=%.3f",
        time.perf_counter() - t0,
    )

    primitive._output["preview"] = {
        "html": output_path_html,
        "png": output_path_png if wrote_png else None,
        "fig": fig,
    }

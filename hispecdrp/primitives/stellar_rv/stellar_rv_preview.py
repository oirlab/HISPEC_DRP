"""
Interactive previews for stellar radial-velocity products.

There are **two** preview surfaces:

1. **Default preview** (``create_stellar_rv_preview``): **two** Plotly HTML embeds per save — (a) RV vs
   λ per fiber with its own mode bar; (b) CCF + residual per fiber (linked velocity axes within each
   fiber), second mode bar, slot dropdown. For ``ccf_mask_kpf`` with
   mask-overlay payload, the top row is the same spectrum/mask content as the standalone
   ``*_mask_overlay_debug_fib*.html`` (fixed ``mask_overlay_velocity_m_s``; order dropdown uses
   ``restyle``). CCF slot dropdown + annotations target the middle and bottom rows only.
2. **Mask-shift debug HTML** (``write_mask_overlay_debug_pages``): per-order spectrum with
   mask segment overlay and velocity slider, for ``algorithm="ccf_mask_kpf"`` only. When the
   pipeline built ``_ccf_preview_by_fiber`` (preview and/or mask-overlay flags), each order
   can include ``ccf_v_m_s`` and ``ccf_err``: weighted SD of per-line ``F*SN`` values about the
   order-mean mask CCF (``ccf_utils._calc_mask_ccf_chunks_numba``), interpolated onto the
   interactive CCF curve. Payload arrays are **downsampled** (uniform stride) before embedding
   in HTML so the file stays small and the browser parses quickly; limits are set by
   ``mask_overlay_max_spectrum_points`` / ``mask_overlay_max_ccf_points`` or env
   ``HISPEC_DRP_MASK_OVERLAY_SPECTRUM_MAX_POINTS`` / ``HISPEC_DRP_MASK_OVERLAY_CCF_MAX_POINTS``.

Gaussian reference curves use :func:`~hispecdrp.primitives.utils.ccf_utils.fit_ccf_rv`;
mask geometry uses :mod:`~hispecdrp.primitives.utils.ccf_utils` (mask loaders).

**Entry points.** :func:`create_stellar_rv_preview` builds the standard multi-panel HTML;
:func:`write_mask_overlay_debug_pages` writes interactive mask-overlay debug pages;
:func:`collect_mask_overlay_payload_for_primitive` and related helpers assemble mask/spectrum
payloads from ``ccf_utils`` and the L2 model. Environment variables
``HISPEC_DRP_MASK_OVERLAY_*`` and ``HISPEC_DRP_SHOW_CCF_PREVIEW`` adjust downsampling and
browser behavior (see in-file constants and call sites).

Notes
-----
Plotly layout code is **HISPEC DRP–original**. For mask file format and upstream
line lists, see ``hispecdrp/data/masks/README.md`` and spirou-ccf/KPF references
in :mod:`~hispecdrp.primitives.utils.ccf_utils`.

References
----------
.. [1] spirou-ccf masks: https://github.com/edermartioli/spirou-ccf/tree/main/ccf_masks
.. [2] KPF radial_velocity ``src``: https://github.com/Keck-DataReductionPipelines/KPF-Pipeline/tree/81609b0ac0ab6d12536fb771bfdd0112001340c3/modules/radial_velocity/src
"""
from __future__ import annotations

import html as html_module
import io
import json
import logging
import math
import os
import tempfile
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from ..utils.ccf_utils import (
    CCFFitKpfConfig,
    _gaussian_ccf_model,
    _mask_line_order_edge_taper_multipliers,
    build_hybrid_merged_velocity_grid,
    clip_gaussian_plot_velocity_range_m_s,
    combine_rv_estimates_median_mad_weighted,
    combine_rvs_median_mad,
    effective_ccf_v_step_for_fit,
    fit_ccf_rv,
    load_mask,
    refine_ccf_rv_coarse_grid_parabolic,
    resolve_ccf_broad_grid_m_s,
    resolve_mask_path,
    restrict_mask_to_chunk,
)

logger = logging.getLogger(__name__)

# Speed of light (m/s); mask wavelength shift matches overlay HTML / ``collect_mask_overlay_payload``.
C_M_S = 299792458.0

# Plotly CCF scatter/residual traces: cap points to keep preview HTML small and write_html fast.
PREVIEW_CCF_SCATTER_MAX_POINTS = 3500

# RV vs λ: ``xref="paper"`` is the axes **inner** box (0–1), not the full figure. Anchor at ``x=1``
# and shift right by this many pixels so “Derived” / “Plot key” sit just outside the frame in the margin.
RV_SIDEBAR_XSHIFT_PX = 32

# Preview HTML: wheel-zoom at cursor; no Plotly logo. RV figure: independent axes. CCF figure:
# CCF and residual rows share velocity zoom/pan via ``matches`` on the residual x-axis.
PREVIEW_PLOTLY_HTML_CONFIG: dict = {
    "scrollZoom": True,
    "displaylogo": False,
    "responsive": True,
}


def _plotly_subplot_x_anchor_name(row_1based: int) -> str:
    """Plotly x-axis anchor for subplot row ``row_1based`` (1-based): ``x``, ``x2``, …"""
    r = int(row_1based)
    return "x" if r <= 1 else f"x{r}"


def _plotly_domain_axis_refs(row_1based: int) -> tuple[str, str]:
    """
    Return (xref, yref) for annotation coordinates in subplot ``row_1based`` (1-based).

    Plotly allows ``x domain`` / ``y domain`` for the **first** row only; ``x1 domain`` is invalid
    (enumerated xref pattern starts at bare ``x`` or ``x2``, …).
    """
    r = int(row_1based)
    if r <= 1:
        return "x domain", "y domain"
    return f"x{r} domain", f"y{r} domain"


def _rv_sidebar_paper_xy(fig: go.Figure, row_1based: int, y_frac: float) -> tuple[float, float]:
    """
    Paper coordinates for RV sidebar text beside the RV vs λ **axes** (not the full SVG).

    For ``xref="paper"`` / ``yref="paper"``, Plotly uses the **inner plotting area** (inside
    margins), with ``x``, ``y`` in ``[0, 1]``. The right edge of the data panel is ``x=1``; we
    Return ``x=1.0``; call sites add ``xshift=RV_SIDEBAR_XSHIFT_PX`` (px) so the box clears the
    axes. Do **not** convert figure margins into ``x`` here — that put
    labels near ``x≈0.6`` and overlapped the scatter.

    ``y_frac`` is 0 = bottom, 1 = top of that subplot row, using that row's ``yaxis`` domain.
    """
    yn = "yaxis" if int(row_1based) <= 1 else f"yaxis{int(row_1based)}"
    dom = getattr(fig.layout, yn).domain
    y0, y1 = float(dom[0]), float(dom[1])
    y_paper = y0 + float(y_frac) * (y1 - y0)
    return 1.0, y_paper


# Mask-overlay debug HTML: full echelle pixels × many orders + long CCF(v) grids inflate inline JSON.
MASK_OVERLAY_SPECTRUM_MAX_POINTS_DEFAULT = 2800
MASK_OVERLAY_CCF_MAX_POINTS_DEFAULT = 4000


def _env_positive_int(name: str, default: int, *, floor: int = 32) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return max(floor, int(default))
    try:
        return max(floor, int(raw))
    except ValueError:
        return max(floor, int(default))


def _mask_overlay_downsample_limits(primitive) -> tuple[int, int]:
    """
    Max points per order for spectrum (wave, spec) and for embedded CCF σ(v) samples.

    Primitive attributes override environment when set to a positive int.
    """
    spec_def = MASK_OVERLAY_SPECTRUM_MAX_POINTS_DEFAULT
    ccf_def = MASK_OVERLAY_CCF_MAX_POINTS_DEFAULT
    spec_attr = getattr(primitive, "mask_overlay_max_spectrum_points", None)
    ccf_attr = getattr(primitive, "mask_overlay_max_ccf_points", None)
    n_spec = int(spec_attr) if spec_attr is not None else None
    n_ccf = int(ccf_attr) if ccf_attr is not None else None
    if n_spec is not None and n_spec > 0:
        spec_lim = max(32, n_spec)
    else:
        spec_lim = _env_positive_int(
            "HISPEC_DRP_MASK_OVERLAY_SPECTRUM_MAX_POINTS", spec_def
        )
    if n_ccf is not None and n_ccf > 0:
        ccf_lim = max(32, n_ccf)
    else:
        ccf_lim = _env_positive_int("HISPEC_DRP_MASK_OVERLAY_CCF_MAX_POINTS", ccf_def)
    return spec_lim, ccf_lim


def _downsample_ccf_for_plot(
    v_m_s: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray | None,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Uniform stride along velocity for long CCF arrays (display only)."""
    v = np.asarray(v_m_s, dtype=np.float64).ravel()
    c = np.asarray(y, dtype=np.float64).ravel()
    n = int(v.size)
    mp = int(max_points)
    if n <= mp or mp < 8:
        return v, c, y_err
    stride = int(math.ceil(n / float(mp)))
    stride = max(1, stride)
    if y_err is not None:
        e = np.asarray(y_err, dtype=np.float64).ravel()
        if e.size == n:
            return v[::stride], c[::stride], e[::stride]
    return v[::stride], c[::stride], None


def _preview_ccf_velocity_range_m_s(primitive) -> tuple[float, float, float]:
    """Same velocity extent as the stellar RV CCF run for previews and overlays (m/s)."""
    bvm, bvx, bvs = resolve_ccf_broad_grid_m_s(
        getattr(primitive, "v_min", None),
        getattr(primitive, "v_max", None),
        getattr(primitive, "v_step", None),
    )
    vc = getattr(primitive, "ccf_input_velocity", None)
    if vc is None or not np.isfinite(float(vc)):
        return bvm, bvx, bvs
    vg = build_hybrid_merged_velocity_grid(
        bvm,
        bvx,
        bvs,
        float(vc),
        float(getattr(primitive, "ccf_velocity_window", 300.0)),
        float(getattr(primitive, "ccf_velocity_step", 0.1)),
    )
    return float(vg[0]), float(vg[-1]), float(effective_ccf_v_step_for_fit(vg))


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


def _preview_gaussian_curve_v_range_m_s(
    v0_m_s: float,
    sigma_m_s: float,
    v_samples_m_s: np.ndarray,
) -> tuple[float, float]:
    """
    Velocity range for drawing the Gaussian CCF model in the preview.

    Do **not** use the LevMar ``fit_v_lo`` / ``fit_v_hi`` window (often hundreds of km/s): that
    stretches the red curve across the panel, hides the peak, and makes residuals look wrong.
    Use a symmetric span around ``v0`` scaled by ``sigma`` instead.
    """
    vg = np.asarray(v_samples_m_s, dtype=np.float64).ravel()
    gmin, gmax = float(np.nanmin(vg)), float(np.nanmax(vg))
    sig = max(float(sigma_m_s), 1e-9)
    half = float(np.clip(max(5000.0, 6.5 * sig), 5000.0, 40_000.0))
    lo = max(gmin, float(v0_m_s) - half)
    hi = min(gmax, float(v0_m_s) + half)
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo >= hi:
        lo, hi = gmin, gmax
    return lo, hi


def _mask_kpf_refine_plotly_traces_from_cached_rmeta(
    fig: go.Figure,
    row: int,
    *,
    rv_r_m_s: float,
    rmeta: dict,
    visible: bool,
    show_legend: bool,
    ylim_lo: float,
    ylim_hi: float,
) -> list[int]:
    """
    Build the five mask-KPF refine overlay traces from ``rmeta`` (no second parabolic solve).

    Used when :func:`~hispecdrp.primitives.utils.ccf_utils.finalize_ccf_preview_dict` already
    ran refinement and stored ``mask_kpf_refine_rmeta`` on the slot.
    """
    trace_indices: list[int] = []

    def _placeholder() -> None:
        fig.add_trace(
            go.Scatter(x=[], y=[], showlegend=False, visible=visible, hoverinfo="skip"),
            row=row,
            col=1,
        )
        trace_indices.append(len(fig.data) - 1)

    if not rmeta.get("ok"):
        for _ in range(5):
            _placeholder()
        return trace_indices

    vk = np.asarray(rmeta["refine_curve_v_m_s"], dtype=np.float64) / 1000.0
    ys = np.asarray(rmeta["refine_curve_ccf_smoothed"], dtype=np.float64)
    ss = np.asarray(rmeta["refine_curve_ccf_sigma"], dtype=np.float64)
    upper = ys + ss
    lower = ys - ss
    x_poly = np.concatenate([vk, vk[::-1]])
    y_poly = np.concatenate([upper, lower[::-1]])
    fig.add_trace(
        go.Scatter(
            x=x_poly,
            y=y_poly,
            fill="toself",
            fillcolor="rgba(0, 114, 178, 0.14)",
            line=dict(width=0),
            mode="lines",
            name="Refine ±1σ (smoothed)",
            legendgroup="mask_kpf_refine",
            showlegend=show_legend,
            visible=visible,
            hoverinfo="skip",
        ),
        row=row,
        col=1,
    )
    trace_indices.append(len(fig.data) - 1)

    fig.add_trace(
        go.Scatter(
            x=vk,
            y=ys,
            mode="lines",
            name="Refine smoothed CCF",
            line=dict(width=2.0, color="rgb(0,114,178)"),
            legendgroup="mask_kpf_refine",
            showlegend=False,
            visible=visible,
        ),
        row=row,
        col=1,
    )
    trace_indices.append(len(fig.data) - 1)

    vd = np.asarray(rmeta["refine_parabola_v_dense_m_s"], dtype=np.float64) / 1000.0
    zd = np.asarray(rmeta["refine_parabola_ccf_dense"], dtype=np.float64)
    if vd.size > 1 and zd.size == vd.size:
        fig.add_trace(
            go.Scatter(
                x=vd,
                y=zd,
                mode="lines",
                name="Refine parabola (3-pt)",
                line=dict(width=2.0, color="rgb(204, 51, 102)", dash="dash"),
                legendgroup="mask_kpf_refine",
                showlegend=False,
                visible=visible,
            ),
            row=row,
            col=1,
        )
        trace_indices.append(len(fig.data) - 1)
    else:
        _placeholder()

    v_lo = float(rmeta.get("parabola_v_lo_m_s", float("nan")))
    v_mid = float(rmeta.get("parabola_v_mid_m_s", float("nan")))
    v_hi = float(rmeta.get("parabola_v_hi_m_s", float("nan")))
    zl = float(rmeta.get("parabola_ccf_smoothed_lo", float("nan")))
    zm = float(rmeta.get("parabola_ccf_smoothed_mid", float("nan")))
    zh = float(rmeta.get("parabola_ccf_smoothed_hi", float("nan")))
    if all(math.isfinite(x) for x in (v_lo, v_mid, v_hi, zl, zm, zh)):
        fig.add_trace(
            go.Scatter(
                x=np.asarray([v_lo, v_mid, v_hi], dtype=np.float64) / 1000.0,
                y=np.asarray([zl, zm, zh], dtype=np.float64),
                mode="markers",
                name="Parabola samples",
                marker=dict(size=9, color="rgb(204, 51, 102)", symbol="diamond"),
                legendgroup="mask_kpf_refine",
                showlegend=False,
                visible=visible,
            ),
            row=row,
            col=1,
        )
        trace_indices.append(len(fig.data) - 1)
    else:
        _placeholder()

    rv_line = float(rv_r_m_s)
    if isinstance(rmeta, dict) and rmeta.get("ok"):
        rv_meta = rmeta.get("rv_refined_m_s")
        if rv_meta is not None and math.isfinite(float(rv_meta)):
            rv_line = float(rv_meta)
    rv_km = rv_line / 1000.0
    err_m = float(rmeta.get("rv_err_m_s", float("nan")))
    v_label = f"Final refined RV = {rv_line:.2f} m/s"
    if math.isfinite(err_m) and err_m > 0:
        v_label += f" (σ={err_m * 100.0:.2f} cm/s)"
    fig.add_trace(
        go.Scatter(
            x=[rv_km, rv_km],
            y=[ylim_lo, ylim_hi],
            mode="lines",
            name=v_label,
            line=dict(width=1.5, color="black", dash="dash"),
            legendgroup="mask_kpf_refine",
            showlegend=False,
            visible=visible,
        ),
        row=row,
        col=1,
    )
    trace_indices.append(len(fig.data) - 1)
    return trace_indices


def _mask_kpf_refine_ccf_add_plotly_traces(
    fig: go.Figure,
    row: int,
    *,
    sl: dict,
    vg: np.ndarray,
    ccf: np.ndarray,
    params: dict | None,
    diag: dict | None,
    visible: bool,
    show_legend: bool,
    ylim_lo: float,
    ylim_hi: float,
) -> list[int]:
    """
    For ``ccf_mask_kpf``, overlay :func:`~hispecdrp.primitives.utils.ccf_utils.refine_ccf_rv_coarse_grid_parabolic`
    curves on the CCF panel (smoothed CCF ±1σ, local parabola, refined RV line). Refinement uses
    the default pipeline behaviour: scan ``smooth_sigma_scale`` to minimize
    ``|RV_refined − v_0|`` versus the Gaussian fit center (unless disabled in that function).

    When the slot carries ``mask_kpf_refine_rmeta`` from :func:`~hispecdrp.primitives.utils.ccf_utils.finalize_ccf_preview_dict`,
    that result is reused so Plotly does not repeat the parabolic refinement.

    Always appends **five** traces (real overlays or empty placeholders) so dropdown indices stay aligned.
    """
    trace_indices: list[int] = []

    def _placeholder() -> None:
        fig.add_trace(
            go.Scatter(x=[], y=[], showlegend=False, visible=visible, hoverinfo="skip"),
            row=row,
            col=1,
        )
        trace_indices.append(len(fig.data) - 1)

    cached = sl.get("mask_kpf_refine_rmeta")
    kpf = sl.get("kpf_fit")
    rv_slot = kpf.get("rv") if isinstance(kpf, dict) else None
    if (
        isinstance(cached, dict)
        and cached.get("ok")
        and rv_slot is not None
        and np.isfinite(float(rv_slot))
    ):
        return _mask_kpf_refine_plotly_traces_from_cached_rmeta(
            fig,
            row,
            rv_r_m_s=float(rv_slot),
            rmeta=cached,
            visible=visible,
            show_legend=show_legend,
            ylim_lo=ylim_lo,
            ylim_hi=ylim_hi,
        )

    try:
        v_step_eff = float(effective_ccf_v_step_for_fit(vg))
        sig_opt: float | None = None
        if params and (diag or {}).get("fit_ok"):
            s = params.get("sigma")
            if s is not None:
                sf = float(s)
                if math.isfinite(sf) and sf > 0:
                    sig_opt = sf
        _ce = sl.get("ccf_err")
        _ce_arr = (
            np.asarray(_ce, dtype=np.float64).ravel()
            if _ce is not None
            else None
        )
        if _ce_arr is not None and _ce_arr.size != vg.size:
            _ce_arr = None
        rv_r, rmeta = refine_ccf_rv_coarse_grid_parabolic(
            vg,
            ccf,
            v_step_eff,
            ccf_err=_ce_arr,
            gaussian_sigma_m_s=sig_opt,
            mask_method="stellar",
            config=CCFFitKpfConfig(),
        )
        if not rmeta.get("ok"):
            for _ in range(5):
                _placeholder()
            return trace_indices

        return _mask_kpf_refine_plotly_traces_from_cached_rmeta(
            fig,
            row,
            rv_r_m_s=float(rv_r),
            rmeta=rmeta,
            visible=visible,
            show_legend=show_legend,
            ylim_lo=ylim_lo,
            ylim_hi=ylim_hi,
        )

    except Exception as e:
        logger.debug("mask-kpf preview: refine_ccf_rv_coarse_grid_parabolic failed: %s", e)
        while len(trace_indices) < 5:
            _placeholder()
        return trace_indices


def _format_rv_value(v: float, *, kind: str) -> str:
    """
    Format RV-related numbers for Plotly annotations.

    * ``mean_ms`` — signed pipeline mean RV (m/s).
    * ``sigma_ms`` — uncertainty in m/s.
    * ``sigma_cm`` — uncertainty in cm/s (e.g. aggregate σ × 100).
    """
    if kind == "sigma_cm":
        if not np.isfinite(v) or v < 0:
            return "—"
        x = float(v)
        ax = abs(x)
        if ax >= 1000.0:
            return f"{x:.2f} cm/s"
        if ax >= 10.0:
            return f"{x:.3f} cm/s"
        return f"{x:.5f} cm/s"
    if not np.isfinite(v):
        return "—"
    a = float(v)
    if kind == "sigma_ms":
        if a < 0:
            return "—"
        if a == 0.0:
            return "0 m/s"
        aval = abs(a)
        if aval >= 10.0:
            return f"{a:.3f} m/s"
        if aval >= 0.1:
            return f"{a:.4f} m/s"
        if aval >= 1e-4:
            return f"{a:.6f} m/s"
        return f"{a:.3e} m/s"
    # mean_ms
    if a == 0.0:
        return "0 m/s"
    aval = abs(a)
    if aval >= 1e5:
        return f"{a:.2f} m/s"
    if aval >= 1000.0:
        return f"{a:.3f} m/s"
    if aval >= 10.0:
        return f"{a:.4f} m/s"
    if aval >= 0.1:
        return f"{a:.5f} m/s"
    if aval >= 1e-4:
        return f"{a:.6f} m/s"
    return f"{a:.3e} m/s"


def _pipeline_rv_avg_from_tables(output_model, fiber: str) -> tuple[float, float]:
    """``RVCCFAVG`` RV / RV_ERR in m/s (same reference as the top RV panel)."""
    avg_tbl = getattr(output_model, f"rvccfavg_fib{fiber}", None)
    if avg_tbl is None or len(avg_tbl) == 0:
        return float("nan"), float("nan")
    return float(avg_tbl["RV"][0]), float(avg_tbl["RV_ERR"][0])


def _per_order_pipeline_rv_iv_mean_m_s(tbl) -> dict[int, tuple[float, float]]:
    """
    Per-echelle-order inverse-variance mean RV ± σ from ``RVCCF`` (m/s).

    Uses every chunk with finite ``RV``, ``RV_ERR`` > 0, and ``ORDER`` ≥ 0 so the
    summary matches the scatter point(s) for that order (inlier or outlier).
    """
    rv = np.asarray(tbl["RV"], dtype=np.float64)
    rv_e = np.asarray(tbl["RV_ERR"], dtype=np.float64)
    if "ORDER" not in tbl.colnames:
        return {}
    ord_col = np.asarray(tbl["ORDER"], dtype=np.int64)
    use = np.isfinite(rv) & np.isfinite(rv_e) & (rv_e > 0) & (ord_col >= 0)
    if not np.any(use):
        return {}
    out: dict[int, tuple[float, float]] = {}
    for o in np.unique(ord_col[use]):
        m = use & (ord_col == int(o))
        if not np.any(m):
            continue
        rw = rv[m]
        re = rv_e[m]
        w = 1.0 / (re * re)
        sw = float(np.sum(w))
        if sw <= 0:
            continue
        mu = float(np.sum(w * rw) / sw)
        sig = float(1.0 / np.sqrt(sw))
        out[int(o)] = (mu, sig)
    return out


def _preview_abbrev_ccf_uncertainty(diag: dict) -> str:
    """Short legend text for how RV uncertainty was estimated (fit_ok only)."""
    wtd = bool(diag.get("used_ccf_pointwise_error"))
    if diag.get("used_kpf_ccf_error"):
        s = "min(KPF, LevMar)" if diag.get("used_covariance_for_error") else "KPF ccf_error_calc"
    elif diag.get("used_covariance_for_error"):
        s = "LevMar cov. (mean)"
    elif diag.get("used_sigma_fallback"):
        s = "σ/√n (fallback)"
    else:
        s = "—"
    if wtd and s != "—":
        return f"{s} · wtd σ_ccf"
    return s


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


def _primitive_output_dir(primitive) -> str:
    return getattr(primitive, "output_dir", None) or os.getcwd()


def _fits_base_path(primitive, output_model) -> str:
    """FITS output path stem (with .fits) used for sibling preview HTML/PNG paths."""
    return output_model.get_output_path(
        output_dir=_primitive_output_dir(primitive),
        suffix=primitive.suffix,
    )


def _first_output_model_from_primitive(primitive):
    """
    Resolve the first L2 output model from ``primitive._output['output']``.

    Returns ``None`` if missing or empty.
    """
    out = primitive._output.get("output") if primitive._output else None
    if out is None:
        return None
    if isinstance(out, (list, tuple)):
        if not out:
            return None
        return out[0]
    return out


def _aggregated_mask_chunk_rv_for_order(
    order_num: int,
    chunk_echelle_order: np.ndarray,
    chunk_rv_m_s: np.ndarray,
    chunk_rv_err_m_s: np.ndarray,
) -> tuple[float | None, float | None]:
    """
    Inverse-variance weighted mean RV and combined σ for all chunks on one echelle order.
    """
    co = np.asarray(chunk_echelle_order, dtype=np.int64).ravel()
    rv = np.asarray(chunk_rv_m_s, dtype=np.float64).ravel()
    er = np.asarray(chunk_rv_err_m_s, dtype=np.float64).ravel()
    if co.size == 0 or co.size != rv.size or co.size != er.size:
        return None, None
    sel = co == int(order_num)
    if not np.any(sel):
        return None, None
    rv_s, er_s = rv[sel], er[sel]
    ok = np.isfinite(rv_s) & np.isfinite(er_s) & (er_s > 0)
    if np.any(ok):
        w = 1.0 / (er_s[ok] ** 2)
        mu = float(np.average(rv_s[ok], weights=w))
        sig = float(1.0 / np.sqrt(np.sum(w))) if np.sum(w) > 0 else None
        return mu, sig
    ok2 = np.isfinite(rv_s)
    if np.any(ok2):
        return float(np.nanmedian(rv_s[ok2])), None
    return None, None


def collect_mask_overlay_payload_for_primitive(
    primitive,
    *,
    input_model,
    fiber: str,
    order_vec: list,
    chunk_rv_m_s: np.ndarray | None = None,
    chunk_rv_err_m_s: np.ndarray | None = None,
    chunk_echelle_order: np.ndarray | None = None,
) -> dict | None:
    """
    Build per-order spectrum + mask payload for mask-overlay debug HTML (ccf_mask_kpf).

    Each order includes ``mask_weight_raw`` (catalog weights) and ``mask_weight`` (after order-edge
    taper when enabled). The overlay uses raw weights for mask-rectangle height; the velocity
    slider CCF uses ``mask_weight``.

    When chunk RV arrays are supplied (same length as ``run_ccf_mask_kpf_for_fiber`` outputs),
    each order entry includes ``fit_rv_m_s`` / ``fit_rv_err_m_s`` for the velocity-slider hint.
    """
    vm, vx, _vs = _preview_ccf_velocity_range_m_s(primitive)
    try:
        resolved_mask = resolve_mask_path(getattr(primitive, "mask_path", None))
        mask_lines = load_mask(
            resolved_mask, mask_width_um=getattr(primitive, "mask_width_um", None)
        )
    except Exception as exc:
        logger.warning(
            "Stellar RV preview: mask overlay payload — cannot load mask: %s", exc
        )
        return None

    wave = getattr(input_model, f"WAVE_FIB{fiber}", None)
    spec = getattr(input_model, f"SPEC_FIB{fiber}", None)
    if wave is None or spec is None:
        logger.debug(
            "Stellar RV preview: mask overlay payload — missing WAVE/SPEC fiber=%s",
            fiber,
        )
        return None

    spec_lim, ccf_lim = _mask_overlay_downsample_limits(primitive)

    entries: list[dict] = []
    # Include every order on the input model so the HTML dropdown can inspect the
    # full BSPEC/RSPEC order set even if RV fitting used a subset via do_orders.
    for oi, order in enumerate(order_vec):
        w = np.asarray(wave[oi], dtype=np.float64)
        s = np.asarray(spec[oi], dtype=np.float64)
        ok = np.isfinite(w) & np.isfinite(s)
        if int(np.sum(ok)) < 10:
            continue
        w = w[ok]
        s = s[ok]
        med = float(np.nanmedian(s))
        if not np.isfinite(med) or med == 0.0:
            med = 1.0
        s_norm = s / med
        w_full = np.asarray(w, dtype=np.float64)
        w_full_lo = float(np.nanmin(w_full))
        w_full_hi = float(np.nanmax(w_full))
        w, s_norm, _ = _downsample_ccf_for_plot(w_full, s_norm, None, spec_lim)
        wl_lo = float(np.nanmin(w))
        wl_hi = float(np.nanmax(w))
        fit_rv: float | None
        fit_err: float | None
        if (
            chunk_echelle_order is not None
            and chunk_rv_m_s is not None
            and chunk_rv_err_m_s is not None
        ):
            fit_rv, fit_err = _aggregated_mask_chunk_rv_for_order(
                int(order),
                chunk_echelle_order,
                chunk_rv_m_s,
                chunk_rv_err_m_s,
            )
        else:
            fit_rv, fit_err = None, None

        mask_o = restrict_mask_to_chunk(
            mask_lines,
            w_full_lo,
            w_full_hi,
            v_min_m_s=vm,
            v_max_m_s=vx,
            zb=0.0,
        )
        if mask_o.line_lambda_um.size > 0:
            ml = np.asarray(mask_o.line_lambda_um, dtype=np.float64)
            # Preview JS still expects paired arrays; delta lines use start == end.
            mstart = ml
            mend = ml
            mw_raw = np.asarray(mask_o.weight, dtype=np.float64)
            mw = mw_raw.copy()
            n_tap = float(getattr(primitive, "mask_order_edge_taper_pixels", 100.0))
            if n_tap > 0.0:
                tap = _mask_line_order_edge_taper_multipliers(
                    w_full,
                    ml,
                    edge_taper_pixels=n_tap,
                    exp_shape=float(
                        getattr(primitive, "mask_order_edge_taper_exp_shape", 5.0)
                    ),
                )
                mw = mw * tap
        else:
            mstart = np.array([], dtype=np.float64)
            mend = np.array([], dtype=np.float64)
            mw_raw = np.array([], dtype=np.float64)
            mw = np.array([], dtype=np.float64)

        entries.append(
            {
                "order": int(order),
                "wl_lo_um": wl_lo,
                "wl_hi_um": wl_hi,
                "wave": np.asarray(w, dtype=np.float64),
                "spec": np.asarray(s_norm, dtype=np.float64),
                "mask_start_um": np.asarray(mstart, dtype=np.float64),
                "mask_end_um": np.asarray(mend, dtype=np.float64),
                "mask_weight_raw": np.asarray(mw_raw, dtype=np.float64),
                "mask_weight": np.asarray(mw, dtype=np.float64),
                "fit_rv_m_s": fit_rv,
                "fit_rv_err_m_s": fit_err,
            }
        )

    if not entries:
        logger.info(
            "Stellar RV preview: mask overlay payload — no valid orders fiber=%s",
            fiber,
        )
        return None

    # Order-mean weight-normalized mask CCF and σ_CCF(v) from rv_per_chunks_mask_kpf
    # when the stellar RV run already built _ccf_preview_by_fiber (preview or mask-overlay flags).
    slot_by_order: dict[int, dict] = {}
    if getattr(primitive, "algorithm", None) == "ccf_mask_kpf":
        prev = getattr(primitive, "_ccf_preview_by_fiber", {}).get(fiber)
        if prev:
            for sl in prev.get("slots", []):
                if sl.get("kind") != "order":
                    continue
                try:
                    oo = int(sl["order"])
                except (TypeError, ValueError, KeyError):
                    continue
                slot_by_order[oo] = sl
    for entry in entries:
        od = int(entry["order"])
        sl = slot_by_order.get(od)
        if sl is None:
            continue
        ce = sl.get("ccf_err")
        vslot = np.asarray(sl.get("v"), dtype=np.float64).ravel()
        if ce is None or vslot.size == 0:
            continue
        cea = np.asarray(ce, dtype=np.float64).ravel()
        if cea.size != vslot.size or not np.any(np.isfinite(cea) & (cea > 0)):
            continue
        v_ds, ce_ds, _ = _downsample_ccf_for_plot(vslot, cea, cea, ccf_lim)
        entry["ccf_v_m_s"] = v_ds
        entry["ccf_err"] = ce_ds

    logger.info(
        "Stellar RV preview: mask overlay payload built fiber=%s n_orders=%d",
        fiber,
        len(entries),
    )
    return {"fiber": str(fiber), "orders": entries}


def _json_opt_float(x: float | None) -> float | None:
    if x is None:
        return None
    xf = float(x)
    return xf if np.isfinite(xf) else None


def _json_round_list(seq, ndigits: int = 8) -> list[float]:
    """Shorter JSON than full float repr; sufficient for Plotly + linear interp in the overlay."""
    return [round(float(x), ndigits) for x in seq]


def _to_serializable_order_payload(order_payload: dict) -> dict:
    out_orders = []
    for row in order_payload.get("orders", []):
        out_orders.append(
            {
                "order": int(row["order"]),
                "wl_lo_um": round(float(row["wl_lo_um"]), 8),
                "wl_hi_um": round(float(row["wl_hi_um"]), 8),
                "wave": _json_round_list(row["wave"], 8),
                "spec": _json_round_list(row["spec"], 7),
                "mask_start_um": _json_round_list(row["mask_start_um"], 9),
                "mask_end_um": _json_round_list(row["mask_end_um"], 9),
                "mask_weight_raw": _json_round_list(row["mask_weight_raw"], 9),
                "mask_weight": _json_round_list(row["mask_weight"], 9),
                "fit_rv_m_s": _json_opt_float(row.get("fit_rv_m_s")),
                "fit_rv_err_m_s": _json_opt_float(row.get("fit_rv_err_m_s")),
            }
        )
        r0 = out_orders[-1]
        cv = row.get("ccf_v_m_s")
        cerr = row.get("ccf_err")
        if cv is not None and cerr is not None:
            cv = np.asarray(cv, dtype=np.float64).ravel()
            cerr = np.asarray(cerr, dtype=np.float64).ravel()
            if cv.size == cerr.size and cv.size > 0 and np.any(np.isfinite(cerr) & (cerr > 0)):
                r0["ccf_v_m_s"] = _json_round_list(cv, 6)
                r0["ccf_err"] = _json_round_list(cerr, 8)
    return {"fiber": str(order_payload["fiber"]), "orders": out_orders}


def _build_mask_overlay_debug_html(
    *,
    title: str,
    payload: dict,
    v_init_m_s: float,
    v_min_m_s: float,
    v_max_m_s: float,
    v_step_m_s: float,
    zb: float = 0.0,
) -> str:
    """
    CCF curve: 50 m/s samples over the full slider range, plus a supersampled band
    around the fitted RV (else ``v_init``): **±10 km/s** at target **0.1 m/s** (capped
    at 200k points in that inner span so step may exceed 0.1 m/s if the cap binds),
    and the remainder of **±100 km/s** (clamped to the slider) at **2 m/s** so the
    wings stay smooth without evaluating ~2e6 points.

    Client-side: slider motion is coalesced with ``requestAnimationFrame`` (one redraw
    per frame at most). When only ``v`` changes, the spectrum panel uses ``Plotly.relayout``
    (mask rectangles + title) and the CCF panel ``Plotly.restyle`` on the marker trace,
    avoiding full ``Plotly.react`` on every tick. Readiness is tracked explicitly because
    ``graphDiv.data`` is not a reliable fast-path check across Plotly versions.
    """
    payload_json = json.dumps(
        _to_serializable_order_payload(payload),
        separators=(",", ":"),
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 10px 14px 14px; }}
    .row {{ display: flex; gap: 12px; align-items: center; margin-bottom: 6px; flex-wrap: wrap; }}
    .row-slider {{ margin-bottom: 2px; justify-content: flex-start; align-items: center; }}
    #vLabel {{
      display: block;
      width: 100%;
      max-width: 100%;
      box-sizing: border-box;
      text-align: left;
      font-size: 12px;
      line-height: 1.4;
      margin: 0 0 10px 0;
      padding: 0;
      color: #222;
    }}
    .row label {{ font-size: 13px; }}
    #plotSpectrum {{ width: 100%; height: 400px; margin-top: 0; }}
    #plotCcf {{ width: 100%; height: 260px; margin-top: 2px; }}
    input[type=number] {{ width: 140px; }}
    select {{ min-width: 280px; }}
    #vSliderWrap {{ position: relative; width: 840px; height: 28px; }}
    #rvFitTrack {{ position: absolute; left: 0; right: 0; top: 10px; height: 8px;
      background: rgba(0,0,0,0.08); border-radius: 4px; pointer-events: none; }}
    #rvFitBand {{ display: none; position: absolute; top: 10px; height: 8px;
      background: rgba(46, 204, 113, 0.55); border-radius: 4px; pointer-events: none; }}
    #rvFitTick {{ display: none; position: absolute; top: 5px; width: 0; height: 18px;
      border-left: 3px solid #1e8449; transform: translateX(-1.5px); pointer-events: none; }}
    #vSlider {{ position: absolute; left: 0; width: 100%; top: 0; height: 28px; margin: 0;
      z-index: 2; background: transparent; }}
  </style>
</head>
<body>
  <div class="row">
    <label>Order
      <select id="orderSelect"></select>
    </label>
    <label>Velocity shift (m/s)
      <input id="vInput" type="number" step="{float(v_step_m_s):.1f}" />
    </label>
  </div>
  <p style="font-size:11px;color:#555;margin:0 0 0 0;max-width:840px;">
    Green band / tick: fitted RV ±1σ for this order (inverse-variance mean over mask-CCF chunks).
  </p>
  <div class="row row-slider">
    <div id="vSliderWrap">
      <div id="rvFitTrack"></div>
      <div id="rvFitBand"></div>
      <div id="rvFitTick"></div>
      <input id="vSlider" type="range" min="{float(v_min_m_s):.1f}" max="{float(v_max_m_s):.1f}" step="{float(v_step_m_s):.1f}" />
    </div>
  </div>
  <div id="vLabel"></div>
  <div id="plotSpectrum"></div>
  <div id="plotCcf"></div>

  <script>
    const C_MS = {C_M_S};
    const ZB = {float(zb)};
    const DATA = {payload_json};
    const V_INIT_M_S = {float(v_init_m_s)};
    let currentOrderIdx = 0;
    let currentV = V_INIT_M_S;

    const orderSelect = document.getElementById("orderSelect");
    const vSlider = document.getElementById("vSlider");
    const vInput = document.getElementById("vInput");
    const vLabel = document.getElementById("vLabel");
    const plotSpectrumDiv = document.getElementById("plotSpectrum");
    const plotCcfDiv = document.getElementById("plotCcf");
    const orderCache = new Map();
    const ccfCurveCache = new Map();
    /** Per-order spectrum traces + static mask metadata (avoid recomputing flux on every v tick). */
    const spectrumStaticByOrderIdx = new Map();
    let spectrumDrawnOrderIdx = -1;
    let ccfDrawnOrderIdx = -1;
    /** Plotly may not expose ``.data`` on the DOM node reliably; use flags after ``react``. */
    let spectrumPlotReady = false;
    let ccfPlotReady = false;
    let sliderPlotRaf = null;

    DATA.orders.forEach((o, i) => {{
      const opt = document.createElement("option");
      opt.value = String(i);
      const lo = Number(o.wl_lo_um);
      const hi = Number(o.wl_hi_um);
      let lab = `Order ${{o.order}}`;
      if (Number.isFinite(lo) && Number.isFinite(hi)) {{
        lab += ` · λ ${{lo.toFixed(4)}}–${{hi.toFixed(4)}} µm`;
      }}
      opt.textContent = lab;
      orderSelect.appendChild(opt);
    }});

    function updateRvFitIndicator(order) {{
      const band = document.getElementById("rvFitBand");
      const tick = document.getElementById("rvFitTick");
      const vMin = parseFloat(vSlider.min);
      const vMax = parseFloat(vSlider.max);
      const span = vMax - vMin;
      if (!band || !tick || !(span > 0)) return;
      const rv = order.fit_rv_m_s;
      const sig = order.fit_rv_err_m_s;
      if (rv == null || !Number.isFinite(Number(rv))) {{
        band.style.display = "none";
        tick.style.display = "none";
        return;
      }}
      const rvN = Number(rv);
      const sigN = Number(sig);
      band.style.display = "block";
      tick.style.display = "block";
      let lo = rvN;
      let hi = rvN;
      if (Number.isFinite(sigN) && sigN > 0) {{
        lo = rvN - sigN;
        hi = rvN + sigN;
      }}
      const pLo = 100 * (lo - vMin) / span;
      const pHi = 100 * (hi - vMin) / span;
      const leftPct = Math.max(0, Math.min(100, Math.min(pLo, pHi)));
      const rightPct = Math.max(0, Math.min(100, Math.max(pLo, pHi)));
      let widthPct = rightPct - leftPct;
      if (widthPct < 0.4) widthPct = 0.4;
      band.style.left = leftPct + "%";
      band.style.width = widthPct + "%";
      const cPct = Math.max(0, Math.min(100, 100 * (rvN - vMin) / span));
      tick.style.left = cPct + "%";
    }}

    function shiftWavelengths(arr, vMs) {{
      const factor = (1.0 + vMs / C_MS) / (1.0 + ZB);
      return arr.map(c => c * factor);
    }}

    function lowerBound(arr, x) {{
      let lo = 0;
      let hi = arr.length;
      while (lo < hi) {{
        const mid = (lo + hi) >> 1;
        if (arr[mid] < x) lo = mid + 1;
        else hi = mid;
      }}
      return lo;
    }}

    function upperBound(arr, x) {{
      let lo = 0;
      let hi = arr.length;
      while (lo < hi) {{
        const mid = (lo + hi) >> 1;
        if (arr[mid] <= x) lo = mid + 1;
        else hi = mid;
      }}
      return lo;
    }}

    function interpLinear(wx, sy, xq) {{
      const nn = wx.length;
      if (nn < 2) return NaN;
      if (xq <= wx[0]) return sy[0];
      if (xq >= wx[nn - 1]) return sy[nn - 1];
      let lo = 0;
      let hi = nn - 1;
      while (hi - lo > 1) {{
        const mid = (lo + hi) >> 1;
        if (wx[mid] <= xq) lo = mid;
        else hi = mid;
      }}
      const i0 = lo;
      const i1 = lo + 1;
      const t = (xq - wx[i0]) / (wx[i1] - wx[i0]);
      return (1.0 - t) * sy[i0] + t * sy[i1];
    }}

    function buildOrderCache(order) {{
      const w = order.wave;
      const s = order.spec;
      const n = w.length;
      if (n < 3) return null;
      const left = new Array(n);
      const right = new Array(n);
      const width = new Array(n);
      const sUse = new Array(n);
      for (let i = 0; i < n; i++) {{
        let l;
        let r;
        if (i === 0) {{
          const d = w[1] - w[0];
          l = w[0] - 0.5 * d;
        }} else {{
          l = 0.5 * (w[i - 1] + w[i]);
        }}
        if (i === n - 1) {{
          const d = w[n - 1] - w[n - 2];
          r = w[n - 1] + 0.5 * d;
        }} else {{
          r = 0.5 * (w[i] + w[i + 1]);
        }}
        left[i] = l;
        right[i] = r;
        const pw = r - l;
        width[i] = Number.isFinite(pw) && pw > 0 ? pw : NaN;
        sUse[i] = Number.isFinite(s[i]) ? s[i] : 0.0;
      }}
      return {{ left, right, width, sUse, n }};
    }}

    function ccfAtVelocity(order, vMs) {{
      let cache = orderCache.get(currentOrderIdx);
      if (!cache) {{
        cache = buildOrderCache(order);
        if (!cache) return NaN;
        orderCache.set(currentOrderIdx, cache);
      }}

      const left = cache.left;
      const right = cache.right;
      const width = cache.width;
      const sUse = cache.sUse;
      const n = cache.n;
      const maskSpectra = new Array(n).fill(0.0);
      const shift = (1.0 + vMs / C_MS) / (1.0 + ZB);
      const wCenters = order.wave;
      let deltaAcc = 0.0;
      let wSumDelta = 0.0;

      for (let i = 0; i < order.mask_start_um.length; i++) {{
        const lnStart = order.mask_start_um[i];
        const lnEnd = order.mask_end_um[i];
        const lnWeight = order.mask_weight[i];
        if (!Number.isFinite(lnStart) || !Number.isFinite(lnEnd) || !Number.isFinite(lnWeight)) continue;
        let lineStart = lnStart * shift;
        let lineEnd = lnEnd * shift;
        if (!Number.isFinite(lineStart) || !Number.isFinite(lineEnd)) continue;
        if (lineEnd <= lineStart) {{
          const lam0 = 0.5 * (lineStart + lineEnd);
          if (lam0 < left[0] || lam0 > right[n - 1]) continue;
          const fv = interpLinear(wCenters, sUse, lam0);
          if (Number.isFinite(fv)) {{
            deltaAcc += lnWeight * fv;
            wSumDelta += lnWeight;
          }}
          continue;
        }}

        const closest = lowerBound(right, lineStart);
        const next = upperBound(left, lineEnd);
        if (next <= 0 || closest >= n - 1) continue;
        const lo = Math.max(0, closest);
        const hi = Math.min(n, next);
        if (hi <= lo) continue;

        for (let p = lo; p < hi; p++) {{
          if (left[p] <= lineEnd && right[p] >= lineStart) {{
            const waveStart = Math.max(left[p], lineStart);
            const waveEnd = Math.min(right[p], lineEnd);
            const overlap = waveEnd - waveStart;
            if (overlap > 0 && Number.isFinite(width[p]) && width[p] > 0) {{
              maskSpectra[p] += lnWeight * (overlap / width[p]);
            }}
          }}
        }}
      }}

      let deltaPart = 0.0;
      if (wSumDelta > 0.0) {{
        deltaPart = deltaAcc / wSumDelta;
      }}
      let boxPart = 0.0;
      for (let i = 0; i < n; i++) {{
        boxPart += sUse[i] * maskSpectra[i];
      }}
      return deltaPart + boxPart;
    }}

    function getCcfCurve(order) {{
      if (ccfCurveCache.has(currentOrderIdx)) {{
        return ccfCurveCache.get(currentOrderIdx);
      }}
      const vMin = parseFloat(vSlider.min);
      const vMax = parseFloat(vSlider.max);
      const coarseStep = 50.0;
      const outerHalfWidth = 100000.0;
      const innerHalfWidth = 10000.0;
      const fineStep = 0.1;
      const wingStep = 2.0;
      const maxInnerPts = 200000;

      let vc = V_INIT_M_S;
      const frv = order.fit_rv_m_s;
      if (frv != null && Number.isFinite(Number(frv))) {{
        vc = Number(frv);
      }}

      const vBins = new Set();
      const addV = (v) => {{
        if (!Number.isFinite(v)) return;
        const c = Math.max(vMin, Math.min(vMax, v));
        vBins.add(Math.round(c * 10) / 10);
      }};

      const nCoarse = Math.max(1, Math.floor((vMax - vMin) / coarseStep) + 1);
      for (let k = 0; k < nCoarse; k++) {{
        addV(vMin + k * coarseStep);
      }}
      addV(vMax);

      const v_lo = Math.max(vMin, vc - outerHalfWidth);
      const v_hi = Math.min(vMax, vc + outerHalfWidth);
      const ilo = Math.max(vMin, vc - innerHalfWidth);
      const ihi = Math.min(vMax, vc + innerHalfWidth);

      if (ihi > ilo) {{
        let nIn = Math.floor((ihi - ilo) / fineStep) + 1;
        if (nIn < 2) nIn = 2;
        if (nIn > maxInnerPts) nIn = maxInnerPts;
        const actualInner = (ihi - ilo) / (nIn - 1);
        for (let i = 0; i < nIn; i++) {{
          addV(ilo + i * actualInner);
        }}
      }}

      if (v_lo < ilo) {{
        for (let w = v_lo; w < ilo - 1e-9; w += wingStep) {{
          addV(w);
        }}
        addV(ilo);
      }}
      if (ihi < v_hi) {{
        addV(ihi);
        for (let w = ihi + wingStep; w < v_hi - 1e-9; w += wingStep) {{
          addV(w);
        }}
        addV(v_hi);
      }}

      const vArr = Array.from(vBins).sort((a, b) => a - b);
      const ccfArr = new Array(vArr.length);
      let ccfErrArr = null;
      const pv = order.ccf_v_m_s;
      const pe = order.ccf_err;
      if (Array.isArray(pv) && Array.isArray(pe) && pv.length === pe.length && pv.length >= 2) {{
        ccfErrArr = new Array(vArr.length);
      }}
      for (let i = 0; i < vArr.length; i++) {{
        ccfArr[i] = ccfAtVelocity(order, vArr[i]);
        if (ccfErrArr) {{
          const ev = interpLinear(pv, pe, vArr[i]);
          ccfErrArr[i] = Number.isFinite(ev) && ev > 0 ? ev : NaN;
        }}
      }}
      const bundle = {{ v: vArr, ccf: ccfArr, ccfErr: ccfErrArr }};
      ccfCurveCache.set(currentOrderIdx, bundle);
      return bundle;
    }}

    function spectrumTitleText(order, vMs) {{
      const lo = Number(order.wl_lo_um);
      const hi = Number(order.wl_hi_um);
      let wl = "";
      if (Number.isFinite(lo) && Number.isFinite(hi)) {{
        wl = ` | λ ${{lo.toFixed(4)}}–${{hi.toFixed(4)}} µm`;
      }}
      return `Fiber ${{DATA.fiber}} | Order ${{order.order}}${{wl}} | v = ${{vMs.toFixed(1)}} m/s`;
    }}

    function getSpectrumStatic(order, idx) {{
      if (spectrumStaticByOrderIdx.has(idx)) {{
        return spectrumStaticByOrderIdx.get(idx);
      }}
      const xSpec = order.wave;
      const ySpec = order.spec;
      let loF = Infinity;
      let hiF = -Infinity;
      for (let i = 0; i < ySpec.length; i++) {{
        const v = ySpec[i];
        if (Number.isFinite(v)) {{
          loF = Math.min(loF, v);
          hiF = Math.max(hiF, v);
        }}
      }}
      const spanF = hiF - loF;
      const ySpec01 = [];
      if (spanF > 0 && Number.isFinite(spanF)) {{
        for (let i = 0; i < ySpec.length; i++) {{
          const v = ySpec[i];
          ySpec01.push(Number.isFinite(v) ? (v - loF) / spanF : NaN);
        }}
      }} else {{
        for (let i = 0; i < ySpec.length; i++) {{
          const v = ySpec[i];
          ySpec01.push(Number.isFinite(v) ? 0.5 : NaN);
        }}
      }}

      let xMin = Infinity;
      let xMax = -Infinity;
      for (let i = 0; i < xSpec.length; i++) {{
        const xv = xSpec[i];
        if (Number.isFinite(xv)) {{
          xMin = Math.min(xMin, xv);
          xMax = Math.max(xMax, xv);
        }}
      }}
      const xSpan = xMax - xMin;
      const minDx = Math.max(1e-12, xSpan * 3e-5);
      // Rectangle height uses raw catalog weights; CCF slider uses deweighted mask_weight.
      let wMask = order.mask_weight;
      if (
        Array.isArray(order.mask_weight_raw) &&
        order.mask_weight_raw.length === order.mask_start_um.length
      ) {{
        wMask = order.mask_weight_raw;
      }}
      let wMax = 0.0;
      for (let j = 0; j < wMask.length; j++) {{
        const t = Number(wMask[j]);
        if (Number.isFinite(t) && t > wMax) wMax = t;
      }}

      const traces = [
        {{
          x: xSpec, y: ySpec01, mode: "lines", name: "Spectrum",
          line: {{ width: 1.2 }},
          yaxis: "y1",
        }},
        {{
          x: [null], y: [null], mode: "markers", name: "Mask lines (shifted)",
          marker: {{ size: 10, color: "rgba(255, 140, 0, 0.7)", symbol: "square" }},
          yaxis: "y2",
          showlegend: true,
        }},
      ];

      const st = {{
        traces,
        xMin,
        xMax,
        minDx,
        wMax,
        wMask,
      }};
      spectrumStaticByOrderIdx.set(idx, st);
      return st;
    }}

    function buildMaskShapes(order, vMs, st) {{
      const xMin = st.xMin;
      const xMax = st.xMax;
      const minDx = st.minDx;
      const wMax = st.wMax;
      const wMask = st.wMask;
      const xMaskStart = shiftWavelengths(order.mask_start_um, vMs);
      const xMaskEnd = shiftWavelengths(order.mask_end_um, vMs);
      const fiducialDepth = 0.5;
      const minBandH = 0.02;
      const shapes = [];
      for (let i = 0; i < xMaskStart.length; i++) {{
        const x0 = xMaskStart[i];
        const x1 = xMaskEnd[i];
        if (!Number.isFinite(x0) || !Number.isFinite(x1)) continue;
        let lo = Math.min(x0, x1);
        let hi = Math.max(x0, x1);
        if (hi < xMin || lo > xMax) continue;
        if (hi <= lo) {{
          const xc = 0.5 * (lo + hi);
          lo = xc - 0.5 * minDx;
          hi = xc + 0.5 * minDx;
        }} else if (hi - lo < minDx) {{
          const xc = 0.5 * (lo + hi);
          lo = xc - 0.5 * minDx;
          hi = xc + 0.5 * minDx;
        }}
        let w = Number(wMask[i]);
        if (!Number.isFinite(w) || w < 0) w = 0;
        const wRel = wMax > 0 ? w / wMax : 0;
        const yTop = 1.0;
        const bandH = Math.min(
          fiducialDepth,
          Math.max(fiducialDepth * wRel, minBandH)
        );
        const yBot = Math.max(0.0, yTop - bandH);
        if (yBot >= yTop) continue;
        shapes.push({{
          type: "rect",
          xref: "x",
          yref: "y2",
          x0: lo,
          x1: hi,
          y0: yBot,
          y1: yTop,
          fillcolor: "rgba(255, 140, 0, 0.42)",
          line: {{ color: "rgba(230, 115, 0, 0.95)", width: 1 }},
          layer: "below",
        }});
      }}
      return shapes;
    }}

    function updatePlot() {{
      const order = DATA.orders[currentOrderIdx];
      if (!order) return;

      const st = getSpectrumStatic(order, currentOrderIdx);
      const shapes = buildMaskShapes(order, currentV, st);
      const titleText = spectrumTitleText(order, currentV);

      const layoutSpectrum = {{
        uirevision: `fib-${{DATA.fiber}}-ord-${{order.order}}`,
        title: {{
          text: titleText,
          y: 1.03,
          yref: "paper",
          yanchor: "top",
          font: {{ size: 11 }},
          pad: {{ b: 2, t: 0 }},
        }},
        xaxis: {{ title: "Wavelength (um)" }},
        yaxis: {{
          title: "Normalized flux",
          range: [0, 1],
          zeroline: true,
        }},
        yaxis2: {{
          title: "Mask weight",
          overlaying: "y",
          side: "right",
          range: [0, 1],
          scaleanchor: "y",
          scaleratio: 1,
          showgrid: false,
        }},
        shapes: shapes,
        legend: {{
          orientation: "h",
          xref: "paper",
          yref: "paper",
          x: 0.5,
          xanchor: "center",
          y: 1.02,
          yanchor: "bottom",
          font: {{ size: 9 }},
          bgcolor: "rgba(255, 255, 255, 0.92)",
          bordercolor: "rgba(0,0,0,0.15)",
          borderwidth: 1,
        }},
        margin: {{ l: 70, r: 70, t: 132, b: 48 }},
      }};

      const ccfVal = ccfAtVelocity(order, currentV);
      const ccfText = Number.isFinite(ccfVal) ? ccfVal.toExponential(3) : "nan";
      let ccfSigText = "";
      if (Array.isArray(order.ccf_v_m_s) && Array.isArray(order.ccf_err) && order.ccf_v_m_s.length === order.ccf_err.length) {{
        const se = interpLinear(order.ccf_v_m_s, order.ccf_err, currentV);
        if (Number.isFinite(se) && se > 0) {{
          ccfSigText = ` ± ${{se.toExponential(2)}} (σ_CCF)`;
        }}
      }}

      const sameSpectrumOrder = spectrumDrawnOrderIdx === currentOrderIdx;
      if (sameSpectrumOrder && spectrumPlotReady) {{
        Plotly.relayout(plotSpectrumDiv, {{
          "title.text": titleText,
          shapes: shapes,
        }});
      }} else {{
        Plotly.react(plotSpectrumDiv, st.traces, layoutSpectrum, {{ responsive: true }});
        spectrumDrawnOrderIdx = currentOrderIdx;
        spectrumPlotReady = true;
      }}

      const ccfValMarker = Number.isFinite(ccfVal) ? ccfVal : null;
      const sameCcfOrder = ccfDrawnOrderIdx === currentOrderIdx;
      if (sameCcfOrder && ccfPlotReady) {{
        Plotly.restyle(plotCcfDiv, {{
          x: [[currentV]],
          y: [[ccfValMarker]],
        }}, [1]);
      }} else {{
        const curve = getCcfCurve(order);
        const tracesCcf = [
          {{
            x: curve.v,
            y: curve.ccf,
            mode: "lines",
            name: "CCF",
            line: {{ color: "#2c3e50", width: 1.5 }},
          }},
          {{
            x: [currentV],
            y: [ccfValMarker],
            mode: "markers",
            name: "Current v",
            marker: {{
              size: 11,
              color: "rgba(231, 76, 60, 0.95)",
              line: {{ color: "rgba(192, 57, 43, 1)", width: 2 }},
            }},
          }},
        ];
        const layoutCcf = {{
          uirevision: `ccf-${{DATA.fiber}}-ord-${{order.order}}`,
          title: {{
            text: "RV versus mask CCF",
            y: 1.02,
            yref: "paper",
            yanchor: "top",
            pad: {{ b: 0, t: 0 }},
            font: {{ size: 13 }},
          }},
          xaxis: {{ title: "Radial velocity (m/s)" }},
          yaxis: {{ title: "CCF (Normalized)" }},
          margin: {{ l: 70, r: 28, t: 44, b: 42 }},
          showlegend: false,
        }};
        Plotly.react(plotCcfDiv, tracesCcf, layoutCcf, {{ responsive: true }});
        ccfDrawnOrderIdx = currentOrderIdx;
        ccfPlotReady = true;
      }}

      let rvAggTxt = "";
      if (order.fit_rv_m_s != null && Number.isFinite(Number(order.fit_rv_m_s))) {{
        const frv = Number(order.fit_rv_m_s);
        const fe = order.fit_rv_err_m_s;
        if (fe != null && Number.isFinite(Number(fe)) && Number(fe) > 0) {{
          rvAggTxt = ` | RV = ${{frv.toFixed(1)}} ± ${{Number(fe).toFixed(1)}} m/s`;
        }} else {{
          rvAggTxt = ` | RV = ${{frv.toFixed(1)}} m/s`;
        }}
      }}
      vLabel.textContent = `v = ${{currentV.toFixed(1)}} m/s | CCF = ${{ccfText}}${{ccfSigText}}${{rvAggTxt}}`;
      updateRvFitIndicator(order);
    }}

    function cancelSliderPlotRaf() {{
      if (sliderPlotRaf !== null) {{
        cancelAnimationFrame(sliderPlotRaf);
        sliderPlotRaf = null;
      }}
    }}

    function schedulePlotFromSlider() {{
      if (sliderPlotRaf !== null) return;
      sliderPlotRaf = requestAnimationFrame(() => {{
        sliderPlotRaf = null;
        updatePlot();
      }});
    }}

    function setVelocity(v, opts) {{
      const immediate = opts && opts.immediate;
      currentV = Number(v);
      if (!Number.isFinite(currentV)) currentV = 0.0;
      vSlider.value = String(currentV);
      vInput.value = String(currentV);
      if (immediate) cancelSliderPlotRaf();
      updatePlot();
    }}

    orderSelect.addEventListener("change", (ev) => {{
      cancelSliderPlotRaf();
      currentOrderIdx = Number(ev.target.value);
      spectrumDrawnOrderIdx = -1;
      ccfDrawnOrderIdx = -1;
      spectrumPlotReady = false;
      ccfPlotReady = false;
      updatePlot();
    }});
    vSlider.addEventListener("input", (ev) => {{
      currentV = Number(ev.target.value);
      if (!Number.isFinite(currentV)) currentV = 0.0;
      vInput.value = String(currentV);
      schedulePlotFromSlider();
    }});
    vSlider.addEventListener("change", (ev) => {{
      cancelSliderPlotRaf();
      setVelocity(ev.target.value, {{ immediate: true }});
    }});
    vInput.addEventListener("change", (ev) => setVelocity(ev.target.value, {{ immediate: true }}));
    vInput.addEventListener("keydown", (ev) => {{
      if (ev.key === "Enter") setVelocity(ev.target.value, {{ immediate: true }});
    }});

    vSlider.value = String(currentV);
    vInput.value = String(currentV);
    orderSelect.value = "0";
    updatePlot();
  </script>
</body>
</html>
"""


def write_mask_overlay_debug_pages(primitive, output_model) -> dict[str, str]:
    """
    Write debug HTML overlays for mask-shift RV and optionally open them.

    Returns
    -------
    dict[str, str]
        fiber -> html path
    """
    out: dict[str, str] = {}
    payloads = getattr(primitive, "_mask_overlay_by_fiber", {}) or {}
    if not payloads:
        logger.info("Stellar RV preview: mask overlay debug — no payloads to plot.")
        return out

    logger.info(
        "Stellar RV preview: mask overlay debug — writing HTML for %d fiber(s)",
        len(payloads),
    )
    base_path = _fits_base_path(primitive, output_model)
    v_init = float(getattr(primitive, "mask_overlay_velocity_m_s", 0.0))
    vm, vx, vs = _preview_ccf_velocity_range_m_s(primitive)

    for fiber, payload in payloads.items():
        html = _build_mask_overlay_debug_html(
            title=f"Stellar RV mask overlay debug (fiber {fiber})",
            payload=payload,
            v_init_m_s=v_init,
            v_min_m_s=vm,
            v_max_m_s=vx,
            v_step_m_s=vs,
        )
        path = base_path.replace(".fits", f"_mask_overlay_debug_fib{fiber}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        out[str(fiber)] = path
        logger.info(
            "Stellar RV preview: mask overlay debug — wrote fiber=%s path=%s n_orders=%d",
            fiber,
            path,
            len(payload.get("orders", [])),
        )
        if bool(getattr(primitive, "show_mask_overlay_debug", False)):
            try:
                webbrowser.open(f"file://{path}")
            except Exception as exc:
                logger.debug("Could not open mask overlay debug HTML: %s", exc)
    logger.info(
        "Stellar RV preview: mask overlay debug — finished n_files=%d",
        len(out),
    )
    return out


def _preview_gaussian_cache_and_slot_legends(
    primitive,
    output_model,
    fibers_with_ccf: list[str],
) -> tuple[
    dict[str, tuple[float, float, dict[int, tuple[float, float]]]],
    dict[int, tuple],
    dict[str, list[str]],
]:
    """Pipeline RV summaries, per-slot Gaussian cache, and CCF-panel HTML legend strings."""
    pipeline_rv_by_fiber: dict[str, tuple[float, float, dict[int, tuple[float, float]]]] = {}
    for fib in fibers_with_ccf:
        tbl_rv = getattr(output_model, f"rvccf_fib{fib}", None)
        rag, rae = _pipeline_rv_avg_from_tables(output_model, fib)
        po: dict[int, tuple[float, float]] = {}
        if tbl_rv is not None:
            po = _per_order_pipeline_rv_iv_mean_m_s(tbl_rv)
        pipeline_rv_by_fiber[fib] = (rag, rae, po)

    fit_slot_cache: dict[int, tuple] = {}
    slot_metadata: dict[str, list[str]] = {}
    for fiber in fibers_with_ccf:
        detail = primitive._ccf_preview_by_fiber.get(fiber)
        if not detail:
            slot_metadata[fiber] = []
            continue
        rv_agg_tbl, rv_agg_err_tbl, per_order_rv = pipeline_rv_by_fiber[fiber]
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
                    _ce = sl.get("ccf_err")
                    _ce_arr = (
                        np.asarray(_ce, dtype=np.float64) if _ce is not None else None
                    )
                    _ccf_arr = np.asarray(sl["ccf"], dtype=np.float64)
                    _rv, _re, params, diag = fit_ccf_rv(
                        sl["v"],
                        _ccf_arr,
                        v_step,
                        return_params=True,
                        ccf_err=_ce_arr,
                    )
                    rv_err = float(_re) if _re is not None and np.isfinite(_re) else np.nan
                    fit_slot_cache[sid] = (params, diag, rv_err)
            params, diag, rv_err = fit_slot_cache[sid]
            is_mask_kpf = getattr(primitive, "algorithm", None) == "ccf_mask_kpf"
            if sl.get("kind") == "mean":
                meta = "Mean · all ord"
                if np.isfinite(rv_agg_tbl):
                    src = "IV" if is_mask_kpf else "RVCCFAVG"
                    meta += (
                        f"<br>⟨RV⟩ ({src}): {rv_agg_tbl / 1000.0:.4f} km/s · "
                        f"σ={_format_rv_value(rv_agg_err_tbl, kind='sigma_ms')}"
                    )
                _ce_mn = sl.get("ccf_err")
                if _ce_mn is not None:
                    _ceam = np.asarray(_ce_mn, dtype=np.float64).ravel()
                    _okm = np.isfinite(_ceam) & (_ceam > 0)
                    if np.any(_okm):
                        meta += f"<br>median σ_CCF ≈ {float(np.median(_ceam[_okm])):.3g}"
            else:
                o = sl.get("order", "?")
                wl_lo, wl_hi = sl.get("wl_lo", np.nan), sl.get("wl_hi", np.nan)
                if np.isfinite(wl_lo) and np.isfinite(wl_hi):
                    meta = f"Ord {o} · λ{wl_lo:.3f}–{wl_hi:.3f} µm"
                else:
                    meta = f"Ord {o}"
                oo: int | None
                try:
                    oo = int(o) if o is not None and str(o).strip() != "?" else None
                except (TypeError, ValueError):
                    oo = None
                if oo is not None and oo in per_order_rv:
                    rp, rpe = per_order_rv[oo]
                    if is_mask_kpf:
                        meta += (
                            f"<br>RV (Gauss=table): {rp / 1000.0:.4f} km/s · "
                            f"σ={_format_rv_value(rpe, kind='sigma_ms')}"
                        )
                    else:
                        meta += (
                            f"<br>Pipe RVCCF: {rp / 1000.0:.4f} km/s · "
                            f"σ={_format_rv_value(rpe, kind='sigma_ms')}"
                        )
                elif oo is not None:
                    lbl = "RV" if is_mask_kpf else "Pipe"
                    meta += f"<br>{lbl}: —"
            if params and diag.get("fit_ok"):
                v0 = float(params["v0"])
                sig = float(params["sigma"])
                amp = float(params["amplitude"])
                if is_mask_kpf and sl.get("kind") != "mean":
                    meta += (
                        f"<br>Gauss σ={sig / 1000.0:.4f} km/s · amp={amp:.3g}"
                    )
                    try:
                        oi = int(sl.get("order"))
                    except (TypeError, ValueError):
                        oi = None
                    if oi is not None and oi in per_order_rv:
                        rp = float(per_order_rv[oi][0])
                        if np.isfinite(rp) and abs(v0 - rp) > 0.5:
                            meta += f"<br><i>ΔRV refit {v0 - rp:+.2f} m/s</i>"
                    meta += f"<br>σ_RV: {_preview_abbrev_ccf_uncertainty(diag)}"
                else:
                    meta += (
                        f"<br>Gauss RV={v0 / 1000.0:.4f} km/s · "
                        f"σ={sig / 1000.0:.4f} km/s · amp={amp:.3g}"
                    )
                    if sl.get("kind") == "mean" and np.isfinite(rv_agg_tbl):
                        dvp = v0 - rv_agg_tbl
                        if np.isfinite(dvp) and abs(dvp) > 250.0:
                            meta += f"<br><i>Gauss−⟨RV⟩tbl {dvp:+.0f} m/s</i>"
                    elif sl.get("kind") != "mean" and not is_mask_kpf:
                        try:
                            oi = int(sl.get("order"))
                        except (TypeError, ValueError):
                            oi = None
                        if oi is not None and oi in per_order_rv:
                            rp = float(per_order_rv[oi][0])
                            if np.isfinite(rp):
                                meta += f"<br><i>Gauss−IV {v0 - rp:+.1f} m/s</i>"
                    meta += (
                        f"<br>formal σ_RV={_format_rv_value(float(rv_err), kind='sigma_ms')} · "
                        f"{_preview_abbrev_ccf_uncertainty(diag)}"
                    )
                    _ce_sl = sl.get("ccf_err")
                    if _ce_sl is not None:
                        _cea = np.asarray(_ce_sl, dtype=np.float64).ravel()
                        _okc = np.isfinite(_cea) & (_cea > 0)
                        if np.any(_okc):
                            meta += f"<br>median σ_CCF ≈ {float(np.median(_cea[_okc])):.3g}"
            else:
                meta += "<br><i>Gauss fit fail</i>"
                fail_msg = (diag.get("message") or "").strip()
                if fail_msg:
                    tail = fail_msg[:40] + "…" if len(fail_msg) > 40 else fail_msg
                    meta += f" · {tail}"
            slot_metadata[fiber].append(meta)

    logger.info(
        "Stellar RV preview: CCF slot metadata and Gaussian cache ready "
        "(cached distinct slot objects=%d)",
        len(fit_slot_cache),
    )
    return pipeline_rv_by_fiber, fit_slot_cache, slot_metadata


def _preview_export_combined_html(
    primitive,
    output_model,
    fig_rv: go.Figure,
    fig_ccf: go.Figure,
    title_unified: str,
) -> tuple[str, str, bool, str]:
    """
    Write unified HTML (and optional stacked PNG), optionally open browser.

    Returns
    -------
    output_path_html, output_path_png, wrote_png, combined_html
    """
    base_path = _fits_base_path(primitive, output_model)
    output_path_html = base_path.replace(".fits", "_preview.html")
    output_path_png = base_path.replace(".fits", "_preview.png")
    wrote_png = False

    esc_title = html_module.escape(title_unified)
    div_rv = pio.to_html(
        fig_rv,
        include_plotlyjs="cdn",
        full_html=False,
        config=PREVIEW_PLOTLY_HTML_CONFIG,
    )
    div_ccf = pio.to_html(
        fig_ccf,
        include_plotlyjs=False,
        full_html=False,
        config=PREVIEW_PLOTLY_HTML_CONFIG,
    )
    combined_html = (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        f"<title>{esc_title}</title>\n"
        "<style>body{margin:0;font-family:system-ui,sans-serif;}"
        "hr{border:none;border-top:1px solid #ccc;margin:0;}</style>\n"
        "</head><body>\n"
        f'<p style="margin:12px 16px 4px;font-size:14px;">{esc_title}</p>\n'
        f"{div_rv}<hr>\n{div_ccf}\n"
        "</body></html>\n"
    )

    if primitive.save_preview:
        logger.info("Stellar RV preview: writing HTML path=%s", output_path_html)
        Path(output_path_html).write_text(combined_html, encoding="utf-8")
        logger.info("Stellar RV preview: HTML write complete")
        if _preview_should_write_png(primitive):
            try:
                import kaleido

                scale = float(getattr(primitive, "preview_png_scale", 2.0))
                scale = max(0.25, min(scale, 8.0))
                logger.info(
                    "Stellar RV preview: writing PNG via Kaleido path=%s scale=%.2f",
                    output_path_png,
                    scale,
                )
                try:
                    from PIL import Image

                    br = fig_rv.to_image(
                        format="png", engine="kaleido", scale=scale
                    )
                    bc = fig_ccf.to_image(
                        format="png", engine="kaleido", scale=scale
                    )
                    im_rv = Image.open(io.BytesIO(br)).convert("RGBA")
                    im_ccf = Image.open(io.BytesIO(bc)).convert("RGBA")
                    w = max(im_rv.width, im_ccf.width)
                    h = im_rv.height + im_ccf.height
                    stacked = Image.new("RGBA", (w, h), (255, 255, 255, 255))
                    stacked.paste(im_rv, (0, 0))
                    stacked.paste(im_ccf, (0, im_rv.height))
                    stacked.convert("RGB").save(output_path_png)
                    wrote_png = True
                except Exception:
                    kaleido.write_fig_sync(
                        fig_ccf, path=output_path_png, opts={"scale": scale}
                    )
                    wrote_png = True
                logger.info("Stellar RV preview: PNG write complete")
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
                "Stellar RV preview: skipping Kaleido PNG (%s)",
                skip_reason,
            )
    if primitive.save_preview:
        logger.info(
            "Stellar RV preview: save_preview phase done html=%s png=%s",
            output_path_html,
            output_path_png if wrote_png else None,
        )
    if primitive.show_preview:
        if primitive.save_preview and os.path.isfile(output_path_html):
            uri = Path(output_path_html).resolve().as_uri()
            logger.info("Stellar RV preview: opening browser %s", uri)
            webbrowser.open(uri)
        else:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix="_stellar_rv_preview.html",
                delete=False,
                encoding="utf-8",
            ) as tf:
                tf.write(combined_html)
                tmp_path = tf.name
            logger.info(
                "Stellar RV preview: opening browser (temp HTML) path=%s", tmp_path
            )
            webbrowser.open(Path(tmp_path).resolve().as_uri())
        logger.info("Stellar RV preview: browser open returned")

    return output_path_html, output_path_png, wrote_png, combined_html


def create_stellar_rv_preview(primitive) -> None:
    """
    Writes ``*_preview.html`` with **two** Plotly embeds:

    1. **RV figure** — Per-chunk pipeline RV vs wavelength (Δ vs ⟨RV⟩) and derived-RV / plot-key
       annotations (own mode bar).
    2. **CCF figure** — Per fiber, CCF vs velocity (y-axis ``CCF``) then **Residual CCF**
       (raw CCF minus the Gaussian model on the full grid, same units). The Gaussian trace is
       drawn over a **narrow symmetric** span around ``v₀`` (not the full LevMar window). For
       ``ccf_mask_kpf``, the black **dashed** vertical line is the final refined RV; for template
       CCF, a dashed line marks Gaussian ``v₀``. **Dropdown** switches slot on both rows;
       velocity zoom/pan is **linked** (``matches``). Own mode bar.

    Normalized spectrum + mask lines are not shown here; use ``*_mask_overlay_debug_fib*.html`` for that.

    Hover is limited to the subplot under the cursor when supported (``hoversubplots='single'``).
    """
    logger.info("Stellar RV preview: create_stellar_rv_preview starting")
    output_model = _first_output_model_from_primitive(primitive)
    if output_model is None:
        logger.warning("Stellar RV preview: no output model — aborting preview.")
        return

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
        logger.warning(
            "Stellar RV preview: no RVCCF tables with valid RV/wavelength — aborting."
        )
        return

    logger.info(
        "Stellar RV preview: fibers with RVCCF data: %s save_preview=%s show_preview=%s",
        fibers_with_ccf,
        getattr(primitive, "save_preview", False),
        getattr(primitive, "show_preview", False),
    )

    pv_min_m_s, pv_max_m_s, _pv_step = _preview_ccf_velocity_range_m_s(primitive)
    v_axis_km_lo = pv_min_m_s / 1000.0
    v_axis_km_hi = pv_max_m_s / 1000.0

    n_fib = len(fibers_with_ccf)
    row_plan_ccf: list[tuple] = []
    for f in fibers_with_ccf:
        row_plan_ccf.append(("ccf", f))
        row_plan_ccf.append(("resid", f))

    # Per fiber: ``fig_rv`` row = RV; ``fig_ccf`` rows = CCF then residual (1-based indices).
    fiber_row_map: dict[str, dict[str, int]] = {}
    for fi, fiber in enumerate(fibers_with_ccf):
        fiber_row_map[fiber] = {
            "rv": fi + 1,
            "ccf": 2 * fi + 1,
            "resid": 2 * fi + 2,
        }
    n_rows_ccf = 2 * n_fib
    logger.info(
        "Stellar RV preview: building Plotly figures fig_rv rows=%d fig_ccf rows=%d (n_fibers=%d)",
        n_fib,
        n_rows_ccf,
        n_fib,
    )
    subplot_titles_rv: list[str] = []
    subplot_titles_ccf: list[str] = []
    for fiber in fibers_with_ccf:
        subplot_titles_rv.append(f"Fiber {fiber} - per chunk RV")
        subplot_titles_ccf.append(
            f"Fiber {fiber} — CCF vs RV (±σ_CCF; Gauss; mask-KPF parabolic refine; slot ↓)"
        )
        subplot_titles_ccf.append(f"Fiber {fiber} — Residual CCF")

    v_spacing_rv = max(0.04, min(0.09, 1.4 / max(n_fib, 8)))
    if n_fib > 1:
        v_spacing_rv = min(v_spacing_rv, 1.0 / (n_fib - 1))
    v_spacing_ccf = max(0.04, min(0.09, 1.4 / max(n_rows_ccf, 8)))
    if n_rows_ccf > 1:
        v_spacing_ccf = min(v_spacing_ccf, 1.0 / (n_rows_ccf - 1))

    fig_rv = make_subplots(
        rows=n_fib,
        cols=1,
        subplot_titles=subplot_titles_rv,
        vertical_spacing=v_spacing_rv,
        shared_xaxes=False,
        shared_yaxes=False,
    )
    fig_ccf = make_subplots(
        rows=n_rows_ccf,
        cols=1,
        subplot_titles=subplot_titles_ccf,
        vertical_spacing=v_spacing_ccf,
        shared_xaxes=False,
        shared_yaxes=False,
    )

    pipeline_rv_by_fiber, fit_slot_cache, slot_metadata = (
        _preview_gaussian_cache_and_slot_legends(primitive, output_model, fibers_with_ccf)
    )

    rv_panel_y_title: dict[str, str] = {}
    rv_panel_y_range: dict[str, tuple[float, float]] = {}
    rv_sidebar_by_fiber: dict[str, str] = {}
    rv_plot_legend_by_fiber: dict[str, str] = {}
    ccf_trace_indices: dict[str, list[list[int]]] = {}
    resid_trace_indices: dict[str, list[list[int]]] = {}
    resid_ylim_by_fiber: dict[str, tuple[float, float]] = {}

    for fi, fiber in enumerate(fibers_with_ccf):
        row_rv = fiber_row_map[fiber]["rv"]

        tbl = getattr(output_model, f"rvccf_fib{fiber}", None)
        avg_tbl = getattr(output_model, f"rvccfavg_fib{fiber}", None)
        if tbl is not None:
            wl = np.asarray(tbl["WAVELENGTH"], dtype=np.float64)
            rv = np.asarray(tbl["RV"], dtype=np.float64)
            rv_err = np.asarray(tbl["RV_ERR"], dtype=np.float64)
            inlier = (
                np.asarray(tbl["INLIER"], dtype=bool)
                if "INLIER" in tbl.colnames
                else np.ones(len(rv), dtype=bool)
            )
            valid = np.isfinite(wl) & np.isfinite(rv)
            if np.any(valid):
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
                rv_pt_kms = rv / 1000.0
                rv_err_pt_kms = rv_err / 1000.0
                if rv_mean_cm_r < 0:
                    y_title = f"RV (cm/s) + {abs(rv_mean_cm_r):.4e}"
                else:
                    y_title = f"RV (cm/s) − {rv_mean_cm_r:.4e}"
                rv_panel_y_title[fiber] = y_title
                if "ORDER" in tbl.colnames:
                    order_col = np.asarray(tbl["ORDER"], dtype=np.int64)[valid]
                else:
                    order_col = np.full(len(wl), -1, dtype=np.int64)
                ord_lbl = np.where(order_col >= 0, order_col.astype(str), "—")
                cd_rv = np.column_stack([ord_lbl.astype(object), rv_pt_kms, rv_err_pt_kms])
                in_ok = inlier & np.isfinite(rv_err) & (rv_err > 0)
                out_ok = ~inlier & np.isfinite(rv_err) & (rv_err > 0)
                if np.any(in_ok):
                    y_clip_src = y_cm_off[in_ok]
                else:
                    y_clip_src = y_cm_off[np.isfinite(y_cm_off)]
                if y_clip_src.size:
                    rv_panel_y_range[fiber] = _iterative_sigma_clip_ylim(y_clip_src)
                if np.any(in_ok):
                    fig_rv.add_trace(
                        go.Scatter(
                            x=wl[in_ok],
                            y=y_cm_off[in_ok],
                            error_y=dict(type="data", array=err_y_cm[in_ok], visible=True),
                            mode="markers",
                            name=f"Fiber {fiber} (inlier)",
                            showlegend=False,
                            marker=dict(size=6, color="royalblue"),
                            customdata=cd_rv[in_ok],
                            hovertemplate=(
                                "λ = %{x:.5f} µm<br>"
                                "Order %{customdata[0]}<br>"
                                "RV (pipeline) = %{customdata[1]:.6f} ± %{customdata[2]:.6g} km/s<br>"
                                "ΔRV vs ⟨RV⟩ = %{y:.2f} ± %{error_y.array:.2f} cm/s<extra></extra>"
                            ),
                        ),
                        row=row_rv,
                        col=1,
                    )
                if np.any(out_ok):
                    fig_rv.add_trace(
                        go.Scatter(
                            x=wl[out_ok],
                            y=y_cm_off[out_ok],
                            error_y=dict(type="data", array=err_y_cm[out_ok], visible=True),
                            mode="markers",
                            name=f"Fiber {fiber} (outlier)",
                            showlegend=False,
                            marker=dict(size=5, color="grey", symbol="circle-open"),
                            customdata=cd_rv[out_ok],
                            hovertemplate=(
                                "λ = %{x:.5f} µm<br>"
                                "Order %{customdata[0]}<br>"
                                "RV (pipeline) = %{customdata[1]:.6f} ± %{customdata[2]:.6g} km/s<br>"
                                "ΔRV vs ⟨RV⟩ = %{y:.2f} ± %{error_y.array:.2f} cm/s<extra></extra>"
                            ),
                        ),
                        row=row_rv,
                        col=1,
                    )
                if not np.any(in_ok) and np.any(valid):
                    fig_rv.add_trace(
                        go.Scatter(
                            x=wl,
                            y=y_cm_off,
                            error_y=dict(type="data", array=err_y_cm, visible=True),
                            mode="markers",
                            name=f"Fiber {fiber} chunks",
                            showlegend=False,
                            marker=dict(size=6, color="royalblue"),
                            customdata=cd_rv,
                            hovertemplate=(
                                "λ = %{x:.5f} µm<br>"
                                "Order %{customdata[0]}<br>"
                                "RV (pipeline) = %{customdata[1]:.6f} ± %{customdata[2]:.6g} km/s<br>"
                                "ΔRV vs ⟨RV⟩ = %{y:.2f} ± %{error_y.array:.2f} cm/s<extra></extra>"
                            ),
                        ),
                        row=row_rv,
                        col=1,
                    )
                x_range = [float(np.nanmin(wl)), float(np.nanmax(wl))] if len(wl) else [0.0, 1.0]
                mean_ms_str = _format_rv_value(float(rv_agg), kind="mean_ms")
                mean_line = f"⟨RV⟩ = {mean_ms_str}"
                if np.isfinite(rv_agg_err):
                    leg = (
                        f"{mean_line}<br>σ(⟨RV⟩) = {_format_rv_value(float(rv_agg_err), kind='sigma_ms')} "
                        f"({_format_rv_value(float(rv_agg_err) * 100.0, kind='sigma_cm')})"
                    )
                else:
                    leg = f"{mean_line}<br>uncertainty = —"
                # RMS bias from mean: sqrt(sum of squared biases over good points / n_good)
                n_good = int(np.sum(inlier))
                rms_bias_m_s = float("nan")
                if n_good > 0:
                    biases_m_s = rv[inlier] - rv_agg
                    rms_bias_m_s = float(np.sqrt(np.sum(biases_m_s**2) / n_good))
                    rms_bias_cm_s = rms_bias_m_s * 100.0
                    leg += (
                        f"<br>RMS(⟨RV⟩) = {_format_rv_value(rms_bias_m_s, kind='sigma_ms')} "
                        f"({_format_rv_value(rms_bias_cm_s, kind='sigma_cm')})"
                    )
                n_chunks_tbl = int(len(rv))
                sb_lines = [
                    f"<b>Derived - fiber {fiber}</b>",
                    f"Chunks in RVCCF table: {n_chunks_tbl} · Inlier: {n_good}",
                    f"⟨RV⟩ = {mean_ms_str}",
                ]
                if np.isfinite(rv_agg_err):
                    sb_lines.append(
                        f"σ(⟨RV⟩) (aggregate uncertainty) = "
                        f"{_format_rv_value(float(rv_agg_err), kind='sigma_ms')} "
                        f"({_format_rv_value(float(rv_agg_err) * 100.0, kind='sigma_cm')})"
                    )
                else:
                    sb_lines.append("σ(⟨RV⟩) (aggregate uncertainty) = —")
                if n_good > 0 and np.isfinite(rms_bias_m_s):
                    rms_bias_cm_s2 = rms_bias_m_s * 100.0
                    sb_lines.append(
                        f"RMS scatter vs ⟨RV⟩ (inlier chunks) = "
                        f"{_format_rv_value(rms_bias_m_s, kind='sigma_ms')} "
                        f"({_format_rv_value(rms_bias_cm_s2, kind='sigma_cm')})"
                    )
                    if n_good > 1:
                        std_c = float(np.std(rv[inlier], ddof=1))
                        if np.isfinite(std_c):
                            sb_lines.append(f"Std (inlier RV, sample) = {std_c:.2f} m/s")
                else:
                    sb_lines.append("RMS scatter vs ⟨RV⟩ (inlier chunks) = —")
                osig = float(getattr(primitive, "outlier_sigma", 4.0))
                _, _, mad_gate = combine_rvs_median_mad(rv, rv_err, osig)
                _, _, inl_c, sig_iv, scatter_se = combine_rv_estimates_median_mad_weighted(
                    rv, rv_err, osig
                )
                thr_m_s = osig * mad_gate if np.isfinite(mad_gate) else float("nan")
                sb_lines.append(
                    f"MAD (chunk RVs, for gate) = {_format_rv_value(mad_gate, kind='sigma_ms')} · "
                    f"cut = σ×MAD = {_format_rv_value(thr_m_s, kind='sigma_ms')} (σ={osig:g})"
                )
                sb_lines.append(f"Chunks flagged outlier (MAD gate): {int(np.sum(~inl_c))}")
                if np.isfinite(sig_iv):
                    sb_lines.append(
                        f"σ_IV (inverse-variance SE, inliers) = "
                        f"{_format_rv_value(sig_iv, kind='sigma_ms')}"
                    )
                if np.isfinite(scatter_se):
                    sb_lines.append(
                        f"scatter_SE (MAD of inlier RVs / √N) = "
                        f"{_format_rv_value(scatter_se, kind='sigma_ms')}"
                    )
                rv_sidebar_by_fiber[fiber] = "<br>".join(sb_lines)
                if np.any(in_ok):
                    _pk = (
                        "<b>Plot key</b><br>"
                        "Blue markers: inlier chunks (pipeline RV vs λ)<br>"
                    )
                    if np.any(out_ok):
                        _pk += "Grey open circles: outlier chunks<br>"
                    _pk += (
                        "Red dashed: ΔRV = 0 vs ⟨RV⟩ (reference)<br>"
                        "Red dotted: ±1σ(⟨RV⟩) on the cm/s offset axis"
                    )
                    rv_plot_legend_by_fiber[fiber] = _pk
                else:
                    rv_plot_legend_by_fiber[fiber] = (
                        "<b>Plot key</b><br>"
                        "Blue markers: per-chunk pipeline RV vs λ<br>"
                        "Red dashed: ΔRV = 0 vs ⟨RV⟩ (reference)<br>"
                        "Red dotted: ±1σ(⟨RV⟩) on the cm/s offset axis"
                    )
                fig_rv.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=[0.0, 0.0],
                        mode="lines",
                        line=dict(dash="dash", color="crimson", width=2),
                        name=leg,
                        showlegend=False,
                    ),
                    row=row_rv,
                    col=1,
                )
                if np.isfinite(rv_agg_err) and rv_agg_err > 0:
                    sig_cm = float(rv_agg_err) * 100.0
                    sig_cm_lbl = _format_rv_value(sig_cm, kind="sigma_cm")
                    for y_val, label in [(sig_cm, f"+1σ ({sig_cm_lbl})"), (-sig_cm, f"-1σ ({sig_cm_lbl})")]:
                        fig_rv.add_trace(
                            go.Scatter(
                                x=x_range,
                                y=[y_val, y_val],
                                mode="lines",
                                line=dict(dash="dot", color="crimson", width=1),
                                name=label,
                                showlegend=False,
                            ),
                            row=row_rv,
                            col=1,
                        )

    for ri, entry in enumerate(row_plan_ccf):
        kind, fiber = entry[0], entry[1]
        row = (
            fiber_row_map[fiber]["ccf"]
            if kind == "ccf"
            else fiber_row_map[fiber]["resid"]
        )

        if kind == "ccf":
            detail = primitive._ccf_preview_by_fiber.get(fiber)
            if not detail:
                continue
            v_step = float(detail["v_step"])
            slots = detail["slots"]
            is_mask_kpf = getattr(primitive, "algorithm", None) == "ccf_mask_kpf"
            ccf_trace_indices[fiber] = []
            ccf_ylim_lo, ccf_ylim_hi = -1.0, 1.0
            for si, sl in enumerate(slots):
                params, diag = fit_slot_cache.get(id(sl), (None, {}, np.nan))[:2]
                vg = np.asarray(sl["v"], dtype=np.float64)
                ccf = np.asarray(sl["ccf"], dtype=np.float64)
                _ce = sl.get("ccf_err")
                vg_plot, ccf_plot, _ = _downsample_ccf_for_plot(
                    vg, ccf, _ce, PREVIEW_CCF_SCATTER_MAX_POINTS
                )
                vg_km = vg_plot / 1000.0  # x-axis in km/s (plot subset)
                visible = si == 0
                if si == 0:
                    ccf_ylim_lo, ccf_ylim_hi = _preview_ccf_y_limits(ccf)
                ccf_band_trace_idx: list[int] = []
                idx_data = len(fig_ccf.data)
                ccf_scatter_kw: dict = dict(
                    x=vg_km,
                    y=ccf_plot,
                    mode="markers",
                    name="CCF data",
                    marker=dict(size=3, color="darkgreen"),
                    showlegend=True,
                    visible=visible,
                )
                fig_ccf.add_trace(go.Scatter(**ccf_scatter_kw), row=row, col=1)
                ccf_band_trace_idx.append(idx_data)
                idx_fit = len(fig_ccf.data)
                if params and diag.get("fit_ok"):
                    # Preview: narrow symmetric window around v₀ — do not use LevMar fit_v_lo/hi
                    # (often ±500 km/s), which makes the Gaussian trace look biased and flat.
                    v_lo_m, v_hi_m = _preview_gaussian_curve_v_range_m_s(
                        float(params["v0"]),
                        float(params["sigma"]),
                        vg,
                    )
                    v_lo_m, v_hi_m = clip_gaussian_plot_velocity_range_m_s(
                        v_lo_m,
                        v_hi_m,
                        v_grid_min_m_s=diag.get("ccf_velocity_grid_min_m_s"),
                        v_grid_max_m_s=diag.get("ccf_velocity_grid_max_m_s"),
                        v_samples_m_s=vg,
                    )
                    n_vf = max(
                        50,
                        min(4000, int((v_hi_m - v_lo_m) / max(float(v_step), 1.0))),
                    )
                    vf_m = np.linspace(v_lo_m, v_hi_m, n_vf)
                    cf = _gaussian_ccf_model(
                        vf_m,
                        float(params["amplitude"]),
                        float(params["v0"]),
                        float(params["sigma"]),
                        float(params["baseline"]),
                    )
                    vf_km = vf_m / 1000.0  # x-axis in km/s (match CCF data trace)
                    fig_ccf.add_trace(
                        go.Scatter(
                            x=vf_km,
                            y=cf,
                            mode="lines",
                            name="Gaussian fit",
                            line=dict(width=2, color="crimson"),
                            showlegend=True,
                            visible=visible,
                        ),
                        row=row,
                        col=1,
                    )
                else:
                    fig_ccf.add_trace(
                        go.Scatter(x=[], y=[], showlegend=False, visible=visible),
                        row=row,
                        col=1,
                    )
                ccf_band_trace_idx.append(idx_fit)
                if (not is_mask_kpf) and params and diag.get("fit_ok"):
                    v0_km = float(params["v0"]) / 1000.0
                    idx_v0 = len(fig_ccf.data)
                    fig_ccf.add_trace(
                        go.Scatter(
                            x=[v0_km, v0_km],
                            y=[ccf_ylim_lo, ccf_ylim_hi],
                            mode="lines",
                            name="Gaussian RV (v₀)",
                            line=dict(width=1.5, color="black", dash="dash"),
                            showlegend=True,
                            visible=visible,
                        ),
                        row=row,
                        col=1,
                    )
                    ccf_band_trace_idx.append(idx_v0)
                _bgp = (params or {}).get("baseline")
                if _bgp is not None and np.isfinite(float(_bgp)):
                    idx_bg = len(fig_ccf.data)
                    _bgy = float(_bgp)
                    vg_km_lo = float(np.nanmin(vg_km))
                    vg_km_hi = float(np.nanmax(vg_km))
                    fig_ccf.add_trace(
                        go.Scatter(
                            x=[vg_km_lo, vg_km_hi],
                            y=[_bgy, _bgy],
                            mode="lines",
                            name="Peak-mask baseline",
                            line=dict(
                                dash="dash",
                                width=1.5,
                                color="rgba(100,100,100,0.9)",
                            ),
                            showlegend=True,
                            visible=visible,
                        ),
                        row=row,
                        col=1,
                    )
                    ccf_band_trace_idx.append(idx_bg)
                if is_mask_kpf:
                    for ti in _mask_kpf_refine_ccf_add_plotly_traces(
                        fig_ccf,
                        row,
                        sl=sl,
                        vg=vg,
                        ccf=ccf,
                        params=params,
                        diag=diag,
                        visible=visible,
                        show_legend=True,
                        ylim_lo=ccf_ylim_lo,
                        ylim_hi=ccf_ylim_hi,
                    ):
                        ccf_band_trace_idx.append(ti)
                ccf_trace_indices[fiber].append(ccf_band_trace_idx)
            fig_ccf.update_yaxes(range=[ccf_ylim_lo, ccf_ylim_hi], autorange=False, row=row, col=1)
            continue

        # resid — CCF data minus the same Gaussian model evaluated on the full velocity grid.
        detail = primitive._ccf_preview_by_fiber.get(fiber)
        if not detail:
            continue
        slots = detail["slots"]
        resid_trace_indices[fiber] = []
        r_abs_peak = 0.0
        for sl in slots:
            params, diag = fit_slot_cache.get(id(sl), (None, {}, np.nan))[:2]
            vg = np.asarray(sl["v"], dtype=np.float64)
            ccf = np.asarray(sl["ccf"], dtype=np.float64)
            if params and diag.get("fit_ok"):
                model_at = _gaussian_ccf_model(
                    vg,
                    float(params["amplitude"]),
                    float(params["v0"]),
                    float(params["sigma"]),
                    float(params["baseline"]),
                )
                residual = ccf - model_at
                fin = residual[np.isfinite(residual)]
                if fin.size:
                    r_abs_peak = max(
                        r_abs_peak,
                        float(np.nanpercentile(np.abs(fin), 99.5)),
                    )
        pad_r = max(r_abs_peak * 0.12, 1e-15)
        if r_abs_peak > 0.0 and math.isfinite(r_abs_peak):
            resid_ylim_by_fiber[fiber] = (-r_abs_peak - pad_r, r_abs_peak + pad_r)
        else:
            resid_ylim_by_fiber[fiber] = (-1.0, 1.0)

        for si, sl in enumerate(slots):
            params, diag = fit_slot_cache.get(id(sl), (None, {}, np.nan))[:2]
            vg = np.asarray(sl["v"], dtype=np.float64)
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
            vg_res, residual_plot, _ = _downsample_ccf_for_plot(
                vg, residual, None, PREVIEW_CCF_SCATTER_MAX_POINTS
            )
            vg_km_res = vg_res / 1000.0
            r_slot_idx: list[int] = []
            resid_kw: dict = dict(
                x=vg_km_res,
                y=residual_plot,
                mode="markers",
                name="Residual CCF",
                marker=dict(size=3, color="darkslateblue"),
                showlegend=False,
                visible=visible,
            )
            fig_ccf.add_trace(go.Scatter(**resid_kw), row=row, col=1)
            r_slot_idx.append(len(fig_ccf.data) - 1)
            resid_trace_indices[fiber].append(r_slot_idx)
        fig_ccf.add_hline(
            y=0.0,
            line_dash="dot",
            line_color="gray",
            line_width=1,
            row=row,
            col=1,
        )

    n_traces = len(fig_ccf.data)

    # CCF mean vs order slot — update visibility + CCF-panel annotation text.
    # Build initial annotations (one per fiber, for CCF panel) — domain refs (x,y) are 0–1 in panel
    annotations: list[dict] = []
    for fi, fiber in enumerate(fibers_with_ccf):
        ccf_row = fiber_row_map[fiber]["ccf"]
        xref, yref = _plotly_domain_axis_refs(ccf_row)
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
                font=dict(size=7, family="Arial, sans-serif"),
                bgcolor="rgba(255,255,255,0.92)",
                borderpad=2,
            )
        )

    # Shared with updatemenus so order-slot changes do not reset legend position / visibility.
    ccf_legend_dict: dict = dict(
        x=1.01,
        xref="paper",
        xanchor="left",
        y=0.88,
        yref="paper",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.92)",
        borderwidth=1,
        bordercolor="rgba(0,0,0,0.12)",
        font=dict(size=8),
        tracegroupgap=4,
        itemwidth=30,
    )

    updatemenus_ccf: list[dict] = []
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
                label = "Mean (all ord)"
            else:
                o = sl.get("order", "?")
                wl_lo, wl_hi = sl.get("wl_lo", np.nan), sl.get("wl_hi", np.nan)
                if np.isfinite(wl_lo) and np.isfinite(wl_hi):
                    label = f"Ord {o} · {wl_lo:.3f}–{wl_hi:.3f} µm"
                else:
                    label = f"Ord {o}"
            visible = [True] * n_traces
            for slot_idx in range(len(slots)):
                if slot_idx != si:
                    if slot_idx < len(ccf_indices):
                        for ti in ccf_indices[slot_idx]:
                            visible[ti] = False
                    if slot_idx < len(res_indices):
                        for ti in res_indices[slot_idx]:
                            visible[ti] = False
            new_annotations = [dict(a) for a in annotations]
            if fi < len(new_annotations) and si < len(meta_list):
                new_annotations[fi]["text"] = meta_list[si]
            buttons.append(
                dict(
                    label=label,
                    method="update",
                    args=[
                        {"visible": visible},
                        {
                            "annotations": new_annotations,
                            "showlegend": True,
                            "legend": ccf_legend_dict,
                        },
                    ],
                )
            )
        if not buttons:
            continue
        # Position dropdown inside this fiber's CCF panel (paper coords), low in the cell
        # so it clears the figure legend (paper x≈1).
        ccf_row_1based = fiber_row_map[fiber]["ccf"]
        row_height = (1.0 - (n_rows_ccf - 1) * v_spacing_ccf) / n_rows_ccf
        from_top = (ccf_row_1based - 0.5) * row_height + (ccf_row_1based - 1) * v_spacing_ccf
        y_dropdown = 1.0 - from_top - 0.90 * row_height
        y_dropdown = max(0.03, min(0.97, y_dropdown))
        updatemenus_ccf.append(
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                showactive=True,
                x=1.06,
                xanchor="left",
                y=y_dropdown,
                yanchor="middle",
                font=dict(size=9),
                bgcolor="rgba(255,255,255,0.9)",
                borderwidth=1,
                bordercolor="gray",
            )
        )

    rv_fig_w = 980
    rv_fig_mr = 420
    annotations_rv: list[dict] = []
    for fi, fiber in enumerate(fibers_with_ccf):
        r = fiber_row_map[fiber]["rv"]
        x_p_der, y_p_der = _rv_sidebar_paper_xy(fig_rv, r, 0.98)
        text = rv_sidebar_by_fiber.get(fiber, "")
        if text and str(text).strip():
            annotations_rv.append(
                dict(
                    xref="paper",
                    yref="paper",
                    x=x_p_der,
                    y=y_p_der,
                    xshift=RV_SIDEBAR_XSHIFT_PX,
                    text=text,
                    showarrow=False,
                    xanchor="left",
                    yanchor="top",
                    font=dict(size=11, family="Arial, sans-serif"),
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="rgba(0,0,0,0.15)",
                    borderwidth=1,
                    borderpad=6,
                    align="left",
                )
            )
        leg_txt = rv_plot_legend_by_fiber.get(fiber, "")
        if leg_txt and str(leg_txt).strip():
            x_p_key, y_p_key = _rv_sidebar_paper_xy(fig_rv, r, 0.30)
            annotations_rv.append(
                dict(
                    xref="paper",
                    yref="paper",
                    x=x_p_key,
                    y=y_p_key,
                    xshift=RV_SIDEBAR_XSHIFT_PX,
                    text=leg_txt,
                    showarrow=False,
                    xanchor="left",
                    yanchor="top",
                    font=dict(size=10, family="Arial, sans-serif"),
                    bgcolor="rgba(255,255,255,0.88)",
                    bordercolor="rgba(0,0,0,0.12)",
                    borderwidth=1,
                    borderpad=5,
                    align="left",
                )
            )

    algo = getattr(primitive, "algorithm", None) or "ccf_stellar_template"
    title_suffix = ""
    if hasattr(output_model, "meta") and getattr(output_model.meta, "filename", None):
        title_suffix = f" — {output_model.meta.filename}"
    title_unified = f"Stellar RV — unified preview — algorithm: {algo}{title_suffix}"

    fig_h_rv = int(min(20000, 380 + max(0, n_fib - 1) * 118))
    fig_h_ccf = int(min(20000, 380 + max(0, n_rows_ccf - 1) * 118))
    # ``make_subplots(..., subplot_titles=...)`` stores titles as layout.annotations. Passing only
    # ``annotations_rv`` would patch those entries by index and replace the title with sidebar
    # content; prepend the existing annotations so Derived / plot-key boxes are appended.
    rv_layout_ann: list[dict] = []
    _rv_la = getattr(fig_rv.layout, "annotations", None)
    if _rv_la:
        for _a in _rv_la:
            rv_layout_ann.append(_a.to_plotly_json())
    fig_rv.update_layout(
        height=fig_h_rv,
        width=rv_fig_w,
        title_text=f"{title_unified} — RV vs λ",
        hovermode="closest",
        dragmode="zoom",
        # Traces use showlegend=False; keep legend off so the right margin is for Derived / plot key.
        showlegend=False,
        margin=dict(l=70, r=rv_fig_mr, t=80, b=50),
        annotations=rv_layout_ann + annotations_rv,
    )
    fig_ccf.update_layout(
        height=fig_h_ccf,
        width=980,
        title_text=f"{title_unified} — CCF & residual",
        hovermode="closest",
        dragmode="zoom",
        showlegend=True,
        legend=ccf_legend_dict,
        margin=dict(l=70, r=320, t=80, b=50),
        annotations=list(annotations),
        updatemenus=updatemenus_ccf,
    )
    # Plotly ≥5.24 / plotly.js: hover labels stay on the subplot under the cursor (no strip across rows).
    try:
        fig_rv.update_layout(hoversubplots="single")
        fig_ccf.update_layout(hoversubplots="single")
    except Exception:
        pass
    fig_rv.update_xaxes(showgrid=True)
    fig_rv.update_yaxes(showgrid=True)
    fig_ccf.update_xaxes(showgrid=True)
    fig_ccf.update_yaxes(showgrid=True)

    for fi, fiber in enumerate(fibers_with_ccf):
        row_rv_ax = fiber_row_map[fiber]["rv"]
        fig_rv.update_xaxes(title_text="Wavelength (µm)", row=row_rv_ax, col=1)
        y_kwargs: dict = {
            "title_text": rv_panel_y_title.get(fiber, "RV (cm/s) − ⟨RV⟩"),
        }
        if fiber in rv_panel_y_range:
            y_kwargs["range"] = list(rv_panel_y_range[fiber])
            y_kwargs["autorange"] = False
        fig_rv.update_yaxes(row=row_rv_ax, col=1, **y_kwargs)

    for ri, entry in enumerate(row_plan_ccf):
        kind, fiber = entry[0], entry[1]
        row_ax = (
            fiber_row_map[fiber]["ccf"]
            if kind == "ccf"
            else fiber_row_map[fiber]["resid"]
        )
        if kind == "ccf":
            fig_ccf.update_xaxes(
                range=[v_axis_km_lo, v_axis_km_hi],
                showticklabels=False,
                row=row_ax,
                col=1,
            )
            fig_ccf.update_yaxes(
                title_text="CCF",
                row=row_ax,
                col=1,
            )
        else:
            # Share velocity zoom/pan with the CCF row above (``matches``); range is set on CCF x-axis.
            fig_ccf.update_xaxes(
                title_text="Velocity shift (km/s)",
                matches=_plotly_subplot_x_anchor_name(fiber_row_map[fiber]["ccf"]),
                row=row_ax,
                col=1,
            )
            ry0, ry1 = resid_ylim_by_fiber.get(fiber, (-1.0, 1.0))
            fig_ccf.update_yaxes(
                title_text="Residual CCF",
                range=[ry0, ry1],
                autorange=False,
                row=row_ax,
                col=1,
            )

    logger.info(
        "Stellar RV preview: two figures ready (n_plotly_traces rv=%d ccf=%d)",
        len(fig_rv.data),
        len(fig_ccf.data),
    )

    output_path_html, output_path_png, wrote_png, _combined_html = (
        _preview_export_combined_html(primitive, output_model, fig_rv, fig_ccf, title_unified)
    )
    logger.info("Stellar RV preview: create_stellar_rv_preview finished")

    primitive._output["preview"] = {
        "html": output_path_html,
        "png": output_path_png if wrote_png else None,
        "fig": fig_rv,
        "fig_rv": fig_rv,
        "fig_ccf": fig_ccf,
        # Deprecated aliases: same combined HTML / stacked PNG as primary keys.
        "html_ccf": output_path_html,
        "png_ccf": output_path_png if wrote_png else None,
    }

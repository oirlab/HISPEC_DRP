"""
Interactive previews for stellar radial-velocity products.

There are **three** preview surfaces:

1. **Default preview** (``create_stellar_rv_preview``): **two** Plotly HTML embeds per save — (a) RV vs
   λ per fiber with its own mode bar; (b) CCF + residual per fiber (linked velocity axes within each
   fiber), second mode bar, slot dropdown. For ``ccf_line_mask`` with
   mask-overlay payload, the top row is the same spectrum/mask content as the standalone
   ``*_mask_overlay_debug_fib*.html`` (initial overlay velocity: finite ``ccf_input_velocity`` if
   set, else ``0`` m/s; chunk dropdown uses ``restyle``). CCF slot dropdown aligns with the CCF
   legend (``x``). CCF panels overlay the Gaussian fit, bisector midpoints / wing chords when
   ``bisector_fit`` is present, and the final Gaussian RV.
2. **Spectrum + CCF overlay debug HTML** (``write_mask_overlay_debug_pages`` /
   ``write_template_overlay_debug_pages``): one shared page implementation
   (:mod:`~hispecdrp.primitives.stellar_rv.previews.ccf_spectrum_overlay_debug_html` + embedded
   ``ccf_spectrum_overlay_unified.html.in``) with an ``overlay_kind`` switch. **Mask** mode
   (``ccf_line_mask``): per-**chunk** spectrum (equal-µm bins from ``window_size_um`` /
   :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.ccf_order_n_chunks_for_span`, or for
   ``window_line_by_line`` the symmetric ``window_size_um`` band about each predicted observed
   mask line at ``ccf_input_velocity`` / :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.mask_line_window_spectrum_band_um`)
   with shifted mask
   segments and the **pipeline** mask CCF per chunk (same Numba grid as RVCCF; the overlay restores
   the **raw** weighted mask statistic when ``LINE_MASK_CCF_STORED_NEGATED_FROM_RAW`` is true in
   ``ccf_utils``, matching the absorption-shaped CCF from the data). **Template** mode (``ccf_stellar_template``):
   same chrome, but the spectrum panel overplots the median-normalized rest-frame template
   Doppler-shifted onto each chunk’s wavelength samples; the CCF panel still uses the **pipeline**
   per-order template CCF from ``_ccf_preview_by_fiber`` (one curve per echelle order).
   Template CCF values are weighted covariances in data units and are often **much smaller than
   10⁻⁷**; they are embedded with **significant-figure** rounding so fixed-decimal JSON does not
   zero the curve (the main Plotly preview passes float64 arrays and is unaffected).
   When the pipeline built ``_ccf_preview_by_fiber``, mask mode can include ``ccf_v_m_s`` and
   ``ccf_err`` (quadrature of flux-error–propagated uncertainty on the mask-weighted mean and
   weighted line-to-line dispersion of ``F*SN``; see ``_calc_mask_ccf_chunks_numba``), interpolated onto
   the interactive CCF curve.
   Payload arrays are **downsampled** before embedding; limits come from env
   ``HISPEC_DRP_MASK_OVERLAY_SPECTRUM_MAX_POINTS`` / ``HISPEC_DRP_MASK_OVERLAY_CCF_MAX_POINTS``
   (template crop length: ``HISPEC_DRP_TEMPLATE_OVERLAY_MAX_TEMPLATE_POINTS``).
  **Lagrange pixel-interpolation** curve on the mask spectrum panel (same stencil as
  ``rv_per_chunks_line_mask`` / ``mask_ccf_pixel_interp_order``) is **always** embedded in mask
  overlay debug HTML. The dense λ grid size is derived from the displayed spectrum point count:
  ``n_dense = 2 * mask_ccf_pixel_interp_order * N_specpoints`` (clamped to ``[24, 8192]``),
  where ``N_specpoints`` is the number of points in the plotted blue spectrum trace.

Gaussian reference curves use :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.fit_ccf_rv`;
mask geometry uses :mod:`~hispecdrp.primitives.utils.ccf.ccf_utils` (mask loaders).

**Entry points.** :func:`create_stellar_rv_preview` builds the standard multi-panel HTML;
:func:`maybe_write_ccf_overlay_debug_html` dispatches from the primitive’s algorithm and overlay
flags into the writers below; :func:`write_mask_overlay_debug_pages` /
:func:`write_template_overlay_debug_pages` write the shared spectrum+CCF overlay pages
(mask vs. template overlay);
:func:`collect_mask_overlay_payload_for_primitive` and related helpers assemble mask/spectrum
payloads from ``ccf_utils`` and the L2 model. Environment variables
``HISPEC_DRP_MASK_OVERLAY_*`` and ``HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS`` adjust downsampling and
browser behavior (see in-file constants and call sites).

Notes
-----
Plotly layout code is **HISPEC DRP–original**. For mask file format and upstream
line lists, see ``hispecdrp/data/masks/README.md`` and spirou-ccf/KPF references
in :mod:`~hispecdrp.primitives.utils.ccf.ccf_utils`.

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
import multiprocessing as mp
import os
import tempfile
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from astropy import units as u

from ....datamodels.dqflags import DQ_FLAGS
from ....utils.line_mask_utils import restrict_mask_to_chunk, stellar_rv_ccf_mask_lines
from ....utils.physical_constants import C_M_S
from ..rv_combine_utils import (
    combine_rv_estimates_median_mad_weighted,
    combine_rvs_median_mad,
)
from ...utils.ccf.ccf_parabolic import (
    ensure_line_mask_lstsq_preview_plot_metadata,
    refine_ccf_rv_coarse_grid_parabolic,
)
from ...utils.ccf.ccf_utils import (
    CCFFitConfig,
    LINE_MASK_CCF_STORED_NEGATED_FROM_RAW,
    MAX_MASK_CCF_PIXEL_INTERP_ORDER,
    _gaussian_ccf_model,
    build_centered_uniform_velocity_grid,
    build_line_mask_chunk_wavelength_intervals_line_by_line,
    ccf_order_n_chunks_for_span,
    clip_gaussian_plot_velocity_range_m_s,
    effective_ccf_v_step_for_fit,
    fit_ccf_rv,
    mask_line_window_spectrum_band_um,
    resolve_ccf_broad_grid_m_s,
    STELLAR_RV_LINE_MASK_ALGORITHMS,
)

from .ccf_spectrum_overlay_debug_html import build_ccf_spectrum_overlay_debug_html

logger = logging.getLogger(__name__)

# Kaleido/Chrome PNG export can hang on large line-mask CCF figures (many Plotly traces).
_DEFAULT_KALEIDO_PNG_MAX_CCF_TRACES = 200
_DEFAULT_KALEIDO_PNG_TIMEOUT_S = 90.0


def _preview_kaleido_png_max_ccf_traces() -> int:
    raw = os.environ.get("HISPEC_DRP_PREVIEW_PNG_MAX_CCF_TRACES")
    if raw is None or str(raw).strip() == "":
        return _DEFAULT_KALEIDO_PNG_MAX_CCF_TRACES
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_KALEIDO_PNG_MAX_CCF_TRACES


def _preview_kaleido_png_timeout_s() -> float:
    raw = os.environ.get("HISPEC_DRP_PREVIEW_PNG_TIMEOUT_S")
    if raw is None or str(raw).strip() == "":
        return _DEFAULT_KALEIDO_PNG_TIMEOUT_S
    try:
        return max(5.0, float(raw))
    except ValueError:
        return _DEFAULT_KALEIDO_PNG_TIMEOUT_S


def _kaleido_render_png_worker(fig_json: str, scale: float) -> bytes:
    """Render one Plotly figure to PNG in an isolated process (for timeout/kill)."""
    fig = pio.from_json(fig_json)
    try:
        return fig.to_image(format="png", engine="kaleido", scale=scale)
    except Exception:
        import kaleido

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            kaleido.write_fig_sync(fig, path=tmp_path, opts={"scale": scale})
            return Path(tmp_path).read_bytes()
        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)


def _figure_to_png_bytes_with_timeout(
    fig: go.Figure,
    *,
    scale: float,
    timeout_s: float,
) -> bytes | None:
    """Return PNG bytes or ``None`` if Kaleido exceeds ``timeout_s`` (process is terminated)."""
    fig_json = fig.to_json()
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(1)
    try:
        async_result = pool.apply_async(
            _kaleido_render_png_worker,
            (fig_json, float(scale)),
        )
        return async_result.get(timeout=float(timeout_s))
    except (TimeoutError, mp.context.TimeoutError):
        logger.warning(
            "Stellar RV preview: Kaleido PNG export timed out after %.0f s; skipping PNG",
            timeout_s,
        )
        pool.terminate()
        pool.join()
        return None
    except Exception as exc:
        logger.warning(
            "Stellar RV preview: Kaleido PNG export failed: %s",
            exc,
        )
        return None
    finally:
        pool.close()


# Inner HTML for ``ccf_spectrum_overlay_unified.html.in`` help paragraph (trusted markup only).
_MASK_OVERLAY_SPECTRUM_HELP_HTML = (
    "Green band/tick: order RV ±1σ.<br>"
    "Dashed green: chunk-center λ.<br>"
    "Orange: mask weights.<br>"
    "Red: Lagrange-interpolated spectrum (same order as ``mask_ccf_pixel_interp_order``).<br>"
)
_TEMPLATE_OVERLAY_SPECTRUM_HELP_HTML = (
    "    Orange curve: rest-frame stellar template (median-normalized per order in the payload), "
    "Doppler-shifted into the observed frame (λ<sub>rest</sub> = λ<sub>obs</sub> / (1 + v/c)), "
    "then min–max scaled to [0, 1] on the plotted echelle pixels (same scaling style as the black spectrum). "
    "Green band: pipeline RV ±1σ for this order (IV mean over template-CCF chunks)."
)

# Plotly CCF scatter/residual traces: cap points to keep preview HTML small and write_html fast.
PREVIEW_CCF_SCATTER_MAX_POINTS = 200
# Max CCF preview slots embedded per fiber (slot 0 = fiber mean). Extra order slots dominate HTML size.
PREVIEW_CCF_MAX_PLOT_SLOTS = 1

# RV vs λ: ``xref="paper"`` is the axes **inner** box (0–1), not the full figure. Anchor at ``x=1``
# and shift right by this many pixels so “Derived” sits just outside the frame in the margin.
RV_SIDEBAR_XSHIFT_PX = 28
# Figure width / right margin must leave room for the Derived sidebar (long labels were clipped).
RV_FIG_WIDTH = 820
RV_FIG_MARGIN_RIGHT = 340

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


# Extra downward shift in normalized paper coordinates (smaller ``y`` = lower on the figure).
# Retained for call-site compatibility; dropdown placement no longer uses this offset.
_CCF_SLOT_DROPDOWN_EXTRA_DROP_PAPER = 0.10


def _ccf_slot_dropdown_paper_xy(
    fig_ccf: go.Figure,
    ccf_row_1based: int,
    *,
    legend: dict,
) -> tuple[float, float]:
    """
    Paper ``(x, y)`` for the CCF slot ``updatemenus`` dropdown (``xref`` / ``yref``: ``paper``).

    Horizontally matches the CCF legend (``legend["x"]``, ``xanchor="left"``).
    Vertically sits near the **top** of this fiber's CCF subplot row so the control
    stays with its panel; the shared legend is placed lower on the figure so the two
    do not overlap.
    """
    legend_x = float(legend.get("x", 1.02))
    yn = "yaxis" if int(ccf_row_1based) <= 1 else f"yaxis{int(ccf_row_1based)}"
    dom = getattr(fig_ccf.layout, yn).domain
    y0, y1 = float(dom[0]), float(dom[1])
    # Near the top of this CCF row (leave a small inset below the subplot title).
    y = y0 + 0.82 * (y1 - y0)
    y = max(y0 + 0.02, min(y, y1 - 0.02))
    return legend_x, y


# Mask-overlay debug HTML: full echelle pixels × many orders + long CCF(v) grids inflate inline JSON.
MASK_OVERLAY_SPECTRUM_MAX_POINTS_DEFAULT = 2800
MASK_OVERLAY_CCF_MAX_POINTS_DEFAULT = 4000
TEMPLATE_OVERLAY_MAX_TEMPLATE_POINTS_DEFAULT = 12000


def _env_positive_int(name: str, default: int, *, floor: int = 32) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return max(floor, int(default))
    try:
        return max(floor, int(raw))
    except ValueError:
        return max(floor, int(default))


def _truthy_env(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


def _mask_unique_sorted_xy(w: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Finite (w,y), ascending ``w``; duplicate ``w`` → mean ``y`` (stable for Lagrange stencil)."""
    w = np.asarray(w, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    m = np.isfinite(w) & np.isfinite(y)
    w = w[m]
    y = y[m]
    if w.size == 0:
        return w, y
    o = np.argsort(w, kind="mergesort")
    w = w[o]
    y = y[o]
    uw, inv, cnt = np.unique(w, return_inverse=True, return_counts=True)
    if int(uw.size) == int(w.size):
        return w, y
    y_sum = np.bincount(inv, weights=y)
    y_mean = y_sum / np.maximum(cnt.astype(np.float64), 1.0)
    return uw, y_mean


def _poly_interp_sorted_1d_numpy(
    xq: float,
    x: np.ndarray,
    y: np.ndarray,
    order: int,
) -> float:
    """
    Lagrange interpolation on sorted ``x`` (ascending); matches Numba
    :func:`~hispecdrp.primitives.utils.ccf.ccf_utils._poly_interp_sorted_1d`.
    """
    n = int(x.size)
    if n <= 0:
        return float("nan")
    if n == 1:
        return float(y[0])
    o = int(order)
    if o < 1:
        o = 1
    if o > MAX_MASK_CCF_PIXEL_INTERP_ORDER:
        o = MAX_MASK_CCF_PIXEL_INTERP_ORDER
    k = o + 1
    if k > n:
        k = n
    if xq <= float(x[0]):
        return float(y[0])
    if xq >= float(x[n - 1]):
        return float(y[n - 1])
    j = int(np.searchsorted(x, xq, side="right")) - 1
    if j < 0:
        j = 0
    if j >= n - 1:
        j = n - 2
    s = j - (k - 1) // 2
    if s < 0:
        s = 0
    if s + k > n:
        s = n - k
    acc = 0.0
    for mm in range(k):
        xm = float(x[s + mm])
        ym = float(y[s + mm])
        lm = 1.0
        for t in range(k):
            if t == mm:
                continue
            xt = float(x[s + t])
            d = xm - xt
            if d == 0.0:
                return ym
            lm *= (xq - xt) / d
        acc += ym * lm
    return float(acc)


def _mask_overlay_spec_interp_curve(
    w_native_um: np.ndarray,
    s_norm: np.ndarray,
    *,
    pixel_interp_order: int,
    n_dense: int,
    spec_downsample_lim: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Dense λ grid + Lagrange-interpolated median-normalized spectrum (same convention as mask CCF).

    Returns downsampled (wave_um, spec) for JSON embedding, or ``None`` if not enough points.
    """
    ws, ys = _mask_unique_sorted_xy(w_native_um, s_norm)
    if ws.size < 2:
        return None
    pio = int(pixel_interp_order)
    if pio < 1:
        pio = 1
    pio = min(pio, MAX_MASK_CCF_PIXEL_INTERP_ORDER)
    lo = float(ws[0])
    hi = float(ws[-1])
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return None
    nd = max(24, min(int(n_dense), 8192))
    ng = min(nd, max(64, 8 * int(ws.size)))
    g = np.linspace(lo, hi, num=ng, dtype=np.float64)
    yi = np.empty_like(g, dtype=np.float64)
    for i in range(int(g.size)):
        yi[i] = _poly_interp_sorted_1d_numpy(float(g[i]), ws, ys, pio)
    w_ds, y_ds, _ = _downsample_ccf_for_plot(g, yi, None, int(spec_downsample_lim))
    if w_ds.size < 2:
        return None
    return w_ds, y_ds


def _mask_overlay_downsample_limits() -> tuple[int, int]:
    """Max points per order for embedded spectrum (wave, spec) and CCF σ(v) in overlay HTML."""
    spec_lim = _env_positive_int(
        "HISPEC_DRP_MASK_OVERLAY_SPECTRUM_MAX_POINTS",
        MASK_OVERLAY_SPECTRUM_MAX_POINTS_DEFAULT,
    )
    ccf_lim = _env_positive_int(
        "HISPEC_DRP_MASK_OVERLAY_CCF_MAX_POINTS",
        MASK_OVERLAY_CCF_MAX_POINTS_DEFAULT,
    )
    return spec_lim, ccf_lim


def _template_overlay_max_template_points_from_env() -> int:
    """Max rest-frame template samples per order embedded in template-overlay debug HTML."""
    return _env_positive_int(
        "HISPEC_DRP_TEMPLATE_OVERLAY_MAX_TEMPLATE_POINTS",
        TEMPLATE_OVERLAY_MAX_TEMPLATE_POINTS_DEFAULT,
        floor=256,
    )


def _overlay_v_init_m_s(primitive) -> float:
    """Initial RV (m/s) for spectrum+CCF overlay HTML: ``ccf_input_velocity`` when finite, else 0."""
    v = getattr(primitive, "ccf_input_velocity", None)
    if v is None:
        return 0.0
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return 0.0
    return fv if np.isfinite(fv) else 0.0


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
    """Velocity extent for CCF preview axes (m/s).

    Matches stellar-RV defaults: ``±v_half_window`` about ``ccf_input_velocity``
    when set, otherwise ``±v_half_window`` about 0 — **not** the ±500 km/s scout
    grid and **not** a per-fiber scout-centered refined window (those can sit
    hundreds of km/s from zero when auto-center fails).
    """
    _bvm, _bvx, bvs = resolve_ccf_broad_grid_m_s(
        getattr(primitive, "v_min_coarse", None),
        getattr(primitive, "v_max_coarse", None),
        getattr(primitive, "v_step_coarse", None),
    )
    half = float(getattr(primitive, "v_half_window", 20000.0))
    vc = getattr(primitive, "ccf_input_velocity", None)
    if vc is None or not np.isfinite(float(vc)):
        return -half, +half, float(bvs)
    vg = build_centered_uniform_velocity_grid(
        float(vc),
        half,
        float(bvs),
    )
    return float(vg[0]), float(vg[-1]), float(effective_ccf_v_step_for_fit(vg))


def _preview_max_plot_slots() -> int:
    """How many CCF slots (mean + orders) to embed in the Plotly HTML preview."""
    return _env_positive_int(
        "HISPEC_DRP_PREVIEW_CCF_MAX_PLOT_SLOTS",
        PREVIEW_CCF_MAX_PLOT_SLOTS,
        floor=1,
    )


def _preview_slots_for_plot(slots: list) -> list:
    """Truncate preview slots so HTML stays small (default: fiber-mean only)."""
    if not slots:
        return []
    return list(slots[: _preview_max_plot_slots()])


def _iterative_sigma_clip_ylim(
    y: np.ndarray,
    *,
    y_err: np.ndarray | None = None,
    sigma: float = 3.0,
    max_iter: int = 5,
    pad_frac: float = 0.05,
) -> tuple[float, float]:
    """
    Default y-axis limits for RV scatter: iterative sigma clipping on finite values,
    then padding so the panel spans the \"good\" points.

    When ``y_err`` is provided (same length as ``y``), limits expand to the full
    extent of ``y ± y_err`` for the sigma-clipped inliers so per-chunk error bars
    are not clipped by the axes. Flagged outliers should be omitted from ``y`` /
    ``y_err`` by the caller.
    """

    v = np.asarray(y, dtype=np.float64).ravel()
    e = None if y_err is None else np.asarray(y_err, dtype=np.float64).ravel()
    if e is not None and e.shape != v.shape:
        raise ValueError("y_err must have the same shape as y")
    finite = np.isfinite(v)
    if e is not None:
        finite = finite & np.isfinite(e) & (e >= 0)
    v = v[finite]
    if e is not None:
        e = e[finite]
    if v.size == 0:
        return -1.0, 1.0
    if v.size == 1:
        m = float(v[0])
        half = float(e[0]) if e is not None else max(abs(m) * 1e-6, 1e-9)
        half = max(half, max(abs(m) * 1e-6, 1e-9))
        return m - half, m + half
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
    ef = e[mask] if e is not None else None
    if vf.size == 0:
        vf = v
        ef = e
    if ef is not None and ef.size:
        lo = float(np.min(vf - ef))
        hi = float(np.max(vf + ef))
    else:
        lo, hi = float(np.min(vf)), float(np.max(vf))
    span = hi - lo
    pad = pad_frac * span if span > 0 else max(abs(lo) * pad_frac, 1e-6)
    return lo - pad, hi + pad


def _preview_ccf_y_limits(ccf: np.ndarray) -> tuple[float, float]:
    """
    Y-limits for CCF / residual panels: robust data range plus symmetric padding.

    Uses the central ~99.9% of finite values as the data span, then expands by
    **10% of that span** (5% margin on each side) so traces do not sit on the
    axis edges.
    """
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
    pad = 0.05 * span
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


def _line_mask_preview_gaussian_overlay_params(
    sl: dict,
    params: dict,
    diag: dict,
    vg: np.ndarray,
    ccf: np.ndarray,
) -> tuple[float, float, float, float] | None:
    """
    Display-only Gaussian (amplitude, v0, sigma, baseline) for line-mask CCF previews.

    LevMar ``params`` can carry a near-zero ``amplitude`` while the reported RV uses
    ``line_mask_refine_rmeta`` (LSTSQ parabolic vertex). The overlay uses ``rv_refined_m_s``
    as ``v0`` when present, applies the same ``sigma`` sanity caps as before, then solves for
    **(amplitude, baseline)** by linear least squares on the slot CCF:

    ``ccf ≈ amplitude * exp(-0.5 * ((v - v0) / sigma)^2) + baseline``.

    That keeps the crimson curve and its asymptote aligned with the green data for the
    chosen ``(v0, sigma)`` (the LevMar baseline alone is not re-used when it would leave a
    vertical offset after changing ``v0``/``sigma`` for display).
    """
    if not params or not (diag or {}).get("fit_ok"):
        return None
    v0 = float(params["v0"])
    amp_lm = float(params["amplitude"])
    sig = float(params["sigma"])
    base_lm = float(params["baseline"])
    rmeta = sl.get("line_mask_refine_rmeta") if isinstance(sl, dict) else None
    if isinstance(rmeta, dict) and rmeta.get("ok"):
        rv_r = rmeta.get("rv_refined_m_s")
        if rv_r is not None and math.isfinite(float(rv_r)):
            v0 = float(rv_r)
    vg_a = np.asarray(vg, dtype=np.float64).ravel()
    ccf_a = np.asarray(ccf, dtype=np.float64).ravel()
    if vg_a.size == 0 or ccf_a.size == 0 or vg_a.size != ccf_a.size:
        return amp_lm, v0, sig, base_lm
    v_span = float(np.nanmax(vg_a) - np.nanmin(vg_a)) if vg_a.size > 1 else 0.0
    if v_span > 0.0 and math.isfinite(sig) and sig > 0.45 * v_span:
        sig = min(sig, max(1500.0, 0.11 * v_span))
    sig = max(float(sig), 1e-9)
    G = np.exp(-0.5 * ((vg_a - v0) / sig) ** 2)
    ok = np.isfinite(ccf_a) & np.isfinite(G)
    n_ok = int(np.count_nonzero(ok))
    amp = amp_lm
    base = base_lm
    if n_ok >= 3:
        Gf = G[ok]
        yf = ccf_a[ok]
        M = np.column_stack((Gf, np.ones(n_ok, dtype=np.float64)))
        coeff, _res, rank, _s = np.linalg.lstsq(M, yf, rcond=None)
        if int(rank) >= 2 and np.all(np.isfinite(coeff)):
            amp = float(coeff[0])
            base = float(coeff[1])
        else:
            pk = float(np.nanmax(ccf_a))
            mn = float(np.nanmin(ccf_a))
            contr = (pk - mn) if math.isfinite(pk) and math.isfinite(mn) else 0.0
            emp = (pk - base_lm) if math.isfinite(pk) and math.isfinite(base_lm) else 0.0
            if emp > 0.0:
                amp = max(amp_lm, emp)
            if contr > 0.0 and amp < 0.05 * contr:
                amp = max(amp, 0.85 * contr)
                sig = min(sig, max(1200.0, 0.09 * v_span)) if v_span > 0.0 else sig
    return amp, v0, sig, base


def _line_mask_refine_plotly_traces_from_cached_rmeta(
    fig: go.Figure,
    row: int,
    *,
    rv_r_m_s: float,
    rmeta: dict,
    visible: bool,
    show_legend: bool,
    ylim_lo: float,
    ylim_hi: float,
    bisector_fit: dict | None = None,
) -> list[int]:
    """
    Build five line-mask refine overlay traces from ``rmeta`` (no parabolic solve).

    Slots: (0) smoothed ±1σ band, (1) smoothed CCF, (2–3) reserved placeholders
    (formerly parabolic diagnostics; left empty), (4) final Gaussian RV line.
    When the slot carries ``line_mask_refine_rmeta``, that result is reused.
    """
    trace_indices: list[int] = []

    def _placeholder() -> None:
        fig.add_trace(
            go.Scatter(x=[], y=[], showlegend=False, visible=visible, hoverinfo="skip"),
            row=row,
            col=1,
        )
        trace_indices.append(len(fig.data) - 1)

    def _final_rv_vertical_line() -> None:
        # Pipeline final chunk RV is the Gaussian fit (``rv_r_m_s``).
        rv_line = float(rv_r_m_s)
        rv_km = rv_line / 1000.0
        err_m = float(rmeta.get("rv_err_m_s", float("nan")))
        v_label = f"Final RV (Gaussian) = {rv_line:.2f} m/s"
        if math.isfinite(err_m) and err_m > 0:
            v_label += f" (σ={err_m:.2f} m/s)"
        bdiag = (
            bisector_fit.get("diagnostics")
            if isinstance(bisector_fit, dict)
            else None
        )
        if isinstance(bdiag, dict):
            bis_m = bdiag.get("bis_m_s")
            bis_e = bdiag.get("bis_err_m_s")
            if bis_m is not None and math.isfinite(float(bis_m)):
                v_label += f"<br>BIS = {float(bis_m):.2f} m/s"
                if bis_e is not None and math.isfinite(float(bis_e)):
                    v_label += f" (σ={float(bis_e):.2f})"
        fig.add_trace(
            go.Scatter(
                x=[rv_km, rv_km],
                y=[ylim_lo, ylim_hi],
                mode="lines",
                name=v_label,
                line=dict(width=1.5, color="black", dash="dash"),
                legendgroup="line_mask_refine",
                showlegend=show_legend,
                visible=visible,
            ),
            row=row,
            col=1,
        )
        trace_indices.append(len(fig.data) - 1)

    if not rmeta.get("ok"):
        for _ in range(4):
            _placeholder()
        _final_rv_vertical_line()
        return trace_indices

    # No smoothed refine curve: keep placeholders for slots 0–3, then final RV.
    if "refine_curve_v_m_s" not in rmeta:
        for _ in range(4):
            _placeholder()
        _final_rv_vertical_line()
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
            legendgroup="line_mask_refine",
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
            name="Smoothed CCF (refine basis)",
            line=dict(width=2.0, color="rgb(0,114,178)"),
            legendgroup="line_mask_refine",
            showlegend=show_legend,
            visible=visible,
        ),
        row=row,
        col=1,
    )
    trace_indices.append(len(fig.data) - 1)

    # Slots 2–3: formerly parabolic fit/samples — omitted (pipeline RV is bisector).
    for _ in range(2):
        _placeholder()

    _final_rv_vertical_line()
    return trace_indices


def _bisector_preview_plotly_traces(
    fig: go.Figure,
    row: int,
    *,
    sl: dict,
    vg: np.ndarray,
    ccf: np.ndarray,
    visible: bool,
    show_legend: bool,
) -> list[int]:
    """
    Overlay bisector cut midpoints (and optional wing chords) from ``sl["bisector_fit"]``.

    Falls back to the Queloz ``bisector`` / ``bis_levels`` curve when NEID percentile
    cuts are absent. Always appends **two** traces (real or empty placeholders) so
    CCF slot dropdown indices stay aligned across slots.
    """
    trace_indices: list[int] = []

    def _placeholder() -> None:
        fig.add_trace(
            go.Scatter(x=[], y=[], showlegend=False, visible=visible, hoverinfo="skip"),
            row=row,
            col=1,
        )
        trace_indices.append(len(fig.data) - 1)

    bi = sl.get("bisector_fit")
    if not isinstance(bi, dict):
        _placeholder()
        _placeholder()
        return trace_indices
    bdiag = bi.get("diagnostics") if isinstance(bi.get("diagnostics"), dict) else {}
    if not isinstance(bdiag, dict):
        bdiag = {}
    cuts = bdiag.get("cuts")
    vg_a = np.asarray(vg, dtype=np.float64).ravel()
    ccf_a = np.asarray(ccf, dtype=np.float64).ravel()
    v_mids: list[float] = []
    y_mids: list[float] = []
    v_chords_x: list[float] = []
    y_chords: list[float] = []

    if cuts:
        base = bdiag.get("baseline")
        peak_amp = bdiag.get("peak_amplitude")
        for cut in cuts:
            if not isinstance(cut, dict):
                continue
            vm = cut.get("v_mid_m_s")
            if vm is None or not math.isfinite(float(vm)):
                continue
            vm_f = float(vm)
            # Horizontal cut level in CCF units (not CCF(v_mid), which sits near the peak).
            y_c = cut.get("ccf_level")
            if y_c is None or not math.isfinite(float(y_c)):
                tf = cut.get("target_flux")
                if (
                    tf is not None
                    and base is not None
                    and math.isfinite(float(tf))
                    and math.isfinite(float(base))
                    and peak_amp is not None
                    and math.isfinite(float(peak_amp))
                    and float(peak_amp) > 0
                ):
                    # Infer emission (+) vs absorption (−) from peak vs baseline.
                    sign = 1.0
                    if ccf_a.size and math.isfinite(float(np.nanmax(ccf_a))):
                        if float(np.nanmax(ccf_a)) < float(base):
                            sign = -1.0
                    y_c = float(base) + sign * float(tf)
                elif vg_a.size >= 2 and ccf_a.size == vg_a.size:
                    y_c = float(np.interp(vm_f, vg_a, ccf_a))
                else:
                    y_c = float("nan")
            else:
                y_c = float(y_c)
            v_mids.append(vm_f / 1000.0)
            y_mids.append(y_c)
            vl = cut.get("v_left_m_s")
            vr = cut.get("v_right_m_s")
            if (
                vl is not None
                and vr is not None
                and math.isfinite(float(vl))
                and math.isfinite(float(vr))
                and math.isfinite(y_c)
            ):
                # Horizontal chord spanning the full CCF width at this depth.
                v_chords_x.extend([float(vl) / 1000.0, float(vr) / 1000.0, float("nan")])
                y_chords.extend([y_c, y_c, float("nan")])
    else:
        # Queloz inverse-slope curve: absolute midpoint velocities (levels unused for y).
        bis_curve = bdiag.get("bisector")
        if bis_curve is not None:
            vc = np.asarray(bis_curve, dtype=np.float64).ravel()
            if vc.size >= 2 and vg_a.size >= 2 and ccf_a.size == vg_a.size:
                for vm in vc:
                    if not math.isfinite(float(vm)):
                        continue
                    vm_f = float(vm)
                    v_mids.append(vm_f / 1000.0)
                    y_mids.append(float(np.interp(vm_f, vg_a, ccf_a)))

    bis_m = bdiag.get("bis_m_s")
    bis_e = bdiag.get("bis_err_m_s")
    n_cuts = bdiag.get("n_cuts")
    if n_cuts is None and cuts:
        n_cuts = len(cuts)
    mid_name = "Bisector midpoints"
    if n_cuts is not None:
        mid_name = f"Bisector midpoints (n={int(n_cuts)}"
        if bis_m is not None and math.isfinite(float(bis_m)):
            mid_name += f", BIS={float(bis_m):.2f} m/s"
            if bis_e is not None and math.isfinite(float(bis_e)):
                mid_name += f" ± {float(bis_e):.2f}"
        mid_name += ")"
    elif bis_m is not None and math.isfinite(float(bis_m)):
        mid_name = f"Bisector midpoints (BIS={float(bis_m):.2f} m/s"
        if bis_e is not None and math.isfinite(float(bis_e)):
            mid_name += f" ± {float(bis_e):.2f}"
        mid_name += ")"

    if v_mids and any(math.isfinite(y) for y in y_mids):
        fig.add_trace(
            go.Scatter(
                x=np.asarray(v_mids, dtype=np.float64),
                y=np.asarray(y_mids, dtype=np.float64),
                mode="markers+lines",
                name=mid_name,
                marker=dict(
                    size=7,
                    color="rgb(230, 159, 0)",
                    symbol="circle",
                    line=dict(width=0.8, color="rgb(80, 50, 0)"),
                ),
                line=dict(width=2.0, color="rgba(230, 159, 0, 0.90)"),
                legendgroup="bisector",
                showlegend=show_legend,
                visible=visible,
            ),
            row=row,
            col=1,
        )
        trace_indices.append(len(fig.data) - 1)
    else:
        _placeholder()

    if v_chords_x and any(math.isfinite(y) for y in y_chords):
        fig.add_trace(
            go.Scatter(
                x=np.asarray(v_chords_x, dtype=np.float64),
                y=np.asarray(y_chords, dtype=np.float64),
                mode="lines+markers",
                name="Bisector wing chords",
                marker=dict(size=4, color="rgba(230, 159, 0, 0.55)", symbol="circle-open"),
                line=dict(width=1.2, color="rgba(230, 159, 0, 0.55)"),
                legendgroup="bisector",
                showlegend=show_legend,
                visible=visible,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )
        trace_indices.append(len(fig.data) - 1)
    else:
        _placeholder()

    return trace_indices


def _line_mask_refine_ccf_add_plotly_traces(
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
    For ``ccf_line_mask``, overlay diagnostic refine curves on the CCF panel
    (smoothed CCF ±1σ when available) plus a vertical line at the **pipeline**
    final RV (Gaussian fit).

    When the slot carries ``line_mask_refine_rmeta``, that result is reused so
    Plotly does not recompute the diagnostic curves.

    Always appends **five** refine traces (real overlays or empty placeholders),
    then **two** bisector traces, so dropdown indices stay aligned.
    """
    trace_indices: list[int] = []

    def _placeholder() -> None:
        fig.add_trace(
            go.Scatter(x=[], y=[], showlegend=False, visible=visible, hoverinfo="skip"),
            row=row,
            col=1,
        )
        trace_indices.append(len(fig.data) - 1)

    cached = sl.get("line_mask_refine_rmeta")
    gaussian = sl.get("gaussian_fit")
    bisector = sl.get("bisector_fit")
    rv_slot = None
    if isinstance(gaussian, dict):
        rv_g = gaussian.get("rv")
        if rv_g is not None and np.isfinite(float(rv_g)):
            rv_slot = float(rv_g)
            cached_err = (
                float(gaussian["rv_err"])
                if gaussian.get("rv_err") is not None
                and np.isfinite(float(gaussian["rv_err"]))
                else float("nan")
            )
            if isinstance(cached, dict) and math.isfinite(cached_err):
                cached = {**cached, "rv_err_m_s": cached_err}
    if rv_slot is None and isinstance(bisector, dict):
        rv_bi = bisector.get("rv")
        if rv_bi is not None and np.isfinite(float(rv_bi)):
            rv_slot = float(rv_bi)
    if (
        isinstance(cached, dict)
        and cached.get("ok")
        and rv_slot is not None
    ):
        rmeta_use = ensure_line_mask_lstsq_preview_plot_metadata(cached, vg, ccf)
        if rmeta_use is None:
            rmeta_use = cached
        refine_idx = _line_mask_refine_plotly_traces_from_cached_rmeta(
            fig,
            row,
            rv_r_m_s=float(rv_slot),
            rmeta=rmeta_use,
            visible=visible,
            show_legend=show_legend,
            ylim_lo=ylim_lo,
            ylim_hi=ylim_hi,
            bisector_fit=bisector if isinstance(bisector, dict) else None,
        )
        bis_idx = _bisector_preview_plotly_traces(
            fig,
            row,
            sl=sl,
            vg=vg,
            ccf=ccf,
            visible=visible,
            show_legend=show_legend,
        )
        return refine_idx + bis_idx

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
            config=CCFFitConfig(),
        )
        if not rmeta.get("ok"):
            for _ in range(5):
                _placeholder()
            bis_idx = _bisector_preview_plotly_traces(
                fig,
                row,
                sl=sl,
                vg=vg,
                ccf=ccf,
                visible=visible,
                show_legend=show_legend,
            )
            return trace_indices + bis_idx

        rv_line = float(rv_slot) if rv_slot is not None else float(rv_r)
        refine_idx = _line_mask_refine_plotly_traces_from_cached_rmeta(
            fig,
            row,
            rv_r_m_s=rv_line,
            rmeta=rmeta,
            visible=visible,
            show_legend=show_legend,
            ylim_lo=ylim_lo,
            ylim_hi=ylim_hi,
            bisector_fit=bisector if isinstance(bisector, dict) else None,
        )
        bis_idx = _bisector_preview_plotly_traces(
            fig,
            row,
            sl=sl,
            vg=vg,
            ccf=ccf,
            visible=visible,
            show_legend=show_legend,
        )
        return refine_idx + bis_idx

    except Exception as e:
        logger.debug("mask-line preview: refine_ccf_rv_coarse_grid_parabolic failed: %s", e)
        while len(trace_indices) < 5:
            _placeholder()
        bis_idx = _bisector_preview_plotly_traces(
            fig,
            row,
            sl=sl,
            vg=vg,
            ccf=ccf,
            visible=visible,
            show_legend=show_legend,
        )
        return trace_indices + bis_idx


def _format_rv_value(v: float, *, kind: str) -> str:
    """
    Format RV-related numbers for Plotly annotations.

    * ``mean_ms`` — signed pipeline mean RV (m/s).
    * ``sigma_ms`` — uncertainty in m/s.
    """
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


def _pipeline_rv_for_line_mask_preview_slot(tbl, sl: dict) -> tuple[float, float] | None:
    """Single-chunk RV / RV_ERR from ``RVCCF`` (m/s) when ``rvccf_chunk_id`` is set on the slot."""
    rid = sl.get("rvccf_chunk_id")
    if tbl is None or rid is None or "CHUNK_ID" not in tbl.colnames:
        return None
    cid = np.asarray(tbl["CHUNK_ID"], dtype=np.int64)
    m = int(rid)
    sel = cid == m
    if not np.any(sel):
        return None
    i = int(np.flatnonzero(sel)[0])
    rv = float(np.asarray(tbl["RV"], dtype=np.float64)[i])
    rv_e = float(np.asarray(tbl["RV_ERR"], dtype=np.float64)[i])
    if not np.isfinite(rv) or not np.isfinite(rv_e) or rv_e <= 0.0:
        return None
    return rv, rv_e


def _slot_by_order_chunk_from_line_mask_preview(prev: dict | None) -> dict[tuple[int, int], dict]:
    """Map (echelle ``ORDER``, chunk index along that order) → CCF preview slot dict."""
    out: dict[tuple[int, int], dict] = {}
    if not prev:
        return out
    for sl in prev.get("slots", []):
        if sl.get("kind") != "order":
            continue
        try:
            oo = int(sl["order"])
        except (TypeError, ValueError, KeyError):
            continue
        ck_raw = sl.get("chunk_index")
        ck = int(ck_raw) if ck_raw is not None else 0
        out[(oo, ck)] = sl
    return out


def _collapse_line_mask_rv_scatter_rows(
    wl_um: np.ndarray,
    rv_m_s: np.ndarray,
    rv_err_m_s: np.ndarray,
    inlier: np.ndarray,
    order_col: np.ndarray,
    *,
    rv_dup_atol_m_s: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Collapse RV vs wavelength preview points when ``ccf_line_mask`` duplicated the **same**
    order-level RV/ERR at every wavelength chunk center.

    Line-mask mode **evaluates** the mask statistic on overlapping wavelength segments of width
    ``window_size_um`` along each order (multiple chunks per order). When the order-mean
    Gaussian + refine path succeeds, the pipeline **stores one RV per echelle order** on
    every chunk row of ``RVCCF``. The RV panel should show **one marker per order** (λ =
    mean of chunk centers for that order), not one marker per chunk.
    """
    n = int(wl_um.size)
    if n == 0 or order_col.size != n:
        return wl_um, rv_m_s, rv_err_m_s, inlier, np.asarray([], dtype=np.int64), False
    wl_l: list[float] = []
    rv_l: list[float] = []
    er_l: list[float] = []
    in_l: list[bool] = []
    ord_l: list[int] = []
    collapsed = False
    for o in np.unique(order_col):
        m = order_col == o
        idx = np.flatnonzero(m)
        if idx.size == 0:
            continue
        if o < 0:
            for i in idx:
                wl_l.append(float(wl_um[i]))
                rv_l.append(float(rv_m_s[i]))
                er_l.append(float(rv_err_m_s[i]))
                in_l.append(bool(inlier[i]))
                ord_l.append(int(o))
            continue
        r_o = rv_m_s[m]
        if r_o.size > 1 and np.allclose(
            r_o,
            r_o[0],
            rtol=0.0,
            atol=float(rv_dup_atol_m_s),
        ):
            wl_l.append(float(np.nanmean(wl_um[m])))
            rv_l.append(float(r_o[0]))
            er_l.append(float(rv_err_m_s[m][0]))
            in_l.append(bool(np.all(inlier[m])))
            ord_l.append(int(o))
            collapsed = True
        else:
            for i in idx:
                wl_l.append(float(wl_um[i]))
                rv_l.append(float(rv_m_s[i]))
                er_l.append(float(rv_err_m_s[i]))
                in_l.append(bool(inlier[i]))
                ord_l.append(int(order_col[i]))
    return (
        np.asarray(wl_l, dtype=np.float64),
        np.asarray(rv_l, dtype=np.float64),
        np.asarray(er_l, dtype=np.float64),
        np.asarray(in_l, dtype=bool),
        np.asarray(ord_l, dtype=np.int64),
        collapsed,
    )


def _rvcc_hover_order_segment_line(
    order_col: np.ndarray,
    wl_um: np.ndarray,
    *,
    collapsed: bool,
) -> np.ndarray:
    """
    One hover line per RVCCF scatter point: ``Order m, Segment k/n``.

    For non-collapsed line-mask rows, segment index follows the same **descending chunk-center
    λ** ordering as the pipeline / mask-overlay HTML (red → blue within each echelle order).
    When :func:`_collapse_line_mask_rv_scatter_rows` merged duplicate order-mean RVs, there is
    no single segment — label with an order-mean note instead.
    """
    oa = np.asarray(order_col, dtype=np.int64).ravel()
    wa = np.asarray(wl_um, dtype=np.float64).ravel()
    n = int(oa.size)
    out = np.empty(n, dtype=object)
    if n == 0:
        return out
    if collapsed:
        for i in range(n):
            oo = int(oa[i])
            out[i] = "—" if oo < 0 else f"Order {oo} (order-mean RV)"
        return out
    for oo in np.unique(oa):
        if oo < 0:
            for i in np.flatnonzero(oa == oo):
                out[i] = "—"
            continue
        idx = np.flatnonzero(oa == int(oo))
        if idx.size == 0:
            continue
        sub_w = wa[idx]
        idx_sorted = idx[np.argsort(-sub_w)]
        nk = int(idx_sorted.size)
        for rank, ii in enumerate(idx_sorted):
            out[int(ii)] = f"Order {int(oo)}, Segment {rank + 1}/{nk}"
    return out


def _preview_abbrev_ccf_uncertainty(diag: dict) -> str:
    """Short legend text for how RV uncertainty was estimated (fit_ok only)."""
    wtd = bool(diag.get("used_ccf_pointwise_error"))
    if diag.get("used_gradient_ccf_error"):
        s = "min(gradient, LevMar)" if diag.get("used_covariance_for_error") else "gradient ccf_error_calc"
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

    When env ``HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS`` is set (typical interactive /
    test preview runs), PNG export is disabled and only HTML is written when
    ``save_preview`` is True.

    Otherwise, when ``save_preview`` is True, PNGs are written alongside HTML unless
    the environment disables them as above.
    """
    if not getattr(primitive, "save_preview", False):
        return False
    if not getattr(primitive, "preview_write_png", True):
        return False
    if _env_flag("HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS"):
        return False
    return True


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


def _chunk_rv_for_overlay_slot(
    order_num: int,
    wl_center_um: float,
    chunk_echelle_order: np.ndarray | None,
    chunk_rv_m_s: np.ndarray | None,
    chunk_rv_err_m_s: np.ndarray | None,
    chunk_wavelength_centers_um: np.ndarray | None,
) -> tuple[float | None, float | None]:
    """Per-chunk RV hint for overlay HTML; falls back to IV-mean over chunks on that order."""
    if chunk_echelle_order is None or chunk_rv_m_s is None or chunk_rv_err_m_s is None:
        return None, None
    co = np.asarray(chunk_echelle_order, dtype=np.int64).ravel()
    rv = np.asarray(chunk_rv_m_s, dtype=np.float64).ravel()
    er = np.asarray(chunk_rv_err_m_s, dtype=np.float64).ravel()
    if co.size != rv.size or co.size != er.size or co.size == 0:
        return None, None
    if chunk_wavelength_centers_um is not None:
        wc = np.asarray(chunk_wavelength_centers_um, dtype=np.float64).ravel()
        if wc.size == co.size:
            atol = max(1e-12, 1e-12 * abs(float(wl_center_um)))
            sel = (
                (co == int(order_num))
                & np.isfinite(wc)
                & np.isfinite(rv)
                & np.isfinite(er)
                & np.isclose(wc, float(wl_center_um), rtol=0.0, atol=atol)
            )
            if np.any(sel):
                ii = int(np.flatnonzero(sel)[0])
                return float(rv[ii]), float(er[ii])
    return _aggregated_mask_chunk_rv_for_order(order_num, co, rv, er)


def collect_mask_overlay_payload_for_primitive(
    primitive,
    *,
    input_model,
    fiber: str,
    order_vec: list,
    chunk_rv_m_s: np.ndarray | None = None,
    chunk_rv_err_m_s: np.ndarray | None = None,
    chunk_echelle_order: np.ndarray | None = None,
    chunk_wavelength_um: np.ndarray | None = None,
) -> dict | None:
    """
    Build per-**chunk** spectrum + mask payload for mask-overlay debug HTML (``ccf_line_mask``).

    Chunk wavelength bins match :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.ccf_order_n_chunks_for_span`
    and ``window_size_um`` on the primitive. Pixel gates follow ``rv_per_chunks_line_mask`` (finite
    flux/error, not ``DO_NOT_USE``); chunks with fewer than ``ccf_min_good_pixels`` included pixels
    are omitted.

    Each row still uses key ``order`` for the echelle order number plus ``chunk_index`` /
    ``chunk_n`` for the segment on that order. Segments on an echelle row are ordered **descending**
    in chunk-center wavelength (red → blue), matching :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.rv_per_chunks_line_mask`.
    Payload rows include ``voronoi_lo_um`` / ``voronoi_hi_um`` (line-window Voronoi chunk bounds),
    ``spectrum_band_lo_um`` / ``spectrum_band_hi_um`` (symmetric ``window_size_um`` band used for
    pixels, slightly padded), ``window_edge_lo_um`` / ``window_edge_hi_um`` (±½``window_size_um`` at
    ``ccf_input_velocity`` for guide lines), and ``chunk_center_um`` (mean predicted observed mask-line
    λ in the chunk). Mask lines and weights match
    :func:`~hispecdrp.utils.line_mask_utils.restrict_mask_to_chunk` for each chunk span.

    When per-chunk pipeline RV arrays are supplied, ``fit_rv_m_s`` / ``fit_rv_err_m_s`` match the
    chunk center wavelength when ``chunk_wavelength_um`` aligns with the run.
    """
    vm, vx, _vs = _preview_ccf_velocity_range_m_s(primitive)
    try:
        mask_lines, _src = stellar_rv_ccf_mask_lines(
            mask_path=getattr(primitive, "mask_path", None),
            mask_wavelength=getattr(primitive, "mask_wavelength", None),
            mask_weight=getattr(primitive, "mask_weight", None),
        )
    except (ValueError, OSError, IOError) as exc:
        logger.warning(
            "Stellar RV preview: mask overlay payload — cannot load mask: %s", exc
        )
        return None

    wave = getattr(input_model, f"WAVE_FIB{fiber}", None)
    spec = getattr(input_model, f"SPEC_FIB{fiber}", None)
    err = getattr(input_model, f"SPEC_ERR_FIB{fiber}", None)
    dq = getattr(input_model, f"SPEC_DQ_FIB{fiber}", None)
    if wave is None or spec is None:
        logger.debug(
            "Stellar RV preview: mask overlay payload — missing WAVE/SPEC fiber=%s",
            fiber,
        )
        return None

    spec_lim, ccf_lim = _mask_overlay_downsample_limits()
    window_um = float(getattr(primitive, "window_size_um", 0.3))
    min_good = max(1, int(getattr(primitive, "ccf_min_good_pixels", 9)))
    do_not_use = int(DQ_FLAGS.get("DO_NOT_USE", 1))
    wbl = bool(getattr(primitive, "window_line_by_line", False))
    v_in = getattr(primitive, "ccf_input_velocity", None)
    from ..stellar_rv_primitive import StellarRVPrimitive

    zb_prim = StellarRVPrimitive.resolve_ccf_zb_for_fiber(primitive, input_model, fiber)
    use_line_window = wbl and v_in is not None and math.isfinite(float(v_in))

    entries: list[dict] = []
    for oi, order in enumerate(order_vec):
        w_row = np.asarray(wave[oi], dtype=np.float64)
        s_row = np.asarray(spec[oi], dtype=np.float64)
        if err is not None:
            e_src = err[oi] if getattr(err, "ndim", 0) >= 2 else err
            e_row = np.asarray(e_src, dtype=np.float64).ravel()
            if e_row.size != w_row.size:
                e_row = np.broadcast_to(e_row, w_row.shape).ravel().astype(np.float64, copy=False)
        else:
            e_row = np.ones_like(w_row, dtype=np.float64)
        if dq is not None:
            d_src = dq[oi] if getattr(dq, "ndim", 0) >= 2 else dq
            dq_row = np.asarray(d_src, dtype=np.uint32).ravel()
            if dq_row.size != w_row.size:
                dq_row = np.broadcast_to(dq_row, w_row.shape).ravel().astype(np.uint32, copy=False)
            d_bad = (dq_row & np.uint32(do_not_use)) != np.uint32(0)
        else:
            d_bad = np.zeros(w_row.shape, dtype=bool)

        w_min = float(np.nanmin(w_row))
        w_max = float(np.nanmax(w_row))
        if not np.isfinite(w_min) or not np.isfinite(w_max) or w_max <= w_min:
            continue
        span_um = float(w_max - w_min)
        _w_edge_eps = max(1e-15, abs(float(w_max)) * 1e-12)
        if use_line_window:
            wl_lo_a, wl_hi_a, wl_ctr_a = build_line_mask_chunk_wavelength_intervals_line_by_line(
                w_row,
                mask_lines,
                v_center_m_s=float(v_in),
                zb=zb_prim,
                min_good_pixels=min_good,
            )
            n_chunk = int(len(wl_lo_a))
            chunk_fwd: list[tuple[float, float, float, bool]] = [
                (
                    float(wl_lo_a[k]),
                    float(wl_hi_a[k]),
                    float(wl_ctr_a[k]),
                    float(wl_hi_a[k]) >= float(w_max) - _w_edge_eps,
                )
                for k in range(n_chunk)
            ]
            chunk_tuples = list(reversed(chunk_fwd))
        else:
            n_chunk = ccf_order_n_chunks_for_span(span_um, window_um)
            chunk_fwd = [
                (
                    w_min + k * span_um / n_chunk,
                    w_min + (k + 1) * span_um / n_chunk,
                    w_min + (k + 0.5) * span_um / n_chunk,
                    (w_min + (k + 1) * span_um / n_chunk) >= float(w_max) - _w_edge_eps,
                )
                for k in range(n_chunk)
            ]
            chunk_tuples = list(reversed(chunk_fwd))

        for k, (lam_lo, lam_hi, wl_center, right_closed) in enumerate(chunk_tuples):
            if right_closed:
                in_voronoi = (
                    (w_row >= lam_lo)
                    & (w_row <= lam_hi)
                    & np.isfinite(s_row)
                    & np.isfinite(e_row)
                    & (~d_bad)
                )
            else:
                in_voronoi = (
                    (w_row >= lam_lo)
                    & (w_row < lam_hi)
                    & np.isfinite(s_row)
                    & np.isfinite(e_row)
                    & (~d_bad)
                )
            spec_band_lo: float | None = None
            spec_band_hi: float | None = None
            w_edge_lo: float | None = None
            w_edge_hi: float | None = None
            if use_line_window:
                band_lo, band_hi = mask_line_window_spectrum_band_um(
                    float(wl_center),
                    float(window_um),
                    w_min,
                    w_max,
                )
                spec_band_lo = float(band_lo)
                spec_band_hi = float(band_hi)
                wc = float(wl_center)
                wsz = float(window_um)
                w_edge_lo = max(w_min, wc - 0.5 * wsz)
                w_edge_hi = min(w_max, wc + 0.5 * wsz)
                in_band = (
                    (w_row >= band_lo)
                    & (w_row <= band_hi)
                    & np.isfinite(s_row)
                    & np.isfinite(e_row)
                    & (~d_bad)
                )
                in_chunk = in_voronoi & in_band
                if int(np.sum(in_chunk)) < min_good:
                    in_chunk = in_voronoi
            else:
                in_chunk = in_voronoi
            if int(np.sum(in_chunk)) < min_good:
                continue
            w_c = w_row[in_chunk]
            s_c = s_row[in_chunk]
            med = float(np.nanmedian(s_c))
            if not np.isfinite(med) or med == 0.0:
                med = 1.0
            s_norm = s_c / med
            w_plot, s_norm_plot, _ = _downsample_ccf_for_plot(w_c, s_norm, None, spec_lim)
            wl_lo = float(np.nanmin(w_plot))
            wl_hi = float(np.nanmax(w_plot))
            vor_lo = float(lam_lo)
            vor_hi = float(lam_hi)
            lam_c_fid = float(wl_center)
            w_lo_m = float(np.nanmin(w_c))
            w_hi_m = float(np.nanmax(w_c))
            fit_rv, fit_err = _chunk_rv_for_overlay_slot(
                int(order),
                wl_center,
                chunk_echelle_order,
                chunk_rv_m_s,
                chunk_rv_err_m_s,
                chunk_wavelength_um,
            )

            mask_o = restrict_mask_to_chunk(
                mask_lines,
                w_lo_m,
                w_hi_m,
                v_min_m_s=vm,
                v_max_m_s=vx,
                zb=StellarRVPrimitive.resolve_ccf_zb_for_fiber(
                    primitive, input_model, fiber
                ),
            )
            if mask_o.line_lambda_um.size > 0:
                # Use full segment widths when present (tophat finite edges). For delta masks,
                # left/right edges coincide with the center and the overlay expands them only by minDx.
                mstart = np.asarray(mask_o.line_left_edge.to_value(u.um), dtype=np.float64)
                mend = np.asarray(mask_o.line_right_edge.to_value(u.um), dtype=np.float64)
                mw_raw = np.asarray(mask_o.weight, dtype=np.float64)
                mw = mw_raw.copy()
            else:
                mstart = np.array([], dtype=np.float64)
                mend = np.array([], dtype=np.float64)
                mw_raw = np.array([], dtype=np.float64)
                mw = np.array([], dtype=np.float64)

            row_payload: dict = {
                "order": int(order),
                "chunk_index": int(k),
                "chunk_n": int(n_chunk),
                "wl_lo_um": wl_lo,
                "wl_hi_um": wl_hi,
                "voronoi_lo_um": vor_lo,
                "voronoi_hi_um": vor_hi,
                "chunk_center_um": lam_c_fid,
                "wave": np.asarray(w_plot, dtype=np.float64),
                "spec": np.asarray(s_norm_plot, dtype=np.float64),
                "mask_start_um": np.asarray(mstart, dtype=np.float64),
                "mask_end_um": np.asarray(mend, dtype=np.float64),
                "mask_weight_raw": np.asarray(mw_raw, dtype=np.float64),
                "mask_weight": np.asarray(mw, dtype=np.float64),
                "fit_rv_m_s": fit_rv,
                "fit_rv_err_m_s": fit_err,
            }
            if spec_band_lo is not None and spec_band_hi is not None:
                row_payload["spectrum_band_lo_um"] = spec_band_lo
                row_payload["spectrum_band_hi_um"] = spec_band_hi
            if w_edge_lo is not None and w_edge_hi is not None:
                row_payload["window_edge_lo_um"] = float(w_edge_lo)
                row_payload["window_edge_hi_um"] = float(w_edge_hi)
            pio_slot = int(getattr(primitive, "mask_ccf_pixel_interp_order", 3) or 3)
            n_dense = 10 * int(pio_slot) * int(len(w_plot))
            icurve = _mask_overlay_spec_interp_curve(
                w_c,
                s_norm,
                pixel_interp_order=pio_slot,
                n_dense=n_dense,
                spec_downsample_lim=spec_lim,
            )
            if icurve is not None:
                wi, si = icurve
                row_payload["interp_wave_um"] = np.asarray(wi, dtype=np.float64)
                row_payload["interp_spec"] = np.asarray(si, dtype=np.float64)
            entries.append(row_payload)

    if not entries:
        logger.info(
            "Stellar RV preview: mask overlay payload — no valid chunks fiber=%s",
            fiber,
        )
        return None

    slot_by_oc = _slot_by_order_chunk_from_line_mask_preview(
        getattr(primitive, "_ccf_preview_by_fiber", {}).get(str(fiber))
        if (getattr(primitive, "algorithm", None) or "").strip().lower()
        in STELLAR_RV_LINE_MASK_ALGORITHMS
        else None
    )
    for entry in entries:
        od = int(entry["order"])
        ck = int(entry.get("chunk_index", 0))
        sl = slot_by_oc.get((od, ck))
        if sl is None:
            continue
        vslot = np.asarray(sl.get("v"), dtype=np.float64).ravel()
        ccf_row = np.asarray(sl.get("ccf"), dtype=np.float64).ravel()
        if vslot.size == 0 or ccf_row.size != vslot.size:
            continue
        # Slots / RVCCF use the stored convention (often negated so parabolic/Gaussian fits target a
        # maximum). The mask-overlay CCF panel should match the raw weighted mask statistic from
        # :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.rv_per_chunks_line_mask` (absorption-like).
        if bool(LINE_MASK_CCF_STORED_NEGATED_FROM_RAW):
            ccf_row = -ccf_row
        ce = sl.get("ccf_err")
        cea = (
            np.asarray(ce, dtype=np.float64).ravel()
            if ce is not None
            else None
        )
        if cea is not None and (
            cea.size != vslot.size or not np.any(np.isfinite(cea) & (cea > 0))
        ):
            cea = None
        v_ds, c_ds, ce_ds = _downsample_ccf_for_plot(vslot, ccf_row, cea, ccf_lim)
        entry["ccf_v_m_s"] = v_ds
        entry["ccf"] = c_ds
        if ce_ds is not None:
            entry["ccf_err"] = ce_ds

    entries.sort(key=lambda e: (int(e["order"]), -float(e["chunk_center_um"])))

    logger.info(
        "Stellar RV preview: mask overlay payload built fiber=%s n_chunks=%d",
        fiber,
        len(entries),
    )
    out_payload: dict = {
        "fiber": str(fiber),
        "orders": entries,
        "window_line_by_line": bool(getattr(primitive, "window_line_by_line", False)),
        "mask_ccf_pixel_interp_order": int(
            getattr(primitive, "mask_ccf_pixel_interp_order", 3) or 3
        ),
    }
    return out_payload


def _json_opt_float(x: float | None) -> float | None:
    if x is None:
        return None
    xf = float(x)
    return xf if np.isfinite(xf) else None


def _json_round_list(seq, ndigits: int = 8) -> list[float]:
    """Shorter JSON than full float repr; sufficient for Plotly + linear interp in the overlay."""
    return [round(float(x), ndigits) for x in seq]


def _json_float_list_sigfig(seq, n_sig: int = 8) -> list[float]:
    """
    JSON-friendly floats using ``n_sig`` significant figures.

    Fixed-decimal :func:`round` (as in :func:`_json_round_list`) maps typical stellar-template
    weighted-covariance CCF amplitudes to **all zeros** when ``|x| < 0.5·10⁻ⁿ`` for the chosen
    ``ndigits``. Significant-figure formatting preserves the CCF shape for the overlay HTML.
    """
    out: list[float] = []
    for x in seq:
        xf = float(x)
        if not math.isfinite(xf):
            out.append(xf)
            continue
        out.append(float(f"{xf:.{int(n_sig)}g}"))
    return out


def _to_serializable_order_payload(order_payload: dict) -> dict:
    out_orders = []
    for row in order_payload.get("orders", []):
        base: dict = {
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
        if "chunk_index" in row and "chunk_n" in row:
            base["chunk_index"] = int(row["chunk_index"])
            base["chunk_n"] = int(row["chunk_n"])
        if row.get("voronoi_lo_um") is not None and row.get("voronoi_hi_um") is not None:
            base["voronoi_lo_um"] = round(float(row["voronoi_lo_um"]), 8)
            base["voronoi_hi_um"] = round(float(row["voronoi_hi_um"]), 8)
        if row.get("spectrum_band_lo_um") is not None and row.get("spectrum_band_hi_um") is not None:
            base["spectrum_band_lo_um"] = round(float(row["spectrum_band_lo_um"]), 8)
            base["spectrum_band_hi_um"] = round(float(row["spectrum_band_hi_um"]), 8)
        if row.get("window_edge_lo_um") is not None and row.get("window_edge_hi_um") is not None:
            base["window_edge_lo_um"] = round(float(row["window_edge_lo_um"]), 8)
            base["window_edge_hi_um"] = round(float(row["window_edge_hi_um"]), 8)
        if row.get("chunk_center_um") is not None:
            base["chunk_center_um"] = round(float(row["chunk_center_um"]), 8)
        out_orders.append(base)
        r0 = out_orders[-1]
        cv = row.get("ccf_v_m_s")
        cc = row.get("ccf")
        cerr = row.get("ccf_err")
        if cv is not None and cc is not None:
            cva = np.asarray(cv, dtype=np.float64).ravel()
            cca = np.asarray(cc, dtype=np.float64).ravel()
            if cva.size == cca.size and cva.size > 0:
                r0["ccf_v_m_s"] = _json_round_list(cva, 6)
                r0["ccf"] = _json_float_list_sigfig(cca, 8)
                if cerr is not None:
                    cea = np.asarray(cerr, dtype=np.float64).ravel()
                    if (
                        cea.size == cva.size
                        and np.any(np.isfinite(cea) & (cea > 0))
                    ):
                        r0["ccf_err"] = _json_float_list_sigfig(cea, 8)
        iw = row.get("interp_wave_um")
        isp = row.get("interp_spec")
        if iw is not None and isp is not None:
            iwa = np.asarray(iw, dtype=np.float64).ravel()
            ispa = np.asarray(isp, dtype=np.float64).ravel()
            if (
                iwa.size == ispa.size
                and iwa.size >= 2
                and np.any(np.isfinite(iwa))
                and np.any(np.isfinite(ispa))
            ):
                r0["interp_wave_um"] = _json_round_list(iwa, 8)
                r0["interp_spec"] = _json_round_list(ispa, 7)
    root: dict = {"fiber": str(order_payload["fiber"]), "orders": out_orders}
    pio_r = order_payload.get("mask_ccf_pixel_interp_order")
    if pio_r is not None:
        root["mask_ccf_pixel_interp_order"] = int(pio_r)
    return root


def _to_serializable_template_overlay_payload(order_payload: dict) -> dict:
    out_orders = []
    for row in order_payload.get("orders", []):
        tbase: dict = {
            "order": int(row["order"]),
            "wl_lo_um": round(float(row["wl_lo_um"]), 8),
            "wl_hi_um": round(float(row["wl_hi_um"]), 8),
            "wave": _json_round_list(row["wave"], 8),
            "spec": _json_round_list(row["spec"], 7),
            "tpl_wave_um": _json_round_list(row["tpl_wave_um"], 8),
            "tpl_flux": _json_round_list(row["tpl_flux"], 7),
            "fit_rv_m_s": _json_opt_float(row.get("fit_rv_m_s")),
            "fit_rv_err_m_s": _json_opt_float(row.get("fit_rv_err_m_s")),
        }
        if "chunk_index" in row and "chunk_n" in row:
            tbase["chunk_index"] = int(row["chunk_index"])
            tbase["chunk_n"] = int(row["chunk_n"])
        out_orders.append(tbase)
        r0 = out_orders[-1]
        cv = row.get("ccf_v_m_s")
        cc = row.get("ccf")
        cerr = row.get("ccf_err")
        if cv is not None and cc is not None:
            cva = np.asarray(cv, dtype=np.float64).ravel()
            cca = np.asarray(cc, dtype=np.float64).ravel()
            if cva.size == cca.size and cva.size > 0:
                r0["ccf_v_m_s"] = _json_round_list(cva, 6)
                r0["ccf"] = _json_float_list_sigfig(cca, 8)
                if cerr is not None:
                    cea = np.asarray(cerr, dtype=np.float64).ravel()
                    if (
                        cea.size == cva.size
                        and np.any(np.isfinite(cea) & (cea > 0))
                    ):
                        r0["ccf_err"] = _json_float_list_sigfig(cea, 8)
    return {"fiber": str(order_payload["fiber"]), "orders": out_orders}


def collect_template_overlay_payload_for_primitive(
    primitive,
    *,
    input_model,
    fiber: str,
    order_vec: list,
    chunk_rv_m_s: np.ndarray | None = None,
    chunk_rv_err_m_s: np.ndarray | None = None,
    chunk_echelle_order: np.ndarray | None = None,
    chunk_wavelength_um: np.ndarray | None = None,
) -> dict | None:
    """
    Build per-**chunk** spectrum + cropped rest-frame template for template-overlay debug HTML.

    Chunking matches ``window_size_um`` / :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.ccf_order_n_chunks_for_span`
    and the same ``DO_NOT_USE`` + finite flux/error gate as mask CCF, with a minimum of
    ``ccf_min_good_pixels`` pixels per chunk.

    ``ccf`` on each row is optional; when present with ``ccf_v_m_s`` it populates the CCF panel
    from the pipeline preview (still **per echelle order** in the preview dict, shared by all
    chunks on that order). ``tpl_wave_um`` / ``tpl_flux`` are the template in **vacuum rest**
    microns over each chunk span (plus margin). Template crop length is capped by env
    ``HISPEC_DRP_TEMPLATE_OVERLAY_MAX_TEMPLATE_POINTS``.
    """
    if getattr(primitive, "algorithm", None) != "ccf_stellar_template":
        return None
    tw = getattr(primitive, "_stellar_template_wave_um", None)
    tf = getattr(primitive, "_stellar_template_flux", None)
    if tw is None or tf is None:
        return None
    tw = np.asarray(tw, dtype=np.float64).ravel()
    tf = np.asarray(tf, dtype=np.float64).ravel()
    if tw.size != tf.size or tw.size < 8:
        return None

    wave = getattr(input_model, f"WAVE_FIB{fiber}", None)
    spec = getattr(input_model, f"SPEC_FIB{fiber}", None)
    err = getattr(input_model, f"SPEC_ERR_FIB{fiber}", None)
    dq = getattr(input_model, f"SPEC_DQ_FIB{fiber}", None)
    if wave is None or spec is None:
        return None

    spec_lim, ccf_lim = _mask_overlay_downsample_limits()
    n_tpl_cap = _template_overlay_max_template_points_from_env()
    window_um = float(getattr(primitive, "window_size_um", 0.3))
    min_good = max(1, int(getattr(primitive, "ccf_min_good_pixels", 9)))
    do_not_use = int(DQ_FLAGS.get("DO_NOT_USE", 1))

    slot_by_oc = _slot_by_order_chunk_from_line_mask_preview(
        getattr(primitive, "_ccf_preview_by_fiber", {}).get(str(fiber))
    )

    entries: list[dict] = []
    for oi, order in enumerate(order_vec):
        w_row = np.asarray(wave[oi], dtype=np.float64)
        s_row = np.asarray(spec[oi], dtype=np.float64)
        if err is not None:
            e_src = err[oi] if getattr(err, "ndim", 0) >= 2 else err
            e_row = np.asarray(e_src, dtype=np.float64).ravel()
            if e_row.size != w_row.size:
                e_row = np.broadcast_to(e_row, w_row.shape).ravel().astype(np.float64, copy=False)
        else:
            e_row = np.ones_like(w_row, dtype=np.float64)
        if dq is not None:
            d_src = dq[oi] if getattr(dq, "ndim", 0) >= 2 else dq
            dq_row = np.asarray(d_src, dtype=np.uint32).ravel()
            if dq_row.size != w_row.size:
                dq_row = np.broadcast_to(dq_row, w_row.shape).ravel().astype(np.uint32, copy=False)
            d_bad = (dq_row & np.uint32(do_not_use)) != np.uint32(0)
        else:
            d_bad = np.zeros(w_row.shape, dtype=bool)

        w_min = float(np.nanmin(w_row))
        w_max = float(np.nanmax(w_row))
        if not np.isfinite(w_min) or not np.isfinite(w_max) or w_max <= w_min:
            continue
        span_um = float(w_max - w_min)
        n_chunk = ccf_order_n_chunks_for_span(span_um, window_um)

        for k in range(n_chunk):
            lam_lo = w_min + k * span_um / n_chunk
            lam_hi = w_min + (k + 1) * span_um / n_chunk
            if k == n_chunk - 1:
                in_chunk = (
                    (w_row >= lam_lo)
                    & (w_row <= lam_hi)
                    & np.isfinite(s_row)
                    & np.isfinite(e_row)
                    & (~d_bad)
                )
            else:
                in_chunk = (
                    (w_row >= lam_lo)
                    & (w_row < lam_hi)
                    & np.isfinite(s_row)
                    & np.isfinite(e_row)
                    & (~d_bad)
                )
            if int(np.sum(in_chunk)) < min_good:
                continue
            w_c = w_row[in_chunk]
            s_c = s_row[in_chunk]
            med = float(np.nanmedian(s_c))
            if not np.isfinite(med) or med == 0.0:
                med = 1.0
            s_norm = s_c / med
            w_plot, s_plot, _ = _downsample_ccf_for_plot(w_c, s_norm, None, spec_lim)
            wl_lo = float(np.nanmin(w_plot))
            wl_hi = float(np.nanmax(w_plot))
            w_lo = float(np.nanmin(w_c))
            w_hi = float(np.nanmax(w_c))
            wl_center = 0.5 * (lam_lo + lam_hi)
            span = max(w_hi - w_lo, 1e-9)
            margin = max(0.02 * span, 1e-6)
            m_tpl = (tw >= (w_lo - margin)) & (tw <= (w_hi + margin)) & np.isfinite(tf)
            if not np.any(m_tpl):
                continue
            tw_o = tw[m_tpl]
            tf_o = tf[m_tpl].astype(np.float64, copy=False)
            if tw_o.size > n_tpl_cap:
                idx = np.linspace(0, tw_o.size - 1, num=n_tpl_cap, dtype=np.int64)
                tw_o = tw_o[idx]
                tf_o = tf_o[idx]
            tf_med = float(np.nanmedian(tf_o))
            if not np.isfinite(tf_med) or tf_med == 0.0:
                tf_med = 1.0
            tf_o = (tf_o / tf_med).astype(np.float64, copy=False)

            fit_rv, fit_err = _chunk_rv_for_overlay_slot(
                int(order),
                wl_center,
                chunk_echelle_order,
                chunk_rv_m_s,
                chunk_rv_err_m_s,
                chunk_wavelength_um,
            )

            entry: dict = {
                "order": int(order),
                "chunk_index": int(k),
                "chunk_n": int(n_chunk),
                "wl_lo_um": wl_lo,
                "wl_hi_um": wl_hi,
                "wave": np.asarray(w_plot, dtype=np.float64),
                "spec": np.asarray(s_plot, dtype=np.float64),
                "tpl_wave_um": np.asarray(tw_o, dtype=np.float64),
                "tpl_flux": np.asarray(tf_o, dtype=np.float64),
                "fit_rv_m_s": fit_rv,
                "fit_rv_err_m_s": fit_err,
            }
            sl = slot_by_oc.get((int(order), int(k)))
            if sl is None:
                sl = slot_by_oc.get((int(order), 0))
            if sl is not None:
                vslot = np.asarray(sl.get("v"), dtype=np.float64).ravel()
                ccf_row = np.asarray(sl.get("ccf"), dtype=np.float64).ravel()
                if vslot.size == ccf_row.size and vslot.size > 1:
                    ce = sl.get("ccf_err")
                    cea = (
                        np.asarray(ce, dtype=np.float64).ravel()
                        if ce is not None
                        else None
                    )
                    if cea is not None and cea.size != vslot.size:
                        cea = None
                    v_ds, c_ds, ce_ds = _downsample_ccf_for_plot(
                        vslot, ccf_row, cea, ccf_lim
                    )
                    entry["ccf_v_m_s"] = v_ds
                    entry["ccf"] = c_ds
                    if ce_ds is not None:
                        entry["ccf_err"] = ce_ds
            entries.append(entry)

    if not entries:
        logger.info(
            "Stellar RV preview: template overlay payload — no valid chunks fiber=%s",
            fiber,
        )
        return None
    logger.info(
        "Stellar RV preview: template overlay payload built fiber=%s n_chunks=%d",
        fiber,
        len(entries),
    )
    return {"fiber": str(fiber), "orders": entries}


def _write_ccf_overlay_debug_html_pages(
    primitive, output_model, *, kind: str
) -> dict[str, str]:
    """Shared writer for :func:`write_mask_overlay_debug_pages` / ``write_template_*``."""
    if kind == "mask":
        payloads = getattr(primitive, "_mask_overlay_by_fiber", {}) or {}
        log_label = "mask overlay"
        serialize = _to_serializable_order_payload
        overlay_kind = "mask"
        help_html = _MASK_OVERLAY_SPECTRUM_HELP_HTML
        path_token = "_mask_overlay_debug_fib"
        show_attr = "show_overlay_debug"
    elif kind == "template":
        payloads = getattr(primitive, "_template_overlay_by_fiber", {}) or {}
        log_label = "template overlay"
        serialize = _to_serializable_template_overlay_payload
        overlay_kind = "template"
        help_html = _TEMPLATE_OVERLAY_SPECTRUM_HELP_HTML
        path_token = "_template_overlay_debug_fib"
        show_attr = "show_overlay_debug"
    else:
        raise ValueError(f"unknown overlay kind: {kind!r}")

    out: dict[str, str] = {}
    if not payloads:
        logger.info(
            "Stellar RV preview: %s debug — no payloads to plot.",
            log_label,
        )
        return out

    logger.info(
        "Stellar RV preview: %s debug — writing HTML for %d fiber(s)",
        log_label,
        len(payloads),
    )
    base_path = _fits_base_path(primitive, output_model)
    v_init = _overlay_v_init_m_s(primitive)
    vm, vx, vs = _preview_ccf_velocity_range_m_s(primitive)
    from ..stellar_rv_primitive import StellarRVPrimitive

    for fiber, payload in payloads.items():
        zb = StellarRVPrimitive.resolve_ccf_zb_for_fiber(
            primitive, output_model, fiber
        )
        payload_json = json.dumps(
            serialize(payload),
            separators=(",", ":"),
        )
        html = build_ccf_spectrum_overlay_debug_html(
            title=f"Stellar RV {log_label} debug (fiber {fiber})",
            payload_json=payload_json,
            v_init_m_s=v_init,
            v_min_m_s=vm,
            v_max_m_s=vx,
            v_step_m_s=vs,
            zb=zb,
            c_ms=C_M_S,
            overlay_kind=overlay_kind,
            spectrum_help_html=help_html,
        )
        path = base_path.replace(".fits", f"{path_token}{fiber}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        out[str(fiber)] = path
        logger.info(
            "Stellar RV preview: %s debug — wrote fiber=%s path=%s n_chunks=%d",
            log_label,
            fiber,
            path,
            len(payload.get("orders", [])),
        )
        if bool(getattr(primitive, show_attr, False)):
            try:
                webbrowser.open(f"file://{path}")
            except Exception as exc:
                logger.debug("Could not open %s debug HTML: %s", log_label, exc)
    logger.info(
        "Stellar RV preview: %s debug — finished n_files=%d",
        log_label,
        len(out),
    )
    return out


def write_mask_overlay_debug_pages(primitive, output_model) -> dict[str, str]:
    """
    Write debug HTML overlays for mask-shift RV and optionally open them.

    Returns
    -------
    dict[str, str]
        fiber -> html path
    """
    return _write_ccf_overlay_debug_html_pages(primitive, output_model, kind="mask")


def write_template_overlay_debug_pages(primitive, output_model) -> dict[str, str]:
    """
    Write debug HTML for ``ccf_stellar_template``: spectrum + Doppler-shifted template overlay.

    Returns
    -------
    dict[str, str]
        fiber -> html path
    """
    return _write_ccf_overlay_debug_html_pages(primitive, output_model, kind="template")


def maybe_write_ccf_overlay_debug_html(
    primitive,
    algo_key: str,
    output_model,
    output_extras: dict,
) -> None:
    """
    When overlay debug is enabled for the active algorithm, write HTML and merge paths into
    ``output_extras`` (``mask_overlay_debug`` or ``template_overlay_debug``).

    Called from :meth:`StellarRVPrimitive._perform` after per-fiber processing so
    ``primitive._mask_overlay_by_fiber`` / ``primitive._template_overlay_by_fiber`` are populated.
    """
    if (algo_key or "").strip().lower() in STELLAR_RV_LINE_MASK_ALGORITHMS and bool(
        getattr(primitive, "show_overlay_debug", False)
        or getattr(primitive, "save_overlay_debug", False)
    ):
        logger.info("Stellar RV preview: writing mask overlay debug HTML pages")
        paths = write_mask_overlay_debug_pages(primitive, output_model)
        if paths:
            output_extras["mask_overlay_debug"] = paths
        logger.info(
            "Stellar RV preview: mask overlay debug HTML done n_files=%d",
            len(paths),
        )

    if algo_key == "ccf_stellar_template" and bool(
        getattr(primitive, "show_overlay_debug", False)
        or getattr(primitive, "save_overlay_debug", False)
    ):
        logger.info("Stellar RV preview: writing template overlay debug HTML pages")
        paths_t = write_template_overlay_debug_pages(primitive, output_model)
        if paths_t:
            output_extras["template_overlay_debug"] = paths_t
        logger.info(
            "Stellar RV preview: template overlay debug HTML done n_files=%d",
            len(paths_t),
        )


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
        detail = primitive._ccf_preview_by_fiber.get(str(fiber))
        if not detail:
            slot_metadata[fiber] = []
            continue
        _prim_algo_lm = (getattr(primitive, "algorithm", None) or "").strip().lower() in STELLAR_RV_LINE_MASK_ALGORITHMS
        rv_agg_tbl, rv_agg_err_tbl, per_order_rv = pipeline_rv_by_fiber[fiber]
        v_step = float(detail["v_step"])
        slots = _preview_slots_for_plot(detail["slots"])
        slot_metadata[fiber] = []
        for sl in slots:
            sid = id(sl)
            if sid not in fit_slot_cache:
                gaussian = sl.get("gaussian_fit")
                bisector = sl.get("bisector_fit")
                if gaussian and isinstance(gaussian, dict) and gaussian.get("params"):
                    params = gaussian.get("params") or {}
                    diag = gaussian.get("diagnostics") or {}
                    rv_err = float(gaussian.get("rv_err", np.nan))
                    rv_fit = gaussian.get("rv")
                    rv_final = (
                        float(rv_fit)
                        if rv_fit is not None and np.isfinite(float(rv_fit))
                        else float(params.get("v0", np.nan))
                    )
                    if isinstance(bisector, dict):
                        rv_bi = bisector.get("rv")
                        err_bi = bisector.get("rv_err")
                        if rv_bi is not None and np.isfinite(float(rv_bi)):
                            rv_final = float(rv_bi)
                        if err_bi is not None and np.isfinite(float(err_bi)) and float(err_bi) > 0:
                            rv_err = float(err_bi)
                    fit_slot_cache[sid] = (params, diag, rv_err, rv_final)
                else:
                    _ce = sl.get("ccf_err")
                    _ce_arr = (
                        np.asarray(_ce, dtype=np.float64) if _ce is not None else None
                    )
                    _ccf_arr = np.asarray(sl["ccf"], dtype=np.float64)
                    _mask_wbl = (
                        bool(getattr(primitive, "window_line_by_line", False))
                        if _prim_algo_lm
                        else False
                    )
                    _gseed = getattr(primitive, "ccf_input_velocity", None)
                    _gs = (
                        float(_gseed)
                        if _gseed is not None and math.isfinite(float(_gseed))
                        else None
                    )
                    _rv, _re, params, diag = fit_ccf_rv(
                        sl["v"],
                        _ccf_arr,
                        v_step,
                        return_params=True,
                        ccf_err=_ce_arr,
                        ccf_line_mask_stored_negated=(
                            LINE_MASK_CCF_STORED_NEGATED_FROM_RAW
                            if _prim_algo_lm
                            else None
                        ),
                        line_mask_line_window_by_line=_mask_wbl,
                        gaussian_seed_mean_m_s=_gs
                        if _prim_algo_lm
                        else None,
                    )
                    rv_err = float(_re) if _re is not None and np.isfinite(_re) else np.nan
                    rv_final = float(_rv) if _rv is not None and np.isfinite(float(_rv)) else float(params.get("v0", np.nan))
                    fit_slot_cache[sid] = (params, diag, rv_err, rv_final)
            params, diag, rv_err, rv_final = fit_slot_cache[sid]
            v_report = rv_final if np.isfinite(rv_final) else float(params.get("v0", np.nan))
            is_line_mask = _prim_algo_lm
            if sl.get("kind") == "mean":
                meta = "Mean · all ord"
                if np.isfinite(rv_agg_tbl):
                    meta += (
                        f"<br>⟨RV⟩ (RVCCFAVG): {rv_agg_tbl:.2f} m/s · "
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
                cn = int(sl.get("chunk_n", 1) or 1)
                cix = sl.get("chunk_index")
                seg = ""
                if cn > 1 and cix is not None:
                    seg = f" · seg {int(cix) + 1}/{cn}"
                if np.isfinite(wl_lo) and np.isfinite(wl_hi):
                    meta = f"Ord {o}{seg} · λ{wl_lo:.3f}–{wl_hi:.3f} µm"
                else:
                    meta = f"Ord {o}{seg}"
                oo: int | None
                try:
                    oo = int(o) if o is not None and str(o).strip() != "?" else None
                except (TypeError, ValueError):
                    oo = None
                chunk_tbl = (
                    _pipeline_rv_for_line_mask_preview_slot(tbl_rv, sl)
                    if is_line_mask
                    else None
                )
                if chunk_tbl is not None:
                    rp, rpe = chunk_tbl
                    meta += (
                        f"<br>RV (chunk, table): {rp:.2f} m/s · "
                        f"σ={_format_rv_value(rpe, kind='sigma_ms')}"
                    )
                elif oo is not None and oo in per_order_rv:
                    rp, rpe = per_order_rv[oo]
                    if is_line_mask:
                        meta += (
                            f"<br>RV (Gauss=table): {rp:.2f} m/s · "
                            f"σ={_format_rv_value(rpe, kind='sigma_ms')}"
                        )
                    else:
                        meta += (
                            f"<br>Pipe RVCCF: {rp:.2f} m/s · "
                            f"σ={_format_rv_value(rpe, kind='sigma_ms')}"
                        )
                elif oo is not None:
                    lbl = "RV" if is_line_mask else "Pipe"
                    meta += f"<br>{lbl}: —"
            if params and diag.get("fit_ok"):
                v0_lm = float(params["v0"])
                sig_lm = float(params["sigma"])
                amp_lm = float(params["amplitude"])
                v0, sig, amp = v0_lm, sig_lm, amp_lm
                if is_line_mask:
                    _ov_leg = _line_mask_preview_gaussian_overlay_params(
                        sl,
                        params,
                        diag,
                        np.asarray(sl["v"], dtype=np.float64),
                        np.asarray(sl["ccf"], dtype=np.float64),
                    )
                    if _ov_leg is not None:
                        amp, v0, sig, _base_leg = _ov_leg
                if is_line_mask and sl.get("kind") != "mean":
                    meta += (
                        f"<br>Gaussian overlay (LSTSQ-centered): "
                        f"center={v0:.2f} m/s · σ={sig:.2f} m/s · amp={amp:.3g}"
                    )
                    if abs(v0 - v0_lm) > 5.0 or abs(amp - amp_lm) > max(1e-38, 0.05 * abs(amp)):
                        meta += (
                            f"<br><i>LM-only: v₀={v0_lm:.2f} m/s · "
                            f"σ={sig_lm:.2f} m/s · amp={amp_lm:.3g}</i>"
                        )
                    meta += (
                        f"<br>RV (bisector, table): {v_report:.2f} m/s"
                    )
                    _bdiag = (
                        bisector.get("diagnostics")
                        if isinstance(bisector, dict)
                        else None
                    )
                    if isinstance(_bdiag, dict):
                        _bis = _bdiag.get("bis_m_s")
                        _bis_e = _bdiag.get("bis_err_m_s")
                        _ncuts = _bdiag.get("n_cuts")
                        if _bis is not None and math.isfinite(float(_bis)):
                            _bis_s = f"<br>BIS = {float(_bis):.2f} m/s"
                            if _bis_e is not None and math.isfinite(float(_bis_e)):
                                _bis_s += f" (σ={float(_bis_e):.2f})"
                            if _ncuts is not None:
                                _bis_s += f" · n_cuts={int(_ncuts)}"
                            meta += _bis_s
                    tbl_rp: float | None = None
                    ct = _pipeline_rv_for_line_mask_preview_slot(tbl_rv, sl)
                    if ct is not None:
                        tbl_rp = float(ct[0])
                    if tbl_rp is None:
                        try:
                            oi = int(sl.get("order"))
                        except (TypeError, ValueError):
                            oi = None
                        if oi is not None and oi in per_order_rv:
                            tbl_rp = float(per_order_rv[oi][0])
                    if tbl_rp is not None and np.isfinite(tbl_rp) and abs(v_report - tbl_rp) > 0.5:
                        meta += f"<br><i>ΔRV vs table {v_report - tbl_rp:+.2f} m/s</i>"
                    meta += f"<br>σ_RV: {_preview_abbrev_ccf_uncertainty(diag)}"
                else:
                    if is_line_mask:
                        meta += (
                            f"<br>RV (bisector)={v_report:.2f} m/s · "
                            f"overlay σ={sig:.2f} m/s · amp={amp:.3g}"
                        )
                    else:
                        meta += (
                            f"<br>RV (bisector)={v_report:.2f} m/s · "
                            f"LM σ={sig:.2f} m/s · amp={amp:.3g}"
                        )
                    _bdiag2 = (
                        bisector.get("diagnostics")
                        if isinstance(bisector, dict)
                        else None
                    )
                    if isinstance(_bdiag2, dict):
                        _bis2 = _bdiag2.get("bis_m_s")
                        _bis_e2 = _bdiag2.get("bis_err_m_s")
                        if _bis2 is not None and math.isfinite(float(_bis2)):
                            _bs = f"<br>BIS = {float(_bis2):.2f} m/s"
                            if _bis_e2 is not None and math.isfinite(float(_bis_e2)):
                                _bs += f" (σ={float(_bis_e2):.2f})"
                            meta += _bs
                    if sl.get("kind") == "mean" and np.isfinite(rv_agg_tbl):
                        dvp = v_report - rv_agg_tbl
                        if np.isfinite(dvp) and abs(dvp) > 250.0:
                            meta += f"<br><i>RV−⟨RV⟩tbl {dvp:+.0f} m/s</i>"
                    elif sl.get("kind") != "mean" and not is_line_mask:
                        try:
                            oi = int(sl.get("order"))
                        except (TypeError, ValueError):
                            oi = None
                        if oi is not None and oi in per_order_rv:
                            rp = float(per_order_rv[oi][0])
                            if np.isfinite(rp):
                                meta += f"<br><i>Gauss−IV {v_report - rp:+.1f} m/s</i>"
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
        # Keep RV Derived sidebar fully visible (do not clip right margin).
        "<style>body{margin:0;font-family:system-ui,sans-serif;}"
        "hr{border:none;border-top:1px solid #ccc;margin:0;}"
        ".js-plotly-plot,.plotly-graph-div{overflow:visible !important;}"
        f".plotly-graph-div:first-of-type{{min-width:{int(RV_FIG_WIDTH)}px;}}"
        "</style>\n"
        "</head><body>\n"
        f'<p style="margin:12px 16px 4px;font-size:14px;">{esc_title}</p>\n'
        f"{div_rv}<hr>\n{div_ccf}\n"
        "</body></html>\n"
    )

    if primitive.save_preview:
        logger.info("Stellar RV preview: writing HTML path=%s", output_path_html)
        Path(output_path_html).write_text(combined_html, encoding="utf-8")
        logger.info("Stellar RV preview: HTML write complete")
        n_ccf_traces = len(fig_ccf.data)
        max_ccf_traces = _preview_kaleido_png_max_ccf_traces()
        if _preview_should_write_png(primitive) and n_ccf_traces > max_ccf_traces:
            logger.info(
                "Stellar RV preview: skipping Kaleido PNG "
                "(ccf traces=%d > HISPEC_DRP_PREVIEW_PNG_MAX_CCF_TRACES=%d; HTML-only)",
                n_ccf_traces,
                max_ccf_traces,
            )
        elif _preview_should_write_png(primitive):
            scale = float(getattr(primitive, "preview_png_scale", 2.0))
            scale = max(0.25, min(scale, 8.0))
            timeout_s = _preview_kaleido_png_timeout_s()
            logger.info(
                "Stellar RV preview: writing PNG via Kaleido path=%s scale=%.2f "
                "timeout=%.0fs ccf_traces=%d",
                output_path_png,
                scale,
                timeout_s,
                n_ccf_traces,
            )
            try:
                from PIL import Image

                br = _figure_to_png_bytes_with_timeout(
                    fig_rv, scale=scale, timeout_s=timeout_s
                )
                bc = _figure_to_png_bytes_with_timeout(
                    fig_ccf, scale=scale, timeout_s=timeout_s
                )
                if br is not None and bc is not None:
                    im_rv = Image.open(io.BytesIO(br)).convert("RGBA")
                    im_ccf = Image.open(io.BytesIO(bc)).convert("RGBA")
                    w = max(im_rv.width, im_ccf.width)
                    h = im_rv.height + im_ccf.height
                    stacked = Image.new("RGBA", (w, h), (255, 255, 255, 255))
                    stacked.paste(im_rv, (0, 0))
                    stacked.paste(im_ccf, (0, im_rv.height))
                    stacked.convert("RGB").save(output_path_png)
                    wrote_png = True
                    logger.info("Stellar RV preview: PNG write complete")
                else:
                    logger.info(
                        "Stellar RV preview: PNG skipped (Kaleido timed out or failed; HTML written)"
                    )
            except Exception as e:
                logger.warning(
                    "Stellar RV preview: could not write preview PNG "
                    "(kaleido/Chrome/PIL may be missing): %s",
                    e,
                )
        else:
            if not getattr(primitive, "preview_write_png", True):
                skip_reason = "preview_write_png=False (HTML-only preview)"
            elif _env_flag("HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS"):
                skip_reason = "HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS (HTML-only preview)"
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

    1. **RV figure** — Pipeline RV vs wavelength (Δ vs ⟨RV⟩). For ``ccf_line_mask`` with the
       order-mean+refine path, **one marker per echelle order** (λ = mean of mask chunk centers)
       when RVCCF repeats the same RV on every chunk row; otherwise one marker per chunk. Own
       mode bar.
    2. **CCF figure** — Per fiber, CCF vs velocity (y-axis ``CCF``) then **Residual CCF**
       (raw CCF minus the Gaussian model on the full grid, same units). The Gaussian trace is
       drawn over a **narrow symmetric** span around ``v₀`` (not the full LevMar window). For
       ``ccf_line_mask``, the black **dashed** vertical line is the final refined RV; for template
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

    # Per fiber: RV vs λ scatter (optionally one point per echelle order for line-mask
    # order-mean duplicate rows) + metadata for subplot titles.
    rv_plot_by_fiber: dict[str, dict] = {}
    for fiber in fibers_with_ccf:
        tbl = getattr(output_model, f"rvccf_fib{fiber}", None)
        avg_tbl = getattr(output_model, f"rvccfavg_fib{fiber}", None)
        entry: dict = {"tbl": tbl, "avg_tbl": avg_tbl, "collapsed": False}
        if tbl is None:
            rv_plot_by_fiber[fiber] = entry
            continue
        wl0 = np.asarray(tbl["WAVELENGTH"], dtype=np.float64)
        rv0 = np.asarray(tbl["RV"], dtype=np.float64)
        rv_e0 = np.asarray(tbl["RV_ERR"], dtype=np.float64)
        inl0 = (
            np.asarray(tbl["INLIER"], dtype=bool)
            if "INLIER" in tbl.colnames
            else np.ones(len(rv0), dtype=bool)
        )
        valid0 = np.isfinite(wl0) & np.isfinite(rv0)
        if not np.any(valid0):
            rv_plot_by_fiber[fiber] = entry
            continue
        wl = wl0[valid0]
        rv = rv0[valid0]
        rv_err = rv_e0[valid0]
        inlier = inl0[valid0]
        order_col = (
            np.asarray(tbl["ORDER"], dtype=np.int64)[valid0]
            if "ORDER" in tbl.colnames
            else np.full(len(wl), -1, dtype=np.int64)
        )
        collapsed = False
        if (getattr(primitive, "algorithm", None) or "").strip().lower() in STELLAR_RV_LINE_MASK_ALGORITHMS and "ORDER" in tbl.colnames:
            wl, rv, rv_err, inlier, order_col, collapsed = _collapse_line_mask_rv_scatter_rows(
                wl, rv, rv_err, inlier, order_col
            )
        if (not collapsed) and wl.size > 1 and "ORDER" in tbl.colnames:
            # Ascending echelle order, descending chunk center λ (red → blue within each order).
            _ord_k = np.asarray(order_col, dtype=np.int64)
            _wl_k = np.asarray(wl, dtype=np.float64)
            _ix = np.lexsort((-_wl_k, _ord_k))
            wl = wl[_ix]
            rv = rv[_ix]
            rv_err = rv_err[_ix]
            inlier = inlier[_ix]
            order_col = order_col[_ix]
        entry["wl"] = wl
        entry["rv"] = rv
        entry["rv_err"] = rv_err
        entry["inlier"] = inlier
        entry["order_col"] = order_col
        entry["collapsed"] = collapsed
        entry["n_rvccf_rows"] = int(np.count_nonzero(valid0))
        rv_plot_by_fiber[fiber] = entry

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
        _rp = rv_plot_by_fiber.get(fiber) or {}
        if _rp.get("collapsed"):
            subplot_titles_rv.append(
                f"Fiber {fiber} — RV per echelle order (λ = mean of {int(_rp.get('n_rvccf_rows', 0))} "
                f"mask chunks)"
            )
        else:
            subplot_titles_rv.append(f"Fiber {fiber} — per chunk RV")
        subplot_titles_ccf.append(
            f"Fiber {fiber} — CCF vs RV (±σ_CCF; Gauss; bisector)"
        )
        subplot_titles_ccf.append(f"Fiber {fiber} — Residual CCF")

    # Compact spacing: enough for subplot titles / axis labels, without large empty gaps.
    v_spacing_rv = 0.12 if n_fib <= 3 else max(0.08, min(0.16, 0.40 / max(n_fib - 1, 1)))
    v_spacing_ccf = 0.08 if n_rows_ccf <= 6 else max(0.05, min(0.11, 0.45 / max(n_rows_ccf - 1, 1)))

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
    ccf_trace_indices: dict[str, list[list[int]]] = {}
    resid_trace_indices: dict[str, list[list[int]]] = {}
    resid_ylim_by_fiber: dict[str, tuple[float, float]] = {}
    # Show CCF legend entries only once (first fiber with CCF data).
    ccf_legend_fiber = fibers_with_ccf[0] if fibers_with_ccf else None

    for fi, fiber in enumerate(fibers_with_ccf):
        row_rv = fiber_row_map[fiber]["rv"]

        rp = rv_plot_by_fiber.get(fiber) or {}
        tbl = rp.get("tbl")
        avg_tbl = rp.get("avg_tbl")
        collapsed_plot = bool(rp.get("collapsed", False))
        if tbl is not None and rp.get("wl") is not None:
            wl = rp["wl"]
            rv = rp["rv"]
            rv_err = rp["rv_err"]
            inlier = rp["inlier"]
            order_col = rp["order_col"]
            # Sidebar / MAD / IV stats use every RVCCF row (chunks), not collapsed plot points.
            wl_a = np.asarray(tbl["WAVELENGTH"], dtype=np.float64)
            rv_a = np.asarray(tbl["RV"], dtype=np.float64)
            rv_err_a = np.asarray(tbl["RV_ERR"], dtype=np.float64)
            inlier_a = (
                np.asarray(tbl["INLIER"], dtype=bool)
                if "INLIER" in tbl.colnames
                else np.ones(len(rv_a), dtype=bool)
            )
            valid_a = np.isfinite(wl_a) & np.isfinite(rv_a)
            rv_all = rv_a[valid_a]
            rv_err_all = rv_err_a[valid_a]
            inlier_a = inlier_a[valid_a]
            if np.any(valid_a):
                if avg_tbl is not None and len(avg_tbl) > 0 and np.isfinite(float(avg_tbl["RV"][0])):
                    rv_agg = float(avg_tbl["RV"][0])
                    rv_agg_err = float(avg_tbl["RV_ERR"][0])
                else:
                    rv_agg = float(np.nanmean(rv_all))
                    rv_agg_err = float(
                        np.nanmedian(rv_err_all[np.isfinite(rv_err_all) & (rv_err_all > 0)])
                    )
                    if not np.isfinite(rv_agg_err):
                        rv_agg_err = np.nan
                rv_mean_ms = rv_agg
                rv_mean_ms_r = round(rv_mean_ms, 3)
                y_ms_off = rv - rv_mean_ms
                err_y_ms = rv_err
                rv_pt_ms = rv
                rv_err_pt_ms = rv_err
                if rv_mean_ms_r < 0:
                    y_title = f"RV (m/s) + {abs(rv_mean_ms_r):.4e}"
                else:
                    y_title = f"RV (m/s) − {rv_mean_ms_r:.4e}"
                rv_panel_y_title[fiber] = y_title
                ord_lbl = np.where(order_col >= 0, order_col.astype(str), "—")
                seg_line = _rvcc_hover_order_segment_line(
                    order_col, wl, collapsed=collapsed_plot
                )
                cd_rv = np.stack(
                    [
                        ord_lbl.astype(object),
                        np.asarray(rv_pt_ms, dtype=np.float64),
                        np.asarray(rv_err_pt_ms, dtype=np.float64),
                        np.asarray(seg_line, dtype=object),
                    ],
                    axis=1,
                )
                in_ok = inlier & np.isfinite(rv_err) & (rv_err > 0)
                out_ok = ~inlier & np.isfinite(rv_err) & (rv_err > 0)
                if np.any(in_ok):
                    y_clip_src = y_ms_off[in_ok]
                    err_clip_src = err_y_ms[in_ok]
                else:
                    fin = np.isfinite(y_ms_off)
                    y_clip_src = y_ms_off[fin]
                    err_clip_src = err_y_ms[fin] if err_y_ms.size == y_ms_off.size else None
                if y_clip_src.size:
                    rv_panel_y_range[fiber] = _iterative_sigma_clip_ylim(
                        y_clip_src, y_err=err_clip_src
                    )
                if np.any(in_ok):
                    fig_rv.add_trace(
                        go.Scatter(
                            x=wl[in_ok],
                            y=y_ms_off[in_ok],
                            error_y=dict(type="data", array=err_y_ms[in_ok], visible=True),
                            mode="markers",
                            name=f"Fiber {fiber} (inlier)",
                            showlegend=False,
                            marker=dict(size=6, color="royalblue"),
                            customdata=cd_rv[in_ok],
                            hovertemplate=(
                                "%{customdata[3]}<br>"
                                "λ = %{x:.5f} µm<br>"
                                "RV (pipeline) = %{customdata[1]:.3f} ± %{customdata[2]:.3g} m/s<br>"
                                "ΔRV vs ⟨RV⟩ = %{y:.4f} ± %{error_y.array:.4f} m/s<extra></extra>"
                            ),
                        ),
                        row=row_rv,
                        col=1,
                    )
                if np.any(out_ok):
                    fig_rv.add_trace(
                        go.Scatter(
                            x=wl[out_ok],
                            y=y_ms_off[out_ok],
                            error_y=dict(type="data", array=err_y_ms[out_ok], visible=True),
                            mode="markers",
                            name=f"Fiber {fiber} (outlier)",
                            showlegend=False,
                            marker=dict(size=5, color="grey", symbol="circle-open"),
                            customdata=cd_rv[out_ok],
                            hovertemplate=(
                                "%{customdata[3]}<br>"
                                "λ = %{x:.5f} µm<br>"
                                "RV (pipeline) = %{customdata[1]:.3f} ± %{customdata[2]:.3g} m/s<br>"
                                "ΔRV vs ⟨RV⟩ = %{y:.4f} ± %{error_y.array:.4f} m/s<extra></extra>"
                            ),
                        ),
                        row=row_rv,
                        col=1,
                    )
                if not np.any(in_ok) and len(wl) > 0:
                    fig_rv.add_trace(
                        go.Scatter(
                            x=wl,
                            y=y_ms_off,
                            error_y=dict(type="data", array=err_y_ms, visible=True),
                            mode="markers",
                            name=f"Fiber {fiber} chunks",
                            showlegend=False,
                            marker=dict(size=6, color="royalblue"),
                            customdata=cd_rv,
                            hovertemplate=(
                                "%{customdata[3]}<br>"
                                "λ = %{x:.5f} µm<br>"
                                "RV (pipeline) = %{customdata[1]:.3f} ± %{customdata[2]:.3g} m/s<br>"
                                "ΔRV vs ⟨RV⟩ = %{y:.4f} ± %{error_y.array:.4f} m/s<extra></extra>"
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
                        f"{mean_line}<br>σ(⟨RV⟩) = {_format_rv_value(float(rv_agg_err), kind='sigma_ms')}"
                    )
                else:
                    leg = f"{mean_line}<br>uncertainty = —"
                # RMS bias from mean: use all chunk rows in RVCCF (not collapsed plot points).
                n_good = int(np.sum(inlier_a))
                rms_bias_m_s = float("nan")
                if n_good > 0:
                    biases_m_s = rv_all[inlier_a] - rv_agg
                    rms_bias_m_s = float(np.sqrt(np.sum(biases_m_s**2) / n_good))
                    leg += (
                        f"<br>RMS(⟨RV⟩) = {_format_rv_value(rms_bias_m_s, kind='sigma_ms')}"
                    )
                n_chunks_tbl = int(rv_all.size)
                # Compact labels so the right-hand Derived box fits in the figure margin.
                sb_lines = [
                    f"<b>Derived · fib {fiber}</b>",
                    f"Chunks {n_chunks_tbl} · inliers {n_good}",
                    f"⟨RV⟩ = {mean_ms_str}",
                ]
                if np.isfinite(rv_agg_err):
                    sb_lines.append(
                        f"σ(⟨RV⟩) = {_format_rv_value(float(rv_agg_err), kind='sigma_ms')}"
                    )
                else:
                    sb_lines.append("σ(⟨RV⟩) = —")
                if n_good > 0 and np.isfinite(rms_bias_m_s):
                    sb_lines.append(
                        f"RMS vs ⟨RV⟩ = {_format_rv_value(rms_bias_m_s, kind='sigma_ms')}"
                    )
                    if n_good > 1:
                        std_c = float(np.std(rv_all[inlier_a], ddof=1))
                        if np.isfinite(std_c):
                            sb_lines.append(f"Std (inliers) = {std_c:.2f} m/s")
                else:
                    sb_lines.append("RMS vs ⟨RV⟩ = —")
                osig = float(getattr(primitive, "outlier_sigma", 4.0))
                _, _, mad_gate = combine_rvs_median_mad(rv_all, rv_err_all, osig)
                _, _, inl_c, sig_iv, scatter_se = combine_rv_estimates_median_mad_weighted(
                    rv_all, rv_err_all, osig
                )
                thr_m_s = osig * mad_gate if np.isfinite(mad_gate) else float("nan")
                sb_lines.append(
                    f"MAD = {_format_rv_value(mad_gate, kind='sigma_ms')} · "
                    f"cut = {_format_rv_value(thr_m_s, kind='sigma_ms')} "
                    f"(×{osig:g} · max(MAD, σᵢ))"
                )
                sb_lines.append(f"Outliers (MAD+σ) = {int(np.sum(~inl_c))}")
                if np.isfinite(sig_iv):
                    sb_lines.append(
                        f"σ_IV = {_format_rv_value(sig_iv, kind='sigma_ms')}"
                    )
                if np.isfinite(scatter_se):
                    sb_lines.append(
                        f"scatter_SE = {_format_rv_value(scatter_se, kind='sigma_ms')}"
                    )
                rv_sidebar_by_fiber[fiber] = "<br>".join(sb_lines)
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
                    sig_ms = float(rv_agg_err)
                    sig_ms_lbl = _format_rv_value(sig_ms, kind="sigma_ms")
                    for y_val, label in [(sig_ms, f"+1σ ({sig_ms_lbl})"), (-sig_ms, f"-1σ ({sig_ms_lbl})")]:
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
            detail = primitive._ccf_preview_by_fiber.get(str(fiber))
            if not detail:
                continue
            v_step = float(detail["v_step"])
            slots = _preview_slots_for_plot(detail["slots"])
            is_line_mask = (getattr(primitive, "algorithm", None) or "").strip().lower() in STELLAR_RV_LINE_MASK_ALGORITHMS
            ccf_trace_indices[fiber] = []
            # Shared y-axis for the whole CCF row: use plotted preview slots (mean + optional
            # order chunks). Limits from slot 0 alone clip mask-line chunk CCFs and LSTSQ overlays
            # when the mean CCF has much smaller dynamic range than individual chunks.
            _ccf_y_parts: list[np.ndarray] = []
            for sl0 in slots:
                c0 = np.asarray(sl0["ccf"], dtype=np.float64).ravel()
                m0 = np.isfinite(c0)
                if np.any(m0):
                    _ccf_y_parts.append(c0[m0])
            if _ccf_y_parts:
                ccf_ylim_lo, ccf_ylim_hi = _preview_ccf_y_limits(
                    np.concatenate(_ccf_y_parts, dtype=np.float64)
                )
            else:
                ccf_ylim_lo, ccf_ylim_hi = -1.0, 1.0
            for si, sl in enumerate(slots):
                params, diag = fit_slot_cache.get(id(sl), (None, {}, np.nan, np.nan))[:2]
                vg = np.asarray(sl["v"], dtype=np.float64)
                ccf = np.asarray(sl["ccf"], dtype=np.float64)
                _ce = sl.get("ccf_err")
                vg_plot, ccf_plot, _ = _downsample_ccf_for_plot(
                    vg, ccf, _ce, PREVIEW_CCF_SCATTER_MAX_POINTS
                )
                vg_km = np.round(vg_plot / 1000.0, 4)  # x-axis in km/s (plot subset)
                ccf_plot = np.round(ccf_plot, 5)
                visible = si == 0
                show_leg = fiber == ccf_legend_fiber and si == 0
                ccf_band_trace_idx: list[int] = []
                idx_data = len(fig_ccf.data)
                ccf_scatter_kw: dict = dict(
                    x=vg_km,
                    y=ccf_plot,
                    mode="lines",
                    name="CCF data",
                    line=dict(width=1.2, color="darkgreen"),
                    showlegend=show_leg,
                    visible=visible,
                )
                fig_ccf.add_trace(go.Scatter(**ccf_scatter_kw), row=row, col=1)
                ccf_band_trace_idx.append(idx_data)
                _ov: tuple[float, float, float, float] | None = None
                _amp_p = _v0_p = _sig_p = _base_p = float("nan")
                if params and diag.get("fit_ok"):
                    _ov = (
                        _line_mask_preview_gaussian_overlay_params(sl, params, diag, vg, ccf)
                        if is_line_mask
                        else None
                    )
                    if _ov is not None:
                        _amp_p, _v0_p, _sig_p, _base_p = _ov
                    else:
                        _amp_p = float(params["amplitude"])
                        _v0_p = float(params["v0"])
                        _sig_p = float(params["sigma"])
                        _base_p = float(params["baseline"])
                _bgp = (
                    float(_base_p)
                    if params
                    and diag.get("fit_ok")
                    and np.isfinite(float(_base_p))
                    else None
                )
                if (
                    _bgp is not None
                    and params
                    and diag.get("fit_ok")
                ):
                    idx_bg = len(fig_ccf.data)
                    _bgy = float(_bgp)
                    vg_km_lo = float(np.nanmin(vg_km))
                    vg_km_hi = float(np.nanmax(vg_km))
                    _bg_legend = (
                        "Preview baseline (Gauss+const LSQ)"
                        if _ov is not None
                        else "CCF background (σ-clipped median)"
                    )
                    fig_ccf.add_trace(
                        go.Scatter(
                            x=[vg_km_lo, vg_km_hi],
                            y=[_bgy, _bgy],
                            mode="lines",
                            name=_bg_legend,
                            line=dict(dash="dash", width=1.5, color="black"),
                            showlegend=show_leg,
                            visible=visible,
                        ),
                        row=row,
                        col=1,
                    )
                    ccf_band_trace_idx.append(idx_bg)
                idx_fit = len(fig_ccf.data)
                if params and diag.get("fit_ok"):
                    # Preview: narrow symmetric window around v₀ — do not use LevMar fit_v_lo/hi
                    # (often ±500 km/s), which makes the Gaussian trace look biased and flat.
                    v_lo_m, v_hi_m = _preview_gaussian_curve_v_range_m_s(
                        _v0_p,
                        _sig_p,
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
                        _amp_p,
                        _v0_p,
                        _sig_p,
                        _base_p,
                    )
                    vf_km = vf_m / 1000.0  # x-axis in km/s (match CCF data trace)
                    _gtrace_name = (
                        "Gaussian (LSTSQ-centered)"
                        if _ov is not None
                        else "Gaussian fit"
                    )
                    fig_ccf.add_trace(
                        go.Scatter(
                            x=vf_km,
                            y=cf,
                            mode="lines",
                            name=_gtrace_name,
                            line=dict(width=2, color="crimson"),
                            showlegend=show_leg,
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
                if (not is_line_mask) and params and diag.get("fit_ok"):
                    _rv_mark = float(params["v0"])
                    _mark_name = f"Final RV (Gaussian) = {_rv_mark:.2f} m/s"
                    _bi_sl = sl.get("bisector_fit")
                    if isinstance(_bi_sl, dict):
                        _bdiag_m = _bi_sl.get("diagnostics")
                        if isinstance(_bdiag_m, dict):
                            _bis_m = _bdiag_m.get("bis_m_s")
                            _bis_e = _bdiag_m.get("bis_err_m_s")
                            if _bis_m is not None and math.isfinite(float(_bis_m)):
                                _mark_name += f"<br>BIS = {float(_bis_m):.2f} m/s"
                                if _bis_e is not None and math.isfinite(float(_bis_e)):
                                    _mark_name += f" (σ={float(_bis_e):.2f})"
                    v0_km = _rv_mark / 1000.0
                    idx_v0 = len(fig_ccf.data)
                    fig_ccf.add_trace(
                        go.Scatter(
                            x=[v0_km, v0_km],
                            y=[ccf_ylim_lo, ccf_ylim_hi],
                            mode="lines",
                            name=_mark_name,
                            line=dict(width=1.5, color="black", dash="dash"),
                            showlegend=show_leg,
                            visible=visible,
                        ),
                        row=row,
                        col=1,
                    )
                    ccf_band_trace_idx.append(idx_v0)
                    for ti in _bisector_preview_plotly_traces(
                        fig_ccf,
                        row,
                        sl=sl,
                        vg=vg,
                        ccf=ccf,
                        visible=visible,
                        show_legend=show_leg,
                    ):
                        ccf_band_trace_idx.append(ti)
                if is_line_mask:
                    for ti in _line_mask_refine_ccf_add_plotly_traces(
                        fig_ccf,
                        row,
                        sl=sl,
                        vg=vg,
                        ccf=ccf,
                        params=params,
                        diag=diag,
                        visible=visible,
                        show_legend=show_leg,
                        ylim_lo=ccf_ylim_lo,
                        ylim_hi=ccf_ylim_hi,
                    ):
                        ccf_band_trace_idx.append(ti)
                ccf_trace_indices[fiber].append(ccf_band_trace_idx)
            fig_ccf.update_yaxes(range=[ccf_ylim_lo, ccf_ylim_hi], autorange=False, row=row, col=1)
            continue

        # resid — CCF data minus the same Gaussian model evaluated on the full velocity grid.
        detail = primitive._ccf_preview_by_fiber.get(str(fiber))
        if not detail:
            continue
        slots = _preview_slots_for_plot(detail["slots"])
        resid_trace_indices[fiber] = []
        is_line_mask_r = (getattr(primitive, "algorithm", None) or "").strip().lower() in STELLAR_RV_LINE_MASK_ALGORITHMS
        # Match CCF row: default y-range from the first slot, same padding rule as
        # :func:`_preview_ccf_y_limits` (~10% wider than robust data span).
        if slots:
            sl0 = slots[0]
            params0, diag0 = fit_slot_cache.get(id(sl0), (None, {}, np.nan, np.nan))[:2]
            vg0 = np.asarray(sl0["v"], dtype=np.float64)
            ccf0 = np.asarray(sl0["ccf"], dtype=np.float64)
            if params0 and diag0.get("fit_ok"):
                _ov0 = (
                    _line_mask_preview_gaussian_overlay_params(
                        sl0, params0, diag0, vg0, ccf0
                    )
                    if is_line_mask_r
                    else None
                )
                if _ov0 is not None:
                    a0, v00, s0, b0 = _ov0
                else:
                    a0 = float(params0["amplitude"])
                    v00 = float(params0["v0"])
                    s0 = float(params0["sigma"])
                    b0 = float(params0["baseline"])
                model0 = _gaussian_ccf_model(vg0, a0, v00, s0, b0)
                resid_ylim_by_fiber[fiber] = _preview_ccf_y_limits(ccf0 - model0)
            else:
                resid_ylim_by_fiber[fiber] = (-1.0, 1.0)
        else:
            resid_ylim_by_fiber[fiber] = (-1.0, 1.0)

        for si, sl in enumerate(slots):
            params, diag = fit_slot_cache.get(id(sl), (None, {}, np.nan, np.nan))[:2]
            vg = np.asarray(sl["v"], dtype=np.float64)
            ccf = np.asarray(sl["ccf"], dtype=np.float64)
            visible = si == 0
            if params and diag.get("fit_ok"):
                _ovr = (
                    _line_mask_preview_gaussian_overlay_params(sl, params, diag, vg, ccf)
                    if is_line_mask_r
                    else None
                )
                if _ovr is not None:
                    ap, vp, sp, bp = _ovr
                else:
                    ap = float(params["amplitude"])
                    vp = float(params["v0"])
                    sp = float(params["sigma"])
                    bp = float(params["baseline"])
                model_at = _gaussian_ccf_model(vg, ap, vp, sp, bp)
                residual = ccf - model_at
            else:
                residual = np.full_like(ccf, np.nan, dtype=np.float64)
            vg_res, residual_plot, _ = _downsample_ccf_for_plot(
                vg, residual, None, PREVIEW_CCF_SCATTER_MAX_POINTS
            )
            vg_km_res = np.round(vg_res / 1000.0, 4)
            residual_plot = np.round(residual_plot, 5)
            r_slot_idx: list[int] = []
            resid_kw: dict = dict(
                x=vg_km_res,
                y=residual_plot,
                mode="lines",
                name="Residual CCF",
                line=dict(width=1.2, color="darkslateblue"),
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

    # No per-panel CCF text overlays (they clutter the CCF curves). Slot metadata still
    # drives the order-slot dropdown labels via ``updatemenus``.
    annotations: list[dict] = []

    # Shared with updatemenus so order-slot changes do not reset legend position / visibility.
    # Keep the legend fully outside the subplot frames, below the top-row slot dropdown.
    ccf_legend_dict: dict = dict(
        x=1.02,
        xref="paper",
        xanchor="left",
        y=0.58,
        yref="paper",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.92)",
        borderwidth=1,
        bordercolor="rgba(0,0,0,0.12)",
        font=dict(size=8),
        tracegroupgap=4,
        itemwidth=30,
        orientation="v",
    )

    updatemenus_ccf: list[dict] = []
    for fi, fiber in enumerate(fibers_with_ccf):
        slots = _preview_slots_for_plot(
            primitive._ccf_preview_by_fiber.get(str(fiber), {}).get("slots", [])
        )
        if not slots:
            continue
        ccf_indices = ccf_trace_indices.get(fiber, [])
        res_indices = resid_trace_indices.get(fiber, [])
        buttons = []
        for si in range(len(slots)):
            sl = slots[si]
            if sl.get("kind") == "mean":
                label = "Mean (all ord)"
            else:
                o = sl.get("order", "?")
                wl_lo, wl_hi = sl.get("wl_lo", np.nan), sl.get("wl_hi", np.nan)
                cn = int(sl.get("chunk_n", 1) or 1)
                cix = sl.get("chunk_index")
                seg = ""
                if cn > 1 and cix is not None:
                    seg = f"{int(cix) + 1}/{cn} · "
                if np.isfinite(wl_lo) and np.isfinite(wl_hi):
                    label = f"Ord {o} · {seg}{wl_lo:.3f}–{wl_hi:.3f} µm"
                else:
                    label = f"Ord {o}" + (f" · {seg}".rstrip(" · ") if seg else "")
            visible = [True] * n_traces
            for slot_idx in range(len(slots)):
                if slot_idx != si:
                    if slot_idx < len(ccf_indices):
                        for ti in ccf_indices[slot_idx]:
                            visible[ti] = False
                    if slot_idx < len(res_indices):
                        for ti in res_indices[slot_idx]:
                            visible[ti] = False
            buttons.append(
                dict(
                    label=label,
                    method="update",
                    args=[
                        {"visible": visible},
                        {
                            "showlegend": True,
                            "legend": ccf_legend_dict,
                        },
                    ],
                )
            )
        if not buttons:
            continue
        ccf_row_1based = fiber_row_map[fiber]["ccf"]
        dx, dy = _ccf_slot_dropdown_paper_xy(
            fig_ccf, ccf_row_1based, legend=ccf_legend_dict
        )
        updatemenus_ccf.append(
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                showactive=True,
                x=dx,
                xanchor=str(ccf_legend_dict.get("xanchor", "left")),
                y=dy,
                yanchor="top",
                font=dict(size=9),
                bgcolor="rgba(255,255,255,0.9)",
                borderwidth=1,
                bordercolor="gray",
            )
        )

    rv_fig_w = int(RV_FIG_WIDTH)
    rv_fig_mr = int(RV_FIG_MARGIN_RIGHT)
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
                    font=dict(size=10, family="Arial, sans-serif"),
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="rgba(0,0,0,0.15)",
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

    fig_h_rv = int(min(20000, 200 + n_fib * 280))
    fig_h_ccf = int(min(20000, 200 + n_rows_ccf * 160))
    # ``make_subplots(..., subplot_titles=...)`` stores titles as layout.annotations. Passing only
    # ``annotations_rv`` would patch those entries by index and replace the title with sidebar
    # content; prepend the existing annotations so Derived boxes are appended.
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
        # Traces use showlegend=False; keep legend off so the right margin is for Derived.
        showlegend=False,
        margin=dict(l=70, r=rv_fig_mr, t=70, b=45),
        annotations=rv_layout_ann + annotations_rv,
    )
    fig_ccf.update_layout(
        height=fig_h_ccf,
        width=720,
        title_text=f"{title_unified} — CCF & residual",
        hovermode="closest",
        dragmode="zoom",
        showlegend=True,
        legend=ccf_legend_dict,
        margin=dict(l=70, r=280, t=70, b=45),
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
            "title_text": rv_panel_y_title.get(fiber, "RV (m/s) − ⟨RV⟩"),
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

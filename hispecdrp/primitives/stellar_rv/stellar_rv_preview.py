"""
Interactive previews for stellar radial-velocity products.

There are **two** preview surfaces:

1. **Default unified figure** (``create_stellar_rv_preview``): Plotly panels for RV vs
   wavelength (per chunk), CCF vs velocity with Gaussian fits, and fit residuals.
2. **Mask-shift debug HTML** (``write_mask_overlay_debug_pages``): per-order spectrum with
   mask segment overlay and velocity slider, for ``algorithm="ccf_mask_kpf"`` only.

Gaussian reference curves use :func:`~hispecdrp.primitives.utils.ccf_utils.fit_ccf_rv`;
mask geometry uses :mod:`~hispecdrp.primitives.utils.ccf_utils` (mask loaders).

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

import json
import logging
import os
import webbrowser

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils.ccf_utils import (
    _V_MAX,
    _V_MIN,
    _V_STEP,
    _gaussian_ccf_model,
    fit_ccf_rv,
    load_mask,
    resolve_mask_path,
    restrict_mask_to_chunk,
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

    When chunk RV arrays are supplied (same length as ``run_ccf_mask_kpf_for_fiber`` outputs),
    each order entry includes ``fit_rv_m_s`` / ``fit_rv_err_m_s`` for the velocity-slider hint.
    """
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
            float(np.nanmin(w)),
            float(np.nanmax(w)),
            v_min_m_s=_V_MIN,
            v_max_m_s=_V_MAX,
            zb=0.0,
        )
        if mask_o.start_um.size > 0:
            mstart = np.asarray(mask_o.start_um, dtype=np.float64)
            mend = np.asarray(mask_o.end_um, dtype=np.float64)
            mw = np.asarray(mask_o.weight, dtype=np.float64)
        else:
            mstart = np.array([], dtype=np.float64)
            mend = np.array([], dtype=np.float64)
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
    logger.info(
        "Stellar RV preview: mask overlay payload built fiber=%s n_orders=%d",
        fiber,
        len(entries),
    )
    return {"fiber": str(fiber), "orders": entries}


C_M_S = 299792458.0


def _json_opt_float(x: float | None) -> float | None:
    if x is None:
        return None
    xf = float(x)
    return xf if np.isfinite(xf) else None


def _to_serializable_order_payload(order_payload: dict) -> dict:
    out_orders = []
    for row in order_payload.get("orders", []):
        out_orders.append(
            {
                "order": int(row["order"]),
                "wl_lo_um": float(row["wl_lo_um"]),
                "wl_hi_um": float(row["wl_hi_um"]),
                "wave": [float(x) for x in row["wave"]],
                "spec": [float(x) for x in row["spec"]],
                "mask_start_um": [float(x) for x in row["mask_start_um"]],
                "mask_end_um": [float(x) for x in row["mask_end_um"]],
                "mask_weight": [float(x) for x in row["mask_weight"]],
                "fit_rv_m_s": _json_opt_float(row.get("fit_rv_m_s")),
                "fit_rv_err_m_s": _json_opt_float(row.get("fit_rv_err_m_s")),
            }
        )
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
    payload_json = json.dumps(_to_serializable_order_payload(payload))
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 14px; }}
    .row {{ display: flex; gap: 12px; align-items: center; margin-bottom: 10px; flex-wrap: wrap; }}
    .row label {{ font-size: 13px; }}
    #plot {{ width: 100%; height: 760px; }}
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
  <p style="font-size:11px;color:#555;margin:0 0 4px 0;max-width:840px;">
    Green band / tick: fitted RV ±1σ for this order (inverse-variance mean over mask-CCF chunks).
  </p>
  <div class="row">
    <div id="vSliderWrap">
      <div id="rvFitTrack"></div>
      <div id="rvFitBand"></div>
      <div id="rvFitTick"></div>
      <input id="vSlider" type="range" min="{float(v_min_m_s):.1f}" max="{float(v_max_m_s):.1f}" step="{float(v_step_m_s):.1f}" />
    </div>
    <span id="vLabel"></span>
  </div>
  <div id="plot"></div>

  <script>
    const C_MS = {C_M_S};
    const ZB = {float(zb)};
    const DATA = {payload_json};
    let currentOrderIdx = 0;
    let currentV = {float(v_init_m_s)};

    const orderSelect = document.getElementById("orderSelect");
    const vSlider = document.getElementById("vSlider");
    const vInput = document.getElementById("vInput");
    const vLabel = document.getElementById("vLabel");
    const plotDiv = document.getElementById("plot");
    const orderCache = new Map();

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

      for (let i = 0; i < order.mask_start_um.length; i++) {{
        const lnStart = order.mask_start_um[i];
        const lnEnd = order.mask_end_um[i];
        const lnWeight = order.mask_weight[i];
        if (!Number.isFinite(lnStart) || !Number.isFinite(lnEnd) || !Number.isFinite(lnWeight)) continue;
        let lineStart = lnStart * shift;
        let lineEnd = lnEnd * shift;
        if (!Number.isFinite(lineStart) || !Number.isFinite(lineEnd)) continue;
        if (lineEnd <= lineStart) continue;

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

      let ccf = 0.0;
      for (let i = 0; i < n; i++) {{
        ccf += sUse[i] * maskSpectra[i];
      }}
      return ccf;
    }}

    function updatePlot() {{
      const order = DATA.orders[currentOrderIdx];
      if (!order) return;

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

      const xMin = Math.min(...xSpec);
      const xMax = Math.max(...xSpec);
      const xMaskStart = shiftWavelengths(order.mask_start_um, currentV);
      const xMaskEnd = shiftWavelengths(order.mask_end_um, currentV);
      const wMask = order.mask_weight;

      const shapes = [];
      for (let i = 0; i < xMaskStart.length; i++) {{
        const x0 = xMaskStart[i];
        const x1 = xMaskEnd[i];
        if (!Number.isFinite(x0) || !Number.isFinite(x1)) continue;
        const lo = Math.min(x0, x1);
        const hi = Math.max(x0, x1);
        if (hi < xMin || lo > xMax) continue;
        let w = Number(wMask[i]);
        if (!Number.isFinite(w) || w < 0) w = 0;
        const yTop = 1.0;
        const yBot = Math.max(0.0, yTop - 2.0 * w);
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

      const layout = {{
        uirevision: `fib-${{DATA.fiber}}-ord-${{order.order}}`,
        title: {{
          text: (() => {{
            const lo = Number(order.wl_lo_um);
            const hi = Number(order.wl_hi_um);
            let wl = "";
            if (Number.isFinite(lo) && Number.isFinite(hi)) {{
              wl = ` | λ ${{lo.toFixed(4)}}–${{hi.toFixed(4)}} µm`;
            }}
            return `Fiber ${{DATA.fiber}} | Order ${{order.order}}${{wl}} | v = ${{currentV.toFixed(1)}} m/s`;
          }})(),
          y: 1.08,
          yref: "paper",
          yanchor: "top",
          pad: {{ b: 20 }},
        }},
        xaxis: {{ title: "Wavelength (um)" }},
        yaxis: {{
          title: "Normalize flux",
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
          y: 1.03,
          yanchor: "bottom",
          bgcolor: "rgba(255, 255, 255, 0.92)",
          bordercolor: "rgba(0,0,0,0.15)",
          borderwidth: 1,
        }},
        margin: {{ l: 70, r: 70, t: 170, b: 55 }},
      }};

      Plotly.react(plotDiv, traces, layout, {{responsive: true}});
      const ccfVal = ccfAtVelocity(order, currentV);
      const ccfText = Number.isFinite(ccfVal) ? ccfVal.toExponential(3) : "nan";
      let fitTxt = "";
      if (order.fit_rv_m_s != null && Number.isFinite(Number(order.fit_rv_m_s))) {{
        const frv = Number(order.fit_rv_m_s);
        const fe = order.fit_rv_err_m_s;
        if (fe != null && Number.isFinite(Number(fe)) && Number(fe) > 0) {{
          fitTxt = ` | fit ${{frv.toFixed(1)}} ± ${{Number(fe).toFixed(1)}} m/s`;
        }} else {{
          fitTxt = ` | fit ${{frv.toFixed(1)}} m/s`;
        }}
      }}
      vLabel.textContent = `v = ${{currentV.toFixed(1)}} m/s | CCF = ${{ccfText}}${{fitTxt}}`;
      updateRvFitIndicator(order);
    }}

    function setVelocity(v) {{
      currentV = Number(v);
      if (!Number.isFinite(currentV)) currentV = 0.0;
      vSlider.value = String(currentV);
      vInput.value = String(currentV);
      updatePlot();
    }}

    orderSelect.addEventListener("change", (ev) => {{
      currentOrderIdx = Number(ev.target.value);
      updatePlot();
    }});
    vSlider.addEventListener("input", (ev) => setVelocity(ev.target.value));
    vInput.addEventListener("change", (ev) => setVelocity(ev.target.value));
    vInput.addEventListener("keydown", (ev) => {{
      if (ev.key === "Enter") setVelocity(ev.target.value);
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

    for fiber, payload in payloads.items():
        html = _build_mask_overlay_debug_html(
            title=f"Stellar RV mask overlay debug (fiber {fiber})",
            payload=payload,
            v_init_m_s=v_init,
            v_min_m_s=_V_MIN,
            v_max_m_s=_V_MAX,
            v_step_m_s=_V_STEP,
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


def create_stellar_rv_preview(primitive) -> None:
    """
    Plotly preview: for each fiber, three panels stacked vertically:
    1) RV vs wavelength (per-chunk with legend),
    2) CCF vs RV shift + Gaussian fit + metadata, with dropdown to select combined or per-order,
    3) Residual (CCF − Gaussian) vs RV shift (same dropdown selection).
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

    # Three rows per fiber: RV vs λ, CCF (dropdown), Residual (dropdown)
    row_plan: list[tuple] = []
    for fiber in fibers_with_ccf:
        row_plan.append(("rv", fiber))
        row_plan.append(("ccf", fiber))
        row_plan.append(("resid", fiber))

    n_rows = len(row_plan)
    logger.info(
        "Stellar RV preview: building Plotly figure n_subplot_rows=%d (3 × n_fibers)",
        n_rows,
    )
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

    logger.info(
        "Stellar RV preview: CCF slot metadata and Gaussian cache ready "
        "(cached distinct slot objects=%d)",
        len(fit_slot_cache),
    )

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

    algo = getattr(primitive, "algorithm", None) or "ccf_stellar_template"
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

    logger.info(
        "Stellar RV preview: figure traces, axes, dropdowns, and layout complete "
        "(n_plotly_traces=%d)",
        len(fig.data),
    )

    base_path = _fits_base_path(primitive, output_model)
    output_path_html = base_path.replace(".fits", "_preview.html")
    output_path_png = base_path.replace(".fits", "_preview.png")
    wrote_png = False

    if primitive.save_preview:
        logger.info("Stellar RV preview: writing HTML path=%s", output_path_html)
        fig.write_html(output_path_html, include_plotlyjs="cdn")
        logger.info("Stellar RV preview: HTML write complete")
        if _preview_should_write_png(primitive):
            try:
                # Use Kaleido v1 API directly so Plotly's write_image(engine=...) path is avoided.
                import kaleido

                scale = float(getattr(primitive, "preview_png_scale", 2.0))
                scale = max(0.25, min(scale, 8.0))
                logger.info(
                    "Stellar RV preview: writing PNG via Kaleido path=%s scale=%.2f",
                    output_path_png,
                    scale,
                )
                kaleido.write_fig_sync(fig, path=output_path_png, opts={"scale": scale})
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
        logger.info("Stellar RV preview: opening interactive figure (fig.show)")
        fig.show()
        logger.info("Stellar RV preview: fig.show returned")
    logger.info("Stellar RV preview: create_stellar_rv_preview finished")

    primitive._output["preview"] = {
        "html": output_path_html,
        "png": output_path_png if wrote_png else None,
        "fig": fig,
    }

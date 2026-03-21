"""Debug-only mask overlay HTML for ccf_mask_kpf."""

from __future__ import annotations

import json
import logging
import os
import webbrowser

import numpy as np

C_M_S = 299792458.0

logger = logging.getLogger(__name__)


def _to_serializable_order_payload(order_payload: dict) -> dict:
    out_orders = []
    for row in order_payload.get("orders", []):
        out_orders.append(
            {
                "order": int(row["order"]),
                "wave": [float(x) for x in row["wave"]],
                "spec": [float(x) for x in row["spec"]],
                "mask_start_um": [float(x) for x in row["mask_start_um"]],
                "mask_end_um": [float(x) for x in row["mask_end_um"]],
                "mask_weight": [float(x) for x in row["mask_weight"]],
            }
        )
    return {"fiber": str(order_payload["fiber"]), "orders": out_orders}


def _build_debug_html(
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
    select {{ min-width: 140px; }}
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
  <div class="row">
    <input id="vSlider" type="range" min="{float(v_min_m_s):.1f}" max="{float(v_max_m_s):.1f}" step="{float(v_step_m_s):.1f}" style="width: 840px;" />
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
      opt.textContent = `Order ${{o.order}}`;
      orderSelect.appendChild(opt);
    }});

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
      // Map flux to [0, 1] so left axis matches mask weight (right): 1 = top, 0 = bottom.
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

      // Orange rectangles on y2: top at mask weight = 1, bottom at (1 − 2*w) using file weight w.
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
        // Invisible anchor so the legend shows mask styling (rectangles are shapes).
        {{
          x: [null], y: [null], mode: "markers", name: "Mask lines (shifted)",
          marker: {{ size: 10, color: "rgba(255, 140, 0, 0.7)", symbol: "square" }},
          yaxis: "y2",
          showlegend: true,
        }},
      ];

      const layout = {{
        // Same uirevision across slider/input updates → keep x/y zoom, pan, and trace toggles.
        // Include order in the key so switching orders does not inherit the previous order's ranges.
        uirevision: `fib-${{DATA.fiber}}-ord-${{order.order}}`,
        title: {{
          text: `Fiber ${{DATA.fiber}} | Order ${{order.order}} | v = ${{currentV.toFixed(1)}} m/s`,
          y: 1.08,
          yref: "paper",
          yanchor: "top",
          pad: {{ b: 20 }},
        }},
        xaxis: {{ title: "Wavelength (um)" }},
        yaxis: {{
          title: "Normalized flux (0–1)",
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
        // Horizontal legend: near top of #plot, below title block, above the axis/data region.
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
      vLabel.textContent = `v = ${{currentV.toFixed(1)}} m/s | CCF = ${{ccfText}}`;
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


def create_stellar_rv_mask_overlay_debug(primitive, output_model) -> dict[str, str]:
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
        logger.info("MASK_OVERLAY_DEBUG no payloads to plot.")
        return out

    output_dir = getattr(primitive, "output_dir", None) or os.getcwd()
    base_path = output_model.get_output_path(output_dir=output_dir, suffix=primitive.suffix)
    v_init = float(getattr(primitive, "mask_overlay_velocity_m_s", 0.0))

    for fiber, payload in payloads.items():
        html = _build_debug_html(
            title=f"Stellar RV mask overlay debug (fiber {fiber})",
            payload=payload,
            v_init_m_s=v_init,
            v_min_m_s=-500000.0,
            v_max_m_s=500000.0,
            v_step_m_s=50.0,
        )
        path = base_path.replace(".fits", f"_mask_overlay_debug_fib{fiber}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        out[str(fiber)] = path
        logger.info("MASK_OVERLAY_DEBUG wrote %s", path)
        if bool(getattr(primitive, "show_mask_overlay_debug", False)):
            try:
                webbrowser.open(f"file://{path}")
            except Exception as exc:
                logger.debug("Could not open mask overlay debug HTML: %s", exc)
    return out


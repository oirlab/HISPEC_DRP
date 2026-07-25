"""
Preview-plot generation and an HTML report wrapper for end-to-end simulations.

After :func:`hispecdrp.simulations.e2e_pipeline.run_e2e_simulation` writes the per-step
FITS products to ``output_dir``, this module renders a browsable set of PNG previews
organized by pipeline **stage** and **primitive**, and writes a single ``index.html``
that the user can open and scroll through.

Preview policy (per user request):

- **Stage 0**, steps *up to* ramp fitting (``*_raw-*.fits`` ramp products): display the
  **average of the detector across all reads** (mean over the read axis of the ``DATA`` cube).
- **Stage 0**, ramp fitting and *after* (``*_rate-*.fits`` products): display the **rate map**
  (the 2-D ``RATE`` image).
- **Stage 1** (``*_specuncal-specextract.fits``): display the **1-D extracted spectra**, one
  panel per echelle order, per fiber.
- **Stage 2**: reuse the pipeline's **default previews** (e.g. the stellar-RV CCF preview and
  mask-overlay debug pages already written next to the L2 products).
- **Fiberflat pipeline** (``cals/*mastercal-tracepos*preview.png``): include the trace-position
  preview written by :class:`~hispecdrp.primitives.trace_fitting.tracepos_primitive.TracePositionPrimitive`
  during ``ProcessFiberflatsPipeline``.

The module reads the saved FITS directly (``astropy.io.fits``) so it does not depend on the
in-memory pipeline models and naturally skips steps that were disabled/skipped.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

PREVIEWS_SUBDIR = "previews"
REPORT_HTML_NAME = "index.html"

# Longest image dimension (pixels) for rendered detector/rate previews.
_MAX_IMAGE_DIM = 1100

# Canonical ordering and display labels for Stage 0 / Stage 1 step tokens. Tokens are the text
# after the suffix dash in the saved filename (e.g. ``HB...._raw-dqinit.fits`` -> ``dqinit``).
_STAGE0_TOKEN_ORDER: dict[str, str] = {
    "dqinit": "DQ Init",
    "satcheck": "Saturation Check",
    "biassub": "Bias Subtraction",
    "refpixcorr": "Reference Pixel Correction",
    "nonlincorr": "Nonlinearity Correction",
    "jumpdetec": "Jump Detection",
    "rampfit": "Ramp Fitting",
    "darksub": "Dark Subtraction",
    "gaincorr": "Gain Correction",
    "detflatcorr": "Detector Flat Correction",
}
_STAGE1_TOKEN_ORDER: dict[str, str] = {
    "specextract": "Spectral Extraction",
    "wavesolinit": "Initial Wavelength Solution",
}
_STAGE2_TOKEN_ORDER: dict[str, str] = {
    "blaze_correct": "Blaze Correction",
    "barycorr": "Barycentric Correction",
    "stellar_rv": "Stellar RV (CCF)",
}

# Filename suffix tokens: <base>_<group>-<token>.fits
_PRODUCT_RE = re.compile(r"^(?P<base>.+?)_(?P<group>raw|rate|specuncal|spec)-(?P<token>[A-Za-z0-9_]+)\.fits$")


@dataclass
class PreviewItem:
    """One rendered preview belonging to a (stage, primitive) section."""

    primitive_label: str
    primitive_order: int
    images: list[str] = field(default_factory=list)  # paths relative to the report dir
    links: list[tuple[str, str]] = field(default_factory=list)  # (label, href)
    caption: str = ""


def infer_exposure_base_name(
    *,
    stage0_model=None,
    stage1_model=None,
    stage2_model=None,
) -> str | None:
    """Best-effort science exposure base name (e.g. ``HB.20260612.57502.95``) from a model."""
    for model in (stage2_model, stage1_model, stage0_model):
        if model is None:
            continue
        meta = getattr(model, "meta", None)
        fname = getattr(meta, "filename", None) if meta is not None else None
        if not fname:
            fname = getattr(model, "_filepath", None)
        if fname:
            base = _strip_product_suffix(os.path.basename(str(fname)))
            if base:
                return base
    return None


def _strip_product_suffix(filename: str) -> str | None:
    m = _PRODUCT_RE.match(filename)
    if m:
        return m.group("base")
    # Fall back to a bare ``<base>_rate.fits`` / ``<base>_specuncal.fits`` style name.
    m2 = re.match(r"^(?P<base>.+?)_(raw|rate|specuncal|spec)\.fits$", filename)
    if m2:
        return m2.group("base")
    return None


def _infer_base_from_dir(output_dir: str) -> str | None:
    """Pick the exposure base with the most products directly under ``output_dir``."""
    counts: dict[str, int] = {}
    for name in os.listdir(output_dir):
        if not name.endswith(".fits"):
            continue
        m = _PRODUCT_RE.match(name)
        if not m:
            continue
        counts[m.group("base")] = counts.get(m.group("base"), 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda k: counts[k])


# --------------------------------------------------------------------------------------
# Image rendering helpers
# --------------------------------------------------------------------------------------


def _downsample_2d(arr: np.ndarray, max_dim: int = _MAX_IMAGE_DIM) -> np.ndarray:
    """Stride-downsample a 2-D array so its longest dimension is <= ``max_dim``."""
    a = np.asarray(arr)
    if a.ndim != 2:
        return a
    ny, nx = a.shape
    step = max(1, int(np.ceil(max(ny, nx) / float(max_dim))))
    if step <= 1:
        return a
    return a[::step, ::step]


def _finite_mean_std(arr: np.ndarray) -> tuple[float, float]:
    """Mean and sample std of finite pixels in a 2-D (or collapsible) array."""
    a = np.asarray(arr, dtype=np.float64).ravel()
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.std(finite))


def _format_2d_frame_title(base_title: str, mean: float, std: float) -> str:
    """Append pixel mean and standard deviation to a matplotlib frame title."""
    return f"{base_title}\nMean={mean:.2f}, Standard Deviation={std:.2f}"


def _save_detector_image_png(arr2d: np.ndarray, out_path: str, title: str) -> None:
    """Render a 2-D detector/rate image with robust percentile scaling to ``out_path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = _downsample_2d(np.asarray(arr2d, dtype=np.float64))
    finite = np.isfinite(a)
    if np.any(finite):
        vmin, vmax = np.nanpercentile(a[finite], [1.0, 99.0])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = float(np.nanmin(a[finite]))
            vmax = float(np.nanmax(a[finite]))
            if vmin == vmax:
                vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    im = ax.imshow(a, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("detector x (downsampled)")
    ax.set_ylabel("detector y (downsampled)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=95)
    plt.close(fig)


def _save_orders_spectra_png(
    flux: np.ndarray,
    dq: np.ndarray | None,
    out_path: str,
    title: str,
) -> None:
    """Render 1-D extracted spectra (one panel per order) for a single fiber."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = np.asarray(flux, dtype=np.float64)
    if f.ndim != 2:
        f = np.atleast_2d(f)
    n_orders, n_pix = f.shape
    dq_arr = np.asarray(dq) if dq is not None else None

    ncols = 4
    nrows = int(np.ceil(n_orders / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 3.4, nrows * 1.7), squeeze=False
    )
    x = np.arange(n_pix)
    for k in range(nrows * ncols):
        ax = axes[k // ncols][k % ncols]
        if k >= n_orders:
            ax.axis("off")
            continue
        y = f[k].copy()
        if dq_arr is not None and dq_arr.shape == f.shape:
            y = np.where(dq_arr[k] != 0, np.nan, y)
        ax.plot(x, y, lw=0.5, color="#1f77b4")
        ax.set_title(f"order {k}", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.margins(x=0.0)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=95)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Per-stage preview builders
# --------------------------------------------------------------------------------------


def _read_hdu_data(path: str, names: list[str]):
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        available = {hdu.name.upper(): i for i, hdu in enumerate(hdul)}
        for name in names:
            idx = available.get(name.upper())
            if idx is not None and hdul[idx].data is not None:
                return np.asarray(hdul[idx].data)
    return None


def _build_stage0_item(
    path: str,
    token: str,
    *,
    kind: str,
    previews_dir: str,
    report_dir: str,
) -> PreviewItem | None:
    label = _STAGE0_TOKEN_ORDER.get(token, token)
    order = list(_STAGE0_TOKEN_ORDER).index(token) if token in _STAGE0_TOKEN_ORDER else 999
    out_png = os.path.join(previews_dir, f"stage0_{token}.png")
    if kind == "ramp_mean":
        cube = _read_hdu_data(path, ["DATA", "SCI"])
        if cube is None or cube.ndim < 3:
            logger.warning("Stage 0 preview: no ramp cube in %s; skipping.", path)
            return None
        # Mean across the read axis (axis 0 for (nreads, ny, nx); collapse leading axes if 4-D).
        img = np.asarray(cube, dtype=np.float64)
        while img.ndim > 2:
            img = np.nanmean(img, axis=0)
        base_title = f"{label}: mean of detector across all reads"
        mean, std = _finite_mean_std(img)
        title = _format_2d_frame_title(base_title, mean, std)
        caption = (
            "Average of the detector counts across all UTR reads. "
            f"Mean={mean:.2f}, Standard Deviation={std:.2f}."
        )
    else:  # rate_map
        img = _read_hdu_data(path, ["RATE", "SCI", "DATA"])
        if img is None:
            logger.warning("Stage 0 preview: no rate image in %s; skipping.", path)
            return None
        img = np.asarray(img, dtype=np.float64)
        while img.ndim > 2:
            img = img[0]
        base_title = f"{label}: rate map"
        mean, std = _finite_mean_std(img)
        title = _format_2d_frame_title(base_title, mean, std)
        caption = (
            "Rate map (e-/s per pixel) after this step. "
            f"Mean={mean:.2f}, Standard Deviation={std:.2f}."
        )
    _save_detector_image_png(img, out_png, title)
    rel = os.path.relpath(out_png, report_dir)
    return PreviewItem(
        primitive_label=label,
        primitive_order=order,
        images=[rel],
        caption=caption,
    )


def _build_stage1_item(
    path: str,
    token: str,
    *,
    previews_dir: str,
    report_dir: str,
) -> PreviewItem | None:
    from astropy.io import fits

    label = _STAGE1_TOKEN_ORDER.get(token, token)
    order = list(_STAGE1_TOKEN_ORDER).index(token) if token in _STAGE1_TOKEN_ORDER else 999
    images: list[str] = []
    with fits.open(path, memmap=False) as hdul:
        hdu_names = [h.name.upper() for h in hdul]
        spec_names = sorted(
            (hdu.name for hdu in hdul if re.match(r"^SPEC_UNCAL_FIB\d+$", hdu.name.upper())),
            key=lambda s: int(re.search(r"\d+", s).group()),
        )
        for spec_name in spec_names:
            fib = int(re.search(r"\d+", spec_name).group())
            flux = hdul[spec_name].data
            if flux is None:
                continue
            dq_name = f"SPEC_DQ_FIB{fib}"
            dq = hdul[dq_name].data if dq_name.upper() in hdu_names else None
            out_png = os.path.join(previews_dir, f"stage1_{token}_fib{fib}.png")
            _save_orders_spectra_png(
                np.asarray(flux),
                np.asarray(dq) if dq is not None else None,
                out_png,
                f"{label}: extracted 1-D spectra, fiber {fib}",
            )
            images.append(os.path.relpath(out_png, report_dir))
    if not images:
        return None
    return PreviewItem(
        primitive_label=label,
        primitive_order=order,
        images=images,
        caption="Extracted 1-D spectrum per echelle order (flux vs. detector column).",
    )


def _collect_stage2_items(
    output_dir: str,
    base: str,
    report_dir: str,
) -> list[PreviewItem]:
    """Collect the pipeline's default Stage 2 previews already written next to the L2 products."""
    items: dict[str, PreviewItem] = {}
    for name in sorted(os.listdir(output_dir)):
        if not name.startswith(base + "_spec"):
            continue
        if not name.lower().endswith((".png", ".html")):
            continue  # default previews only (skip .fits products)
        path = os.path.join(output_dir, name)
        token = None
        for tok in _STAGE2_TOKEN_ORDER:
            if f"_spec-{tok}" in name or f"-{tok}_" in name or f"-{tok}." in name:
                token = tok
                break
        if "stellar_rv" in name:
            token = "stellar_rv"
        if token is None:
            continue
        label = _STAGE2_TOKEN_ORDER.get(token, token)
        order = list(_STAGE2_TOKEN_ORDER).index(token) if token in _STAGE2_TOKEN_ORDER else 999
        item = items.setdefault(
            token, PreviewItem(primitive_label=label, primitive_order=order, caption="")
        )
        rel = os.path.relpath(path, report_dir)
        if name.lower().endswith(".png"):
            item.images.append(rel)
        elif name.lower().endswith(".html"):
            item.links.append((name.replace(base + "_", ""), rel))
    return [it for it in items.values() if it.images or it.links]


def _collect_fiberflat_items(output_dir: str, report_dir: str) -> list[PreviewItem]:
    """Collect trace-position previews from the fiberflat pipeline under ``output_dir/cals/``."""
    cals_dir = os.path.join(output_dir, "cals")
    if not os.path.isdir(cals_dir):
        return []

    previews: list[tuple[float, str]] = []
    for name in os.listdir(cals_dir):
        lower = name.lower()
        if "mastercal-tracepos" not in lower or not lower.endswith("_preview.png"):
            continue
        path = os.path.join(cals_dir, name)
        if os.path.isfile(path):
            previews.append((os.path.getmtime(path), path))

    if not previews:
        return []

    # Use the most recently written trace-pos preview (current E2E run).
    _, best_path = max(previews, key=lambda t: t[0])
    rel = os.path.relpath(best_path, report_dir)
    return [
        PreviewItem(
            primitive_label="Trace Positions",
            primitive_order=0,
            images=[rel],
            caption=(
                "Fitted echelle trace centroids on simulated fiberflat rate maps "
                "(ProcessFiberflatsPipeline / TracePositionPrimitive)."
            ),
        )
    ]


# --------------------------------------------------------------------------------------
# HTML report
# --------------------------------------------------------------------------------------

_HTML_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       margin: 0; padding: 1.5rem; line-height: 1.4; }
h1 { font-size: 1.5rem; margin: 0 0 0.25rem 0; }
.subtitle { color: #888; margin-bottom: 1.25rem; font-size: 0.9rem; }
details.stage { border: 1px solid #8884; border-radius: 8px; margin: 0 0 1rem 0; padding: 0.25rem 0.75rem; }
details.stage > summary { font-size: 1.15rem; font-weight: 600; cursor: pointer; padding: 0.5rem 0; }
details.prim { border-top: 1px dashed #8884; margin: 0.25rem 0; padding: 0.25rem 0.25rem 0.5rem 0.75rem; }
details.prim > summary { font-size: 1rem; font-weight: 500; cursor: pointer; padding: 0.4rem 0; }
.caption { color: #888; font-size: 0.85rem; margin: 0.1rem 0 0.5rem 0; }
.gallery { display: flex; flex-wrap: wrap; gap: 0.75rem; max-height: 80vh; overflow: auto;
           padding: 0.25rem; border: 1px solid #8882; border-radius: 6px; }
.gallery img { max-width: 100%; height: auto; border: 1px solid #8884; border-radius: 4px; }
.gallery figure { margin: 0; }
.links a { display: inline-block; margin: 0.2rem 0.6rem 0.2rem 0; }
.rv-summary { margin: 1.25rem 0 0.5rem 0; }
.rv-summary h2 { font-size: 1.15rem; margin: 0 0 0.5rem 0; }
.rv-summary table { border-collapse: collapse; width: 100%; max-width: 960px; font-size: 0.9rem; }
.rv-summary th, .rv-summary td { border: 1px solid #8884; padding: 0.45rem 0.55rem; text-align: right; }
.rv-summary th:first-child, .rv-summary td:first-child { text-align: center; }
.rv-summary th { background: #8882; font-weight: 600; }
.rv-summary caption { caption-side: top; text-align: left; color: #888; font-size: 0.85rem; margin-bottom: 0.35rem; }
"""


def _render_item_html(item: PreviewItem) -> str:
    parts = [f'<details class="prim" open><summary>{_esc(item.primitive_label)}</summary>']
    if item.caption:
        parts.append(f'<div class="caption">{_esc(item.caption)}</div>')
    if item.images:
        parts.append('<div class="gallery">')
        for img in item.images:
            parts.append(
                f'<figure><a href="{_esc(img)}" target="_blank">'
                f'<img src="{_esc(img)}" loading="lazy"></a></figure>'
            )
        parts.append("</div>")
    if item.links:
        parts.append('<div class="links">')
        for label, href in item.links:
            parts.append(f'<a href="{_esc(href)}" target="_blank">{_esc(label)} &#8599;</a>')
        parts.append("</div>")
    parts.append("</details>")
    return "\n".join(parts)


def _render_stage_html(stage_title: str, items: list[PreviewItem]) -> str:
    items = sorted(items, key=lambda it: (it.primitive_order, it.primitive_label))
    body = "\n".join(_render_item_html(it) for it in items)
    return (
        f'<details class="stage" open><summary>{_esc(stage_title)}</summary>\n{body}\n</details>'
    )


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt_num(value: float | None, *, decimals: int = 3, unit: str = "") -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{decimals}f}{unit}"


def _render_rv_performance_table_html(
    perf_rows: list,
    *,
    injected_rv_km_s: float,
    injected_berv_km_s: float | None = None,
) -> str:
    """HTML table of E2E RV recovery metrics (from :func:`collect_e2e_rv_performance_rows`)."""
    from . import e2e_support

    if not perf_rows:
        return ""

    header = (
        f"Injected stellar RV = {_fmt_num(injected_rv_km_s, decimals=3)} km/s "
        f"({_fmt_num(injected_rv_km_s * 1000.0, decimals=1)} m/s); "
        "excludes BERV (simulation Doppler uses topocentric ``Gaia RV - BERV``; "
        "Stage 2 CCF converts measured RV to barycentric via ``ccf_zb``)"
    )
    if injected_berv_km_s is not None and np.isfinite(injected_berv_km_s):
        header += (
            f". Reference BERV at ``mjd_mid`` = {_fmt_num(injected_berv_km_s, decimals=3)} km/s "
            f"({_fmt_num(injected_berv_km_s * 1000.0, decimals=1)} m/s)"
        )
    header += (
        ". Measured RV and σ(⟨RV⟩) from RVCCFAVG; BERV column from Stage 2 "
        "``BARYCORR_FIB*``; inlier scatter is RMS of "
        "inlier chunk RVs about ⟨RV⟩ (same metric as the Stage 2 preview sidebar)."
    )
    body_rows: list[str] = []
    for row in perf_rows:
        if not isinstance(row, e2e_support.E2ERVPerformanceRow):
            continue
        body_rows.append(
            "<tr>"
            f"<td>{row.fiber}</td>"
            f"<td>{_fmt_num(row.injected_rv_km_s, decimals=3)}</td>"
            f"<td>{_fmt_num(row.berv_km_s, decimals=3)}</td>"
            f"<td>{_fmt_num(row.measured_rv_km_s, decimals=3)}</td>"
            f"<td>{_fmt_num(row.bias_m_s, decimals=1)}</td>"
            f"<td>{_fmt_num(row.rv_err_m_s, decimals=3)}</td>"
            f"<td>{_fmt_num(row.inlier_scatter_m_s, decimals=3)}</td>"
            f"<td>{row.n_inlier_chunks} / {row.n_total_chunks}</td>"
            "</tr>"
        )
    if not body_rows:
        return ""

    return f"""
<section class="rv-summary">
<h2>RV performance summary</h2>
<table>
<caption>{_esc(header)}</caption>
<thead>
<tr>
<th>Fiber</th>
<th>Injected stellar RV (km/s)</th>
<th>BERV (km/s)</th>
<th>Measured RV (km/s)</th>
<th>Bias (m/s)</th>
<th>σ(⟨RV⟩) (m/s)</th>
<th>Inlier scatter (m/s)</th>
<th>Inlier chunks</th>
</tr>
</thead>
<tbody>
{os.linesep.join(body_rows)}
</tbody>
</table>
</section>
"""


def generate_e2e_preview_report(
    output_dir: str,
    *,
    exposure_base_name: str | None = None,
    gaia_target=None,
    stage2_model=None,
    injected_rv_km_s: float | None = None,
    fibers: list[int] | None = None,
) -> str:
    """
    Render PNG previews and write ``index.html`` summarizing the E2E run.

    Parameters
    ----------
    output_dir : str
        Directory containing the saved per-step FITS products (the E2E ``output_dir``).
    exposure_base_name : str or None
        Science exposure base name (e.g. ``HB.20260612.57502.95``). If ``None``, inferred
        from the products in ``output_dir``.
    gaia_target : optional
        :class:`~hispecdrp.simulations.e2e_gaia.E2EGaiaTarget` for a short header summary.
    stage2_model : optional
        Stage 2 L2 model for the RV performance table. When omitted, the table is skipped.
    injected_rv_km_s : float or None
        Template Doppler RV injected before rate-map generation. Defaults to
        ``gaia_target.injected_rv_km_s`` when available.
    fibers : list of int or None
        Fibers to include in the RV table (default ``[1, 2, 3]``).

    Returns
    -------
    str
        Absolute path to the written ``index.html``.
    """
    output_dir = os.path.abspath(output_dir)
    base = exposure_base_name or _infer_base_from_dir(output_dir)
    if not base:
        raise RuntimeError(f"Could not determine exposure base name in {output_dir!r}")

    previews_dir = os.path.join(output_dir, PREVIEWS_SUBDIR)
    os.makedirs(previews_dir, exist_ok=True)
    report_dir = output_dir
    report_path = os.path.join(output_dir, REPORT_HTML_NAME)

    stage0_items: list[PreviewItem] = []
    stage1_items: list[PreviewItem] = []

    for name in sorted(os.listdir(output_dir)):
        if not name.startswith(base + "_") or not name.endswith(".fits"):
            continue
        m = _PRODUCT_RE.match(name)
        if not m:
            continue
        group = m.group("group")
        token = m.group("token")
        path = os.path.join(output_dir, name)
        try:
            if group == "raw":
                item = _build_stage0_item(
                    path, token, kind="ramp_mean",
                    previews_dir=previews_dir, report_dir=report_dir,
                )
                if item:
                    stage0_items.append(item)
            elif group == "rate":
                item = _build_stage0_item(
                    path, token, kind="rate_map",
                    previews_dir=previews_dir, report_dir=report_dir,
                )
                if item:
                    stage0_items.append(item)
            elif group == "specuncal" and token == "specextract":
                item = _build_stage1_item(
                    path, token, previews_dir=previews_dir, report_dir=report_dir,
                )
                if item:
                    stage1_items.append(item)
        except Exception:
            logger.exception("Failed to build preview for %s", name)

    stage2_items = _collect_stage2_items(output_dir, base, report_dir)
    fiberflat_items = _collect_fiberflat_items(output_dir, report_dir)

    # Header summary
    summary_bits = [f"Exposure: <code>{_esc(base)}</code>"]
    if gaia_target is not None:
        meta = getattr(gaia_target, "target_meta", {}) or {}
        summary_bits.append(f"Target: <code>{_esc(meta.get('target_name', '?'))}</code>")
        summary_bits.append(f"Gaia DR3 <code>{_esc(getattr(gaia_target, 'source_id', '?'))}</code>")
        summary_bits.append(f"injected RV = {getattr(gaia_target, 'injected_rv_km_s', float('nan')):.3f} km/s")

    stages_html = []
    if stage0_items:
        stages_html.append(_render_stage_html("Stage 0 — detector reduction", stage0_items))
    if fiberflat_items:
        stages_html.append(
            _render_stage_html("Fiberflat calibration — trace positions", fiberflat_items)
        )
    if stage1_items:
        stages_html.append(_render_stage_html("Stage 1 — spectral extraction", stage1_items))
    if stage2_items:
        stages_html.append(_render_stage_html("Stage 2 — spectral calibration & RV", stage2_items))

    eff_injected_rv = injected_rv_km_s
    if eff_injected_rv is None and gaia_target is not None:
        eff_injected_rv = getattr(gaia_target, "injected_rv_km_s", None)
    eff_injected_berv = None
    if gaia_target is not None:
        eff_injected_berv = getattr(gaia_target, "berv_km_s", None)
    rv_table_html = ""
    if stage2_model is not None and eff_injected_rv is not None:
        from . import e2e_support

        perf_rows = e2e_support.collect_e2e_rv_performance_rows(
            stage2_model,
            injected_rv_km_s=float(eff_injected_rv),
            injected_berv_km_s=(
                float(eff_injected_berv)
                if eff_injected_berv is not None and np.isfinite(eff_injected_berv)
                else None
            ),
            fibers=fibers,
        )
        rv_table_html = _render_rv_performance_table_html(
            perf_rows,
            injected_rv_km_s=float(eff_injected_rv),
            injected_berv_km_s=(
                float(eff_injected_berv)
                if eff_injected_berv is not None and np.isfinite(eff_injected_berv)
                else None
            ),
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HISPEC E2E previews — {_esc(base)}</title>
<style>{_HTML_CSS}</style>
</head>
<body>
<h1>HISPEC end-to-end simulation previews</h1>
<div class="subtitle">{' &middot; '.join(summary_bits)}</div>
{os.linesep.join(stages_html)}
{rv_table_html}
</body>
</html>
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(
        "E2E previews: wrote %d stage-0, %d fiberflat, %d stage-1, %d stage-2 sections to %s",
        len(stage0_items), len(fiberflat_items), len(stage1_items), len(stage2_items), report_path,
    )
    return report_path

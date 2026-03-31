"""
Stellar RV tests: **ccf_stellar_template** (log-λ CCF, median/MAD rejection, IV weighted mean).

Synthetic L2 uses :func:`_order_wavelength_grid` and Gaussian lines; rest templates use
**3×** L2 pixels per order to reduce log-λ CCF bias.

Run::

  pytest tests/test_stellar_rv.py -v

**Optional previews (env or module variables):**

- ``HISPEC_DRP_SHOW_CCF_PREVIEW=1`` — default stellar RV Plotly preview (RV vs λ, CCF + Gaussian
  fits, residuals) in ``test_stellar_rv_recovers_rv_per_chunk`` (or ``SHOW_PRIMITIVE_CCF_PREVIEW``).
- ``HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS=1`` — after ``test_fit_ccf_kpf_style_btsettl_bspec_all_chunks`` /
  ``test_fit_ccf_kpf_style_btsettl_bspec_all_chunks_parallel`` assertions, run lte049 + template-derived
  ``lte049.0*.mas`` ``ccf_mask_kpf`` (serial chunk test previews still use ``K2_all.mas``): **HTML-only** previews (no ``fig.show()``):
  mask-overlay via ``write_mask_overlay_debug_pages`` (auto-open) and unified RV/CCF preview via
  ``create_stellar_rv_preview`` → ``fig.write_html`` then ``webbrowser.open`` on that file. Prints a step timing
  summary (``-s``).

**Parallel BT-Settl BSPEC chunk test** — ``test_fit_ccf_kpf_style_btsettl_bspec_all_chunks_parallel``
runs the same science setup via ``process_orders_rvs_numba_parallel`` (requests ``max_cores=6``;
actual Numba threads are clamped to the wheel/CPU limit, e.g. GitHub Actions) and prints a timing summary.

**One-liner (mask overlay + default CCF/RV preview, from repo root)::

  HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS=1 pytest tests/test_stellar_rv.py::test_fit_ccf_kpf_style_btsettl_bspec_all_chunks -s

``HISPEC_DRP_SAVE_CCF_PLOTS=1`` sets ``save_preview`` and ``output_dir`` for the echelle test so the
primitive writes a single default preview HTML under ``tests/output/stellar_rv_plots/`` (no extra figures).

**BT-Settl RV injection sweep (Plotly)** — ``test_btsettl_rv_injection_sweep_recovery`` uses the
**lte049** BT-Settl template with the template-derived ``lte049.0*.mas`` mask: per-order
``rv_per_chunks_mask_kpf`` + median/MAD + IV mean (mask-line CCF, same combination spirit as the
parallel BSPEC test). For an interactive browser
window, run the popup test with (BSPEC order 132 only; broad vs merged-hybrid sweep; bias/scatter
annotation; second HTML is two-panel CCF + Gaussian fits). Both HTML files are opened in the browser)::

  HISPEC_DRP_BTSETTL_RV_SWEEP_PLOT=1 pytest tests/test_stellar_rv.py::test_btsettl_rv_injection_sweep_plotly_popup -s
"""
from __future__ import annotations

import glob
import os
import time
import webbrowser
from unittest.mock import patch

import numpy as np
import pytest

import hispecdrp
from hispecdrp import datamodels
from hispecdrp.datamodels import hispec_config
from hispecdrp.datamodels.extensions import RVCCF_HDU
from hispecdrp.primitives import StellarRVPrimitive
from hispecdrp.utils.parallelization_utils import resolve_numba_threads
from hispecdrp.primitives.utils.ccf_utils import (
    MaskLines,
    _V_MAX,
    _V_MIN,
    _gaussian_ccf_model,
    build_hybrid_merged_velocity_grid,
    combine_rvs_median_mad,
    effective_ccf_v_step_for_fit,
    dv_per_pixel_m_s,
    fit_ccf_rv,
    load_mask,
    process_orders_rvs_numba_parallel,
    resolve_ccf_broad_grid_m_s,
    rv_per_chunks_mask_kpf,
    rv_per_chunks,
    weighted_mean_rv,
)
from hispecdrp.utils.spectral_templates_utils import get_spectrum_path, load_btsettl_spectrum

TEMPLATE_DIR = os.path.join(os.path.dirname(hispecdrp.__file__), "data", "templates")

C_KM_S = 299792.458
LAMBDA_ORDER_COEF_UM = 147.0
INPUT_RV_KM_S = -110.0
SIGMA_V_KM_S = 5.0
ABSORPTION_DEPTH = 0.05
FLAT_FLUX = 100.0
_TEMPLATE_PIXELS_PER_ORDER_FACTOR = 3
WINDOW_UM = 0.3
_BTSETTL_WL_RANGE_UM = (0.95, 1.4)
# Popup test: single-order sweep + CCF figure (BSPEC echelle order must be in HISPECConfig list).
_BTSETTL_RV_SWEEP_PLOT_ECHELLE_ORDER = 132
_BTSETTL_RV_SWEEP_PLOT_CCF_DEMO_KM_S = -115.0
# lte049 + polynomial L2 (same construction as mask-KPF smoke scripts)
_BTSETTL_LTE049_POLY_COEFS = np.array([0.0001, -0.001, 0.01, 0.05, 1.0])
MASK_DIR = os.path.join(os.path.dirname(hispecdrp.__file__), "data", "masks")


def _lte049_btsettl_template_mask_path() -> str | None:
    """Bundled template-derived mask ``lte049.0*.mas`` under ``data/masks`` (if present)."""
    matches = sorted(glob.glob(os.path.join(MASK_DIR, "lte049.0*.mas")))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Optional preview toggles: set to True/False here to override environment, or
# leave as None to use only HISPEC_DRP_* env vars documented in the module doc.
# ---------------------------------------------------------------------------
SHOW_PRIMITIVE_CCF_PREVIEW: bool | None = None
SHOW_BSPEC_BTSETTL_PREVIEWS: bool | None = None


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _want_primitive_ccf_preview() -> bool:
    if SHOW_PRIMITIVE_CCF_PREVIEW is not None:
        return SHOW_PRIMITIVE_CCF_PREVIEW
    return _truthy_env("HISPEC_DRP_SHOW_CCF_PREVIEW")


def _want_bspec_btsettl_all_previews() -> bool:
    """Master switch: lte049/K2 ``ccf_mask_kpf`` — mask-overlay HTML + default CCF/RV preview only."""
    if SHOW_BSPEC_BTSETTL_PREVIEWS is not None:
        return SHOW_BSPEC_BTSETTL_PREVIEWS
    return _truthy_env("HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS")


def _print_btsettl_test_timing_summary(
    rows: list[tuple[str, float, str, bool]],
    *,
    title: str = "test_fit_ccf_kpf_style_btsettl_bspec_all_chunks — wall time by step (seconds)",
) -> None:
    """
    Pretty-print wall times (seconds) and the main functions involved per step.

    Each row is ``(label, seconds, functions, include_in_total)``. Rows with
    ``include_in_total=False`` are a breakdown inside a larger step (not added to SUM).
    """
    width = 72
    print("\n" + "=" * width, flush=True)
    print(title, flush=True)
    print("=" * width, flush=True)
    total = 0.0
    for label, dt, funcs, inc in rows:
        if inc:
            total += dt
        mark = "*" if inc else " "
        print(f"{mark} {dt:10.3f}  {label}", flush=True)
        print(f"            └ {funcs}", flush=True)
    print("-" * width, flush=True)
    print(f"* {total:10.3f}  SUM (* = included in total; subsets of apply are not added)", flush=True)
    print("=" * width + "\n", flush=True)


def _bspec_btsettl_ge20_mask_flat(
    wave_2d: np.ndarray,
    order_tasks: list[tuple[int, int]],
    window_um: float,
) -> np.ndarray:
    """
    Flattened mask: True where a wavelength chunk has >=20 pixels (same binning as
    ``rv_per_chunks`` / :func:`process_orders_rvs_numba_parallel`).
    """
    bits: list[bool] = []
    for oi, _m in order_tasks:
        w = wave_2d[oi]
        w_min = float(np.nanmin(w))
        w_max = float(np.nanmax(w))
        n_chunk = max(1, int((w_max - w_min) / window_um))
        span = w_max - w_min
        for k in range(n_chunk):
            lam_lo = w_min + k * span / n_chunk
            lam_hi = w_min + (k + 1) * span / n_chunk
            in_chunk = (w >= lam_lo) & (w < lam_hi)
            bits.append(bool(np.sum(in_chunk) >= 20))
    return np.asarray(bits, dtype=bool)


def _order_wavelength_grid(order: int, n_pix: int) -> tuple[np.ndarray, float]:
    lam_center = LAMBDA_ORDER_COEF_UM / order
    delta_lam = lam_center / order
    wave = np.linspace(
        lam_center - delta_lam / 2.0,
        lam_center + delta_lam / 2.0,
        n_pix,
        dtype=np.float64,
    )
    return wave, lam_center


def _sigma_lambda_um(lam_center_um: float, sigma_v_km_s: float) -> float:
    return lam_center_um * (sigma_v_km_s / C_KM_S)


def _flux_with_gaussian_absorption(
    lam_um: np.ndarray,
    lam_line_um: float,
    sigma_lam_um: float,
    depth: float,
    flat_flux: float,
) -> np.ndarray:
    if sigma_lam_um <= 0:
        return np.full_like(lam_um, flat_flux)
    g = np.exp(-0.5 * ((lam_um - lam_line_um) / sigma_lam_um) ** 2)
    return flat_flux * (1.0 - depth * g)


def make_synthetic_l2_spectrum(
    spectrograph: str,
    n_pix: int = 4096,
    rv_km_s: float = INPUT_RV_KM_S,
    sigma_v_km_s: float = SIGMA_V_KM_S,
    depth: float = ABSORPTION_DEPTH,
    flat_flux: float = FLAT_FLUX,
) -> datamodels.L2Model:
    if spectrograph not in ("BSPEC", "RSPEC"):
        raise ValueError("spectrograph must be BSPEC or RSPEC")
    orders = hispec_config.HISPECConfig.get_order_list(spectrograph)
    if not orders:
        raise ValueError(f"No orders for {spectrograph}")
    order_start, order_end = int(min(orders)), int(max(orders))
    model = datamodels.L2Model(
        meta={
            "order_start": order_start,
            "order_end": order_end,
            "spectrograph": spectrograph,
            "instrument_name": "HISPEC",
        }
    )
    model.meta.order_start = order_start
    model.meta.order_end = order_end

    n_orders = len(orders)
    wave = np.zeros((n_orders, n_pix), dtype=np.float64)
    spec = np.zeros((n_orders, n_pix), dtype=np.float32)
    for oi, m in enumerate(orders):
        w, lam_center = _order_wavelength_grid(m, n_pix)
        wave[oi] = w
        lam_obs = lam_center * (1.0 + rv_km_s / C_KM_S)
        spec[oi] = _flux_with_gaussian_absorption(
            w, lam_obs, _sigma_lambda_um(lam_center, sigma_v_km_s), depth, flat_flux
        )
    model.spec_fib1 = spec
    model.spec_err_fib1 = np.maximum(spec * 0.01, 0.1).astype(np.float32)
    model.wave_fib1 = wave
    model.spec_dq_fib1 = np.zeros((n_orders, n_pix), dtype=np.uint32)
    return model


def make_rest_frame_template(
    spectrograph: str,
    n_pix: int = 4096,
    depth: float = ABSORPTION_DEPTH,
    flat_flux: float = FLAT_FLUX,
    sigma_v_km_s: float = SIGMA_V_KM_S,
    *,
    template_pixels_factor: int = _TEMPLATE_PIXELS_PER_ORDER_FACTOR,
) -> tuple[np.ndarray, np.ndarray]:
    orders = hispec_config.HISPECConfig.get_order_list(spectrograph)
    if not orders:
        raise ValueError(f"No orders for {spectrograph}")
    n_t = max(int(template_pixels_factor) * int(n_pix), 2)
    wl_list, flux_list = [], []
    for m in orders:
        lam_center = LAMBDA_ORDER_COEF_UM / float(m)
        delta_lam = lam_center / float(m)
        w = np.linspace(
            lam_center - delta_lam / 2.0,
            lam_center + delta_lam / 2.0,
            n_t,
            dtype=np.float64,
        )
        sigma_lam = _sigma_lambda_um(lam_center, sigma_v_km_s)
        wl_list.append(w)
        flux_list.append(
            _flux_with_gaussian_absorption(w, lam_center, sigma_lam, depth, flat_flux)
        )
    template_wl = np.concatenate(wl_list)
    template_flux = np.concatenate(flux_list)
    sort_idx = np.argsort(template_wl)
    return template_wl[sort_idx], template_flux[sort_idx]


def _build_bspec_l2_btsettl_doppler_poly(
    w_rest: np.ndarray,
    flux_rest: np.ndarray,
    rv_km_s: float,
    n_pix: int,
    poly: np.poly1d,
) -> datamodels.L2Model:
    """BSPEC echelle L2: BT-Settl at rest λ = λ_obs / (1 + v/c), times polynomial in observed µm."""
    orders = hispec_config.HISPECConfig.get_order_list("BSPEC")
    order_start, order_end = int(min(orders)), int(max(orders))
    n_orders = len(orders)
    model = datamodels.L2Model(
        meta={
            "order_start": order_start,
            "order_end": order_end,
            "spectrograph": "BSPEC",
            "instrument_name": "HISPEC",
        }
    )
    model.meta.order_start = order_start
    model.meta.order_end = order_end
    wave = np.zeros((n_orders, n_pix), dtype=np.float64)
    spec = np.zeros((n_orders, n_pix), dtype=np.float32)
    beta = 1.0 + rv_km_s / C_KM_S
    for oi, m in enumerate(orders):
        wave_ord, _ = _order_wavelength_grid(int(m), n_pix)
        wave[oi] = wave_ord
        fl = np.interp(wave_ord / beta, w_rest, flux_rest).astype(np.float32)
        spec[oi] = (fl * poly(wave_ord)).astype(np.float32)
    model.spec_fib1 = spec
    model.spec_err_fib1 = np.maximum(spec * 0.01, 0.1).astype(np.float32)
    model.wave_fib1 = wave
    model.spec_dq_fib1 = np.zeros((n_orders, n_pix), dtype=np.uint32)
    return model


def _assert_rvccf_table_matches_hdu_schema(tbl, *, label: str = "rvccf_fib1") -> None:
    """Require ``tbl`` columns and dtypes to match :data:`RVCCF_HDU` (``extensions.py``)."""
    expected = RVCCF_HDU["columns"]
    assert tbl is not None, f"{label}: table is None"
    exp_names = list(expected.keys())
    assert set(tbl.colnames) == set(exp_names), (
        f"{label}: column name mismatch\n  schema: {exp_names}\n  table:  {list(tbl.colnames)}"
    )
    for name in exp_names:
        exp = expected[name]
        got_dt = tbl[name].dtype
        exp_dt = np.dtype(exp)
        if exp_dt.kind == "O":
            assert got_dt.kind == "O", (
                f"{label}: column {name!r} expected object dtype, got {got_dt}"
            )
        else:
            assert got_dt == exp_dt or np.can_cast(got_dt, exp_dt, casting="equiv"), (
                f"{label}: column {name!r} dtype {got_dt} incompatible with schema {exp_dt}"
            )


def _print_rvccf_hdu_validation_report(tbl, *, label: str = "rvccf_fib1") -> None:
    """Human-readable column summary for ``pytest -s`` / logs."""
    lines = [f"RVCCF_HDU validation — {label}: n_rows={len(tbl)}"]
    for name in tbl.colnames:
        col = tbl[name]
        line = f"  {name}: dtype={col.dtype}"
        if len(tbl) > 0 and name == "CCF":
            c0 = col[0]
            if isinstance(c0, np.ndarray):
                line += f"  |  row0 len={c0.size} elem_dtype={c0.dtype}"
        lines.append(line)
    print("\n".join(lines), flush=True)


def _run_lte049_k2all_mask_kpf_previews(
    tmp_path: str,
    *,
    mask_path: str | None = None,
) -> list[tuple[str, float, str, bool]]:
    """
    BSPEC L2 from lte049 + Doppler + polynomial; ``ccf_mask_kpf`` with a packaged ``.mas`` mask.

    ``mask_path`` defaults to ``K2_all.mas`` (serial BT-Settl preview). Pass the template-derived
    ``lte049.0*.mas`` path for the parallel test.

    HTML-only: ``save_preview=True``, ``show_preview=False`` so ``create_stellar_rv_preview`` writes
    ``*_preview.html`` without ``fig.show()``; the test then ``webbrowser.open``s that file. Mask overlay uses
    ``write_mask_overlay_debug_pages`` (HTML + auto-open when ``show_mask_overlay_debug=True``).

    Returns timing rows ``(label, seconds, functions, include_in_total)`` for merge into the test summary.
    """
    import hispecdrp.primitives.stellar_rv.stellar_rv_preview as srv_prev

    rows: list[tuple[str, float, str, bool]] = []

    lte049 = get_spectrum_path(
        "lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5",
        base_dir=TEMPLATE_DIR,
    )
    if not os.path.isfile(lte049):
        pytest.skip(f"lte049 template not found: {lte049}")
    if mask_path is None:
        mask_path = os.path.join(MASK_DIR, "K2_all.mas")
    if not os.path.isfile(mask_path):
        pytest.skip(f"mask not found: {mask_path}")

    wl_min, wl_max = _BTSETTL_WL_RANGE_UM
    t0 = time.perf_counter()
    w_rest, flux_rest = _load_btsettl_sorted_micron(lte049, wl_min, wl_max)
    rows.append(
        (
            "lte049: load BT-Settl spectrum (µm)",
            time.perf_counter() - t0,
            "get_spectrum_path, load_btsettl_spectrum, _load_btsettl_sorted_micron",
            True,
        )
    )
    assert len(w_rest) > 50
    poly = np.poly1d(_BTSETTL_LTE049_POLY_COEFS)
    n_pix = 4096
    t0 = time.perf_counter()
    model = _build_bspec_l2_btsettl_doppler_poly(
        w_rest, flux_rest, INPUT_RV_KM_S, n_pix, poly
    )
    rows.append(
        (
            "lte049: build BSPEC L2Model (interp + polynomial)",
            time.perf_counter() - t0,
            "_build_bspec_l2_btsettl_doppler_poly, hispec_config.HISPECConfig.get_order_list, np.interp",
            True,
        )
    )

    out_dir = os.path.join(str(tmp_path), "lte049_mask_kpf_previews")
    os.makedirs(out_dir, exist_ok=True)

    t_mask_html = 0.0
    t_unified_html = 0.0
    orig_mask = srv_prev.write_mask_overlay_debug_pages
    orig_create = srv_prev.create_stellar_rv_preview

    def _timed_mask(*args, **kwargs):
        nonlocal t_mask_html
        t1 = time.perf_counter()
        try:
            return orig_mask(*args, **kwargs)
        finally:
            t_mask_html += time.perf_counter() - t1

    def _timed_create(*args, **kwargs):
        nonlocal t_unified_html
        t1 = time.perf_counter()
        try:
            return orig_create(*args, **kwargs)
        finally:
            t_unified_html += time.perf_counter() - t1

    prev_env = os.environ.get("HISPEC_DRP_SHOW_CCF_PREVIEW")
    try:
        os.environ["HISPEC_DRP_SHOW_CCF_PREVIEW"] = "1"
        prim = StellarRVPrimitive(
            algorithm="ccf_mask_kpf",
            mask_path=mask_path,
            do_fibers=["1"],
            window_size_um=WINDOW_UM,
            show_preview=False,
            save_preview=True,
            save_mask_overlay_debug=True,
            show_mask_overlay_debug=True,
            output_dir=out_dir,
            suffix="lte049_mask_kpf",
            max_cores=6,
        )
        t0 = time.perf_counter()
        with patch.object(srv_prev, "write_mask_overlay_debug_pages", _timed_mask):
            with patch.object(srv_prev, "create_stellar_rv_preview", _timed_create):
                out = prim.apply(model)
        apply_wall = time.perf_counter() - t0
    finally:
        if prev_env is None:
            os.environ.pop("HISPEC_DRP_SHOW_CCF_PREVIEW", None)
        else:
            os.environ["HISPEC_DRP_SHOW_CCF_PREVIEW"] = prev_env

    assert out is not None

    rv_remainder = max(apply_wall - t_mask_html - t_unified_html, 0.0)
    rows.append(
        (
            "StellarRVPrimitive.apply — total wall",
            apply_wall,
            "StellarRVPrimitive.apply → _perform, save_output, _create_preview; includes mask+unified HTML",
            True,
        )
    )
    rows.append(
        (
            "  └ subset: mask-overlay HTML + webbrowser.open",
            t_mask_html,
            "write_mask_overlay_debug_pages, _build_mask_overlay_debug_html",
            False,
        )
    )
    rows.append(
        (
            "  └ subset: unified stellar RV preview HTML (Plotly write_html only)",
            t_unified_html,
            "create_stellar_rv_preview, fig.write_html",
            False,
        )
    )
    rows.append(
        (
            "  └ subset: RV computation + tables + save + overhead (approx.)",
            rv_remainder,
            "run_ccf_mask_kpf_for_fiber, collect_mask_overlay_payload_for_primitive, save_output, etc.",
            False,
        )
    )

    prev = (prim._output or {}).get("preview") or {}
    html_ccf = prev.get("html")
    if html_ccf and os.path.isfile(html_ccf):
        ap = os.path.abspath(html_ccf)
        print(f"LTE049_K2ALL_CCF_PREVIEW_HTML={ap}", flush=True)
        t0 = time.perf_counter()
        webbrowser.open(f"file://{ap}")
        rows.append(
            (
                "open unified preview HTML in default browser (post-apply)",
                time.perf_counter() - t0,
                "webbrowser.open (file://…*_preview.html)",
                True,
            )
        )

    mod = prim._output or {}
    for _fib, p in (mod.get("mask_overlay_debug") or {}).items():
        if p and os.path.isfile(p):
            print(f"LTE049_K2ALL_MASK_OVERLAY_HTML={os.path.abspath(p)}", flush=True)

    return rows


def _load_btsettl_sorted_micron(path: str, wl_min_um: float, wl_max_um: float) -> tuple[np.ndarray, np.ndarray]:
    w_rest, flux_rest = load_btsettl_spectrum(path, wl_min_um=wl_min_um, wl_max_um=wl_max_um)
    w = np.asarray(w_rest.to_value("micron"), dtype=np.float64)
    f = np.asarray(flux_rest.value, dtype=np.float64)
    ok = np.isfinite(w) & np.isfinite(f) & (w > 0)
    w, f = w[ok], f[ok]
    idx = np.argsort(w)
    return w[idx], f[idx]


@pytest.fixture(scope="module")
def template_path():
    path = get_spectrum_path(
        "lte025.0-5.5-0.0a+0.0.BT-Settl.spec.7.h5",
        base_dir=TEMPLATE_DIR,
    )
    if not os.path.isfile(path):
        pytest.skip(f"Template not found: {path}")
    return path


@pytest.fixture(scope="module")
def btsettl_rv_sweep_template_path():
    """lte049 template paired with ``lte049.0*.mas`` for mask-line RV injection sweeps."""
    path = get_spectrum_path(
        "lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5",
        base_dir=TEMPLATE_DIR,
    )
    if not os.path.isfile(path):
        pytest.skip(f"Template not found: {path}")
    return path


@pytest.mark.parametrize("spectrograph", ["BSPEC", "RSPEC"])
def test_stellar_rv_recovers_rv_per_chunk(spectrograph):
    """Full echelle synthetic L2 + template; combined RV matches inject; chunk sanity."""
    algorithm = "ccf_stellar_template"
    n_pix = 4096
    input_rv_m_s = INPUT_RV_KM_S * 1000.0
    l2 = make_synthetic_l2_spectrum(
        spectrograph, n_pix=n_pix, rv_km_s=INPUT_RV_KM_S,
        sigma_v_km_s=SIGMA_V_KM_S, depth=ABSORPTION_DEPTH, flat_flux=FLAT_FLUX,
    )
    template_wl, template_flux = make_rest_frame_template(
        spectrograph, n_pix=n_pix, depth=ABSORPTION_DEPTH, flat_flux=FLAT_FLUX, sigma_v_km_s=SIGMA_V_KM_S,
    )
    show = _want_primitive_ccf_preview()
    save_plots = _truthy_env("HISPEC_DRP_SAVE_CCF_PLOTS")
    plot_out_dir = None
    if save_plots:
        plot_out_dir = os.path.join(os.path.dirname(__file__), "output", "stellar_rv_plots")
        os.makedirs(plot_out_dir, exist_ok=True)
    prim = StellarRVPrimitive(
        template_wavelength=template_wl,
        template_flux=template_flux,
        do_fibers=["1"],
        window_size_um=WINDOW_UM,
        debug_ccf=False,
        algorithm=algorithm,
        show_preview=show,
        save_preview=save_plots or show,
        output_dir=plot_out_dir,
    )
    out = prim.apply(l2)
    assert out is not None

    tbl = getattr(out, "rvccf_fib1", None)
    assert tbl is not None and "RV" in tbl.colnames and "RV_ERR" in tbl.colnames
    rvs = np.asarray(tbl["RV"], dtype=np.float64)
    rv_errs = np.asarray(tbl["RV_ERR"], dtype=np.float64)
    valid = np.isfinite(rvs) & np.isfinite(rv_errs) & (rv_errs > 0)
    assert np.any(valid), f"No valid chunk RVs for {spectrograph}"

    avg_tbl = getattr(out, "rvccfavg_fib1", None)
    assert avg_tbl is not None and len(avg_tbl) > 0
    rv_agg = float(avg_tbl["RV"][0])
    rv_agg_err = float(avg_tbl["RV_ERR"][0])
    tol = max(3.0 * rv_agg_err, 600.0)
    assert abs(rv_agg - input_rv_m_s) <= tol

    if "INLIER" in tbl.colnames:
        inlier = np.asarray(tbl["INLIER"], dtype=bool)[valid]
        if np.any(inlier):
            res = np.abs(rvs[valid][inlier] - input_rv_m_s)
            loose = np.maximum(3.0 * rv_errs[valid][inlier], 2000.0)
            assert np.sum(res <= loose) >= np.sum(inlier) * 0.5

    assert np.all(np.isfinite(rvs[valid]))
    assert np.all((rv_errs[valid] > 0) & (rv_errs[valid] < 1e6))

    if _truthy_env("HISPEC_DRP_STELLAR_RV_REPORT"):
        n_in = int(np.sum(tbl["INLIER"][valid])) if "INLIER" in tbl.colnames else int(np.sum(valid))
        print(
            f"STELLAR_RV_REPORT|{spectrograph}|{algorithm}|PASS|"
            f"rv_agg_km_s={rv_agg/1000:.4f}|rv_agg_err_m_s={rv_agg_err:.2f}|"
            f"bias_m_s={rv_agg - input_rv_m_s:.2f}|n_valid={np.sum(valid)}|n_inlier={n_in}"
        )


def test_fit_ccf_kpf_style_btsettl_bspec_all_chunks(template_path, tmp_path):
    """
    For every BSPEC echelle order, run ``rv_per_chunks`` on Doppler-shifted BT-Settl data
    (lte025 fixture); for **each** wavelength chunk with a usable CCF, run KPF-style
    ``fit_ccf_rv``. Require most fits to recover the injected RV within a loose tolerance.

    Optional: ``HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS=1`` (or ``SHOW_BSPEC_BTSETTL_PREVIEWS``)
    runs lte049 + polynomial BSPEC L2 with ``ccf_mask_kpf`` + ``K2_all.mas`` — mask-overlay debug HTML
    and the default unified RV/CCF/residual preview (HTML only; browsers open when files are written).

    At the end, prints a wall-time summary (see ``_print_btsettl_test_timing_summary``).
    """
    timing_rows: list[tuple[str, float, str, bool]] = []
    input_rv_m_s = INPUT_RV_KM_S * 1000.0
    v_step = 50.0
    n_pix = 4096
    wl_min, wl_max = _BTSETTL_WL_RANGE_UM
    t0 = time.perf_counter()
    w_rest, flux_rest = _load_btsettl_sorted_micron(template_path, wl_min, wl_max)
    assert len(w_rest) > 50
    template_ln = np.log(w_rest)
    orders = hispec_config.HISPECConfig.get_order_list("BSPEC")
    assert orders
    timing_rows.append(
        (
            "lte025: load BT-Settl + ln(template) + BSPEC order list",
            time.perf_counter() - t0,
            "_load_btsettl_sorted_micron, load_btsettl_spectrum, HISPECConfig.get_order_list",
            True,
        )
    )

    n_fit_ok = 0
    n_in_tol = 0
    n_attempt = 0

    t0 = time.perf_counter()
    for order in orders:
        m = int(order)
        wave_ord, lam_c = _order_wavelength_grid(m, n_pix)
        spec = np.interp(
            wave_ord / (1.0 + INPUT_RV_KM_S / C_KM_S), w_rest, flux_rest
        ).astype(np.float64)
        spec_err = np.maximum(spec * 0.01, 0.1)
        _rvs, _e, _centers, v_grid, ccf_2d = rv_per_chunks(
            wave_ord,
            spec,
            spec_err,
            np.zeros(n_pix, dtype=bool),
            template_ln,
            flux_rest,
            WINDOW_UM,
            v_min=_V_MIN,
            v_max=_V_MAX,
            v_step=v_step,
            return_ccf=True,
        )
        vspan = dv_per_pixel_m_s(float(lam_c))
        vspan_arg = vspan if np.isfinite(vspan) else None
        for k in range(ccf_2d.shape[0]):
            row = ccf_2d[k]
            if not np.any(np.isfinite(row)):
                continue
            ok = np.isfinite(row) & np.isfinite(v_grid)
            if np.sum(ok) < 20:
                continue
            n_attempt += 1
            rv, rv_err, _params, diag = fit_ccf_rv(
                v_grid[ok],
                row[ok],
                v_step,
                return_params=True,
                vel_span_m_s_per_pixel=vspan_arg,
            )
            fit_ok = bool(diag.get("fit_ok"))
            tol = max(8000.0, 3.0 * rv_err) if np.isfinite(rv_err) else np.inf
            in_tol = fit_ok and np.isfinite(rv) and np.isfinite(rv_err) and rv_err > 0 and abs(rv - input_rv_m_s) < tol
            if fit_ok:
                n_fit_ok += 1
            if in_tol:
                n_in_tol += 1

    timing_rows.append(
        (
            "lte025: BSPEC echelle chunk loop (rv_per_chunks + fit_ccf_rv)",
            time.perf_counter() - t0,
            "rv_per_chunks, fit_ccf_rv (all orders × wavelength chunks)",
            True,
        )
    )

    assert n_attempt > 10, "Expected many wavelength chunks across BSPEC orders"
    assert n_fit_ok >= 0.85 * n_attempt, f"fit_ok fraction too low: {n_fit_ok}/{n_attempt}"
    assert n_in_tol >= 0.50 * n_attempt, f"RV recovery fraction too low: {n_in_tol}/{n_attempt}"

    if _want_bspec_btsettl_all_previews():
        timing_rows.extend(_run_lte049_k2all_mask_kpf_previews(str(tmp_path)))

    _print_btsettl_test_timing_summary(timing_rows)


def test_fit_ccf_kpf_style_btsettl_bspec_all_chunks_parallel(template_path, tmp_path):
    """
    Same BT-Settl lte025 + BSPEC setup as ``test_fit_ccf_kpf_style_btsettl_bspec_all_chunks``,
    but one call to :func:`process_orders_rvs_numba_parallel` with ``max_cores=6`` (clamped
    to Numba’s allowed thread count on the host, e.g. 2 on small CI runners) and ``ccf_v_step=50`` m/s
    (same as the serial test and ``ccf_utils._V_STEP``). Phase-B Gaussian fits run serially in Python.

    ``ccf_mask_kpf`` RVCCF validation and the mask-KPF chunk benchmark use the template-derived
    ``lte049.0*.mas`` mask in ``data/masks`` (paired with the lte049 BT-Settl template), not the
    default ``M2_filt_allRV.mas``.

    Assertions use the same recovery fractions as the serial test, on chunks that are eligible
    (>=20 pixels in the bin and written by the parallel path, ``np.isfinite(all_wl)``).
    """
    timing_rows: list[tuple[str, float, str, bool]] = []
    mas_lte049 = _lte049_btsettl_template_mask_path()
    if mas_lte049 is None:
        pytest.skip(f"lte049.0*.mas mask not found under {MASK_DIR}")
    input_rv_m_s = INPUT_RV_KM_S * 1000.0
    v_step = 50.0
    n_pix = 4096
    wl_min, wl_max = _BTSETTL_WL_RANGE_UM
    t0 = time.perf_counter()
    w_rest, flux_rest = _load_btsettl_sorted_micron(template_path, wl_min, wl_max)
    assert len(w_rest) > 50
    template_ln = np.log(w_rest)
    orders = hispec_config.HISPECConfig.get_order_list("BSPEC")
    assert orders
    timing_rows.append(
        (
            "lte025: load BT-Settl + ln(template) + BSPEC order list",
            time.perf_counter() - t0,
            "_load_btsettl_sorted_micron, load_btsettl_spectrum, HISPECConfig.get_order_list",
            True,
        )
    )

    n_orders = len(orders)
    wave_2d = np.zeros((n_orders, n_pix), dtype=np.float64)
    spec_2d = np.zeros((n_orders, n_pix), dtype=np.float64)
    spec_err_2d = np.zeros((n_orders, n_pix), dtype=np.float64)
    dq_2d = np.zeros((n_orders, n_pix), dtype=np.uint32)
    order_tasks: list[tuple[int, int]] = []
    t0 = time.perf_counter()
    for oi, order in enumerate(orders):
        m = int(order)
        wave_ord, _lam_c = _order_wavelength_grid(m, n_pix)
        wave_2d[oi] = wave_ord
        spec_2d[oi] = np.interp(
            wave_ord / (1.0 + INPUT_RV_KM_S / C_KM_S),
            w_rest,
            flux_rest,
        ).astype(np.float64)
        spec_err_2d[oi] = np.maximum(spec_2d[oi] * 0.01, 0.1)
        order_tasks.append((oi, m))
    timing_rows.append(
        (
            "lte025: stack BSPEC 2-D model (interp Doppler-shifted BT-Settl per order)",
            time.perf_counter() - t0,
            "_order_wavelength_grid, np.interp (build wave_2d, spec_2d, spec_err_2d)",
            True,
        )
    )

    ge20_flat = _bspec_btsettl_ge20_mask_flat(wave_2d, order_tasks, WINDOW_UM)
    do_not_use = int(datamodels.DQ_FLAGS["DO_NOT_USE"])
    nthr = resolve_numba_threads(6)
    t0 = time.perf_counter()
    all_rvs, all_rv_errs, all_wl, _n_chunks_per_order, _chunk_start = (
        process_orders_rvs_numba_parallel(
            wave_2d,
            spec_2d,
            spec_err_2d,
            dq_2d,
            do_not_use,
            template_ln,
            flux_rest,
            WINDOW_UM,
            order_tasks,
            max_cores=6,
            return_ccf=False,
            ccf_v_step=50.0,
        )
    )
    parallel_wall = time.perf_counter() - t0
    timing_rows.append(
        (
            f"lte025: process_orders_rvs_numba_parallel (numba_threads={nthr}, ccf_v_step=50 m/s)",
            parallel_wall,
            "process_orders_rvs_numba_parallel, _process_all_orders_ccf_numba (prange), "
            "process_all_orders_rvs_numba (_ccf_rv_from_ccf / fit_ccf_rv, serial)",
            True,
        )
    )

    assert len(all_rvs) == len(ge20_flat), "flat chunk layout mismatch"
    eligible = ge20_flat & np.isfinite(all_wl)
    n_attempt = int(np.sum(eligible))
    assert n_attempt > 10, "Expected many wavelength chunks across BSPEC orders"

    sel_fit = eligible & np.isfinite(all_rvs) & np.isfinite(all_rv_errs) & (all_rv_errs > 0)
    n_fit_ok = int(np.sum(sel_fit))
    tol_vec = np.maximum(8000.0, 3.0 * all_rv_errs)
    in_tol = (
        sel_fit
        & (np.abs(all_rvs - input_rv_m_s) < tol_vec)
        & (all_rv_errs < 1e6)
    )
    n_in_tol = int(np.sum(in_tol))

    assert n_fit_ok >= 0.85 * n_attempt, f"fit fraction too low: {n_fit_ok}/{n_attempt}"
    assert n_in_tol >= 0.50 * n_attempt, f"RV recovery fraction too low: {n_in_tol}/{n_attempt}"

    # ------------------------------------------------------------------
    # RVCCF_HDU: validate output L2 ``rvccf_fib1`` vs datamodel schema (ccf_mask_kpf)
    # ------------------------------------------------------------------
    order_start, order_end = int(min(orders)), int(max(orders))
    rvccf_l2 = datamodels.L2Model(
        meta={
            "order_start": order_start,
            "order_end": order_end,
            "spectrograph": "BSPEC",
            "instrument_name": "HISPEC",
        }
    )
    rvccf_l2.meta.order_start = order_start
    rvccf_l2.meta.order_end = order_end
    rvccf_l2.spec_fib1 = spec_2d.astype(np.float32)
    rvccf_l2.spec_err_fib1 = spec_err_2d.astype(np.float32)
    rvccf_l2.wave_fib1 = wave_2d
    rvccf_l2.spec_dq_fib1 = dq_2d

    t0 = time.perf_counter()
    prim_rvccf = StellarRVPrimitive(
        algorithm="ccf_mask_kpf",
        mask_path=mas_lte049,
        do_fibers=["1"],
        window_size_um=WINDOW_UM,
        outlier_sigma=4.0,
        max_cores=6,
        save_preview=False,
        show_preview=False,
        output_dir=str(tmp_path),
        suffix="parallel_rvccf_hdu_validate",
    )
    out_rvccf = prim_rvccf.apply(rvccf_l2)
    timing_rows.append(
        (
            "RVCCF_HDU: StellarRVPrimitive ccf_mask_kpf → rvccf_fib1 schema check",
            time.perf_counter() - t0,
            "StellarRVPrimitive.apply, _write_rv_tables, assemble_rvccf_fib_table",
            True,
        )
    )
    assert out_rvccf is not None
    tbl_rvccf = getattr(out_rvccf, "rvccf_fib1", None)
    _assert_rvccf_table_matches_hdu_schema(tbl_rvccf, label="rvccf_fib1")
    _print_rvccf_hdu_validation_report(tbl_rvccf, label="rvccf_fib1")
    assert tbl_rvccf is not None and len(tbl_rvccf) > 0, "expected non-empty rvccf_fib1 from ccf_mask_kpf"
    for i in range(len(tbl_rvccf)):
        nv = int(tbl_rvccf["N_VEL"][i])
        ccf = tbl_rvccf["CCF"][i]
        if isinstance(ccf, np.ndarray) and ccf.size > 0:
            assert ccf.size == nv, f"rvccf_fib1 row {i}: N_VEL={nv} vs len(CCF)={ccf.size}"

    # ------------------------------------------------------------------
    # ccf_mask_kpf chunk-parallel benchmark (rv_per_chunks_mask_kpf path)
    # ------------------------------------------------------------------
    # With WINDOW_UM=0.3, many single-order calls have only 1 chunk, which can
    # hide parallel scaling. Use a smaller benchmark window to force many chunks.
    #
    # Each synthetic echelle row has span ~147/order^2 µm (see _order_wavelength_grid).
    # A fixed 0.01 µm window can be *wider* than the order span, so
    # ``n_chunk = max(1, int(span/window))`` is 1 and Numba ``prange`` over chunks
    # never parallelizes. Tie the window to a representative span so each order
    # contributes many chunks (``_calc_mask_ccf_chunks_numba`` prange).
    bench_orders = order_tasks[: min(12, len(order_tasks))]
    oi0 = bench_orders[0][0]
    span0 = float(np.ptp(wave_2d[oi0]))
    bench_window_um = min(0.01, WINDOW_UM / 20.0, max(1e-6, span0 / 16.0))
    mask_lines = load_mask(mas_lte049, mask_width_um=None)
    bench_times: dict[int, float] = {}
    bench_chunks: dict[int, int] = {}
    eff_threads: dict[int, int] = {}
    for req_cores in (1, 2, 6):
        eff_threads[req_cores] = resolve_numba_threads(req_cores)
        t0 = time.perf_counter()
        n_chunks_total = 0
        for oi, _order in bench_orders:
            d_bad = (np.asarray(dq_2d[oi], dtype=np.uint32) & do_not_use) != 0
            rvs_b, _errs_b, _centers_b, _v_grid_b, _ccf_b, _wlo, _whi, _ng = rv_per_chunks_mask_kpf(
                wave_2d[oi],
                spec_2d[oi],
                spec_err_2d[oi],
                d_bad,
                mask_lines,
                window_size_um=bench_window_um,
                v_min_m_s=_V_MIN,
                v_max_m_s=_V_MAX,
                v_step_m_s=v_step,
                max_cores=req_cores,
                return_ccf=False,
            )
            n_chunks_total += int(len(rvs_b))
        bench_times[req_cores] = time.perf_counter() - t0
        bench_chunks[req_cores] = n_chunks_total
        timing_rows.append(
            (
                f"lte025: ccf_mask_kpf chunk benchmark req_cores={req_cores} "
                f"(effective={eff_threads[req_cores]}, window={bench_window_um:.4f} um)",
                bench_times[req_cores],
                "rv_per_chunks_mask_kpf, _calc_mask_ccf_chunks_numba (prange over chunks), "
                "numba_thread_scope, rv_from_ccf_mask_kpf",
                True,
            )
        )

    # Verify measurable speed-up whenever effective thread count exceeds 1.
    t1 = bench_times[1]
    multi = [c for c in (2, 6) if eff_threads[c] > 1]
    if multi:
        tbest = min(bench_times[c] for c in multi)
        # Require at least modest speed-up relative to 1 core.
        assert tbest < 0.98 * t1, (
            "Expected ccf_mask_kpf chunk-parallel speed-up with >1 effective thread: "
            f"t1={t1:.3f}s, t2={bench_times[2]:.3f}s (eff={eff_threads[2]}), "
            f"t6={bench_times[6]:.3f}s (eff={eff_threads[6]}), "
            f"n_chunks={bench_chunks[1]}"
        )
    else:
        print(
            "ccf_mask_kpf speed-up check skipped: host allows only one effective Numba thread "
            f"(eff threads: 1->{eff_threads[1]}, 2->{eff_threads[2]}, 6->{eff_threads[6]}).",
            flush=True,
        )

    if _want_bspec_btsettl_all_previews():
        timing_rows.extend(
            _run_lte049_k2all_mask_kpf_previews(str(tmp_path), mask_path=mas_lte049)
        )

    _print_btsettl_test_timing_summary(
        timing_rows,
        title=(
            "test_fit_ccf_kpf_style_btsettl_bspec_all_chunks_parallel — "
            "wall time by step (seconds)"
        ),
    )


def _collect_btsettl_rv_injection_sweep(
    template_path: str,
    *,
    use_ccf_input_velocity_supersample: bool = False,
    single_echelle_order: int | None = None,
) -> dict:
    """
    For each injected RV in [-105, -125] km/s step -2, build BSPEC BT-Settl cubes and recover
    a single RV via :func:`rv_per_chunks_mask_kpf` on the **lte049** template-derived ``.mas``
    (see :func:`_lte049_btsettl_template_mask_path`) plus median/MAD outlier rejection and
    inverse-variance weighted mean (same combination as :class:`StellarRVPrimitive` for mask KPF).

    Parameters
    ----------
    use_ccf_input_velocity_supersample
        If False (default), use the broad CCF grid (``resolve_ccf_broad_grid_m_s`` at 50 m/s).
        If True, for each injection use a **merged** grid: full broad range plus denser samples on
        ``injected_RV ± 300`` m/s at 0.1 m/s (:func:`~hispecdrp.primitives.utils.ccf_utils.build_hybrid_merged_velocity_grid`).
    single_echelle_order
        If not ``None``, restrict processing to this BSPEC echelle order (one row in the
        synthetic 2-D echelle arrays).

    Returns
    -------
    dict
        ``injected_km_s``, ``recovered_m_s``, ``recovered_err_m_s`` (arrays), ``ols_slope``,
        ``ols_intercept_m_s``, ``rms_residual_m_s``, ``mean_offset_m_s`` (mean of recovered − injected),
        ``ccf_grid_mode`` (``\"broad\"`` or ``\"merged_hybrid_per_injection\"``), and
        ``single_echelle_order`` (echo of the argument).
    """
    mas_path = _lte049_btsettl_template_mask_path()
    if mas_path is None:
        raise FileNotFoundError(
            f"BT-Settl RV sweep requires lte049.0*.mas under {MASK_DIR} (see line_mask_utils)"
        )
    mask_lines = load_mask(mas_path, mask_width_um=None)

    wl_min, wl_max = _BTSETTL_WL_RANGE_UM
    w_rest, flux_rest = _load_btsettl_sorted_micron(template_path, wl_min, wl_max)
    assert len(w_rest) > 50
    orders = hispec_config.HISPECConfig.get_order_list("BSPEC")
    assert orders
    n_pix = 4096
    do_not_use = int(datamodels.DQ_FLAGS["DO_NOT_USE"])
    nthr = resolve_numba_threads(6)
    vb, vx, vs_broad = resolve_ccf_broad_grid_m_s()

    if single_echelle_order is not None:
        so = int(single_echelle_order)
        if so not in orders:
            raise AssertionError(
                f"single_echelle_order={so} not in BSPEC order list ({orders[0]} … {orders[-1]})"
            )
        order_nums = [so]
        n_orders = 1
    else:
        order_nums = [int(o) for o in orders]
        n_orders = len(order_nums)

    injected_km_s = np.arange(-105, -126, -2, dtype=np.float64)
    recovered_m_s: list[float] = []
    recovered_err_m_s: list[float] = []

    for rv_km_s in injected_km_s:
        wave_2d = np.zeros((n_orders, n_pix), dtype=np.float64)
        spec_2d = np.zeros((n_orders, n_pix), dtype=np.float64)
        spec_err_2d = np.zeros((n_orders, n_pix), dtype=np.float64)
        dq_2d = np.zeros((n_orders, n_pix), dtype=np.uint32)
        beta = 1.0 + float(rv_km_s) / C_KM_S
        for oi, m in enumerate(order_nums):
            wave_ord, _lam_c = _order_wavelength_grid(m, n_pix)
            wave_2d[oi] = wave_ord
            spec_2d[oi] = np.interp(wave_ord / beta, w_rest, flux_rest).astype(np.float64)
            spec_err_2d[oi] = np.maximum(spec_2d[oi] * 0.01, 0.1)

        if use_ccf_input_velocity_supersample:
            inj_m_s = float(rv_km_s) * 1000.0
            v_grid = build_hybrid_merged_velocity_grid(vb, vx, vs_broad, inj_m_s, 300.0, 0.1)
            vg_kw: dict = dict(
                v_grid_m_s=v_grid,
                v_min_m_s=vb,
                v_max_m_s=vx,
                v_step_m_s=vs_broad,
            )
        else:
            vg_kw = dict(v_min_m_s=vb, v_max_m_s=vx, v_step_m_s=50.0)

        chunk_rvs: list[np.ndarray] = []
        chunk_errs: list[np.ndarray] = []
        for oi in range(n_orders):
            d_bad = (np.asarray(dq_2d[oi], dtype=np.uint32) & do_not_use) != 0
            rvs_o, errs_o, _cen, _vg, _ccf, _wlo, _whi, _ng = rv_per_chunks_mask_kpf(
                wave_2d[oi],
                spec_2d[oi],
                spec_err_2d[oi],
                d_bad,
                mask_lines,
                window_size_um=WINDOW_UM,
                max_cores=nthr,
                return_ccf=False,
                **vg_kw,
            )
            chunk_rvs.append(np.asarray(rvs_o, dtype=np.float64).ravel())
            chunk_errs.append(np.asarray(errs_o, dtype=np.float64).ravel())

        ar = np.concatenate(chunk_rvs) if chunk_rvs else np.asarray([], dtype=np.float64)
        ae = np.concatenate(chunk_errs) if chunk_errs else np.asarray([], dtype=np.float64)
        inlier, _med, _mad = combine_rvs_median_mad(ar, ae, 4.0)
        if not np.any(inlier):
            rv_hat, rv_hat_err = np.nan, np.nan
        else:
            rv_hat, rv_hat_err = weighted_mean_rv(ar, ae, inlier)
        recovered_m_s.append(float(rv_hat))
        recovered_err_m_s.append(float(rv_hat_err))

    inj_m = injected_km_s * 1000.0
    rec = np.asarray(recovered_m_s, dtype=np.float64)
    okf = np.isfinite(rec) & np.isfinite(inj_m)
    if np.sum(okf) < 2:
        raise AssertionError("Need at least two finite recovered RVs for linear fit.")
    coef = np.polyfit(inj_m[okf], rec[okf], 1)
    slope = float(coef[0])
    intercept_m_s = float(coef[1])
    pred = slope * inj_m[okf] + intercept_m_s
    rms_residual_m_s = float(np.sqrt(np.mean((rec[okf] - pred) ** 2)))
    mean_offset_m_s = float(np.mean(rec[okf] - inj_m[okf]))

    return {
        "injected_km_s": injected_km_s,
        "recovered_m_s": rec,
        "recovered_err_m_s": np.asarray(recovered_err_m_s, dtype=np.float64),
        "ols_slope": slope,
        "ols_intercept_m_s": intercept_m_s,
        "rms_residual_m_s": rms_residual_m_s,
        "mean_offset_m_s": mean_offset_m_s,
        "ccf_grid_mode": (
            "merged_hybrid_per_injection"
            if use_ccf_input_velocity_supersample
            else "broad"
        ),
        "single_echelle_order": single_echelle_order,
    }


def _btsettl_single_order_ccf_snapshot(
    template_path: str,
    echelle_order: int,
    rv_km_s: float,
    *,
    use_broad_ccf_grid: bool,
) -> dict:
    """
    Mean mask-line CCF vs velocity for one BSPEC echelle order and one injected RV (km/s),
    using ``lte049.0*.mas`` and either the broad default search (50 m/s step) or the **merged**
    broad + refined grid (hybrid CCF matching :class:`StellarRVPrimitive` with
    ``ccf_input_velocity`` set).
    """
    mas_path = _lte049_btsettl_template_mask_path()
    if mas_path is None:
        raise FileNotFoundError(
            f"CCF snapshot requires lte049.0*.mas under {MASK_DIR} (see line_mask_utils)"
        )
    mask_lines = load_mask(mas_path, mask_width_um=None)

    wl_min, wl_max = _BTSETTL_WL_RANGE_UM
    w_rest, flux_rest = _load_btsettl_sorted_micron(template_path, wl_min, wl_max)
    orders = hispec_config.HISPECConfig.get_order_list("BSPEC")
    if echelle_order not in orders:
        raise AssertionError(
            f"echelle_order={echelle_order} not in BSPEC list ({orders[0]} … {orders[-1]})"
        )

    n_pix = 4096
    beta = 1.0 + float(rv_km_s) / C_KM_S
    wave_ord, _lam_c = _order_wavelength_grid(echelle_order, n_pix)
    spec_1d = np.interp(wave_ord / beta, w_rest, flux_rest).astype(np.float64)
    spec_err_1d = np.maximum(spec_1d * 0.01, 0.1)

    do_not_use = int(datamodels.DQ_FLAGS["DO_NOT_USE"])
    nthr = resolve_numba_threads(6)
    vb, vx, vs_broad = resolve_ccf_broad_grid_m_s()
    d_bad = np.zeros(n_pix, dtype=bool)

    if use_broad_ccf_grid:
        vg_kw = dict(v_min_m_s=vb, v_max_m_s=vx, v_step_m_s=50.0)
    else:
        inj_m_s = float(rv_km_s) * 1000.0
        v_grid_h = build_hybrid_merged_velocity_grid(vb, vx, vs_broad, inj_m_s, 300.0, 0.1)
        vg_kw = dict(
            v_grid_m_s=v_grid_h,
            v_min_m_s=vb,
            v_max_m_s=vx,
            v_step_m_s=vs_broad,
        )

    _rvs, _errs, _cen, v_grid, all_ccf, _wlo, _whi, all_n_good = rv_per_chunks_mask_kpf(
        wave_ord,
        spec_1d,
        spec_err_1d,
        d_bad,
        mask_lines,
        window_size_um=WINDOW_UM,
        max_cores=nthr,
        return_ccf=True,
        **vg_kw,
    )

    vg = np.asarray(v_grid, dtype=np.float64)
    ac = np.asarray(all_ccf, dtype=np.float64)
    ccf_mean = np.nanmean(ac, axis=0)
    v_step = (
        float(effective_ccf_v_step_for_fit(vg))
        if vg.size > 1
        else float(ccf_kw.get("ccf_v_step", 50.0))
    )
    wl_c = float(np.nanmean(wave_ord))
    vsp = dv_per_pixel_m_s(wl_c)
    n_good_fit = int(np.max(all_n_good)) if np.asarray(all_n_good).size > 0 else 0

    rv_f, rv_e, params, diag = fit_ccf_rv(
        vg,
        ccf_mean,
        v_step,
        n_good=n_good_fit,
        return_params=True,
        vel_span_m_s_per_pixel=vsp if np.isfinite(vsp) else None,
    )

    return {
        "echelle_order": int(echelle_order),
        "injected_km_s": float(rv_km_s),
        "v_grid_m_s": vg,
        "ccf_mean": ccf_mean,
        "v_step_m_s": v_step,
        "rv_fit_m_s": rv_f,
        "rv_err_fit_m_s": rv_e,
        "params": params,
        "diag": diag,
    }


def _btsettl_ccf_gaussian_legend_label(snap: dict, *, panel_tag: str) -> str:
    """Multiline legend text for the Gaussian model in a CCF panel."""
    p, d = snap["params"], snap["diag"]
    rv_f, rv_e = snap["rv_fit_m_s"], snap["rv_err_fit_m_s"]
    if not d.get("fit_ok"):
        return f"{panel_tag} Gaussian (fit failed): {d.get('message', '—')}"
    return (
        f"{panel_tag} Gaussian: v₀={p['v0']:.2f} σ={p['sigma']:.2f} m/s<br>"
        f"A={p['amplitude']:.4g} baseline={p['baseline']:.4g}<br>"
        f"RV={rv_f:.2f}±{rv_e:.2f} m/s"
    )


def _btsettl_add_ccf_subplot_traces(fig, row: int, snap: dict, *, panel_tag: str) -> None:
    import plotly.graph_objects as go

    vg = np.asarray(snap["v_grid_m_s"], dtype=np.float64)
    ccf = np.asarray(snap["ccf_mean"], dtype=np.float64)
    vk = vg / 1000.0
    fig.add_trace(
        go.Scatter(
            x=vk,
            y=ccf,
            mode="lines",
            name=f"{panel_tag}: CCF (chunk mean)",
            line=dict(width=1.2),
        ),
        row=row,
        col=1,
    )
    diag = snap["diag"]
    p = snap["params"] or {}
    if diag.get("fit_ok") and p and all(
        k in p for k in ("amplitude", "v0", "sigma", "baseline")
    ):
        v_lo, v_hi = float(vg[0]), float(vg[-1])
        v_dense = np.linspace(v_lo, v_hi, max(256, int(vg.size) * 4))
        y_m = _gaussian_ccf_model(
            v_dense,
            float(p["amplitude"]),
            float(p["v0"]),
            float(p["sigma"]),
            float(p["baseline"]),
        )
        fig.add_trace(
            go.Scatter(
                x=v_dense / 1000.0,
                y=y_m,
                mode="lines",
                name=_btsettl_ccf_gaussian_legend_label(snap, panel_tag=panel_tag),
                line=dict(width=2, dash="dash"),
            ),
            row=row,
            col=1,
        )
    else:
        mid = int(len(vk) // 2)
        fig.add_trace(
            go.Scatter(
                x=[float(vk[mid])],
                y=[float(np.nanmedian(ccf))],
                mode="markers",
                marker=dict(size=6, opacity=0),
                name=_btsettl_ccf_gaussian_legend_label(snap, panel_tag=panel_tag),
            ),
            row=row,
            col=1,
        )


def _build_btsettl_order_ccf_dual_panel_figure(snap_broad: dict, snap_refined: dict):
    """Two stacked panels: broad-only vs merged hybrid CCF (same order + injection)."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.11,
        subplot_titles=(
            "Broad grid (50 m/s step, full RV search)",
            "Merged broad + refined (hybrid: full search + ±300 m/s @ 0.1 m/s)",
        ),
    )
    eo = int(snap_broad["echelle_order"])
    rvk = float(snap_broad["injected_km_s"])
    assert eo == int(snap_refined["echelle_order"])
    assert rvk == float(snap_refined["injected_km_s"])

    _btsettl_add_ccf_subplot_traces(fig, 1, snap_broad, panel_tag="Broad")
    _btsettl_add_ccf_subplot_traces(fig, 2, snap_refined, panel_tag="Merged hybrid")

    fig.update_layout(
        title_text=(
            f"BT-Settl BSPEC order {eo}: CCF vs RV @ injected {rvk:.1f} km s⁻¹ "
            "(broad vs merged hybrid)"
        ),
        height=820,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.92)",
            font=dict(size=10),
        ),
        margin=dict(l=60, r=200, t=80, b=50),
    )
    fig.update_xaxes(title_text="Velocity shift (km s⁻¹)", row=1, col=1)
    fig.update_xaxes(title_text="Velocity shift (km s⁻¹)", row=2, col=1)
    fig.update_yaxes(title_text="CCF (mask)", row=1, col=1)
    fig.update_yaxes(title_text="CCF (mask)", row=2, col=1)
    return fig


def _build_btsettl_rv_injection_figure(sweep: dict):
    """Plotly injected vs recovered RV with OLS fit, y=x, and annotation (bias, scatter)."""
    import plotly.graph_objects as go

    x_km = sweep["injected_km_s"]
    y_km = sweep["recovered_m_s"] / 1000.0
    b = sweep["ols_slope"]
    a_m = sweep["ols_intercept_m_s"]
    rms = sweep["rms_residual_m_s"]
    dmean = sweep["mean_offset_m_s"]

    x_line = np.linspace(float(np.min(x_km)), float(np.max(x_km)), 50)
    y_fit_km = (b * (x_line * 1000.0) + a_m) / 1000.0

    ann = (
        f"OLS (m/s): recovered = {b:.6f} × injected + {a_m:.2f}<br>"
        f"Mean(rec − inj) = {dmean:.2f} m/s<br>"
        f"RMS about OLS = {rms:.2f} m/s"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_km,
            y=y_km,
            mode="markers",
            name="recovered (IV mean)",
            marker=dict(size=10),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_fit_km,
            mode="lines",
            name="OLS fit",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=x_line,
            mode="lines",
            name="y = x",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title="BT-Settl BSPEC: injected vs recovered RV (template CCF, parallel Numba)",
        xaxis_title="Injected RV (km s⁻¹)",
        yaxis_title="Recovered RV (km s⁻¹)",
        annotations=[
            dict(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
                text=ann,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="gray",
                borderwidth=1,
            )
        ],
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
        margin=dict(t=60, l=60, r=40, b=60),
    )
    return fig


def _build_btsettl_rv_injection_comparison_figure(sweep_broad: dict, sweep_refined: dict):
    """
    Plotly overlay: broad default CCF grid vs per-injection **merged hybrid** CCF (same as
    :class:`StellarRVPrimitive` with ``ccf_input_velocity`` set). Annotation compares bias
    (``mean_offset_m_s``) and scatter (``rms_residual_m_s``) for each method.

    Two stacked panels share the injected-RV x-axis: (1) recovered vs injected with OLS and
    y = x; (2) residuals in m/s of each point minus its OLS prediction (same linear fits as the
    bias / RMS statistics).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    x_km = sweep_broad["injected_km_s"]
    assert np.allclose(sweep_refined["injected_km_s"], x_km)

    yb_km = sweep_broad["recovered_m_s"] / 1000.0
    yr_km = sweep_refined["recovered_m_s"] / 1000.0

    bb, ba = sweep_broad["ols_slope"], sweep_broad["ols_intercept_m_s"]
    rb, ra = sweep_refined["ols_slope"], sweep_refined["ols_intercept_m_s"]
    rms_b, rms_r = sweep_broad["rms_residual_m_s"], sweep_refined["rms_residual_m_s"]
    off_b, off_r = sweep_broad["mean_offset_m_s"], sweep_refined["mean_offset_m_s"]

    inj_m = np.asarray(x_km, dtype=np.float64) * 1000.0
    pred_b_m = bb * inj_m + ba
    pred_r_m = rb * inj_m + ra
    res_b_m = np.asarray(sweep_broad["recovered_m_s"], dtype=np.float64) - pred_b_m
    res_r_m = np.asarray(sweep_refined["recovered_m_s"], dtype=np.float64) - pred_r_m

    x_line = np.linspace(float(np.min(x_km)), float(np.max(x_km)), 50)
    y_fit_b = (bb * (x_line * 1000.0) + ba) / 1000.0
    y_fit_r = (rb * (x_line * 1000.0) + ra) / 1000.0

    ann = (
        "<b>Broad grid</b> (default search, 50 m/s step):<br>"
        f"Mean(rec − inj) = {off_b:.2f} m/s &nbsp;|&nbsp; RMS about OLS = {rms_b:.2f} m/s<br><br>"
        "<b>Merged hybrid</b> (broad search + ±300 m/s @ 0.1 m/s around injection; "
        "same as StellarRVPrimitive with ccf_input_velocity):<br>"
        f"Mean(rec − inj) = {off_r:.2f} m/s &nbsp;|&nbsp; RMS about OLS = {rms_r:.2f} m/s"
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.62, 0.38],
    )

    fig.add_trace(
        go.Scatter(
            x=x_km,
            y=yb_km,
            mode="markers",
            name="Broad grid (50 m/s step)",
            marker=dict(size=9, color="#1f77b4"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_km,
            y=yr_km,
            mode="markers",
            name="Merged hybrid (per injection)",
            marker=dict(size=9, color="#ff7f0e", symbol="diamond"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_fit_b,
            mode="lines",
            name="OLS broad",
            line=dict(color="#1f77b4", dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_fit_r,
            mode="lines",
            name="OLS merged hybrid",
            line=dict(color="#ff7f0e", dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=x_line,
            mode="lines",
            name="y = x",
            line=dict(dash="dash", color="gray"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x_km,
            y=res_b_m,
            mode="markers",
            name="Residual broad (rec − OLS)",
            marker=dict(size=8, color="#1f77b4"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_km,
            y=res_r_m,
            mode="markers",
            name="Residual merged hybrid (rec − OLS)",
            marker=dict(size=8, color="#ff7f0e", symbol="diamond"),
        ),
        row=2,
        col=1,
    )

    so = sweep_broad.get("single_echelle_order")
    title_core = (
        "BT-Settl BSPEC: broad CCF grid vs merged hybrid (template CCF, parallel Numba)"
    )
    if so is not None:
        title_core += f" — order {int(so)} only"

    fig.update_layout(
        title=title_core,
        annotations=[
            dict(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
                text=ann,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1,
            )
        ],
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
        margin=dict(t=60, l=60, r=40, b=60),
        height=780,
    )
    fig.update_yaxes(title_text="Recovered RV (km s⁻¹)", row=1, col=1)
    fig.update_yaxes(title_text="Residual vs OLS (m s⁻¹)", row=2, col=1)
    fig.update_xaxes(title_text="Injected RV (km s⁻¹)", row=2, col=1)
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray", line_width=1, row=2, col=1)
    return fig


def test_btsettl_rv_injection_sweep_recovery(btsettl_rv_sweep_template_path, tmp_path):
    """
    Scan injected stellar RV from -105 to -125 km s⁻¹ in steps of 2 km s⁻¹ on the **lte049**
    BT-Settl template; recover RV using ``rv_per_chunks_mask_kpf`` with the template-derived
    ``lte049.0*.mas`` mask and the same median/MAD + IV mean as mask-KPF stellar RV. Fit a linear
    relation; save Plotly HTML under ``tmp_path``.
    """
    try:
        sweep = _collect_btsettl_rv_injection_sweep(btsettl_rv_sweep_template_path)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    assert sweep["single_echelle_order"] is None
    assert sweep["ccf_grid_mode"] == "broad"
    assert len(sweep["injected_km_s"]) == 11
    assert np.all(np.isfinite(sweep["recovered_m_s"]))
    assert 0.7 < sweep["ols_slope"] < 1.3, f"slope out of range: {sweep['ols_slope']}"
    assert sweep["rms_residual_m_s"] < 20_000.0, f"RMS too large: {sweep['rms_residual_m_s']}"

    fig = _build_btsettl_rv_injection_figure(sweep)
    out_html = os.path.join(str(tmp_path), "btsettl_rv_injection_sweep.html")
    fig.write_html(out_html)
    assert os.path.isfile(out_html)


@pytest.mark.skipif(
    not _truthy_env("HISPEC_DRP_BTSETTL_RV_SWEEP_PLOT"),
    reason="Set HISPEC_DRP_BTSETTL_RV_SWEEP_PLOT=1 to run browser Plotly (see module docstring).",
)
def test_btsettl_rv_injection_sweep_plotly_popup(btsettl_rv_sweep_template_path, tmp_path):
    """
    Runs the BT-Settl injection sweep twice on **BSPEC echelle order 132 only** (lte049 template +
    ``lte049.0*.mas``): broad default CCF grid vs **merged hybrid** (full broad search + denser
    sampling ±300 m/s @ 0.1 m/s).
    Writes (1) injected-vs-recovered comparison HTML and (2) a two-panel CCF-vs-RV figure with
    Gaussian fits and parameter legends. Opens **both** in the browser when
    ``HISPEC_DRP_BTSETTL_RV_SWEEP_PLOT=1``.
    """
    pytest.importorskip("plotly")
    try:
        sweep_broad = _collect_btsettl_rv_injection_sweep(
            btsettl_rv_sweep_template_path,
            use_ccf_input_velocity_supersample=False,
            single_echelle_order=_BTSETTL_RV_SWEEP_PLOT_ECHELLE_ORDER,
        )
        sweep_refined = _collect_btsettl_rv_injection_sweep(
            btsettl_rv_sweep_template_path,
            use_ccf_input_velocity_supersample=True,
            single_echelle_order=_BTSETTL_RV_SWEEP_PLOT_ECHELLE_ORDER,
        )
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    eo = _BTSETTL_RV_SWEEP_PLOT_ECHELLE_ORDER
    assert sweep_broad["ccf_grid_mode"] == "broad"
    assert sweep_refined["ccf_grid_mode"] == "merged_hybrid_per_injection"
    assert sweep_broad["single_echelle_order"] == eo
    fig_inj = _build_btsettl_rv_injection_comparison_figure(sweep_broad, sweep_refined)
    out_inj = os.path.join(str(tmp_path), "btsettl_rv_injection_sweep_popup.html")
    fig_inj.write_html(out_inj)

    rv_demo = _BTSETTL_RV_SWEEP_PLOT_CCF_DEMO_KM_S
    try:
        snap_broad = _btsettl_single_order_ccf_snapshot(
            btsettl_rv_sweep_template_path, eo, rv_demo, use_broad_ccf_grid=True
        )
        snap_refined = _btsettl_single_order_ccf_snapshot(
            btsettl_rv_sweep_template_path, eo, rv_demo, use_broad_ccf_grid=False
        )
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    fig_ccf = _build_btsettl_order_ccf_dual_panel_figure(snap_broad, snap_refined)
    out_ccf = os.path.join(str(tmp_path), "btsettl_rv_injection_order_ccf_popup.html")
    fig_ccf.write_html(out_ccf)

    webbrowser.open("file://" + os.path.abspath(out_inj), new=1)
    webbrowser.open("file://" + os.path.abspath(out_ccf), new=2)


def test_ccf_mask_kpf_excludes_partial_edge_line_overlap():
    """
    For mask CCF, a line is counted only when its shifted segment is fully inside
    the chunk wavelength support; partially clipped edge overlaps are excluded.
    """
    wave = np.linspace(1.0, 1.1, 512, dtype=np.float64)
    spec = np.ones_like(wave)
    spec_err = np.ones_like(wave)
    dq_bad = np.zeros_like(wave, dtype=bool)

    # One fully contained line and one partially overlapping edge line.
    full_only = MaskLines(
        start_um=np.array([1.020], dtype=np.float64),
        end_um=np.array([1.030], dtype=np.float64),
        weight=np.array([1.0], dtype=np.float64),
    )
    full_plus_partial = MaskLines(
        start_um=np.array([1.020, 0.995], dtype=np.float64),
        end_um=np.array([1.030, 1.005], dtype=np.float64),
        weight=np.array([1.0, 1.0], dtype=np.float64),
    )

    out_full = rv_per_chunks_mask_kpf(
        wave,
        spec,
        spec_err,
        dq_bad,
        full_only,
        window_size_um=1.0,  # force a single chunk over this order
        v_min_m_s=-100.0,
        v_max_m_s=100.0,
        v_step_m_s=50.0,
        max_cores=2,
        return_ccf=True,
    )
    out_mix = rv_per_chunks_mask_kpf(
        wave,
        spec,
        spec_err,
        dq_bad,
        full_plus_partial,
        window_size_um=1.0,  # force a single chunk over this order
        v_min_m_s=-100.0,
        v_max_m_s=100.0,
        v_step_m_s=50.0,
        max_cores=2,
        return_ccf=True,
    )

    _rvs_a, _errs_a, _centers_a, v_a, ccf_a, _wla, _wha, _nga = out_full
    _rvs_b, _errs_b, _centers_b, v_b, ccf_b, _wlb, _whb, _ngb = out_mix
    assert ccf_a is not None and ccf_b is not None
    assert ccf_a.shape == ccf_b.shape == (1, len(v_a))
    assert np.allclose(v_a, v_b)
    assert np.allclose(ccf_a, ccf_b, rtol=0.0, atol=1e-12), (
        "Partially overlapping edge line should not contribute to ccf_mask_kpf CCF."
    )

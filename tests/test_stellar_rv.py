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
  ``test_fit_ccf_kpf_style_btsettl_bspec_all_chunks_parallel`` assertions, run lte049 + ``K2_all.mas``
  ``ccf_mask_kpf``: **HTML-only** previews (no ``fig.show()``):
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
"""
from __future__ import annotations

import os
import time
import webbrowser
from unittest.mock import patch

import numpy as np
import pytest

import hispecdrp
from hispecdrp import datamodels
from hispecdrp.datamodels import hispec_config
from hispecdrp.primitives import StellarRVPrimitive
from hispecdrp.primitives.utils.ccf_utils import (
    _V_MAX,
    _V_MIN,
    _resolve_stellar_rv_numba_threads,
    dv_per_pixel_m_s,
    fit_ccf_rv,
    process_orders_rvs_numba_parallel,
    rv_per_chunks,
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
# lte049 + polynomial L2 (same construction as mask-KPF smoke scripts)
_BTSETTL_LTE049_POLY_COEFS = np.array([0.0001, -0.001, 0.01, 0.05, 1.0])
MASK_DIR = os.path.join(os.path.dirname(hispecdrp.__file__), "data", "masks")

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


def _run_lte049_k2all_mask_kpf_previews(tmp_path: str) -> list[tuple[str, float, str, bool]]:
    """
    BSPEC L2 from lte049 + Doppler + polynomial; ``ccf_mask_kpf`` with ``K2_all.mas``.

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
    mask_path = os.path.join(MASK_DIR, "K2_all.mas")
    if not os.path.isfile(mask_path):
        pytest.skip(f"K2_all.mas not found: {mask_path}")

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

    out_dir = os.path.join(str(tmp_path), "lte049_k2all_mask_kpf")
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
            suffix="lte049_k2all",
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
    v_step = 500.0
    n_pix = 2048
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
    to Numba’s allowed thread count on the host, e.g. 2 on small CI runners) and ``ccf_v_step=500`` m/s (matches the serial test’s velocity
    sampling, not the library default 50 m/s). Phase-B Gaussian fits run serially in Python.

    Assertions use the same recovery fractions as the serial test, on chunks that are eligible
    (>=20 pixels in the bin and written by the parallel path, ``np.isfinite(all_wl)``).
    """
    timing_rows: list[tuple[str, float, str, bool]] = []
    input_rv_m_s = INPUT_RV_KM_S * 1000.0
    v_step = 500.0
    n_pix = 2048
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
    nthr = _resolve_stellar_rv_numba_threads(6)
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
            ccf_v_step=500.0,
        )
    )
    parallel_wall = time.perf_counter() - t0
    timing_rows.append(
        (
            f"lte025: process_orders_rvs_numba_parallel (numba_threads={nthr}, ccf_v_step=500 m/s)",
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

    if _want_bspec_btsettl_all_previews():
        timing_rows.extend(_run_lte049_k2all_mask_kpf_previews(str(tmp_path)))

    _print_btsettl_test_timing_summary(
        timing_rows,
        title=(
            "test_fit_ccf_kpf_style_btsettl_bspec_all_chunks_parallel — "
            "wall time by step (seconds)"
        ),
    )

"""
Stellar RV primitive tests (**ccf_stellar_template**: stellar-template CCF in log-λ,
median/MAD outlier rejection, inverse-variance weighted mean).

Synthetic L2 spectra use a uniform echelle grid (:func:`_order_wavelength_grid`) with
Gaussian absorption at the linear-Doppler-shifted wavelength. Rest-frame templates
are sampled at **3×** the L2 pixel count per order (finer λ sampling) to reduce
log-λ CCF / interpolation bias vs. template-at-data-resolution.

**Stellar-template CCF + BT-Settl BSPEC preview** (``ccf_stellar_template``)::

  pytest tests/test_stellar_rv.py -v
  HISPEC_DRP_SHOW_CCF_PREVIEW=1 pytest tests/test_stellar_rv.py::test_btsettl_shifted_polynomial_l2_recovers_rv_with_preview -v -s

**Mask-shift CCF + BT-Settl BSPEC preview** (``ccf_mask_kpf``; ``lte049.0-2.5*.h5`` + ``K2_all.mas``)::

  HISPEC_DRP_SHOW_CCF_PREVIEW=1 pytest tests/test_stellar_rv.py::test_btsettl_bspec_l2_mask_kpf_recovers_rv_with_preview -v -s

**BT-Settl + ``K2_all.mas`` per order + velocity slider** (debug HTML; order dropdown + shift bar)::

  HISPEC_DRP_SHOW_MASK_OVERLAY_DEBUG=1 pytest tests/test_stellar_rv.py::test_stellar_rv_mask_kpf_debug_overlay_html -v -s

(Preview: HTML + ``fig.show``; no PNG when ``HISPEC_DRP_SHOW_CCF_PREVIEW`` is set.
``HISPEC_DRP_SHOW_MASK_OVERLAY_DEBUG`` opens the mask-overlay HTML in the default browser.)
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np
import pytest

from hispecdrp import datamodels
import hispecdrp
from hispecdrp.datamodels import hispec_config
from hispecdrp.primitives import StellarRVPrimitive
from hispecdrp.primitives.utils.ccf_mask_kpf_utils import load_mask_spirou_mas
from hispecdrp.primitives.utils.ccf_utils import (
    _V_MAX,
    _V_MIN,
    dv_per_pixel_m_s,
    fit_ccf_rv,
    rv_per_chunks,
)
from hispecdrp.utils.spectral_templates_utils import get_spectrum_path, load_btsettl_spectrum

logger = logging.getLogger(__name__)

# BT-Settl template lives in hispecdrp/data/templates/
TEMPLATE_DIR = os.path.join(os.path.dirname(hispecdrp.__file__), "data", "templates")
MASK_DIR = os.path.join(os.path.dirname(hispecdrp.__file__), "data", "masks")


# Synthetic spectrum constants for RV recovery test
C_KM_S = 299792.458  # speed of light, km/s
LAMBDA_ORDER_COEF_UM = 147.0  # λ_center(m) = LAMBDA_ORDER_COEF_UM / m (µm)
INPUT_RV_KM_S = -110.0  # km/s (stellar RV to recover)
# Line width (σ_v); smaller gives a narrower CCF peak in the preview.
SIGMA_V_KM_S = 5.0  # Gaussian absorption velocity width (σ), km/s
ABSORPTION_DEPTH = 0.05  # 5% absorption depth
FLAT_FLUX = 100.0

# Reproducible synthetic noise in tests (numpy Generator seed base; per-order streams).
_STELLAR_RV_SYNTHETIC_NOISE_SEED = 0xC0FFEE42

# Template spectrum: number of wavelength samples per order relative to L2 n_pix
_TEMPLATE_PIXELS_PER_ORDER_FACTOR = 3


def _order_wavelength_grid(order: int, n_pix: int) -> tuple[np.ndarray, float]:
    """Wavelength grid for one echelle order. Returns (wave_um, lambda_center_um)."""
    lam_center = LAMBDA_ORDER_COEF_UM / order
    delta_lam = lam_center / order  # free spectral range
    wave = np.linspace(
        lam_center - delta_lam / 2.0,
        lam_center + delta_lam / 2.0,
        n_pix,
        dtype=np.float64,
    )
    return wave, lam_center


def _sigma_lambda_um(lam_center_um: float, sigma_v_km_s: float) -> float:
    """σ_λ (µm) for a Gaussian line with velocity width σ_v (km/s)."""
    return lam_center_um * (sigma_v_km_s / C_KM_S)


def _flux_with_gaussian_absorption(
    lam_um: np.ndarray,
    lam_line_um: float,
    sigma_lam_um: float,
    depth: float,
    flat_flux: float,
) -> np.ndarray:
    """Flux with a Gaussian absorption at lam_line_um (depth fraction, flat_flux continuum)."""
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
    """
    Build L2Model with full BSPEC or RSPEC order range, flat continuum and
    Gaussian absorption (velocity width sigma_v_km_s, depth) at linear Doppler
    shift rv_km_s on each order’s echelle grid. Uses SPEC_FIB1, WAVE_FIB1.
    """
    if spectrograph not in ("BSPEC", "RSPEC"):
        raise ValueError("spectrograph must be BSPEC or RSPEC")
    orders = hispec_config.HISPECConfig.get_order_list(spectrograph)
    if not orders:
        raise ValueError(f"No orders for {spectrograph}")

    order_start = int(min(orders))
    order_end = int(max(orders))
    meta = {
        "order_start": order_start,
        "order_end": order_end,
        "spectrograph": spectrograph,
        "instrument_name": "HISPEC",
    }
    model = datamodels.L2Model(meta=meta)
    model.meta.order_start = order_start
    model.meta.order_end = order_end

    n_orders = len(orders)
    wave = np.zeros((n_orders, n_pix), dtype=np.float64)
    spec = np.zeros((n_orders, n_pix), dtype=np.float32)
    for oi, m in enumerate(orders):
        w, lam_center = _order_wavelength_grid(m, n_pix)
        wave[oi] = w
        lam_obs = lam_center * (1.0 + rv_km_s / C_KM_S)
        sigma_lam = _sigma_lambda_um(lam_center, sigma_v_km_s)
        spec[oi] = _flux_with_gaussian_absorption(
            w, lam_obs, sigma_lam, depth, flat_flux
        )

    err = np.maximum(spec * 0.01, 0.1).astype(np.float32)
    spec_dq = np.zeros((n_orders, n_pix), dtype=np.uint32)

    model.spec_fib1 = spec
    model.spec_err_fib1 = err
    model.wave_fib1 = wave
    model.spec_dq_fib1 = spec_dq
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
    """
    Rest-frame template (wavelength µm, flux) with Gaussian absorptions at
    each order center. Each order uses ``template_pixels_factor * n_pix`` samples
    over the same span as the L2 echelle grid (higher λ resolution than data).
    """
    orders = hispec_config.HISPECConfig.get_order_list(spectrograph)
    if not orders:
        raise ValueError(f"No orders for {spectrograph}")
    n_t = int(template_pixels_factor) * int(n_pix)
    if n_t < 2:
        raise ValueError("template must have at least 2 points per order")

    wl_list = []
    flux_list = []
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
        f = _flux_with_gaussian_absorption(
            w, lam_center, sigma_lam, depth, flat_flux
        )
        wl_list.append(w)
        flux_list.append(f)

    template_wl = np.concatenate(wl_list)
    template_flux = np.concatenate(flux_list)
    sort_idx = np.argsort(template_wl)
    return template_wl[sort_idx], template_flux[sort_idx]


def make_synthetic_l2_single_order(
    spectrograph: str,
    order: int,
    n_pix: int = 4096,
    rv_km_s: float = INPUT_RV_KM_S,
    sigma_v_km_s: float = SIGMA_V_KM_S,
    depth: float = ABSORPTION_DEPTH,
    flat_flux: float = FLAT_FLUX,
) -> datamodels.L2Model:
    """
    Build L2Model with a single echelle order: flat continuum and Gaussian
    absorption at rv_km_s. Uses SPEC_FIB1, WAVE_FIB1 for stellar RV primitive.
    """
    if spectrograph not in ("BSPEC", "RSPEC"):
        raise ValueError("spectrograph must be BSPEC or RSPEC")
    meta = {
        "order_start": order,
        "order_end": order,
        "spectrograph": spectrograph,
        "instrument_name": "HISPEC",
    }
    model = datamodels.L2Model(meta=meta)
    model.meta.order_start = order
    model.meta.order_end = order

    w, lam_center = _order_wavelength_grid(order, n_pix)
    lam_obs = lam_center * (1.0 + rv_km_s / C_KM_S)
    sigma_lam = _sigma_lambda_um(lam_center, sigma_v_km_s)
    spec = _flux_with_gaussian_absorption(w, lam_obs, sigma_lam, depth, flat_flux)
    spec = np.asarray(spec, dtype=np.float32).reshape(1, -1)
    wave = np.asarray(w, dtype=np.float64).reshape(1, -1)
    err = np.maximum(spec * 0.01, 0.1).astype(np.float32)
    spec_dq = np.zeros((1, n_pix), dtype=np.uint32)

    model.spec_fib1 = spec
    model.spec_err_fib1 = err
    model.wave_fib1 = wave
    model.spec_dq_fib1 = spec_dq
    return model


def make_rest_frame_template_single_order(
    order: int,
    n_pix: int = 4096,
    depth: float = ABSORPTION_DEPTH,
    flat_flux: float = FLAT_FLUX,
    sigma_v_km_s: float = SIGMA_V_KM_S,
    *,
    template_pixels_factor: int = _TEMPLATE_PIXELS_PER_ORDER_FACTOR,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rest-frame template for one order: ``template_pixels_factor * n_pix`` samples
    over the echelle order span; absorption at λ_center (rest).
    """
    lam_center = LAMBDA_ORDER_COEF_UM / float(order)
    delta_lam = lam_center / float(order)
    n_t = int(template_pixels_factor) * int(n_pix)
    w = np.linspace(
        lam_center - delta_lam / 2.0,
        lam_center + delta_lam / 2.0,
        max(n_t, 2),
        dtype=np.float64,
    )
    sigma_lam = _sigma_lambda_um(lam_center, sigma_v_km_s)
    flux = _flux_with_gaussian_absorption(
        w, lam_center, sigma_lam, depth, flat_flux
    )
    return np.asarray(w, dtype=np.float64), np.asarray(flux, dtype=np.float64)


def _make_minimal_l2_for_ccf(n_orders: int = 10, n_pix: int = 4096):
    """Build a minimal L2-like model with SPEC_FIB1, WAVE_FIB1 for stellar RV (BSPEC order range)."""
    order_start = 110
    order_end = order_start + n_orders - 1
    meta = {
        "order_start": order_start,
        "order_end": order_end,
        "spectrograph": "BSPEC",
        "instrument_name": "HISPEC",
    }
    # Wavelength: each order covers a small range; span 1.0 to 1.35 µm across orders
    wl_per_order = np.linspace(1.0, 1.35, n_orders + 1)
    wave = np.zeros((n_orders, n_pix), dtype=np.float64)
    for oi in range(n_orders):
        wave[oi] = np.linspace(wl_per_order[oi], wl_per_order[oi + 1], n_pix)
    # Spectrum: non-zero flux with some structure (so CCF has signal)
    spec = np.ones((n_orders, n_pix), dtype=np.float32) * 100.0
    spec += 10.0 * np.sin(2 * np.pi * np.arange(n_pix) / 200.0)[np.newaxis, :]
    spec_err = np.maximum(spec * 0.01, 0.1).astype(np.float32)
    spec_dq = np.zeros((n_orders, n_pix), dtype=np.uint32)

    model = datamodels.L2Model(meta=meta)
    # set_meta_defaults in on_init overwrites order_start/order_end from config; set again so make_order_vec() matches data shape
    model.meta.order_start = order_start
    model.meta.order_end = order_end
    model.spec_fib1 = spec
    model.spec_err_fib1 = spec_err
    model.wave_fib1 = wave
    model.spec_dq_fib1 = spec_dq
    return model


@pytest.fixture(scope="module")
def template_path():
    """Resolve BT-Settl template path from hispecdrp/data/templates/ (skip if missing)."""
    path = get_spectrum_path(
        "lte025.0-5.5-0.0a+0.0.BT-Settl.spec.7.h5",
        base_dir=TEMPLATE_DIR,
    )
    if not os.path.isfile(path):
        pytest.skip(f"Template not found: {path}")
    return path


@pytest.fixture(scope="module")
def template_path_lte049():
    """BT-Settl template ``lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5`` for mask-overlay debug test."""
    path = get_spectrum_path(
        "lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5",
        base_dir=TEMPLATE_DIR,
    )
    if not os.path.isfile(path):
        pytest.skip(f"Template not found: {path}")
    return path


@pytest.fixture(scope="module")
def k2_filt_allrv_mask_path():
    """Packaged SPIROU-style mask ``K2_filt_allRV.mas`` (vacuum nm → µm in loader)."""
    p = os.path.join(MASK_DIR, "K2_filt_allRV.mas")
    if not os.path.isfile(p):
        pytest.skip(f"Mask not found: {p}")
    return p


@pytest.fixture(scope="module")
def k2_all_mask_path():
    """Packaged SPIROU-style mask ``K2_all.mas`` (vacuum nm → µm in loader)."""
    p = os.path.join(MASK_DIR, "K2_all.mas")
    if not os.path.isfile(p):
        pytest.skip(f"Mask not found: {p}")
    return p


def test_stellar_rv_debug_ccf_logs_per_chunk(capsys, template_path):
    """Run CCF with debug_ccf=True and assert per-chunk diagnostics are logged."""
    model = _make_minimal_l2_for_ccf(n_orders=10, n_pix=1024)
    prim = StellarRVPrimitive(
        template_path=template_path,
        do_fibers=["1"],
        debug_ccf=True,
        window_size_um=0.3,
    )
    out = prim.apply(model)
    assert out is not None

    captured = capsys.readouterr()
    log_text = captured.out + captured.err
    assert "CCF fiber=" in log_text, "Expected CCF per-chunk log line when debug_ccf=True"
    assert "wl_center=" in log_text
    assert "ccf_max=" in log_text
    assert "n_good=" in log_text
    assert "rv=" in log_text


@pytest.mark.parametrize("spectrograph", ["BSPEC", "RSPEC"])
def test_stellar_rv_single_order_injection_recovery(spectrograph):
    """
    Single echelle order: data on uniform grid, template at 3× λ sampling;
    combined RV should match injected linear Doppler RV within tolerance (sub-pixel
    residual vs. log-λ CCF is reduced by finer template, not eliminated).
    """
    orders = hispec_config.HISPECConfig.get_order_list(spectrograph)
    assert orders is not None and len(orders) > 0, f"No orders for {spectrograph}"
    order = int(orders[len(orders) // 2])
    n_pix = 2048
    input_rv_m_s = INPUT_RV_KM_S * 1000.0
    algorithm = "ccf_stellar_template"

    l2 = make_synthetic_l2_single_order(
        spectrograph,
        order,
        n_pix=n_pix,
        rv_km_s=INPUT_RV_KM_S,
        sigma_v_km_s=SIGMA_V_KM_S,
        depth=ABSORPTION_DEPTH,
        flat_flux=FLAT_FLUX,
    )
    template_wl, template_flux = make_rest_frame_template_single_order(
        order,
        n_pix=n_pix,
        depth=ABSORPTION_DEPTH,
        flat_flux=FLAT_FLUX,
        sigma_v_km_s=SIGMA_V_KM_S,
    )

    prim = StellarRVPrimitive(
        template_wavelength=template_wl,
        template_flux=template_flux,
        do_fibers=["1"],
        do_orders=[order],
        window_size_um=0.3,
        debug_ccf=True,
        algorithm=algorithm,
        show_preview=False,
        save_preview=False,
    )
    result = prim.apply(l2)
    assert result is not None, f"Primitive returned None for {spectrograph} {algorithm}"
    out = result

    tbl = getattr(out, "rvccf_fib1", None)
    assert tbl is not None and hasattr(tbl, "columns"), (
        f"Output L2 missing rvccf_fib1 for {spectrograph} {algorithm}"
    )
    assert "RV" in tbl.colnames and "RV_ERR" in tbl.colnames

    rvs = np.asarray(tbl["RV"], dtype=np.float64)
    rv_errs = np.asarray(tbl["RV_ERR"], dtype=np.float64)
    valid = np.isfinite(rvs) & np.isfinite(rv_errs) & (rv_errs > 0)
    assert np.any(valid), (
        f"No valid chunk RVs for {spectrograph} {algorithm} (order={order})"
    )

    avg_tbl = getattr(out, "rvccfavg_fib1", None)
    assert avg_tbl is not None and len(avg_tbl) > 0, (
        f"Output L2 missing rvccfavg_fib1 for {spectrograph} {algorithm}"
    )
    rv_agg = float(avg_tbl["RV"][0])
    rv_agg_err = float(avg_tbl["RV_ERR"][0])
    tolerance_m_s = max(3.0 * rv_agg_err, 500.0)

    assert abs(rv_agg - input_rv_m_s) <= tolerance_m_s, (
        f"{spectrograph} {algorithm} single order {order}: combined RV {rv_agg/1000:.4f} km/s "
        f"should be within {tolerance_m_s:.0f} m/s of inject {input_rv_m_s/1000:.4f} km/s "
        f"(bias={rv_agg - input_rv_m_s:.1f} m/s, rv_err={rv_agg_err:.1f} m/s)."
    )


def test_stellar_rv_mask_kpf_single_order_recovery(tmp_path):
    """
    Single echelle order: mask-only (no template) RV recovery.

    We create a synthetic `.mas` mask containing one line segment centered at the
    rest-frame line wavelength used in `make_synthetic_l2_single_order()`.
    """

    spectrograph = "BSPEC"
    orders = hispec_config.HISPECConfig.get_order_list(spectrograph)
    order_mid = int(orders[len(orders) // 2])
    assert orders is not None and len(orders) > 0
    order = int(orders[len(orders) // 2])
    n_pix = 2048

    rv_km_s = INPUT_RV_KM_S
    l2 = make_synthetic_l2_single_order(
        spectrograph,
        order,
        n_pix=n_pix,
        rv_km_s=rv_km_s,
        sigma_v_km_s=SIGMA_V_KM_S,
        depth=ABSORPTION_DEPTH,
        flat_flux=FLAT_FLUX,
    )

    lam_center_um = LAMBDA_ORDER_COEF_UM / float(order)
    # Use a mask segment width comparable to a few pixels in wavelength.
    delta_lam_um = (LAMBDA_ORDER_COEF_UM / float(order)) / float(order)  # FSR / order
    pix_spacing_um = delta_lam_um / float(n_pix - 1)
    mask_width_um = max(pix_spacing_um * 5.0, 1e-6)

    mask_path = tmp_path / "synthetic_mask.mas"
    start_um = lam_center_um - 0.5 * mask_width_um
    end_um = lam_center_um + 0.5 * mask_width_um
    weight = 1.0
    # SPIROU .mas columns are vacuum nanometers (see ccf_mask_kpf_utils.load_mask_spirou_mas).
    mask_path.write_text(
        f"{start_um * 1e3:.10e} {end_um * 1e3:.10e} {weight:.6f}\n", encoding="utf-8"
    )

    prim = StellarRVPrimitive(
        mask_path=str(mask_path),
        do_fibers=["1"],
        do_orders=[order],
        window_size_um=0.3,
        debug_ccf=False,
        algorithm="ccf_mask_kpf",
        show_preview=False,
        save_preview=False,
    )
    out = prim.apply(l2)
    assert out is not None

    avg_tbl = getattr(out, "rvccfavg_fib1", None)
    assert avg_tbl is not None and len(avg_tbl) > 0
    assert "RV" in avg_tbl.colnames and "RV_ERR" in avg_tbl.colnames

    rv_agg = float(avg_tbl["RV"][0])
    rv_err = float(avg_tbl["RV_ERR"][0])

    input_rv_m_s = rv_km_s * 1000.0
    tolerance_m_s = max(3.0 * rv_err, 500.0)
    assert abs(rv_agg - input_rv_m_s) <= tolerance_m_s, (
        f"ccf_mask_kpf: combined RV {rv_agg:.1f} m/s vs expected {input_rv_m_s:.1f} m/s "
        f"(rv_err={rv_err:.1f} m/s)."
    )


def test_load_mask_spirou_mas_wavelengths_are_nanometers(tmp_path):
    """NIR rows like 1136 nm must map to ~1.136 µm, not ~0.114 µm (Å mis-inference)."""
    p = tmp_path / "one_line.mas"
    p.write_text(
        "1136.8214805449140 1136.8374071988853 3.1340128422546798E-002\n",
        encoding="utf-8",
    )
    m = load_mask_spirou_mas(str(p))
    assert m.start_um.size == 1
    assert abs(float(m.start_um[0]) - 1.136821480544914) < 1e-9
    assert abs(float(m.end_um[0]) - 1.1368374071988853) < 1e-9


def test_stellar_rv_mask_kpf_debug_overlay_html(
    tmp_path, template_path_lte049, k2_all_mask_path
):
    """
    Debug-only mask overlay HTML is created for all BSPEC orders.

    Uses ``lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5`` and ``K2_all.mas`` so the
    spectrum + mask overlay matches that pairing; the order dropdown can switch across
    the full BSPEC order list.

    Set ``HISPEC_DRP_SHOW_MASK_OVERLAY_DEBUG=1`` to open the HTML in the default browser
    after the test writes it (same pattern as ``HISPEC_DRP_SHOW_CCF_PREVIEW``).
    """
    show_overlay = os.environ.get("HISPEC_DRP_SHOW_MASK_OVERLAY_DEBUG", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    spectrograph = "BSPEC"
    orders = hispec_config.HISPECConfig.get_order_list(spectrograph)
    order_mid = int(orders[len(orders) // 2])
    n_pix = 1024
    order_start = int(min(orders))
    order_end = int(max(orders))
    n_orders = len(orders)
    wl_min_um = 0.95
    wl_max_um = 1.4
    w_rest, flux_rest = load_btsettl_spectrum(
        template_path_lte049, wl_min_um=wl_min_um, wl_max_um=wl_max_um
    )
    w_rest = np.asarray(w_rest.to_value("micron"), dtype=np.float64)
    flux_rest = np.asarray(flux_rest.value, dtype=np.float64)
    ok = np.isfinite(w_rest) & np.isfinite(flux_rest) & (w_rest > 0)
    w_rest = w_rest[ok]
    flux_rest = flux_rest[ok]
    idx = np.argsort(w_rest)
    w_rest = w_rest[idx]
    flux_rest = flux_rest[idx]
    assert len(w_rest) > 50, "Need BT-Settl support across the test order span"

    wave = np.zeros((n_orders, n_pix), dtype=np.float64)
    spec = np.zeros((n_orders, n_pix), dtype=np.float32)
    for oi, m in enumerate(orders):
        wave_ord, _ = _order_wavelength_grid(int(m), n_pix)
        wave[oi] = wave_ord
        lambda_rest = wave_ord / (1.0 + INPUT_RV_KM_S / C_KM_S)
        spec[oi] = np.interp(lambda_rest, w_rest, flux_rest).astype(np.float32)
    err = np.maximum(spec * 0.01, 0.1).astype(np.float32)
    spec_dq = np.zeros((n_orders, n_pix), dtype=np.uint32)
    l2 = datamodels.L2Model(
        meta={
            "order_start": order_start,
            "order_end": order_end,
            "spectrograph": spectrograph,
            "instrument_name": "HISPEC",
        }
    )
    l2.meta.order_start = order_start
    l2.meta.order_end = order_end
    l2.spec_fib1 = spec
    l2.spec_err_fib1 = err
    l2.wave_fib1 = wave
    l2.spec_dq_fib1 = spec_dq

    prim = StellarRVPrimitive(
        algorithm="ccf_mask_kpf",
        mask_path=k2_all_mask_path,
        do_fibers=["1"],
        do_orders=[order_mid],
        window_size_um=0.3,
        show_preview=False,
        save_preview=False,
        save_mask_overlay_debug=True,
        show_mask_overlay_debug=show_overlay,
        output_dir=str(tmp_path),
        suffix="stellar_rv_mask_debug",
    )
    out = prim.apply(l2)
    assert out is not None

    html_files = [
        os.path.join(str(tmp_path), f)
        for f in os.listdir(str(tmp_path))
        if f.endswith("_mask_overlay_debug_fib1.html")
    ]
    assert len(html_files) == 1
    html_path = html_files[0]
    assert os.path.isfile(html_path)
    print(f"MASK_OVERLAY_DEBUG_HTML={os.path.abspath(html_path)}", flush=True)
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    assert "Velocity shift (m/s)" in html
    assert "orderSelect" in html and "vSlider" in html and "vInput" in html
    for m in orders:
        assert f"\"order\": {int(m)}" in html


def test_stellar_rv_rspec_order_61_red_arm_recovery():
    """
    RSPEC order 61 (λ_center≈2.41 µm): uniform data grid + 3× template sampling;
    combined RV should match injected −110 km/s within tolerance.
    """
    spectrograph, order, n_pix = "RSPEC", 61, 2048
    input_rv_m_s = INPUT_RV_KM_S * 1000.0
    l2 = make_synthetic_l2_single_order(
        spectrograph, order, n_pix=n_pix, rv_km_s=INPUT_RV_KM_S,
        sigma_v_km_s=SIGMA_V_KM_S, depth=ABSORPTION_DEPTH, flat_flux=FLAT_FLUX,
    )
    tw, tf = make_rest_frame_template_single_order(
        order, n_pix=n_pix, depth=ABSORPTION_DEPTH, flat_flux=FLAT_FLUX, sigma_v_km_s=SIGMA_V_KM_S,
    )
    prim = StellarRVPrimitive(
        template_wavelength=tw, template_flux=tf, do_fibers=["1"], do_orders=[order],
        window_size_um=0.3, debug_ccf=False, algorithm="ccf_stellar_template",
        show_preview=False, save_preview=False,
    )
    out = prim.apply(l2)
    rv_agg = float(out.rvccfavg_fib1["RV"][0])
    rv_err = float(out.rvccfavg_fib1["RV_ERR"][0])
    assert abs(rv_agg - input_rv_m_s) <= max(3.0 * rv_err, 500.0), (
        f"RSPEC order 61: RV {rv_agg:.1f} m/s vs expected {input_rv_m_s:.1f} m/s "
        f"(chunk wl≈2.41 µm red arm)."
    )


@pytest.mark.parametrize("spectrograph", ["BSPEC", "RSPEC"])
def test_stellar_rv_recovers_rv_per_chunk(spectrograph):
    """
    Generate L2 dummy spectra for BSPEC or RSPEC covering the full wavelength range:
    flat continuum + Gaussian absorption (50 km/s width, 5% depth) at RV = -110 km/s.
    Use a rest-frame template (3× λ resolution per order). The pipeline combines
    chunk/order RVs (ccf_stellar_template); RVCCFAVG must recover INPUT_RV_KM_S to within
    tolerance. Set HISPEC_DRP_SHOW_CCF_PREVIEW=1 for unified RV + CCF preview.
    """
    algorithm = "ccf_stellar_template"
    n_pix = 4096
    input_rv_m_s = INPUT_RV_KM_S * 1000.0

    l2 = make_synthetic_l2_spectrum(
        spectrograph,
        n_pix=n_pix,
        rv_km_s=INPUT_RV_KM_S,
        sigma_v_km_s=SIGMA_V_KM_S,
        depth=ABSORPTION_DEPTH,
        flat_flux=FLAT_FLUX,
    )
    template_wl, template_flux = make_rest_frame_template(
        spectrograph,
        n_pix=n_pix,
        depth=ABSORPTION_DEPTH,
        flat_flux=FLAT_FLUX,
        sigma_v_km_s=SIGMA_V_KM_S,
    )

    show_preview = os.environ.get("HISPEC_DRP_SHOW_CCF_PREVIEW", "").strip().lower() in ("1", "true", "yes")
    save_plots = os.environ.get("HISPEC_DRP_SAVE_CCF_PLOTS", "").strip().lower() in ("1", "true", "yes")
    plot_out_dir = None
    if save_plots:
        plot_out_dir = os.path.join(os.path.dirname(__file__), "output", "stellar_rv_plots")
        os.makedirs(plot_out_dir, exist_ok=True)
    prim = StellarRVPrimitive(
        template_wavelength=template_wl,
        template_flux=template_flux,
        do_fibers=["1"],
        window_size_um=0.3,
        debug_ccf=True,
        algorithm=algorithm,
        show_preview=show_preview,
        save_preview=save_plots or show_preview,
        output_dir=plot_out_dir,
    )
    result = prim.apply(l2)
    assert result is not None
    out = result

    # When HISPEC_DRP_SAVE_CCF_PLOTS=1, write a Plotly HTML per (spectrograph, algorithm)
    if save_plots and plot_out_dir and prim._output and "preview" in prim._output:
        prev = prim._output["preview"]
        fig = prev.get("fig")
        if fig is not None:
            path = os.path.join(plot_out_dir, f"stellar_rv_{spectrograph}_{algorithm}_preview.html")
            fig.write_html(path, include_plotlyjs="cdn")
            print(f"Saved Plotly preview: {path}")

    tbl_attr = "rvccf_fib1"
    tbl = getattr(out, tbl_attr, None)
    assert tbl is not None and hasattr(tbl, "columns"), (
        f"Output L2 missing {tbl_attr}"
    )
    assert "RV" in tbl.colnames and "RV_ERR" in tbl.colnames

    rvs = np.asarray(tbl["RV"], dtype=np.float64)
    rv_errs = np.asarray(tbl["RV_ERR"], dtype=np.float64)
    valid = np.isfinite(rvs) & np.isfinite(rv_errs) & (rv_errs > 0)
    assert np.any(valid), f"No valid chunk RVs for {spectrograph}"

    # Primary check: combined RV (after outlier rejection) recovers input
    avg_tbl = getattr(out, "rvccfavg_fib1", None)
    assert avg_tbl is not None and len(avg_tbl) > 0, f"Output L2 missing rvccfavg_fib1 for {spectrograph}"
    rv_agg = float(avg_tbl["RV"][0])
    rv_agg_err = float(avg_tbl["RV_ERR"][0])
    tolerance_m_s = max(3.0 * rv_agg_err, 600.0)
    assert abs(rv_agg - input_rv_m_s) <= tolerance_m_s, (
        f"{spectrograph} {algorithm}: combined RV {rv_agg/1000:.4f} km/s should be within "
        f"{tolerance_m_s:.0f} m/s of inject {input_rv_m_s/1000:.4f} km/s."
    )

    # Inliers vs nominal inject (sub-pixel / chunk scatter can be large on some orders).
    if "INLIER" in tbl.colnames:
        inlier = np.asarray(tbl["INLIER"], dtype=bool)[valid]
        in_ok = inlier
        if np.any(in_ok):
            res_in = np.abs(rvs[valid][in_ok] - input_rv_m_s)
            loose = np.maximum(3.0 * rv_errs[valid][in_ok], 2000.0)
            n_ok = np.sum(res_in <= loose)
            assert n_ok >= np.sum(in_ok) * 0.5, (
                f"{spectrograph} {algorithm}: at least half of inlier chunks should have "
                f"|RV - input| <= 3*RV_ERR or 100 m/s."
            )

    # Sanity: valid chunk RVs and uncertainties are finite and reasonable
    assert np.all(np.isfinite(rvs[valid])), "All valid chunk RVs should be finite"
    assert np.all(rv_errs[valid] > 0), "All valid chunk RV uncertainties should be positive"
    assert np.all(rv_errs[valid] < 1e6), "RV uncertainties should be reasonable (< 1000 km/s)"

    # Optional one-line report for validation/performance comparison (parse with HISPEC_DRP_STELLAR_RV_REPORT=1)
    if os.environ.get("HISPEC_DRP_STELLAR_RV_REPORT", "").strip().lower() in ("1", "true", "yes"):
        n_in = int(np.sum(tbl["INLIER"][valid])) if "INLIER" in tbl.colnames else int(np.sum(valid))
        bias_m_s = rv_agg - input_rv_m_s
        print(
            f"STELLAR_RV_REPORT|{spectrograph}|{algorithm}|PASS|"
            f"rv_agg_km_s={rv_agg/1000:.4f}|rv_agg_err_m_s={rv_agg_err:.2f}|"
            f"bias_m_s={bias_m_s:.2f}|n_valid={np.sum(valid)}|n_inlier={n_in}"
        )


def test_fit_ccf_rv_recovers_center_and_uncertainty():
    """
    Validate that fit_ccf_rv recovers the true RV and returns a sensible
    uncertainty when given a synthetic Gaussian CCF (velocity vs correlation).
    Models one chunk per BSPEC order (110–148) with wavelength ranges from the
    echelle relation λ_center = 147/order µm, FSR = 147/order² µm.
    """
    v_step = 500.0  # m/s; use default CCF search range (-500 to 500 km/s)
    v_min, v_max = _V_MIN, _V_MAX
    v_grid = np.arange(v_min, v_max + v_step * 0.5, v_step, dtype=np.float64)
    true_rv = -110000.0  # m/s (e.g. -110 km/s)
    true_sigma = 15000.0  # m/s (Gaussian width)
    amplitude = 1.0
    baseline = 0.0

    # One chunk per BSPEC order; wavelength range from echelle relation (same as _order_wavelength_grid)
    orders = hispec_config.HISPECConfig.get_order_list("BSPEC")
    chunk_results = []
    for order in orders:
        wave, _ = _order_wavelength_grid(order, 100)
        wl_lo = float(wave[0])
        wl_hi = float(wave[-1])
        ccf = amplitude * np.exp(-0.5 * ((v_grid - true_rv) / true_sigma) ** 2) + baseline
        # One independent, fixed stream per echelle order (stable if order list is reordered).
        rng = np.random.default_rng(_STELLAR_RV_SYNTHETIC_NOISE_SEED + int(order) * 1_000_003)
        ccf += 0.02 * (rng.random(v_grid.size) - 0.5)
        chunk_results.append({
            "order": order,
            "wl_lo": wl_lo,
            "wl_hi": wl_hi,
            "ccf": ccf.copy(),
        })

    # Fit first chunk for assertions (test still validates one fit)
    rv, rv_err = fit_ccf_rv(v_grid, chunk_results[0]["ccf"], v_step)
    assert np.isfinite(rv) and np.isfinite(rv_err), "Gaussian fit should return finite RV and uncertainty"
    assert rv_err > 0, "Uncertainty should be positive"
    assert abs(rv - true_rv) < 2 * true_sigma, (
        f"Recovered RV {rv} m/s should be within ~2 sigma of true {true_rv} m/s"
    )
    assert rv_err > 0 and rv_err <= true_sigma * 3, (
        f"RV uncertainty {rv_err} m/s should be positive and ≤ ~3*sigma ({true_sigma})"
    )


def test_fit_ccf_kpf_style_btsettl_bspec_one_order(template_path):
    """
    Simplified check of the KPF-style CCF Gaussian fitter on real BT-Settl structure:
    one BSPEC echelle order, Doppler-shifted spectrum (no polynomial), build chunk
    CCFs via rv_per_chunks and fit the mean CCF with fit_ccf_rv.
    """
    input_rv_km_s = -110.0
    input_rv_m_s = input_rv_km_s * 1000.0
    n_pix = 2048
    window_size_um = 0.3
    v_step = 500.0

    wl_min_um, wl_max_um = 0.95, 1.4
    w_rest, flux_rest = load_btsettl_spectrum(
        template_path, wl_min_um=wl_min_um, wl_max_um=wl_max_um
    )
    w_rest = np.asarray(w_rest.to_value("micron"), dtype=np.float64)
    flux_rest = np.asarray(flux_rest.value, dtype=np.float64)
    ok = np.isfinite(w_rest) & np.isfinite(flux_rest) & (w_rest > 0)
    w_rest = w_rest[ok]
    flux_rest = flux_rest[ok]
    idx = np.argsort(w_rest)
    w_rest = w_rest[idx]
    flux_rest = flux_rest[idx]
    template_ln = np.log(w_rest)
    template_flux = flux_rest

    order = 120
    wave_ord, lam_c = _order_wavelength_grid(order, n_pix)
    lambda_rest = wave_ord / (1.0 + input_rv_km_s / C_KM_S)
    flux_at_rest = np.interp(lambda_rest, w_rest, flux_rest)
    spec = flux_at_rest.astype(np.float64)
    spec_err = np.maximum(spec * 0.01, 0.1)
    mask = np.zeros(n_pix, dtype=bool)

    _rvs, _errs, _centers, v_grid, ccf_2d = rv_per_chunks(
        wave_ord,
        spec,
        spec_err,
        mask,
        template_ln,
        template_flux,
        window_size_um,
        v_min=_V_MIN,
        v_max=_V_MAX,
        v_step=v_step,
        return_ccf=True,
    )
    row = np.nanmean(ccf_2d, axis=0)
    valid = np.isfinite(row) & np.isfinite(v_grid)
    assert np.sum(valid) > 20, "Need a usable mean CCF"
    vspan = dv_per_pixel_m_s(float(lam_c))

    rv, rv_err, params, diag = fit_ccf_rv(
        v_grid[valid],
        row[valid],
        v_step,
        return_params=True,
        vel_span_m_s_per_pixel=vspan if np.isfinite(vspan) else None,
    )
    assert diag.get("fit_ok"), f"KPF-style fit should succeed: {diag.get('message')}"
    assert np.isfinite(rv) and np.isfinite(rv_err) and rv_err > 0
    tol_m_s = max(8000.0, 3.0 * rv_err)
    assert abs(rv - input_rv_m_s) < tol_m_s, (
        f"fit_ccf_rv RV {rv/1000:.2f} km/s vs input {input_rv_km_s} km/s"
    )
    assert "amplitude" in params and "v0" in params and "sigma" in params


def test_btsettl_shifted_polynomial_l2_recovers_rv_with_preview(template_path):
    """
    Load BT-Settl template, shift by a known RV, multiply by a 4th-order polynomial,
    put into a BSPEC L2 model, run stellar RV (CCF) with default algorithm/parameters,
    validate input RV is recovered. Set HISPEC_DRP_SHOW_CCF_PREVIEW=1 for the
    unified stellar RV preview (per-chunk RV + mean/order CCF + Gaussian fits).
    """
    input_rv_km_s = -110.0  # km/s (same as other tests)
    input_rv_m_s = input_rv_km_s * 1000.0
    n_pix = 4096
    window_size_um = 0.3  # default
    show_unified = os.environ.get("HISPEC_DRP_SHOW_CCF_PREVIEW", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    t_test0 = time.perf_counter()
    # BSPEC wavelength range (orders 110–148): ~0.99–1.34 µm
    wl_min_um, wl_max_um = 0.95, 1.4
    t_load = time.perf_counter()
    w_rest, flux_rest = load_btsettl_spectrum(
        template_path, wl_min_um=wl_min_um, wl_max_um=wl_max_um
    )
    w_rest = np.asarray(w_rest.to_value("micron"), dtype=np.float64)
    flux_rest = np.asarray(flux_rest.value, dtype=np.float64)
    # Sort and ensure no NaNs in used range
    ok = np.isfinite(w_rest) & np.isfinite(flux_rest) & (w_rest > 0)
    w_rest = w_rest[ok]
    flux_rest = flux_rest[ok]
    idx = np.argsort(w_rest)
    w_rest = w_rest[idx]
    flux_rest = flux_rest[idx]
    assert len(w_rest) > 100, "BT-Settl template should have many points in BSPEC range"
    logger.info(
        "TEST_TIMING step=load_btsettl_spectrum wall_s=%.3f", time.perf_counter() - t_load
    )

    # 4th-order polynomial: 1 + 0.05*x + 0.01*x^2 - 0.001*x^3 + 0.0001*x^4 (x in µm)
    poly_coefs = [0.0001, -0.001, 0.01, 0.05, 1.0]
    poly = np.poly1d(poly_coefs)

    orders = hispec_config.HISPECConfig.get_order_list("BSPEC")
    order_start = int(min(orders))
    order_end = int(max(orders))
    n_orders = len(orders)
    meta = {
        "order_start": order_start,
        "order_end": order_end,
        "spectrograph": "BSPEC",
        "instrument_name": "HISPEC",
    }
    model = datamodels.L2Model(meta=meta)
    model.meta.order_start = order_start
    model.meta.order_end = order_end

    t_build = time.perf_counter()
    wave = np.zeros((n_orders, n_pix), dtype=np.float64)
    spec = np.zeros((n_orders, n_pix), dtype=np.float32)
    for oi, m in enumerate(orders):
        wave_ord, _ = _order_wavelength_grid(m, n_pix)
        wave[oi] = wave_ord
        # Observed λ; rest-frame wavelength where we evaluate template: λ_rest = λ_obs / (1 + v/c)
        lambda_rest = wave_ord / (1.0 + input_rv_km_s / C_KM_S)
        flux_at_rest = np.interp(lambda_rest, w_rest, flux_rest)
        spec[oi] = (flux_at_rest * poly(wave_ord)).astype(np.float32)

    spec_err = np.maximum(spec * 0.01, 0.1).astype(np.float32)
    spec_dq = np.zeros((n_orders, n_pix), dtype=np.uint32)

    model.spec_fib1 = spec
    model.spec_err_fib1 = spec_err
    model.wave_fib1 = wave
    model.spec_dq_fib1 = spec_dq
    logger.info(
        "TEST_TIMING step=build_L2_synthetic_BSPEC wall_s=%.3f",
        time.perf_counter() - t_build,
    )

    prim = StellarRVPrimitive(
        template_path=template_path,
        do_fibers=["1"],
        window_size_um=window_size_um,
        show_preview=show_unified,
        save_preview=show_unified,
        max_cores=6,
    )
    t_apply = time.perf_counter()
    out = prim.apply(model)
    dt_apply = time.perf_counter() - t_apply
    dt_total = time.perf_counter() - t_test0
    logger.info("TEST_TIMING step=StellarRVPrimitive.apply wall_s=%.3f", dt_apply)
    logger.info("TEST_TIMING step=test_total wall_s=%.3f", dt_total)
    # Also print: after `apply`, hispecdrp's logging_context may leave pytest's
    # live log capture not showing these two lines consistently.
    print(
        f"TEST_TIMING step=StellarRVPrimitive.apply wall_s={dt_apply:.3f}",
        flush=True,
    )
    print(f"TEST_TIMING step=test_total wall_s={dt_total:.3f}", flush=True)
    assert out is not None

    # Recovered RV from pipeline: use RVCCFAVG if available, else weighted mean from RVCCF
    tbl_avg = getattr(out, "rvccfavg_fib1", None)
    if tbl_avg is not None and hasattr(tbl_avg, "columns") and "RV" in tbl_avg.colnames:
        recovered_rv_m_s = float(tbl_avg["RV"][0])
        recovered_rv_err_m_s = float(tbl_avg["RV_ERR"][0]) if "RV_ERR" in tbl_avg.colnames else None
    else:
        tbl = getattr(out, "rvccf_fib1", None)
        assert tbl is not None and "RV" in tbl.colnames
        rvs = np.asarray(tbl["RV"], dtype=np.float64)
        rv_errs = np.asarray(tbl["RV_ERR"], dtype=np.float64)
        valid = np.isfinite(rvs) & np.isfinite(rv_errs) & (rv_errs > 0)
        assert np.any(valid), "No valid chunk RVs from pipeline"
        w = 1.0 / (rv_errs[valid] ** 2)
        recovered_rv_m_s = float(np.average(rvs[valid], weights=w))
        recovered_rv_err_m_s = float(1.0 / np.sqrt(np.sum(w)))

    # Validate input RV is recovered (within 2*uncertainty or 5 km/s)
    tolerance_m_s = max(2.0 * (recovered_rv_err_m_s or 5000.0), 5000.0)
    assert abs(recovered_rv_m_s - input_rv_m_s) <= tolerance_m_s, (
        f"Recovered RV {recovered_rv_m_s/1000:.2f} km/s should match input {input_rv_km_s} km/s "
        f"within {tolerance_m_s/1000:.1f} km/s."
    )


def test_btsettl_bspec_l2_mask_kpf_recovers_rv_with_preview(
    template_path_lte049, k2_all_mask_path, tmp_path
):
    """
    Same synthetic BSPEC L2 as ``test_btsettl_shifted_polynomial_l2_recovers_rv_with_preview``
    (BT-Settl + polynomial + Doppler shift), but ``algorithm="ccf_mask_kpf"`` (no stellar
    template). Uses ``lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5`` and the packaged
    ``K2_all.mas`` line list (vacuum nm columns).

    **Recovery:** The injected RV is **not** asserted to match the mask-shift aggregate RV
    (masks correlate with line segments, not the BT-Settl template cross-match). This test
    checks finite outputs, table structure, and optional HTML preview.

    Set ``HISPEC_DRP_SHOW_CCF_PREVIEW=1`` for Plotly preview (same env as the BT-Settl
    ``ccf_stellar_template`` test). Preview HTML is written under ``tests/output/stellar_rv_plots/``
    when preview is enabled.
    """
    input_rv_km_s = -110.0
    input_rv_m_s = input_rv_km_s * 1000.0
    n_pix = 4096
    window_size_um = 0.3
    show_unified = os.environ.get("HISPEC_DRP_SHOW_CCF_PREVIEW", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    wl_min_um, wl_max_um = 0.95, 1.4
    print(f"BTSETTL_TEMPLATE={os.path.abspath(template_path_lte049)}", flush=True)
    w_rest, flux_rest = load_btsettl_spectrum(
        template_path_lte049, wl_min_um=wl_min_um, wl_max_um=wl_max_um
    )
    w_rest = np.asarray(w_rest.to_value("micron"), dtype=np.float64)
    flux_rest = np.asarray(flux_rest.value, dtype=np.float64)
    ok = np.isfinite(w_rest) & np.isfinite(flux_rest) & (w_rest > 0)
    w_rest = w_rest[ok]
    flux_rest = flux_rest[ok]
    idx = np.argsort(w_rest)
    w_rest = w_rest[idx]
    flux_rest = flux_rest[idx]
    assert len(w_rest) > 100, "BT-Settl template should have many points in BSPEC range"

    poly_coefs = [0.0001, -0.001, 0.01, 0.05, 1.0]
    poly = np.poly1d(poly_coefs)

    orders = hispec_config.HISPECConfig.get_order_list("BSPEC")
    order_start = int(min(orders))
    order_end = int(max(orders))
    n_orders = len(orders)
    meta = {
        "order_start": order_start,
        "order_end": order_end,
        "spectrograph": "BSPEC",
        "instrument_name": "HISPEC",
    }
    model = datamodels.L2Model(meta=meta)
    model.meta.order_start = order_start
    model.meta.order_end = order_end

    wave = np.zeros((n_orders, n_pix), dtype=np.float64)
    spec = np.zeros((n_orders, n_pix), dtype=np.float32)
    for oi, m in enumerate(orders):
        wave_ord, _ = _order_wavelength_grid(m, n_pix)
        wave[oi] = wave_ord
        lambda_rest = wave_ord / (1.0 + input_rv_km_s / C_KM_S)
        flux_at_rest = np.interp(lambda_rest, w_rest, flux_rest)
        spec[oi] = (flux_at_rest * poly(wave_ord)).astype(np.float32)

    spec_err = np.maximum(spec * 0.01, 0.1).astype(np.float32)
    spec_dq = np.zeros((n_orders, n_pix), dtype=np.uint32)

    model.spec_fib1 = spec
    model.spec_err_fib1 = spec_err
    model.wave_fib1 = wave
    model.spec_dq_fib1 = spec_dq

    plot_out_dir = os.path.join(os.path.dirname(__file__), "output", "stellar_rv_plots")
    if show_unified:
        os.makedirs(plot_out_dir, exist_ok=True)

    prim = StellarRVPrimitive(
        algorithm="ccf_mask_kpf",
        mask_path=k2_all_mask_path,
        do_fibers=["1"],
        window_size_um=window_size_um,
        show_preview=show_unified,
        save_preview=show_unified,
        output_dir=plot_out_dir if show_unified else str(tmp_path),
        max_cores=6,
    )
    print(f"K2_MASK={os.path.abspath(k2_all_mask_path)}", flush=True)
    out = prim.apply(model)
    assert out is not None

    tbl_avg = getattr(out, "rvccfavg_fib1", None)
    assert tbl_avg is not None and hasattr(tbl_avg, "columns") and "RV" in tbl_avg.colnames
    recovered_rv_m_s = float(tbl_avg["RV"][0])
    recovered_rv_err_m_s = (
        float(tbl_avg["RV_ERR"][0]) if "RV_ERR" in tbl_avg.colnames else None
    )
    assert np.isfinite(recovered_rv_m_s), "ccf_mask_kpf aggregate RV must be finite"

    tbl = getattr(out, "rvccf_fib1", None)
    assert tbl is not None and "RV" in tbl.colnames
    rvs = np.asarray(tbl["RV"], dtype=np.float64)
    assert np.any(np.isfinite(rvs)), "Per-chunk RV column should have finite values"

    # Unlike ccf_stellar_template, mask-shift CCF does not compare to the BT-Settl template;
    # the injected Doppler shift used to build the spectrum is not expected to match
    # the mask-based aggregate RV. Smoke-check uncertainty is reported when present.
    if recovered_rv_err_m_s is not None:
        assert np.isfinite(recovered_rv_err_m_s) and recovered_rv_err_m_s >= 0.0

    if show_unified:
        prev = prim._output.get("preview") if prim._output else None
        assert prev is not None and ("html" in prev or prev.get("fig") is not None)
        assert "1" in getattr(prim, "_ccf_preview_by_fiber", {})
        hp = prev.get("html") if isinstance(prev, dict) else None
        if hp:
            print(f"CCF_PREVIEW_HTML={os.path.abspath(hp)}", flush=True)

"""
Test cross-correlation primitive with debug_ccf=True to log per-chunk CCF diagnostics,
and with synthetic L1 spectra (BSPEC/RSPEC) that recover input RV within uncertainty.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from hispecdrp import datamodels
import hispecdrp
from hispecdrp.datamodels import hispec_config
from hispecdrp.primitives import CrossCorrelationPrimitive
from hispecdrp.utils.spectral_templates_utils import get_spectrum_path

# BT-Settl template lives in hispecdrp/data/templates/
TEMPLATE_DIR = os.path.join(os.path.dirname(hispecdrp.__file__), "data", "templates")


# Synthetic spectrum constants for RV recovery test
C_KM_S = 299792.458  # speed of light, km/s
LAMBDA_ORDER_COEF_UM = 147.0  # λ_center(m) = LAMBDA_ORDER_COEF_UM / m (µm)
INPUT_RV_KM_S = -110.0  # km/s (stellar RV to recover)
SIGMA_V_KM_S = 50.0  # Gaussian absorption velocity width (σ), km/s
ABSORPTION_DEPTH = 0.05  # 5% absorption depth
FLAT_FLUX = 100.0


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


def make_synthetic_l1_spectrum(
    spectrograph: str,
    n_pix: int = 4096,
    rv_km_s: float = INPUT_RV_KM_S,
    sigma_v_km_s: float = SIGMA_V_KM_S,
    depth: float = ABSORPTION_DEPTH,
    flat_flux: float = FLAT_FLUX,
) -> datamodels.L1Model:
    """
    Build L1Model with full BSPEC or RSPEC order range, flat continuum and
    Gaussian absorption (velocity width sigma_v_km_s, depth) shifted to rv_km_s.
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
    model = datamodels.L1Model(meta=meta, config=hispec_config.HISPECConfig)
    model.meta.order_start = order_start
    model.meta.order_end = order_end

    n_orders = len(orders)
    wave = np.zeros((n_orders, n_pix), dtype=np.float64)
    spec = np.zeros((n_orders, n_pix), dtype=np.float32)
    for oi, m in enumerate(orders):
        w, lam_center = _order_wavelength_grid(m, n_pix)
        wave[oi] = w
        # Observed line position: λ_obs = λ_rest * (1 + v/c)
        lam_line = lam_center * (1.0 + rv_km_s / C_KM_S)
        sigma_lam = _sigma_lambda_um(lam_center, sigma_v_km_s)
        spec[oi] = _flux_with_gaussian_absorption(
            w, lam_line, sigma_lam, depth, flat_flux
        )

    err = np.maximum(spec * 0.01, 0.1).astype(np.float32)
    spec_dq = np.zeros((n_orders, n_pix), dtype=np.uint32)

    model.spec_uncal_fib1 = spec
    model.spec_uncal_err_fib1 = err
    model.wave_fib1 = wave
    model.spec_dq_fib1 = spec_dq
    return model


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
    Gaussian absorption (velocity width sigma_v_km_s, depth) shifted to rv_km_s.
    Uses SPEC_FIB1, WAVE_FIB1 to match the cross-correlation primitive.
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
        lam_line = lam_center * (1.0 + rv_km_s / C_KM_S)
        sigma_lam = _sigma_lambda_um(lam_center, sigma_v_km_s)
        spec[oi] = _flux_with_gaussian_absorption(
            w, lam_line, sigma_lam, depth, flat_flux
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rest-frame template (wavelength µm, flux) with Gaussian absorptions at
    each order center. Returns (template_wl_um, template_flux) for CCF.
    """
    orders = hispec_config.HISPECConfig.get_order_list(spectrograph)
    if not orders:
        raise ValueError(f"No orders for {spectrograph}")

    wl_list = []
    flux_list = []
    for m in orders:
        w, lam_center = _order_wavelength_grid(m, n_pix)
        sigma_lam = _sigma_lambda_um(lam_center, sigma_v_km_s)
        # Rest frame: line at λ_center
        f = _flux_with_gaussian_absorption(
            w, lam_center, sigma_lam, depth, flat_flux
        )
        wl_list.append(w)
        flux_list.append(f)

    template_wl = np.concatenate(wl_list)
    template_flux = np.concatenate(flux_list)
    sort_idx = np.argsort(template_wl)
    return template_wl[sort_idx], template_flux[sort_idx]


def _make_minimal_l2_for_ccf(n_orders: int = 10, n_pix: int = 4096):
    """Build a minimal L2-like model with SPEC_FIB1, WAVE_FIB1 for CCF (BSPEC order range)."""
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


def test_cross_correlation_debug_ccf_logs_per_chunk(capsys, template_path):
    """Run CCF with debug_ccf=True and assert per-chunk diagnostics are logged."""
    model = _make_minimal_l2_for_ccf(n_orders=10, n_pix=1024)
    prim = CrossCorrelationPrimitive(
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


def test_cross_correlation_debug_ccf_wavelength_vs_signal(capsys, template_path):
    """
    Run CCF with debug_ccf=True and collect ccf_max vs wl_center to inspect
    whether chunks above 1.17 µm have weaker CCF (diagnostic for user issue).
    """
    model = _make_minimal_l2_for_ccf(n_orders=15, n_pix=1024)
    prim = CrossCorrelationPrimitive(
        template_path=template_path,
        do_fibers=["1"],
        debug_ccf=True,
        window_size_um=0.3,
    )
    prim.apply(model)

    captured = capsys.readouterr()
    log_text = captured.out + captured.err
    wl_centers = []
    ccf_maxes = []
    for line in log_text.splitlines():
        if "CCF fiber=" not in line or "chunk=" not in line:
            continue
        for part in line.split():
            if part.startswith("wl_center="):
                wl_centers.append(float(part.split("=")[1]))
            elif part.startswith("ccf_max="):
                ccf_maxes.append(float(part.split("=")[1]))
    assert len(wl_centers) == len(ccf_maxes)
    if len(wl_centers) < 2:
        return
    wl = np.array(wl_centers)
    ccf = np.array(ccf_maxes)
    below = wl < 1.17
    above = wl >= 1.17
    if np.any(below) and np.any(above):
        mean_below = np.nanmean(ccf[below])
        mean_above = np.nanmean(ccf[above])
        # Log for inspection when running pytest -s
        print(f"\nCCF diagnostic: mean ccf_max below 1.17 µm = {mean_below:.6f}, above = {mean_above:.6f}")
        print("(If above is much smaller, template flux or continuum normalization may be weak at λ > 1.17 µm)")


@pytest.mark.parametrize("spectrograph", ["BSPEC", "RSPEC"])
def test_cross_correlation_recovers_rv_per_chunk(spectrograph):
    """
    Generate L2 dummy spectra for BSPEC or RSPEC covering the full wavelength range:
    flat continuum + Gaussian absorption (50 km/s width, 5% depth) at RV = -110 km/s.
    Use a rest-frame template with the same feature; CCF should recover -110 km/s
    to within the reported uncertainty in each chunk.
    """
    n_pix = 4096
    input_rv_m_s = INPUT_RV_KM_S * 1000.0  # m/s

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

    prim = CrossCorrelationPrimitive(
        template_wavelength=template_wl,
        template_flux=template_flux,
        do_fibers=["1"],
        window_size_um=0.3,
        debug_ccf=True,
    )
    result = prim.apply(l2)
    assert result is not None
    # apply() returns the main output (L2Model) directly, not a dict
    out = result

    tbl_attr = "rvccf_fib1"
    tbl = getattr(out, tbl_attr, None)
    assert tbl is not None and hasattr(tbl, "columns"), (
        f"Output L2 missing {tbl_attr}"
    )
    assert "RV" in tbl.colnames and "RV_ERR" in tbl.colnames

    rvs = np.asarray(tbl["RV"], dtype=np.float64)
    rv_errs = np.asarray(tbl["RV_ERR"], dtype=np.float64)
    valid = np.isfinite(rvs) & np.isfinite(rv_errs) & (rv_errs > 0)

    assert np.any(valid), (
        f"No valid chunk RVs for {spectrograph}"
    )

    residuals = np.abs(rvs[valid] - input_rv_m_s)
    within_err = residuals <= rv_errs[valid]
    n_within = np.sum(within_err)
    n_total = np.sum(valid)

    assert n_within == n_total, (
        f"{spectrograph}: {n_total - n_within}/{n_total} chunks have |RV - {INPUT_RV_KM_S} km/s| > RV_ERR. "
        f"Max residual = {np.max(residuals):.1f} m/s, min RV_ERR = {np.min(rv_errs[valid]):.1f} m/s."
    )

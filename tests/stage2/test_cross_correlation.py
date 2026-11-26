"""
Test stellar template CCF (``StellarRVPrimitive`` / ``ccf_stellar_template``) with
``debug_ccf=True`` for per-chunk diagnostics, and synthetic L2 spectra (BSPEC/RSPEC)
that recover input RV within uncertainty.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from hispecdrp import datamodels
from hispecdrp.datamodels import hispec_config
from hispecdrp.primitives import StellarRVPrimitive
from hispecdrp.utils.spectral_template_utils import get_spectral_template_path
from tests.utils import fast_ccf_orders


def _template_ccf_primitive(model, **kwargs) -> StellarRVPrimitive:
    """``StellarRVPrimitive`` configured for template cross-correlation."""
    return StellarRVPrimitive.call(
        model,
        algorithm="ccf_stellar_template",
        save_preview=False,
        **kwargs,
    )


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
    orders: list[int] | None = None,
) -> datamodels.L2Model:
    """
    Build L2Model with BSPEC/RSPEC orders, flat continuum and Gaussian absorption
    shifted to ``rv_km_s``. Pass ``orders`` to limit echelle coverage for faster tests.
    """
    if spectrograph not in ("BSPEC", "RSPEC"):
        raise ValueError("spectrograph must be BSPEC or RSPEC")
    all_orders = hispec_config.HISPECConfig.get_order_list(spectrograph)
    if not all_orders:
        raise ValueError(f"No orders for {spectrograph}")
    if orders is None:
        orders = list(all_orders)
    else:
        orders = [int(o) for o in orders]

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
    orders: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rest-frame template (wavelength µm, flux) with Gaussian absorptions at
    each order center. Returns (template_wl_um, template_flux) for CCF.
    """
    all_orders = hispec_config.HISPECConfig.get_order_list(spectrograph)
    if not all_orders:
        raise ValueError(f"No orders for {spectrograph}")
    orders = list(all_orders) if orders is None else [int(o) for o in orders]

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
    """Resolve BT-Settl template path (may skip if file missing)."""
    path = get_spectral_template_path("lte025.0-5.5-0.0a+0.0.BT-Settl.spec.7.h5")
    if not os.path.isfile(path):
        pytest.skip(f"Template not found: {path}")
    return path


def test_stellar_rv_primitive_debug_ccf_logs_per_chunk(capsys, template_path):
    """Run template CCF with debug_ccf=True and assert per-chunk diagnostics are logged."""
    model = _make_minimal_l2_for_ccf(n_orders=2, n_pix=128)
    out = _template_ccf_primitive(
        model,
        template_path=template_path,
        do_fibers=["1"],
        debug_ccf=True,
        window_size_um=0.5,
    )
    assert out is not None

    captured = capsys.readouterr()
    log_text = captured.out + captured.err
    assert "CCF fiber=" in log_text, "Expected CCF per-chunk log line when debug_ccf=True"
    assert "wl_center=" in log_text
    assert "ccf_max=" in log_text
    assert "n_good=" in log_text
    assert "rv=" in log_text


def test_stellar_rv_primitive_debug_ccf_wavelength_vs_signal(capsys, template_path):
    """Diagnostic: collect ccf_max vs wl_center (legacy debug path)."""
    model = _make_minimal_l2_for_ccf(n_orders=4, n_pix=256)
    out = _template_ccf_primitive(
        model,
        template_path=template_path,
        do_fibers=["1"],
        debug_ccf=True,
        window_size_um=0.3,
    )

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
def test_stellar_rv_primitive_recovers_rv_per_chunk_synthetic(spectrograph):
    """
    Synthetic L2 + rest template: CCF recovers -110 km/s within uncertainty.
    Uses three mid-band orders and 1024 pix (full echelle is covered by slow tests).
    """
    n_pix = 1024
    orders = fast_ccf_orders(spectrograph)
    input_rv_m_s = INPUT_RV_KM_S * 1000.0  # m/s

    l2 = make_synthetic_l2_spectrum(
        spectrograph,
        n_pix=n_pix,
        rv_km_s=INPUT_RV_KM_S,
        sigma_v_km_s=SIGMA_V_KM_S,
        depth=ABSORPTION_DEPTH,
        flat_flux=FLAT_FLUX,
        orders=orders,
    )
    template_wl, template_flux = make_rest_frame_template(
        spectrograph,
        n_pix=n_pix,
        depth=ABSORPTION_DEPTH,
        flat_flux=FLAT_FLUX,
        sigma_v_km_s=SIGMA_V_KM_S,
        orders=orders,
    )

    result = _template_ccf_primitive(
        l2,
        template_wavelength=template_wl,
        template_flux=template_flux,
        do_fibers=["1"],
        do_orders=orders,
        window_size_um=0.3,
        debug_ccf=False,
    )
    assert result is not None
    # apply() returns the main output (L2Model) directly, not a dict
    out = result

    tbl_attr = f"rvccf_fib1"
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

    avg_tbl = getattr(out, "rvccfavg_fib1", None)
    assert avg_tbl is not None and len(avg_tbl) > 0
    rv_agg = float(avg_tbl["RV"][0])
    assert abs(rv_agg - input_rv_m_s) < 50.0, (
        f"{spectrograph}: aggregate RV {rv_agg:.1f} m/s differs from injected "
        f"{input_rv_m_s:.1f} m/s by > 50 m/s"
    )

    residuals = np.abs(rvs[valid] - input_rv_m_s)
    within_3sigma = residuals <= (3.0 * rv_errs[valid])
    frac_within = float(np.mean(within_3sigma))
    assert frac_within >= 0.5, (
        f"{spectrograph}: only {frac_within:.0%} of chunks within 3×RV_ERR of injected RV "
        f"(max residual = {np.max(residuals):.1f} m/s)"
    )

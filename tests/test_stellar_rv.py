"""
Test stellar RV primitive with debug_ccf=True to log per-chunk CCF diagnostics,
and with synthetic L1 spectra (BSPEC/RSPEC) that recover input RV within uncertainty.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from hispecdrp import datamodels
import hispecdrp
from hispecdrp.datamodels import hispec_config
from hispecdrp.primitives import StellarRVPrimitive
from hispecdrp.primitives.utils.ccf_utils import (
    _V_MAX,
    _V_MIN,
    _gaussian_ccf_model,
    fit_gaussian_to_ccf,
    rv_per_chunks_with_ccf,
)
from hispecdrp.utils.spectral_templates_utils import get_spectrum_path, load_btsettl_spectrum

# BT-Settl template lives in hispecdrp/data/templates/
TEMPLATE_DIR = os.path.join(os.path.dirname(hispecdrp.__file__), "data", "templates")


# Synthetic spectrum constants for RV recovery test
C_KM_S = 299792.458  # speed of light, km/s
LAMBDA_ORDER_COEF_UM = 147.0  # λ_center(m) = LAMBDA_ORDER_COEF_UM / m (µm)
INPUT_RV_KM_S = -110.0  # km/s (stellar RV to recover)
# Line width (σ_v); smaller gives a narrower CCF peak in the preview.
SIGMA_V_KM_S = 5.0  # Gaussian absorption velocity width (σ), km/s
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
    Uses SPEC_FIB1, WAVE_FIB1 to match the stellar RV primitive.
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


def test_stellar_rv_debug_ccf_wavelength_vs_signal(capsys, template_path):
    """
    Run CCF with debug_ccf=True and collect ccf_max vs wl_center to inspect
    whether chunks above 1.17 µm have weaker CCF (diagnostic for user issue).
    """
    model = _make_minimal_l2_for_ccf(n_orders=15, n_pix=1024)
    prim = StellarRVPrimitive(
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
def test_stellar_rv_recovers_rv_per_chunk(spectrograph):
    """
    Generate L2 dummy spectra for BSPEC or RSPEC covering the full wavelength range:
    flat continuum + Gaussian absorption (50 km/s width, 5% depth) at RV = -110 km/s.
    Use a rest-frame template with the same feature. The pipeline uses median/MAD
    outlier rejection; the combined RV (RVCCFAVG) must recover the input to within
    tolerance. Outlier chunks are allowed and are masked in the preview (grey points).
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

    show_preview = os.environ.get("HISPEC_DRP_SHOW_CCF_PREVIEW", "").strip().lower() in ("1", "true", "yes")
    prim = StellarRVPrimitive(
        template_wavelength=template_wl,
        template_flux=template_flux,
        do_fibers=["1"],
        window_size_um=0.3,
        debug_ccf=True,
        show_preview=show_preview,
    )
    result = prim.apply(l2)
    assert result is not None
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
    assert np.any(valid), f"No valid chunk RVs for {spectrograph}"

    # Primary check: combined RV (after outlier rejection) recovers input
    avg_tbl = getattr(out, "rvccfavg_fib1", None)
    assert avg_tbl is not None and len(avg_tbl) > 0, f"Output L2 missing rvccfavg_fib1 for {spectrograph}"
    rv_agg = float(avg_tbl["RV"][0])
    rv_agg_err = float(avg_tbl["RV_ERR"][0])
    tolerance_m_s = max(3.0 * rv_agg_err, 100.0)
    assert abs(rv_agg - input_rv_m_s) <= tolerance_m_s, (
        f"{spectrograph}: combined RV {rv_agg/1000:.2f} km/s should be within {tolerance_m_s/1000:.1f} km/s of input {INPUT_RV_KM_S} km/s."
    )

    # Inlier chunks (if INLIER column present) should mostly agree with input; outliers are allowed
    if "INLIER" in tbl.colnames:
        inlier = np.asarray(tbl["INLIER"], dtype=bool)[valid]
        in_ok = inlier
        if np.any(in_ok):
            res_in = np.abs(rvs[valid][in_ok] - input_rv_m_s)
            # Allow inlier chunks to be within 3*RV_ERR or 100 m/s (whichever is larger)
            loose = np.maximum(3.0 * rv_errs[valid][in_ok], 100.0)
            n_ok = np.sum(res_in <= loose)
            assert n_ok >= np.sum(in_ok) * 0.5, (
                f"{spectrograph}: at least half of inlier chunks should have |RV - input| <= 3*RV_ERR or 100 m/s."
            )

    # Sanity: valid chunk RVs and uncertainties are finite and reasonable
    assert np.all(np.isfinite(rvs[valid])), "All valid chunk RVs should be finite"
    assert np.all(rv_errs[valid] > 0), "All valid chunk RV uncertainties should be positive"
    assert np.all(rv_errs[valid] < 1e6), "RV uncertainties should be reasonable (< 1000 km/s)"


def test_fit_gaussian_to_ccf_recovers_center_and_uncertainty():
    """
    Validate that fit_gaussian_to_ccf recovers the true RV and returns a sensible
    uncertainty when given a synthetic Gaussian CCF (velocity vs correlation).
    Models one chunk per BSPEC order (110–148) with wavelength ranges from the
    echelle relation λ_center = 147/order µm, FSR = 147/order² µm.
    Set HISPEC_DRP_SHOW_CCF_PREVIEW=1 to display a preview with a dropdown to select chunk.
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
    for idx, order in enumerate(orders):
        wave, _ = _order_wavelength_grid(order, 100)
        wl_lo = float(wave[0])
        wl_hi = float(wave[-1])
        ccf = amplitude * np.exp(-0.5 * ((v_grid - true_rv) / true_sigma) ** 2) + baseline
        rng = np.random.default_rng(42 + idx)
        ccf += 0.02 * (rng.random(v_grid.size) - 0.5)
        chunk_results.append({
            "order": order,
            "wl_lo": wl_lo,
            "wl_hi": wl_hi,
            "ccf": ccf.copy(),
        })

    # Fit first chunk for assertions (test still validates one fit)
    rv, rv_err = fit_gaussian_to_ccf(
        v_grid, chunk_results[0]["ccf"], v_step
    )
    assert np.isfinite(rv) and np.isfinite(rv_err), "Gaussian fit should return finite RV and uncertainty"
    assert rv_err > 0, "Uncertainty should be positive"
    assert abs(rv - true_rv) < 2 * true_sigma, (
        f"Recovered RV {rv} m/s should be within ~2 sigma of true {true_rv} m/s"
    )
    assert rv_err > 0 and rv_err <= true_sigma * 3, (
        f"RV uncertainty {rv_err} m/s should be positive and ≤ ~3*sigma ({true_sigma})"
    )

    show_preview = os.environ.get("HISPEC_DRP_SHOW_CCF_PREVIEW", "").strip().lower() in ("1", "true", "yes")
    if show_preview:
        # Fit all chunks and build per-chunk metadata for dropdown
        v_fine = np.linspace(v_min, v_max, 800)
        for cr in chunk_results:
            rv_i, rv_err_i, params_i, diag_i = fit_gaussian_to_ccf(
                v_grid, cr["ccf"], v_step, return_params=True
            )
            cr["rv"] = rv_i
            cr["rv_err"] = rv_err_i
            cr["params"] = params_i
            cr["diagnostics"] = diag_i
            if params_i:
                v_lo_fit = diag_i.get("fit_v_lo", v_min)
                v_hi_fit = diag_i.get("fit_v_hi", v_max)
                v_fit = np.linspace(v_lo_fit, v_hi_fit, 200)
                cr["ccf_fit"] = _gaussian_ccf_model(
                    v_fit,
                    params_i["amplitude"],
                    params_i["v0"],
                    params_i["sigma"],
                    params_i["baseline"],
                )
                cr["v_fit"] = v_fit
            else:
                cr["ccf_fit"] = np.array([])
                cr["v_fit"] = np.array([])

        # Optimal combination: mean CCF across orders (same RV assumption -> variance reduction)
        combined_ccf = np.nanmean(np.array([cr["ccf"] for cr in chunk_results], dtype=np.float64), axis=0)
        rv_comb_fit, rv_err_comb_fit, params_comb, diag_comb = fit_gaussian_to_ccf(
            v_grid, combined_ccf, v_step, return_params=True
        )
        # Combined RV and uncertainty from same method as per-order: fit to (mean) CCF, no floor
        combined_result = {
            "rv": rv_comb_fit,
            "rv_err": rv_err_comb_fit,
            "params": params_comb,
            "diagnostics": diag_comb,
        }
        if params_comb:
            v_lo_fit = diag_comb.get("fit_v_lo", v_min)
            v_hi_fit = diag_comb.get("fit_v_hi", v_max)
            v_fit_comb = np.linspace(v_lo_fit, v_hi_fit, 200)
            combined_result["ccf_fit"] = _gaussian_ccf_model(
                v_fit_comb,
                params_comb["amplitude"],
                params_comb["v0"],
                params_comb["sigma"],
                params_comb["baseline"],
            )
            combined_result["v_fit"] = v_fit_comb
        else:
            combined_result["ccf_fit"] = np.array([])
            combined_result["v_fit"] = np.array([])

        import plotly.graph_objects as go

        def make_annotation_text(cr, idx, is_combined=False):
            if is_combined:
                p, d = cr.get("params") or {}, cr.get("diagnostics") or {}
                if p:
                    n_ord = len(chunk_results)
                    return (
                        f"<b>Combined</b> (all BSPEC orders, N={n_ord})<br>"
                        f"<b>RV</b> = {cr['rv']:.3f} m/s (from fit to mean CCF)<br>"
                        f"<b>RV uncertainty</b> = {cr['rv_err']:.3f} m/s (empirical, no floor)<br>"
                        f"v0 = {p['v0']:.3f}, σ = {p['sigma']:.3f} m/s<br>"
                        f"amplitude = {p['amplitude']:.4f}, baseline = {p['baseline']:.4f}<br>"
                        f"<i>Y-axis: CCF = weighted cov.(spec, template), not normalized; flux².</i><br>"
                        f"<b>Error source:</b> {d.get('message', '—')}"
                    )
                return (
                    f"<b>Combined</b> (all BSPEC orders)<br>"
                    f"<b>Fit status:</b> FAILED<br>"
                    f"<b>Message:</b> {d.get('message', '—')}"
                )
            order = cr.get("order", idx)
            wl_lo, wl_hi = cr["wl_lo"], cr["wl_hi"]
            p, d = cr.get("params") or {}, cr.get("diagnostics") or {}
            if p:
                return (
                    f"<b>Chunk {idx}</b> (BSPEC order {order})<br>"
                    f"λ {wl_lo:.4f}–{wl_hi:.4f} µm<br>"
                    f"<b>Fit status:</b> OK<br>"
                    f"<b>RV (v0)</b> = {p['v0']:.3f} m/s<br>"
                    f"<b>RV uncertainty</b> = {cr['rv_err']:.3f} m/s<br>"
                    f"amplitude = {p['amplitude']:.4f}<br>"
                    f"sigma = {p['sigma']:.1f} m/s<br>"
                    f"baseline = {p['baseline']:.4f}<br>"
                    f"<b>Error source:</b> {d.get('message', '—')}"
                )
            return (
                f"<b>Chunk {idx}</b> (BSPEC order {order})<br>"
                f"λ {wl_lo:.4f}–{wl_hi:.4f} µm<br>"
                f"<b>Fit status:</b> FAILED<br>"
                f"<b>Message:</b> {d.get('message', '—')}"
            )

        n_chunks = len(chunk_results)
        fig = go.Figure()
        # Trace 0,1: combined CCF (default visible)
        fig.add_trace(
            go.Scatter(
                x=v_grid,
                y=combined_ccf,
                mode="markers",
                name="Combined data",
                marker=dict(size=4, symbol="circle"),
                visible=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=combined_result.get("v_fit", np.array([])),
                y=combined_result["ccf_fit"],
                mode="lines",
                name="Combined fit",
                line=dict(width=2),
                visible=True,
            )
        )
        # Traces 2..2*n_chunks+1: per-order [data, fit]; fit only over fitted velocity range
        for idx, cr in enumerate(chunk_results):
            order = cr["order"]
            fig.add_trace(
                go.Scatter(
                    x=v_grid,
                    y=cr["ccf"],
                    mode="markers",
                    name=f"Order {order} data",
                    marker=dict(size=4, symbol="circle"),
                    visible=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=cr.get("v_fit", np.array([])),
                    y=cr["ccf_fit"],
                    mode="lines",
                    name=f"Order {order} fit",
                    line=dict(width=2),
                    visible=False,
                )
            )

        n_traces = 2 + 2 * n_chunks
        default_annotation = dict(
            text=make_annotation_text(combined_result, 0, is_combined=True),
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.8)",
        )
        fig.add_annotation(**default_annotation)

        # Dropdown: Combined (default) then one button per order
        buttons = []
        # Combined: show traces 0,1 only
        visible_combined = [False] * n_traces
        visible_combined[0] = True
        visible_combined[1] = True
        buttons.append(
            dict(
                label="Combined (all orders)",
                method="update",
                args=[
                    {"visible": visible_combined},
                    {"annotations": [dict(
                        default_annotation,
                        text=make_annotation_text(combined_result, 0, is_combined=True),
                    )]},
                ],
            )
        )
        for idx in range(n_chunks):
            visible = [False] * n_traces
            visible[2 + 2 * idx] = True
            visible[2 + 2 * idx + 1] = True
            cr = chunk_results[idx]
            buttons.append(
                dict(
                    label=f"Order {cr['order']} (λ {cr['wl_lo']:.3f}–{cr['wl_hi']:.3f} µm)",
                    method="update",
                    args=[
                        {"visible": visible},
                        {"annotations": [dict(
                            default_annotation,
                            text=make_annotation_text(chunk_results[idx], idx),
                        )]},
                    ],
                )
            )
        fig.update_layout(
            title="Gaussian fit to CCF: combined or per BSPEC order (select from dropdown)",
            xaxis_title="Velocity shift (m/s)",
            yaxis_title="CCF (weighted covariance, not normalized; units of flux²)",
            xaxis=dict(range=[v_min, v_max]),
            height=500,
            showlegend=True,
            updatemenus=[
                dict(
                    active=0,
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=0.0,
                    xanchor="left",
                    y=1.12,
                    yanchor="top",
                )
            ],
        )
        fig.show()


def test_btsettl_shifted_polynomial_l2_recovers_rv_with_preview(template_path):
    """
    Load BT-Settl template, shift by a known RV, multiply by a 4th-order polynomial,
    put into a BSPEC L2 model, run stellar RV (CCF) with default algorithm/parameters,
    output Gaussian preview (per order + combined), and validate input RV is recovered.

    Set HISPEC_DRP_SHOW_CCF_PREVIEW=1 to display the dropdown preview.
    """
    input_rv_km_s = -110.0  # km/s (same as other tests)
    input_rv_m_s = input_rv_km_s * 1000.0
    n_pix = 4096
    v_step = 500.0  # m/s
    v_min, v_max = _V_MIN, _V_MAX
    window_size_um = 0.3  # default

    # BSPEC wavelength range (orders 110–148): ~0.99–1.34 µm
    wl_min_um, wl_max_um = 0.95, 1.4
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

    # Run stellar RV with default algorithm and parameters (template_path = unshifted BT-Settl)
    prim = StellarRVPrimitive(
        template_path=template_path,
        do_fibers=["1"],
        window_size_um=window_size_um,
    )
    out = prim.apply(model)
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

    # Build Gaussian preview: per-order CCF (mean over chunks) + combined, fit Gaussian, dropdown
    # CCF API: mask = True for BAD pixels (excluded); good = ~mask & finite(spec) & err>0
    template_ln = np.log(w_rest)
    do_not_use = 0
    chunk_results = []
    for oi, order in enumerate(orders):
        wave_ord = wave[oi]
        spec_ord = spec[oi]
        err_ord = spec_err[oi]
        dq_ord = spec_dq[oi]
        good_pixels = (dq_ord == do_not_use) & (err_ord > 0) & np.isfinite(spec_ord)
        mask_bad = ~good_pixels  # True = bad, so CCF uses good = ~mask_bad
        rvs_o, rv_errs_o, chunk_wl, v_grid, ccf_2d = rv_per_chunks_with_ccf(
            wave_ord,
            spec_ord,
            err_ord,
            mask_bad,
            template_ln,
            flux_rest,
            window_size_um=window_size_um,
            v_min=v_min,
            v_max=v_max,
            v_step=v_step,
        )
        ccf_order = np.nanmean(ccf_2d, axis=0)
        wl_lo = float(np.nanmin(wave_ord))
        wl_hi = float(np.nanmax(wave_ord))
        chunk_results.append({
            "order": order,
            "wl_lo": wl_lo,
            "wl_hi": wl_hi,
            "ccf": ccf_order,
        })

    v_grid = np.arange(v_min, v_max + v_step * 0.5, v_step, dtype=np.float64)
    for cr in chunk_results:
        rv_i, rv_err_i, params_i, diag_i = fit_gaussian_to_ccf(
            v_grid, cr["ccf"], v_step, return_params=True
        )
        cr["rv"] = rv_i
        cr["rv_err"] = rv_err_i
        cr["params"] = params_i
        cr["diagnostics"] = diag_i
        if params_i:
            v_lo_fit = diag_i.get("fit_v_lo", v_min)
            v_hi_fit = diag_i.get("fit_v_hi", v_max)
            v_fit = np.linspace(v_lo_fit, v_hi_fit, 200)
            cr["ccf_fit"] = _gaussian_ccf_model(
                v_fit,
                params_i["amplitude"],
                params_i["v0"],
                params_i["sigma"],
                params_i["baseline"],
            )
            cr["v_fit"] = v_fit
        else:
            cr["ccf_fit"] = np.array([])
            cr["v_fit"] = np.array([])

    ccf_array = np.array([cr["ccf"] for cr in chunk_results], dtype=np.float64)
    # Avoid "Mean of empty slice" when some orders have all-NaN CCF
    if ccf_array.size > 0 and np.any(np.isfinite(ccf_array)):
        combined_ccf = np.nanmean(ccf_array, axis=0)
    else:
        combined_ccf = np.full(ccf_array.shape[1] if ccf_array.ndim > 1 else len(v_grid), np.nan)
    rv_comb_fit, rv_err_comb_fit, params_comb, diag_comb = fit_gaussian_to_ccf(
        v_grid, combined_ccf, v_step, return_params=True
    )
    combined_result = {
        "rv": rv_comb_fit,
        "rv_err": rv_err_comb_fit,
        "params": params_comb,
        "diagnostics": diag_comb,
    }
    if params_comb:
        v_lo_fit = diag_comb.get("fit_v_lo", v_min)
        v_hi_fit = diag_comb.get("fit_v_hi", v_max)
        v_fit_comb = np.linspace(v_lo_fit, v_hi_fit, 200)
        combined_result["ccf_fit"] = _gaussian_ccf_model(
            v_fit_comb,
            params_comb["amplitude"],
            params_comb["v0"],
            params_comb["sigma"],
            params_comb["baseline"],
        )
        combined_result["v_fit"] = v_fit_comb
    else:
        combined_result["ccf_fit"] = np.array([])
        combined_result["v_fit"] = np.array([])

    show_preview = os.environ.get("HISPEC_DRP_SHOW_CCF_PREVIEW", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if show_preview:
        import plotly.graph_objects as go

        def make_annotation_text(cr, idx, is_combined=False):
            if is_combined:
                p, d = cr.get("params") or {}, cr.get("diagnostics") or {}
                if p:
                    n_ord = len(chunk_results)
                    return (
                        f"<b>Combined</b> (all BSPEC orders, N={n_ord})<br>"
                        f"<b>RV</b> = {cr['rv']:.3f} m/s (from fit to mean CCF)<br>"
                        f"<b>RV uncertainty</b> = {cr['rv_err']:.3f} m/s<br>"
                        f"Input RV = {input_rv_km_s} km/s<br>"
                        f"v0 = {p['v0']:.3f}, σ = {p['sigma']:.3f} m/s<br>"
                        f"amplitude = {p['amplitude']:.4f}, baseline = {p['baseline']:.4f}<br>"
                        f"<i>BT-Settl template, shifted + 4th-order polynomial.</i>"
                    )
                return (
                    f"<b>Combined</b> (all BSPEC orders)<br>"
                    f"<b>Fit status:</b> FAILED<br>"
                    f"<b>Message:</b> {d.get('message', '—')}"
                )
            order = cr.get("order", idx)
            wl_lo, wl_hi = cr["wl_lo"], cr["wl_hi"]
            p, d = cr.get("params") or {}, cr.get("diagnostics") or {}
            if p:
                return (
                    f"<b>Order {order}</b><br>"
                    f"λ {wl_lo:.4f}–{wl_hi:.4f} µm<br>"
                    f"<b>RV (v0)</b> = {p['v0']:.3f} m/s<br>"
                    f"<b>RV uncertainty</b> = {cr['rv_err']:.3f} m/s<br>"
                    f"amplitude = {p['amplitude']:.4f}, sigma = {p['sigma']:.1f} m/s<br>"
                )
            return (
                f"<b>Order {order}</b><br>"
                f"λ {wl_lo:.4f}–{wl_hi:.4f} µm<br>"
                f"<b>Fit status:</b> FAILED<br>"
                f"<b>Message:</b> {d.get('message', '—')}"
            )

        n_chunks = len(chunk_results)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=v_grid,
                y=combined_ccf,
                mode="markers",
                name="Combined data",
                marker=dict(size=4, symbol="circle"),
                visible=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=combined_result.get("v_fit", np.array([])),
                y=combined_result["ccf_fit"],
                mode="lines",
                name="Combined fit",
                line=dict(width=2),
                visible=True,
            )
        )
        for idx, cr in enumerate(chunk_results):
            fig.add_trace(
                go.Scatter(
                    x=v_grid,
                    y=cr["ccf"],
                    mode="markers",
                    name=f"Order {cr['order']} data",
                    marker=dict(size=4, symbol="circle"),
                    visible=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=cr.get("v_fit", np.array([])),
                    y=cr["ccf_fit"],
                    mode="lines",
                    name=f"Order {cr['order']} fit",
                    line=dict(width=2),
                    visible=False,
                )
            )
        n_traces = 2 + 2 * n_chunks
        default_annotation = dict(
            text=make_annotation_text(combined_result, 0, is_combined=True),
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.8)",
        )
        fig.add_annotation(**default_annotation)
        buttons = []
        visible_combined = [False] * n_traces
        visible_combined[0] = True
        visible_combined[1] = True
        buttons.append(
            dict(
                label="Combined (all orders)",
                method="update",
                args=[
                    {"visible": visible_combined},
                    {
                        "annotations": [
                            dict(
                                default_annotation,
                                text=make_annotation_text(combined_result, 0, is_combined=True),
                            )
                        ]
                    },
                ],
            )
        )
        for idx in range(n_chunks):
            visible = [False] * n_traces
            visible[2 + 2 * idx] = True
            visible[2 + 2 * idx + 1] = True
            cr = chunk_results[idx]
            buttons.append(
                dict(
                    label=f"Order {cr['order']} (λ {cr['wl_lo']:.3f}–{cr['wl_hi']:.3f} µm)",
                    method="update",
                    args=[
                        {"visible": visible},
                        {
                            "annotations": [
                                dict(
                                    default_annotation,
                                    text=make_annotation_text(chunk_results[idx], idx),
                                )
                            ]
                        },
                    ],
                )
            )
        fig.update_layout(
            title="BT-Settl shifted + polynomial in L2: Gaussian fit to CCF (per order + combined)",
            xaxis_title="Velocity shift (m/s)",
            yaxis_title="CCF (weighted covariance, not normalized; units of flux²)",
            xaxis=dict(range=[v_min, v_max]),
            height=500,
            showlegend=True,
            updatemenus=[
                dict(
                    active=0,
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=0.0,
                    xanchor="left",
                    y=1.12,
                    yanchor="top",
                )
            ],
        )
        fig.show()

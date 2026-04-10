"""
Stellar RV tests: **ccf_stellar_template** (log-λ CCF, median/MAD rejection, IV weighted mean).

Synthetic L2 uses :func:`_order_wavelength_grid` and Gaussian lines; rest templates use
**3×** L2 pixels per order to reduce log-λ CCF bias.

Run::

  pytest tests/test_stellar_rv.py -v

Faster CI / local iteration (skips tests marked ``slow``: BT-Settl parallel integration + injection sweep recovery)::

  pytest tests/test_stellar_rv.py -v -m "not slow"

**Optional previews (env or module variables):**

- ``HISPEC_DRP_SHOW_CCF_PREVIEW=1`` — default stellar RV Plotly preview (RV vs λ, CCF + Gaussian
  fits, residuals) in ``test_stellar_rv_recovers_rv_per_chunk`` (or ``SHOW_PRIMITIVE_CCF_PREVIEW``).
- ``HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS=1`` — after ``test_fit_ccf_kpf_style_btsettl_bspec_all_chunks_parallel``
  assertions, run lte049 + template-derived ``lte049.0*.mas`` ``ccf_mask_kpf``: **HTML-only** previews (no ``fig.show()``):
  mask-overlay via ``write_mask_overlay_debug_pages`` (auto-open) and stellar RV preview via
  ``create_stellar_rv_preview`` → unified ``*_preview.html`` (RV vs λ, CCF, residual per fiber; CCF slot
  dropdown), then ``webbrowser.open``. Prints a step timing
  summary (``-s``).

**Parallel BT-Settl BSPEC chunk test** — ``test_fit_ccf_kpf_style_btsettl_bspec_all_chunks_parallel``
runs the same science setup via ``process_orders_rvs_numba_parallel`` (requests ``max_cores=6``;
actual Numba threads are clamped to the wheel/CPU limit, e.g. GitHub Actions) and prints a timing summary.
By default it uses **one** echelle order (**139**). To run on **all** BSPEC echelle orders from
``HISPECConfig``, either set ``HISPEC_DRP_BSPEC_BTSETTL_ALL_ORDERS=1`` or pass pytest flag
``--bspec-all-orders`` (the flag sets that env var). Use ``HISPEC_DRP_BSPEC_BTSETTL_SINGLE_ORDER=<n>``
to pick a different single order instead.

Simulated ``spec_err`` uses a center-to-edge throughput taper so SNR is high in the order center
and degrades toward the blue/red ends (see ``_bspec_simulated_flux_err``).

**One-liner — mask-KPF parallel test + both previews (mask overlay + unified CCF/RV HTML), from repo root**::

  HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS=1 pytest tests/test_stellar_rv.py::test_fit_ccf_kpf_style_btsettl_bspec_all_chunks_parallel -s

**Same, but all BSPEC echelle orders** (longer runtime)::

  HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS=1 pytest tests/test_stellar_rv.py::test_fit_ccf_kpf_style_btsettl_bspec_all_chunks_parallel -s --bspec-all-orders

  # Equivalent:
  HISPEC_DRP_BSPEC_BTSETTL_ALL_ORDERS=1 HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS=1 pytest tests/test_stellar_rv.py::test_fit_ccf_kpf_style_btsettl_bspec_all_chunks_parallel -s

**One-liner — same test, JSON + NPZ CCF debug dump only** (no mask-overlay / Plotly HTML; patched no-op preview)::

  HISPEC_DRP_BSPEC_BTSETTL_SERIALIZE_MASK_CCF=1 pytest tests/test_stellar_rv.py::test_fit_ccf_kpf_style_btsettl_bspec_all_chunks_parallel -s

``HISPEC_DRP_SAVE_CCF_PLOTS=1`` sets ``save_preview`` and ``output_dir`` for the echelle test so the
primitive writes the unified preview HTML under ``tests/output/stellar_rv_plots/`` (no extra figures).

``HISPEC_DRP_BSPEC_BTSETTL_SERIALIZE_MASK_CCF=1`` — during ``test_fit_ccf_kpf_style_btsettl_bspec_all_chunks_parallel``,
after the lte049 ``ccf_mask_kpf`` preview path (or a lightweight variant if previews are off), write JSON + NPZ
debug dumps of per-order/mean ``v`` (m/s), ``ccf``, ``ccf_err`` (if present), and ``kpf_fit`` (Gaussian RV / err /
params). Prints ``HISPEC_DRP_BSPEC_BTSETTL_MASK_CCF_SERIALIZE=`` with the JSON path (the JSON references the NPZ).

**Post-process CCF JSON (optional)** — ``examples/mask_kpf_ccf_postprocess_uncertainty.py`` re-runs the high-pass
σ estimate and ``fit_ccf_rv`` on serialized debug files (e.g. different Savitzky–Golay settings or archived runs). Mask-KPF
order-mean ``ccf_err`` in the pipeline is already produced by
:func:`~hispecdrp.primitives.utils.ccf_utils.estimate_ccf_uncertainty_highpass_residual` inside
:func:`~hispecdrp.primitives.utils.ccf_utils.run_ccf_mask_kpf_for_fiber`. Prints ``HISPEC_DRP_MASK_CCF_POSTPROCESS_JSON=``::

  PYTHONPATH=. python examples/mask_kpf_ccf_postprocess_uncertainty.py /path/to/lte049_mask_kpf_ccf_rv_debug.json

**Speed / coverage toggles (optional env):**

- ``HISPEC_DRP_BTSETTL_RV_SWEEP_FULL_INJECTIONS=1`` — injection sweep uses 11 RV steps (−105 … −125 km/s
  step 2); default is 6 steps (step 4) for faster CI.
- ``HISPEC_DRP_BTSETTL_RV_SWEEP_ALL_ORDERS=1`` — injection sweep uses **all** BSPEC echelle orders;
  default is a **single** order (same default as mask-KPF BT tests) for speed.

**BT-Settl RV injection sweep (Plotly)** — ``test_btsettl_rv_injection_sweep_recovery`` uses the
**lte049** BT-Settl template with the template-derived ``lte049.0*.mas`` mask: per-order
``rv_per_chunks_mask_kpf`` + median/MAD + IV mean on a **merged hybrid** velocity grid (broad search
plus ±300 m/s @ 0.1 m/s around each injection), matching :class:`StellarRVPrimitive` when
``ccf_input_velocity`` refines the CCF grid. Writes Plotly HTML under ``tmp_path``.
"""
from __future__ import annotations

import glob
import json
import math
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
    CCFFitKpfConfig,
    MaskLines,
    _V_MAX,
    _V_MIN,
    build_hybrid_merged_velocity_grid,
    combine_rvs_median_mad,
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

# Mask-line ``ccf_mask_kpf`` BT-Settl tests: default to this echelle order unless
# ``HISPEC_DRP_BSPEC_BTSETTL_ALL_ORDERS=1`` (use full BSPEC list) or
# ``HISPEC_DRP_BSPEC_BTSETTL_SINGLE_ORDER=<n>`` (override default).
_DEFAULT_BSPEC_BTSETTL_CCF_MASK_KPF_ORDER = 139
# lte049 + polynomial L2 (same construction as mask-KPF smoke scripts)
_BTSETTL_LTE049_POLY_COEFS = np.array([0.0001, -0.001, 0.01, 0.05, 1.0])
MASK_DIR = os.path.join(os.path.dirname(hispecdrp.__file__), "data", "masks")

# --- Test-only performance knobs (same assertions / tolerances; coarser grids are fine for loose RV tests) ---
# Fewer pixels per echelle row cuts CCF cost ~linearly; 2048 still satisfies ≥20 px/chunk with narrow windows.
_TEST_ECHELLE_N_PIX = 2048
# Coarser uniform broad grid for template-CCF and mask-CCF *test* paths (production defaults often 50 m/s).
_TEST_TEMPLATE_CCF_V_STEP_M_S = 100.0
_TEST_MASK_CCF_BROAD_STEP_M_S = 100.0
# Injection sweep: 6 steps (−105 … −125 km/s, step 4); full 11-point sweep via
# ``HISPEC_DRP_BTSETTL_RV_SWEEP_FULL_INJECTIONS=1``. Single echelle order by default; all orders via
# ``HISPEC_DRP_BTSETTL_RV_SWEEP_ALL_ORDERS=1``.
_BTSETTL_INJECTION_KM_S_FAST = np.arange(-105, -126, -4, dtype=np.float64)
_BTSETTL_INJECTION_KM_S_FULL = np.arange(-105, -126, -2, dtype=np.float64)


def _btsettl_injection_km_s_grid() -> np.ndarray:
    if _truthy_env("HISPEC_DRP_BTSETTL_RV_SWEEP_FULL_INJECTIONS"):
        return _BTSETTL_INJECTION_KM_S_FULL
    return _BTSETTL_INJECTION_KM_S_FAST


def _btsettl_injection_sweep_default_single_order() -> int | None:
    """``None`` = all BSPEC echelle rows (slow); otherwise default single order for speed."""
    if _truthy_env("HISPEC_DRP_BTSETTL_RV_SWEEP_ALL_ORDERS"):
        return None
    return int(_DEFAULT_BSPEC_BTSETTL_CCF_MASK_KPF_ORDER)


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
SERIALIZE_BSPEC_BTSETTL_MASK_CCF: bool | None = None


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


def _want_bspec_btsettl_serialize_mask_ccf() -> bool:
    """Serialize mask-KPF CCF preview arrays (``v``, ``ccf``, ``ccf_err``, ``kpf_fit``) to JSON+NPZ."""
    if SERIALIZE_BSPEC_BTSETTL_MASK_CCF is not None:
        return SERIALIZE_BSPEC_BTSETTL_MASK_CCF
    return _truthy_env("HISPEC_DRP_BSPEC_BTSETTL_SERIALIZE_MASK_CCF")


def _bspec_btsettl_ccf_mask_kpf_order_ids() -> list[int]:
    """Echelle orders for BSPEC mask-KPF tests (default single order: :data:`_DEFAULT_BSPEC_BTSETTL_CCF_MASK_KPF_ORDER`)."""
    full = hispec_config.HISPECConfig.get_order_list("BSPEC")
    if not full:
        pytest.skip("No BSPEC orders in HISPECConfig")
    if _truthy_env("HISPEC_DRP_BSPEC_BTSETTL_ALL_ORDERS"):
        return list(full)
    raw = os.environ.get(
        "HISPEC_DRP_BSPEC_BTSETTL_SINGLE_ORDER",
        str(_DEFAULT_BSPEC_BTSETTL_CCF_MASK_KPF_ORDER),
    ).strip()
    o = int(raw)
    if o not in full:
        pytest.skip(
            f"BSPEC echelle order {o} not in configured list ({full[0]} … {full[-1]})"
        )
    return [o]


def _mask_kpf_preview_do_orders() -> list[int] | None:
    """``do_orders`` for :class:`StellarRVPrimitive` previews: ``None`` means all echelle orders."""
    if _truthy_env("HISPEC_DRP_BSPEC_BTSETTL_ALL_ORDERS"):
        return None
    return _bspec_btsettl_ccf_mask_kpf_order_ids()


def _btsettl_ccf_mask_kpf_window_um(wave_first_order_1d: np.ndarray) -> float:
    """
    Wavelength chunk width for mask-KPF BT-Settl tests. Single-order runs use a smaller window
    so one echelle order produces many chunks (keeps ``n_attempt`` assertions meaningful).
    """
    if _truthy_env("HISPEC_DRP_BSPEC_BTSETTL_ALL_ORDERS"):
        return WINDOW_UM
    span = float(np.ptp(np.asarray(wave_first_order_1d, dtype=np.float64)))
    if not np.isfinite(span) or span <= 0:
        return WINDOW_UM
    return float(min(WINDOW_UM, max(1e-6, span / 16.0)))


def _print_btsettl_test_timing_summary(
    rows: list[tuple[str, float, str, bool]],
    *,
    title: str = "BT-Settl BSPEC stellar RV test — wall time by step (seconds)",
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


# Echelle throughput → 0 at order edges: scale σ so effective SNR is high in the center only.
_BSPEC_EDGE_THROUGHPUT_FLOOR = 0.02


def _bspec_order_edge_snr_loss_factor(n_pix: int) -> np.ndarray:
    """
    Length-``n_pix`` multiplier (≥1) applied to flux errors at the blue/red ends of the order.

    Uses :math:`T(u)=\\sin^2(\\pi u)` along normalized pixel coordinate :math:`u\\in[0,1]`,
    with a floor so σ grows by up to :math:`\\sim 1/` :data:`_BSPEC_EDGE_THROUGHPUT_FLOOR` at the edges.
    """
    n = int(n_pix)
    u = np.linspace(0.0, 1.0, n, dtype=np.float64)
    throughput = np.sin(np.pi * u) ** 2
    return 1.0 / np.maximum(throughput, float(_BSPEC_EDGE_THROUGHPUT_FLOOR))


def _bspec_simulated_flux_err(
    spec: np.ndarray,
    *,
    rel_floor: float = 0.01,
    abs_floor: float = 0.1,
    dtype_out=np.float64,
) -> np.ndarray:
    """
    Simulated flux uncertainty for BSPEC-style echelle tests: proportional noise + constant
    floor in the order center, inflated toward pixels 0 and ``n_pix-1`` (poor SNR at edges).
    """
    s = np.asarray(spec, dtype=np.float64)
    base = np.maximum(np.abs(s) * float(rel_floor), float(abs_floor))
    npix = int(s.shape[-1])
    edge = _bspec_order_edge_snr_loss_factor(npix)
    if s.ndim == 1:
        out = base * edge
    elif s.ndim == 2:
        out = base * edge[np.newaxis, :]
    else:
        raise ValueError("spec must be 1-D or 2-D")
    return np.asarray(out, dtype=dtype_out)


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
    model.spec_err_fib1 = _bspec_simulated_flux_err(spec, dtype_out=np.float32)
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
    model.spec_err_fib1 = _bspec_simulated_flux_err(spec, dtype_out=np.float32)
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


def _jsonify_mask_ccf_debug(obj: object) -> object:
    """Convert preview payloads (numpy, nested dicts) to JSON-safe structures."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonify_mask_ccf_debug(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify_mask_ccf_debug(x) for x in obj]
    if isinstance(obj, int):
        return int(obj)
    return str(obj)


def _serialize_mask_kpf_ccf_preview_debug(prim: StellarRVPrimitive, out_dir: str) -> str | None:
    """
    Write per-fiber mask-KPF CCF preview data: large arrays in ``.npz``, metadata + ``kpf_fit`` in ``.json``.

    Returns absolute path to the JSON sidecar, or ``None`` if there is nothing to save.
    """
    by_fiber = getattr(prim, "_ccf_preview_by_fiber", None) or {}
    if not by_fiber:
        print(
            "HISPEC_DRP_BSPEC_BTSETTL_MASK_CCF_SERIALIZE=<empty: no _ccf_preview_by_fiber; "
            "use save_preview or mask-overlay debug so mask-KPF collects CCF preview>",
            flush=True,
        )
        return None
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "lte049_mask_kpf_ccf_rv_debug")
    npz_path = base + ".npz"
    json_path = base + ".json"

    npz_arrays: dict[str, np.ndarray] = {}
    fibers_meta: dict[str, object] = {}
    for fib_key, detail in by_fiber.items():
        fib_s = str(fib_key)
        slots_out: list[dict[str, object]] = []
        slots = (detail or {}).get("slots") or []
        for si, sl in enumerate(slots):
            if not isinstance(sl, dict):
                continue
            kind = sl.get("kind")
            order = sl.get("order")
            prefix = f"fib{fib_s}_s{si}_{kind}_order{order}"
            v = np.asarray(sl.get("v"), dtype=np.float64)
            ccf = np.asarray(sl.get("ccf"), dtype=np.float64)
            npz_arrays[f"{prefix}_v_m_s"] = v
            npz_arrays[f"{prefix}_ccf"] = ccf
            if sl.get("ccf_err") is not None:
                npz_arrays[f"{prefix}_ccf_err"] = np.asarray(sl["ccf_err"], dtype=np.float64)
            kpf = sl.get("kpf_fit")
            slot_meta: dict[str, object] = {
                "kind": kind,
                "order": None if order is None else int(order),
                "wl_lo": float(sl["wl_lo"]) if sl.get("wl_lo") is not None else None,
                "wl_hi": float(sl["wl_hi"]) if sl.get("wl_hi") is not None else None,
                "npz_keys": {
                    "v_m_s": f"{prefix}_v_m_s",
                    "ccf": f"{prefix}_ccf",
                    **(
                        {"ccf_err": f"{prefix}_ccf_err"}
                        if f"{prefix}_ccf_err" in npz_arrays
                        else {}
                    ),
                },
                "kpf_fit": _jsonify_mask_ccf_debug(kpf) if kpf is not None else None,
            }
            slots_out.append(slot_meta)
        fibers_meta[fib_s] = {
            "v_step": (detail or {}).get("v_step"),
            "v_min": (detail or {}).get("v_min"),
            "v_max": (detail or {}).get("v_max"),
            "slots": slots_out,
        }

    payload = {
        "schema": "hispecdrp.mask_kpf_ccf_preview.v1",
        "npz_path": os.path.abspath(npz_path),
        "fibers": fibers_meta,
    }
    np.savez_compressed(npz_path, **npz_arrays)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify_mask_ccf_debug(payload), f, indent=2, allow_nan=False)
    ap = os.path.abspath(json_path)
    print(f"HISPEC_DRP_BSPEC_BTSETTL_MASK_CCF_SERIALIZE={ap}", flush=True)
    return ap


def _run_lte049_k2all_mask_kpf_previews(
    tmp_path: str,
    *,
    mask_path: str | None = None,
    do_orders: list[int] | None = None,
    serialize_only: bool = False,
) -> list[tuple[str, float, str, bool]]:
    """
    BSPEC L2 from lte049 + Doppler + polynomial; ``ccf_mask_kpf`` with a packaged ``.mas`` mask.

    ``mask_path`` defaults to ``K2_all.mas`` (serial BT-Settl preview). Pass the template-derived
    ``lte049.0*.mas`` path for the parallel test.

    ``do_orders`` is forwarded to :class:`StellarRVPrimitive` (``None`` = all echelle orders). When
    ``None``, uses :func:`_mask_kpf_preview_do_orders` (same selection as mask-KPF BT-Settl tests:
    default single order 139 unless ``HISPEC_DRP_BSPEC_BTSETTL_ALL_ORDERS=1``).

    When ``serialize_only`` is False (default): HTML-only — ``save_preview=True``, ``show_preview=False``
    so ``create_stellar_rv_preview`` writes ``*_preview.html`` without ``fig.show()``;
    the test then ``webbrowser.open``s that file. Mask overlay uses ``write_mask_overlay_debug_pages`` (HTML + auto-open
    when ``show_mask_overlay_debug=True``).

    When ``serialize_only`` is True: ``save_preview=True`` so mask-KPF collects CCF preview arrays, but
    ``create_stellar_rv_preview`` is patched to a no-op (no unified Plotly HTML); mask-overlay HTML is not
    written. Used when ``HISPEC_DRP_BSPEC_BTSETTL_SERIALIZE_MASK_CCF=1`` without full previews.

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
    n_pix = _TEST_ECHELLE_N_PIX
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
    if do_orders is None:
        do_orders = _mask_kpf_preview_do_orders()
    try:
        os.environ["HISPEC_DRP_SHOW_CCF_PREVIEW"] = "1"
        if serialize_only:
            prim = StellarRVPrimitive(
                algorithm="ccf_mask_kpf",
                mask_path=mask_path,
                do_fibers=["1"],
                do_orders=do_orders,
                window_size_um=WINDOW_UM,
                show_preview=False,
                save_preview=True,
                save_mask_overlay_debug=False,
                show_mask_overlay_debug=False,
                output_dir=out_dir,
                suffix="lte049_mask_kpf",
                max_cores=6,
            )
            t0 = time.perf_counter()
            with patch.object(srv_prev, "create_stellar_rv_preview", lambda _p: None):
                out = prim.apply(model)
            apply_wall = time.perf_counter() - t0
        else:
            prim = StellarRVPrimitive(
                algorithm="ccf_mask_kpf",
                mask_path=mask_path,
                do_fibers=["1"],
                do_orders=do_orders,
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
            "  └ subset: stellar RV preview HTML (Plotly write_html ×2)",
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
    html_unified = prev.get("html")
    if html_unified and os.path.isfile(html_unified):
        ap = os.path.abspath(html_unified)
        print(f"LTE049_K2ALL_STELLAR_RV_PREVIEW_HTML={ap}", flush=True)
        t0 = time.perf_counter()
        webbrowser.open(f"file://{ap}")
        rows.append(
            (
                "open unified stellar RV preview HTML in default browser (post-apply)",
                time.perf_counter() - t0,
                "webbrowser.open (file://…*_preview.html)",
                True,
            )
        )

    mod = prim._output or {}
    for _fib, p in (mod.get("mask_overlay_debug") or {}).items():
        if p and os.path.isfile(p):
            print(f"LTE049_K2ALL_MASK_OVERLAY_HTML={os.path.abspath(p)}", flush=True)

    if _want_bspec_btsettl_serialize_mask_ccf():
        _serialize_mask_kpf_ccf_preview_debug(prim, out_dir)

    return rows


def _load_btsettl_sorted_micron(path: str, wl_min_um: float, wl_max_um: float) -> tuple[np.ndarray, np.ndarray]:
    w_rest, flux_rest = load_btsettl_spectrum(path, wl_min_um=wl_min_um, wl_max_um=wl_max_um)
    w = np.asarray(w_rest.to_value("micron"), dtype=np.float64)
    f = np.asarray(flux_rest.value, dtype=np.float64)
    ok = np.isfinite(w) & np.isfinite(f) & (w > 0)
    w, f = w[ok], f[ok]
    idx = np.argsort(w)
    return w[idx], f[idx]


@pytest.fixture(scope="session")
def template_path():
    path = get_spectrum_path(
        "lte025.0-5.5-0.0a+0.0.BT-Settl.spec.7.h5",
        base_dir=TEMPLATE_DIR,
    )
    if not os.path.isfile(path):
        pytest.skip(f"Template not found: {path}")
    return path


@pytest.fixture(scope="session")
def btsettl_rv_sweep_template_path():
    """lte049 template paired with ``lte049.0*.mas`` for mask-line RV injection sweeps."""
    path = get_spectrum_path(
        "lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5",
        base_dir=TEMPLATE_DIR,
    )
    if not os.path.isfile(path):
        pytest.skip(f"Template not found: {path}")
    return path


@pytest.fixture(scope="session")
def lte025_btsettl_session_cache(template_path):
    """Load lte025 BT-Settl once per session (shared by parallel + ``rv_per_chunks`` tests)."""
    wl_min, wl_max = _BTSETTL_WL_RANGE_UM
    w_rest, flux_rest = _load_btsettl_sorted_micron(template_path, wl_min, wl_max)
    return {
        "w_rest": w_rest,
        "flux_rest": flux_rest,
        "template_ln": np.log(w_rest),
    }


@pytest.fixture(scope="session")
def lte049_btsettl_session_cache(btsettl_rv_sweep_template_path):
    """Load lte049 BT-Settl once per session (injection sweep + snapshot helpers)."""
    wl_min, wl_max = _BTSETTL_WL_RANGE_UM
    w_rest, flux_rest = _load_btsettl_sorted_micron(btsettl_rv_sweep_template_path, wl_min, wl_max)
    return {
        "w_rest": w_rest,
        "flux_rest": flux_rest,
    }


def test_fit_ccf_rv_failure_includes_debug_context():
    """Failed :func:`fit_ccf_rv` runs attach ``debug_context`` (and log a JSON suffix)."""
    vm_short = np.array([0.0, 1.0], dtype=np.float64)
    ccf_short = np.array([1.0, 1.0], dtype=np.float64)
    _, _, _, d_ins = fit_ccf_rv(
        vm_short,
        ccf_short,
        1.0,
        return_params=True,
        config=CCFFitKpfConfig(),
    )
    assert d_ins.get("message") == "insufficient valid points"
    dc_ins = d_ins.get("debug_context") or {}
    assert dc_ins.get("fit_stage") == "input_grid"
    assert dc_ins.get("n_finite_pairs") == 2

    vm = np.linspace(-200, 200, 120) * 1000.0
    ccf = np.linspace(0.0, 1.0, vm.size)
    _, _, _, d_mean = fit_ccf_rv(vm, ccf, 1000.0, return_params=True)
    assert d_mean.get("message") == "mean outside window"
    dc_mean = d_mean.get("debug_context") or {}
    assert dc_mean.get("fit_stage") == "pass1_post_lm"
    assert dc_mean.get("mean_side") in (
        "below_sample_min",
        "above_sample_max",
        "non_finite_mean",
    )
    assert "fitted_mean_km_s" in dc_mean
    assert "fit_samples_v_km_min" in dc_mean and "fit_samples_v_km_max" in dc_mean


@pytest.mark.parametrize("spectrograph", ["BSPEC", "RSPEC"])
def test_stellar_rv_recovers_rv_per_chunk(spectrograph):
    """Full echelle synthetic L2 + template; combined RV matches inject; chunk sanity."""
    algorithm = "ccf_stellar_template"
    n_pix = _TEST_ECHELLE_N_PIX
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


def test_rv_per_chunks_btsettl_one_order_recovers_injected_rv(lte025_btsettl_session_cache):
    """
    Pure-Python :func:`rv_per_chunks` + :func:`fit_ccf_rv` on **one** BSPEC echelle order
    (lte025 BT-Settl fixture). Uses the same narrow chunk window as
    ``test_fit_ccf_kpf_style_btsettl_bspec_all_chunks_parallel`` so chunk statistics stay
    meaningful without looping all orders (that path is covered by the Numba parallel test).
    """
    input_rv_m_s = INPUT_RV_KM_S * 1000.0
    v_step = _TEST_TEMPLATE_CCF_V_STEP_M_S
    n_pix = _TEST_ECHELLE_N_PIX
    w_rest = lte025_btsettl_session_cache["w_rest"]
    flux_rest = lte025_btsettl_session_cache["flux_rest"]
    template_ln = lte025_btsettl_session_cache["template_ln"]
    assert len(w_rest) > 50
    orders = _bspec_btsettl_ccf_mask_kpf_order_ids()
    order = int(orders[0])
    wave_ord, lam_c = _order_wavelength_grid(order, n_pix)
    spec = np.interp(
        wave_ord / (1.0 + INPUT_RV_KM_S / C_KM_S), w_rest, flux_rest
    ).astype(np.float64)
    spec_err = _bspec_simulated_flux_err(spec)
    span = float(np.ptp(wave_ord))
    window_um_eff = float(min(WINDOW_UM, max(1e-6, span / 16.0)))
    _rvs, _e, _centers, v_grid, ccf_2d = rv_per_chunks(
        wave_ord,
        spec,
        spec_err,
        np.zeros(n_pix, dtype=bool),
        template_ln,
        flux_rest,
        window_um_eff,
        v_min=_V_MIN,
        v_max=_V_MAX,
        v_step=v_step,
        return_ccf=True,
    )
    vspan = dv_per_pixel_m_s(float(lam_c))
    vspan_arg = vspan if np.isfinite(vspan) else None
    n_fit_ok = 0
    n_in_tol = 0
    n_attempt = 0
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
        in_tol = (
            fit_ok
            and np.isfinite(rv)
            and np.isfinite(rv_err)
            and rv_err > 0
            and abs(rv - input_rv_m_s) < tol
        )
        if fit_ok:
            n_fit_ok += 1
        if in_tol:
            n_in_tol += 1

    assert n_attempt > 5, "Expected multiple wavelength chunks on one order with narrow window"
    assert n_fit_ok >= 0.85 * n_attempt, f"fit_ok fraction too low: {n_fit_ok}/{n_attempt}"
    assert n_in_tol >= 0.50 * n_attempt, f"RV recovery fraction too low: {n_in_tol}/{n_attempt}"


@pytest.mark.slow
def test_fit_ccf_kpf_style_btsettl_bspec_all_chunks_parallel(
    template_path, tmp_path, lte025_btsettl_session_cache
):
    """
    BT-Settl lte025 + BSPEC echelle model: one call to :func:`process_orders_rvs_numba_parallel`
    with ``max_cores=6`` (clamped
    to Numba’s allowed thread count on the host, e.g. 2 on small CI runners) and
    ``ccf_v_step`` = :data:`_TEST_TEMPLATE_CCF_V_STEP_M_S` (100 m/s in tests; production default is often 50 m/s).
    Phase-B Gaussian fits run serially in Python.

    By default only **one** BSPEC echelle order is used (default **139**). For all echelle orders
    from ``HISPECConfig``, set ``HISPEC_DRP_BSPEC_BTSETTL_ALL_ORDERS=1`` or pass pytest
    ``--bspec-all-orders``. Use ``HISPEC_DRP_BSPEC_BTSETTL_SINGLE_ORDER=n`` to override the
    default single order. A narrower chunk window is applied in the single-order case so the
    order still contributes many wavelength bins.

    ``ccf_mask_kpf`` RVCCF validation and the mask-KPF chunk benchmark use the template-derived
    ``lte049.0*.mas`` mask in ``data/masks`` (paired with the lte049 BT-Settl template), not the
    default ``M2_filt_allRV.mas``.

    Assertions use the same recovery fractions as
    ``test_rv_per_chunks_btsettl_one_order_recovers_injected_rv``, on chunks that are eligible
    (>=20 pixels in the bin and written by the parallel path, ``np.isfinite(all_wl)``).

    After building the synthetic spectra, zero-flux pixels (including forced order endpoints)
    are assigned ``DQ`` ``DO_NOT_USE`` so CCF and mask-KPF paths exclude them via the standard
    bad-pixel mask.
    """
    timing_rows: list[tuple[str, float, str, bool]] = []
    mas_lte049 = _lte049_btsettl_template_mask_path()
    if mas_lte049 is None:
        pytest.skip(f"lte049.0*.mas mask not found under {MASK_DIR}")
    input_rv_m_s = INPUT_RV_KM_S * 1000.0
    v_step = _TEST_TEMPLATE_CCF_V_STEP_M_S
    n_pix = _TEST_ECHELLE_N_PIX
    _ = template_path  # fixture ensures file exists; spectrum comes from session cache
    t0 = time.perf_counter()
    w_rest = lte025_btsettl_session_cache["w_rest"]
    flux_rest = lte025_btsettl_session_cache["flux_rest"]
    template_ln = lte025_btsettl_session_cache["template_ln"]
    assert len(w_rest) > 50
    orders = _bspec_btsettl_ccf_mask_kpf_order_ids()
    timing_rows.append(
        (
            "lte025: session cache (w_rest, flux_rest, template_ln) + BSPEC mask-KPF order selection",
            time.perf_counter() - t0,
            "lte025_btsettl_session_cache, _bspec_btsettl_ccf_mask_kpf_order_ids",
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
        spec_err_2d[oi] = _bspec_simulated_flux_err(spec_2d[oi])
        order_tasks.append((oi, m))

    # Zero-flux pixels: DO_NOT_USE so template CCF + mask-KPF skip them (``d_bad`` / DQ gate).
    # Force order endpoints to flux 0 and mark bad; also flag any non-finite or non-positive flux.
    do_not_use = int(datamodels.DQ_FLAGS["DO_NOT_USE"])
    do_not_use_u = np.uint32(do_not_use)
    for oi in range(n_orders):
        spec_2d[oi, 0] = 0.0
        spec_2d[oi, -1] = 0.0
    bad_flux = ~np.isfinite(spec_2d) | (spec_2d <= 0.0)
    dq_2d[bad_flux] |= do_not_use_u

    window_um_eff = _btsettl_ccf_mask_kpf_window_um(wave_2d[0])
    timing_rows.append(
        (
            "lte025: stack BSPEC 2-D model (interp Doppler-shifted BT-Settl per order)",
            time.perf_counter() - t0,
            "_order_wavelength_grid, np.interp (build wave_2d, spec_2d, spec_err_2d)",
            True,
        )
    )

    ge20_flat = _bspec_btsettl_ge20_mask_flat(wave_2d, order_tasks, window_um_eff)
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
            window_um_eff,
            order_tasks,
            max_cores=6,
            return_ccf=False,
            ccf_v_step=_TEST_TEMPLATE_CCF_V_STEP_M_S,
        )
    )
    parallel_wall = time.perf_counter() - t0
    timing_rows.append(
        (
            f"lte025: process_orders_rvs_numba_parallel (numba_threads={nthr}, "
            f"ccf_v_step={_TEST_TEMPLATE_CCF_V_STEP_M_S} m/s, "
            f"window_um={window_um_eff:.6f})",
            parallel_wall,
            "process_orders_rvs_numba_parallel, _process_all_orders_ccf_numba (prange), "
            "process_all_orders_rvs_numba (_ccf_rv_from_ccf / fit_ccf_rv, serial)",
            True,
        )
    )

    assert len(all_rvs) == len(ge20_flat), "flat chunk layout mismatch"
    eligible = ge20_flat & np.isfinite(all_wl)
    n_attempt = int(np.sum(eligible))
    assert n_attempt > 10, (
        "Expected many wavelength chunks (raise HISPEC_DRP_BSPEC_BTSETTL_ALL_ORDERS or widen wl window)"
    )

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
        do_orders=orders,
        window_size_um=window_um_eff,
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
    bench_orders = order_tasks[: min(3, len(order_tasks))]
    oi0 = bench_orders[0][0]
    span0 = float(np.ptp(wave_2d[oi0]))
    bench_window_um = min(0.01, WINDOW_UM / 20.0, max(1e-6, span0 / 16.0))
    mask_lines = load_mask(mas_lte049, mask_width_um=None)
    bench_times: dict[int, float] = {}
    bench_chunks: dict[int, int] = {}
    eff_threads: dict[int, int] = {}
    for req_cores in (1, 6):
        eff_threads[req_cores] = resolve_numba_threads(req_cores)
        t0 = time.perf_counter()
        n_chunks_total = 0
        for oi, _order in bench_orders:
            d_bad = (np.asarray(dq_2d[oi], dtype=np.uint32) & do_not_use) != 0
            rvs_b, _errs_b, _centers_b, _v_grid_b, _ccf_b, _wlo, _whi, _ng, _, _ = rv_per_chunks_mask_kpf(
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

    # Verify measurable speed-up whenever effective thread count exceeds 1 (1 vs 6 only in tests).
    t1 = bench_times[1]
    if eff_threads[6] > 1:
        t6 = bench_times[6]
        assert t6 < 0.98 * t1, (
            "Expected ccf_mask_kpf chunk-parallel speed-up with >1 effective thread: "
            f"t1={t1:.3f}s, t6={t6:.3f}s (eff={eff_threads[6]}), "
            f"n_chunks={bench_chunks[1]}"
        )
    else:
        print(
            "ccf_mask_kpf speed-up check skipped: host allows only one effective Numba thread "
            f"(eff threads: 1->{eff_threads[1]}, 6->{eff_threads[6]}).",
            flush=True,
        )

    if _want_bspec_btsettl_all_previews() or _want_bspec_btsettl_serialize_mask_ccf():
        timing_rows.extend(
            _run_lte049_k2all_mask_kpf_previews(
                str(tmp_path),
                mask_path=mas_lte049,
                do_orders=orders,
                serialize_only=(
                    _want_bspec_btsettl_serialize_mask_ccf()
                    and not _want_bspec_btsettl_all_previews()
                ),
            )
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
    lte049_cache: dict | None = None,
    n_pix: int = _TEST_ECHELLE_N_PIX,
    injected_km_s: np.ndarray | None = None,
) -> dict:
    """
    For each injected RV in km/s (default: :func:`_btsettl_injection_km_s_grid`), build BSPEC
    BT-Settl cubes and recover a single RV via :func:`rv_per_chunks_mask_kpf` on the **lte049**
    template-derived ``.mas`` (see :func:`_lte049_btsettl_template_mask_path`) plus median/MAD
    outlier rejection and inverse-variance weighted mean (same combination as
    :class:`StellarRVPrimitive` for mask KPF).

    Parameters
    ----------
    use_ccf_input_velocity_supersample
        If False (default), use the broad CCF grid with step :data:`_TEST_MASK_CCF_BROAD_STEP_M_S`
        in tests (100 m/s; production defaults are often 50 m/s).
        If True, for each injection use a **merged** grid: full broad range plus denser samples on
        ``injected_RV ± 300`` m/s at 0.1 m/s (:func:`~hispecdrp.primitives.utils.ccf_utils.build_hybrid_merged_velocity_grid`).
    single_echelle_order
        If not ``None``, restrict processing to this BSPEC echelle order (one row in the
        synthetic 2-D echelle arrays).
    lte049_cache
        Optional ``{"w_rest", "flux_rest"}`` from :func:`lte049_btsettl_session_cache` to skip
        reloading the lte049 spectrum.
    n_pix
        Echelle pixels per order (tests use :data:`_TEST_ECHELLE_N_PIX`).
    injected_km_s
        Injected stellar RV grid (km/s); default from :func:`_btsettl_injection_km_s_grid`.

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
    if lte049_cache is not None:
        w_rest = lte049_cache["w_rest"]
        flux_rest = lte049_cache["flux_rest"]
    else:
        w_rest, flux_rest = _load_btsettl_sorted_micron(template_path, wl_min, wl_max)
    assert len(w_rest) > 50
    orders = hispec_config.HISPECConfig.get_order_list("BSPEC")
    assert orders
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

    if injected_km_s is None:
        injected_km_s = _btsettl_injection_km_s_grid()
    injected_km_s = np.asarray(injected_km_s, dtype=np.float64).ravel()
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
            spec_err_2d[oi] = _bspec_simulated_flux_err(spec_2d[oi])

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
            vg_kw = dict(
                v_min_m_s=vb,
                v_max_m_s=vx,
                v_step_m_s=_TEST_MASK_CCF_BROAD_STEP_M_S,
            )

        chunk_rvs: list[np.ndarray] = []
        chunk_errs: list[np.ndarray] = []
        for oi in range(n_orders):
            d_bad = (np.asarray(dq_2d[oi], dtype=np.uint32) & do_not_use) != 0
            rvs_o, errs_o, _cen, _vg, _ccf, _wlo, _whi, _ng, _, _ = rv_per_chunks_mask_kpf(
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


@pytest.mark.slow
def test_btsettl_rv_injection_sweep_recovery(
    btsettl_rv_sweep_template_path, lte049_btsettl_session_cache, tmp_path
):
    """
    Scan injected stellar RVs on the **lte049** BT-Settl template (grid from
    :func:`_btsettl_injection_km_s_grid`; default 6 points, full 11 via env); recover RV using
    ``rv_per_chunks_mask_kpf`` with the template-derived ``lte049.0*.mas`` mask and the same
    median/MAD + IV mean as mask-KPF stellar RV. Uses the **merged hybrid** CCF velocity grid
    (broad + dense sampling near each injection) so peak finding matches the refined pipeline;
    fit a linear relation; save Plotly HTML under ``tmp_path``.
    """
    so = _btsettl_injection_sweep_default_single_order()
    try:
        sweep = _collect_btsettl_rv_injection_sweep(
            btsettl_rv_sweep_template_path,
            use_ccf_input_velocity_supersample=True,
            single_echelle_order=so,
            lte049_cache=lte049_btsettl_session_cache,
        )
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    assert sweep["single_echelle_order"] == so
    assert sweep["ccf_grid_mode"] == "merged_hybrid_per_injection"
    assert len(sweep["injected_km_s"]) == len(_btsettl_injection_km_s_grid())
    assert np.all(np.isfinite(sweep["recovered_m_s"]))
    assert 0.7 < sweep["ols_slope"] < 1.3, f"slope out of range: {sweep['ols_slope']}"
    assert sweep["rms_residual_m_s"] < 20_000.0, f"RMS too large: {sweep['rms_residual_m_s']}"

    fig = _build_btsettl_rv_injection_figure(sweep)
    out_html = os.path.join(str(tmp_path), "btsettl_rv_injection_sweep.html")
    fig.write_html(out_html)
    assert os.path.isfile(out_html)


def test_to_serializable_order_payload_includes_mask_weight_raw():
    """Mask overlay JSON carries raw weights for drawing and tapered weights for CCF."""
    from hispecdrp.primitives.stellar_rv.stellar_rv_preview import (
        _to_serializable_order_payload,
    )

    payload = {
        "fiber": "1",
        "orders": [
            {
                "order": 1,
                "wl_lo_um": 1.0,
                "wl_hi_um": 1.1,
                "wave": [1.0, 1.05],
                "spec": [1.0, 1.0],
                "mask_start_um": [1.02],
                "mask_end_um": [1.02],
                "mask_weight_raw": [1.0],
                "mask_weight": [0.1],
            }
        ]
    }
    out = _to_serializable_order_payload(payload)
    row = out["orders"][0]
    assert row["mask_weight_raw"] == [1.0]
    assert row["mask_weight"] == [0.1]


def test_mask_order_edge_taper_multipliers_center_vs_edges():
    """ccf_mask_kpf edge deweight: unity at ±N pixels from edges, ~0 at order ends."""
    from hispecdrp.primitives.utils.ccf_utils import _mask_line_order_edge_taper_multipliers

    w = np.linspace(1.0, 1.1, 201, dtype=np.float64)
    lam = np.array([w[100], w[0], w[-1]], dtype=np.float64)
    m = _mask_line_order_edge_taper_multipliers(
        w, lam, edge_taper_pixels=100.0, exp_shape=5.0
    )
    assert m[0] >= 0.999
    assert m[1] < 1e-10 and m[2] < 1e-10


def test_ccf_mask_kpf_excludes_partial_edge_line_overlap():
    """
    For mask CCF (narrow-line model), a line contributes only when its **center**
    wavelength (mean of start and end) after Doppler shift lies inside the chunk
    pixel-edge span. A segment whose endpoints straddle the chunk but whose center
    falls outside does not contribute.
    """
    wave = np.linspace(1.0, 1.1, 512, dtype=np.float64)
    spec = np.ones_like(wave)
    spec_err = np.ones_like(wave)
    dq_bad = np.zeros_like(wave, dtype=bool)

    # One line with center inside the chunk; a second line with center below the chunk
    # (segment endpoints could overlap the blue edge in a finite-width model, but the
    # narrow-line kernel uses center only).
    full_only = MaskLines(
        line_lambda_um=np.array([1.025], dtype=np.float64),
        weight=np.array([1.0], dtype=np.float64),
    )
    full_plus_partial = MaskLines(
        line_lambda_um=np.array([1.025, 0.995], dtype=np.float64),
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
        mask_ccf_detrend=False,
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
        mask_ccf_detrend=False,
    )

    _rvs_a, _errs_a, _centers_a, v_a, ccf_a, _wla, _wha, _nga, std_a, _pv_a = out_full
    _rvs_b, _errs_b, _centers_b, v_b, ccf_b, _wlb, _whb, _ngb, std_b, _pv_b = out_mix
    assert ccf_a is not None and ccf_b is not None
    assert std_a is not None and std_b is not None
    assert std_a.shape == ccf_a.shape and std_b.shape == ccf_b.shape
    assert np.all(std_a >= 0.0) and np.all(std_b >= 0.0)
    assert ccf_a.shape == ccf_b.shape == (1, len(v_a))
    assert np.allclose(v_a, v_b)
    assert np.allclose(ccf_a, ccf_b, rtol=0.0, atol=1e-12), (
        "Partially overlapping edge line should not contribute to ccf_mask_kpf CCF."
    )


def test_mask_kpf_ccf_peak_prefers_prominent_local_maximum():
    """
    RV extraction must not always use the raw global CCF extremum. Mask-line CCF uses a
    **fixed absorption convention** (minimum of weighted flux×mask near true RV): we fit on
    ``-CCF``, so the physical signal is a **broad minimum** vs a tall narrow wrong feature.

    A tall narrow spike in raw CCF (misleading local maximum in ``-CCF`` with low prominence)
    should lose to the broader absorption dip at the true velocity.
    """
    from hispecdrp.primitives.utils.ccf_utils import _rv_from_ccf_mask_kpf_numba_single

    n = 2001
    v = np.linspace(-1.0e6, 1.0e6, n, dtype=np.float64)
    j = np.arange(n, dtype=np.float64)
    j_true = 1550
    sig = 28.0
    # Absorption-shaped mask CCF: broad dip at j_true; competing plateau + narrow bump elsewhere.
    ccf = np.zeros(n, dtype=np.float64)
    ccf[100:200] = 90.0
    ccf[155] = 110.0
    ccf = ccf + 200.0 - 85.0 * np.exp(-0.5 * ((j - float(j_true)) / sig) ** 2)

    rv, _err, flip = _rv_from_ccf_mask_kpf_numba_single(v, ccf, 4096, 11)
    assert flip == 1
    assert np.isfinite(rv)
    assert abs(rv - float(v[j_true])) < abs(rv - float(v[155]))


def test_nanmean_axis0_ignore_empty_columns_matches_nanmean_no_runtime_warning():
    """Stacked CCF columns can be all-NaN; avoid numpy nanmean empty-slice RuntimeWarning."""
    import warnings

    from hispecdrp.primitives.utils.ccf_utils import _nanmean_axis0_ignore_empty_columns

    x = np.array([[1.0, np.nan], [np.nan, np.nan], [3.0, np.nan]], dtype=np.float64)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always", RuntimeWarning)
        y = _nanmean_axis0_ignore_empty_columns(x)
    assert not rec, f"unexpected warnings: {[str(w.message) for w in rec]}"
    np.testing.assert_allclose(y, np.array([2.0, np.nan]), rtol=0, atol=0, equal_nan=True)


def test_refine_ccf_rv_coarse_grid_parabolic_default_rv_err_is_bouchy_scale():
    """
    Coarse-grid parabolic refinement should report a Bouchy-style σ on the smoothed CCF,
    not the order-10 m/s scatter from diagonal per-bin bootstrap on the raw grid.
    """
    from hispecdrp.primitives.utils.ccf_utils import (
        CCFFitKpfConfig,
        refine_ccf_rv_coarse_grid_parabolic,
    )

    v = np.arange(-150000, 150000, 50.0, dtype=np.float64)
    v0 = 1234.5
    sig = 2800.0
    ccf = np.exp(-0.5 * ((v - v0) / sig) ** 2) * 1e-15
    ccf_err = np.full_like(ccf, 1e-18)

    rv, meta = refine_ccf_rv_coarse_grid_parabolic(
        v,
        ccf,
        50.0,
        ccf_err=ccf_err,
        gaussian_sigma_m_s=sig,
        mask_method="stellar",
        config=CCFFitKpfConfig(),
        bootstrap_rv_err_samples=0,
    )
    assert meta["ok"]
    assert abs(rv - v0) < 0.02
    err = float(meta["rv_err_m_s"])
    assert np.isfinite(err) and 0.05 < err < 0.5
    assert not np.isfinite(meta["rv_err_bootstrap_m_s"])

    _rv2, meta_bs = refine_ccf_rv_coarse_grid_parabolic(
        v,
        ccf,
        50.0,
        ccf_err=ccf_err,
        gaussian_sigma_m_s=sig,
        mask_method="stellar",
        config=CCFFitKpfConfig(),
        bootstrap_rv_err_samples=32,
        bootstrap_seed=0,
    )
    assert np.isfinite(meta_bs["rv_err_m_s"]) and meta_bs["rv_err_m_s"] < 0.5
    assert np.isfinite(meta_bs["rv_err_bootstrap_m_s"])
    assert meta_bs["rv_err_bootstrap_m_s"] > 5.0 * meta_bs["rv_err_m_s"]

    assert meta["refine_curve_v_m_s"].shape == meta["refine_curve_ccf_smoothed"].shape
    assert meta["refine_parabola_v_dense_m_s"].size >= 3
    assert np.isfinite(meta["parabola_a"]) and meta["parabola_a"] < 0

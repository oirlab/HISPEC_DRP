"""
Stellar RV tests: **ccf_stellar_template** (log-λ CCF, median/MAD rejection, IV weighted mean).

Synthetic L2 uses :func:`_order_wavelength_grid` and Gaussian lines; rest templates use
**3×** L2 pixels per order to reduce log-λ CCF bias.

Run::

  pytest tests/test_stellar_rv.py -v

Faster CI / local iteration (skips tests marked ``slow``: BT-Settl parallel mask-KPF integration)::

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
always uses spectrum ``lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5`` and mask
``lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.mas``. It runs the same science setup via
``process_orders_rvs_numba_parallel`` (requests ``max_cores=6``;
actual Numba threads are clamped to the wheel/CPU limit, e.g. GitHub Actions) and prints a timing summary.
By default it uses **one** echelle order (**139**). To run on **all** BSPEC echelle orders from
``HISPECConfig``, either set ``HISPEC_DRP_BSPEC_BTSETTL_ALL_ORDERS=1`` or pass pytest flag
``--bspec-all-orders`` (the flag sets that env var). Use ``HISPEC_DRP_BSPEC_BTSETTL_SINGLE_ORDER=<n>``
to pick a different single order instead.

Simulated ``spec_err`` uses a center-to-edge throughput taper so SNR is high in the order center
and degrades toward the blue/red ends (see ``_bspec_simulated_flux_err``).

**One-liner — mask-KPF parallel test + both previews (mask overlay + unified CCF/RV HTML), from repo root**::

  HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS=1 pytest tests/test_stellar_rv.py::test_fit_ccf_kpf_style_btsettl_bspec_all_chunks_parallel -s

The ``-s`` flag is required to see **printed** diagnostics (pytest otherwise captures stdout). For this
test, the injected vs recovered RV table is emitted **last**, after the wall-time summary, so scroll to
the bottom of the run to read it.

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

"""
from __future__ import annotations

import json
import math
import os
import time
import webbrowser
from unittest.mock import patch

import numpy as np
import pytest
from astropy.stats import mad_std

import hispecdrp
from hispecdrp import datamodels
from hispecdrp.datamodels import hispec_config
from hispecdrp.datamodels.extensions import RVCCF_HDU
from hispecdrp.primitives import StellarRVPrimitive
from hispecdrp.utils.parallelization_utils import resolve_numba_threads
from hispecdrp.primitives.utils.ccf_utils import (
    MASK_KPF_STORED_CCF_NEGATED_FROM_RAW,
    _V_MAX,
    _V_MIN,
    _require_mask_kpf_parabolic_vertex_n_samples,
    ccf_order_n_chunks_for_span,
    combine_rv_estimates_median_mad_weighted,
    load_mask,
    mask_kpf_parabolic_vertex_rv_err_lstsq_homoscedastic_m_s,
    process_orders_rvs_numba_parallel,
    rv_per_chunks_mask_kpf,
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

# Fixed lte049 BT-Settl template + paired mask (``test_fit_ccf_kpf_style_btsettl_bspec_all_chunks_parallel`` only).
LTE049_BTSETTL_SPEC_H5 = "lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5"
LTE049_BTSETTL_MASK_MAS = "lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.mas"

# ``mask_kpf_parabolic_vertex_n_samples`` (odd, LSTSQ parabolic vertex on the mask CCF).
# Offline sweep: ``python tests/_scan_btsettl_parabolic_vertex_n_samples.py`` on the default
# BSPEC lte049 order(s), 50 m/s grid, :data:`WINDOW_UM`. Among odd *n* in 11..71, *n* = 31
# minimizes |RVCCFAVG − injected RV| with |bias| < 1 m/s. Very small *n* (3–9) use a
# degenerate sampling window and report ~0 bias; they are not used as pipeline defaults.
_BTSETTL_MASK_KPF_PARABOLIC_VERTEX_N_SAMPLES = 31


def _lte049_btsettl_spec_h5_path() -> str:
    return get_spectrum_path(LTE049_BTSETTL_SPEC_H5, base_dir=TEMPLATE_DIR)


def _lte049_btsettl_paired_mask_path() -> str:
    return os.path.join(MASK_DIR, LTE049_BTSETTL_MASK_MAS)


# --- Test-only performance knobs (same assertions / tolerances; coarser grids are fine for loose RV tests) ---
# Fewer pixels per echelle row cuts CCF cost ~linearly; 2048 still satisfies ≥20 px/chunk with narrow windows.
_TEST_ECHELLE_N_PIX = 2048
# Uniform broad CCF grid step (m/s) for template-CCF Numba path and mask-CCF benchmarks in this file.
# Match `ccf_utils._V_STEP` (50 m/s) — no hybrid grid; broad scan only.
_TEST_TEMPLATE_CCF_V_STEP_M_S = 50.0


def test_mask_kpf_parabolic_vertex_n_samples_must_be_odd():
    assert _require_mask_kpf_parabolic_vertex_n_samples(11) == 11
    with pytest.raises(ValueError, match="odd"):
        _require_mask_kpf_parabolic_vertex_n_samples(10)
    with pytest.raises(ValueError, match=">= 3"):
        _require_mask_kpf_parabolic_vertex_n_samples(2)


def test_mask_kpf_parabolic_vertex_homoscedastic_sigma_noiseless_quadratic():
    """
    Homoscedastic LSTSQ + delta-method σ collapses toward the numeric floor when the
    parabola is exact on the coarse grid; this is the refinement-limited scale tied to
    ``n_samples`` and :math:`(X^{\\mathsf T}X)^{-1}`, unlike the pipeline ``RV_ERR`` heuristic.
    """
    v_step = 50.0
    v0 = -110000.0
    v = v0 + np.arange(-750.0, 750.0 + v_step, v_step, dtype=np.float64)
    k = 1e-6
    y_work = -k * (v - v0) ** 2
    ccf_raw = -y_work
    i_peak = int(np.argmin(np.abs(v - v0)))
    sig, meta = mask_kpf_parabolic_vertex_rv_err_lstsq_homoscedastic_m_s(
        v,
        ccf_raw,
        i_peak=i_peak,
        n_pt_want=31,
        v_step_m_s_for_floor=v_step,
    )
    assert sig is not None
    assert float(meta["rss"]) < 1e-12
    assert float(sig) < 1e-3


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
        span = float(w_max - w_min)
        n_chunk = ccf_order_n_chunks_for_span(span, float(window_um))
        for k in range(n_chunk):
            lam_lo = w_min + k * span / n_chunk
            lam_hi = w_min + (k + 1) * span / n_chunk
            if k == n_chunk - 1:
                in_chunk = (w >= lam_lo) & (w <= lam_hi)
            else:
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


def _fiber_agg_rvs_from_rvccf_table(
    tbl,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct fiber-level ``(rv, rv_err)`` series matching mask-KPF aggregation:

    - Order-mean CCF path duplicates the same RV per chunk → one pair per echelle order.
    - Otherwise one pair per chunk (multiple rows per order possible).
    """
    ord_col = np.asarray(tbl["ORDER"])
    rv_col = np.asarray(tbl["RV"], dtype=np.float64)
    err_col = np.asarray(tbl["RV_ERR"], dtype=np.float64)
    agg_rv: list[float] = []
    agg_er: list[float] = []
    agg_order: list[int] = []
    for o in np.unique(ord_col):
        m = ord_col == o
        rvs = rv_col[m]
        ers = err_col[m]
        if rvs.size == 0:
            continue
        if np.allclose(rvs, rvs[0]):
            agg_rv.append(float(rvs[0]))
            agg_er.append(float(ers[0]))
            agg_order.append(int(o))
        else:
            for i in range(int(rvs.size)):
                agg_rv.append(float(rvs[i]))
                agg_er.append(float(ers[i]))
                agg_order.append(int(o))
    return (
        np.asarray(agg_rv, dtype=np.float64),
        np.asarray(agg_er, dtype=np.float64),
        np.asarray(agg_order, dtype=np.int64),
    )


def _print_btsettl_mask_kpf_recovery_report(
    output_model,
    tbl_rvccf,
    *,
    injected_rv_m_s: float,
    outlier_sigma: float,
) -> None:
    """
    Print injected vs recovered RVs for ``ccf_mask_kpf`` BSPEC lte049 test: per-order/chunk
    rows, bias vs truth, ``RVCCFAVG`` aggregate, and recomputed robust weighted mean
    (``combine_rv_estimates_median_mad_weighted``) with IV vs MAD/√N components.
    """
    avg = getattr(output_model, "rvccfavg_fib1", None)
    avg_rv = float(avg["RV"][0]) if avg is not None and len(avg) else float("nan")
    avg_err = float(avg["RV_ERR"][0]) if avg is not None and len(avg) else float("nan")

    ord_col = np.asarray(tbl_rvccf["ORDER"])
    rv_col = np.asarray(tbl_rvccf["RV"], dtype=np.float64)
    err_col = np.asarray(tbl_rvccf["RV_ERR"], dtype=np.float64)
    inl_col = np.asarray(tbl_rvccf["INLIER"], dtype=np.uint8)

    lines: list[str] = [
        "",
        "=" * 72,
        "test_fit_ccf_kpf_style_btsettl_bspec_all_chunks_parallel — RV recovery (m/s)",
        "=" * 72,
        "Mask-line CCF vs fit_ccf_rv: stored RVCCF/preview rows use the pipeline convention "
        f"ccf_mask_kpf_stored_negated={MASK_KPF_STORED_CCF_NEGATED_FROM_RAW} "
        "(negated from raw weighted mask statistic so the line peak is near true RV; "
        "see ccf_utils.MASK_KPF_STORED_CCF_NEGATED_FROM_RAW and fit_ccf_rv diagnostics).",
        f"Injected RV (constant per order): {injected_rv_m_s:.3f}",
        "",
        "Per echelle ORDER (one RVCCF row per order when order-mean CCF duplicates chunks):",
        f"{'ORDER':>6}  {'inj':>14}  {'RV':>14}  {'RV_ERR':>12}  {'bias':>12}  INLIER",
    ]
    for o in np.unique(ord_col):
        m = ord_col == o
        rvs = rv_col[m]
        ers = err_col[m]
        inl = inl_col[m]
        inj = injected_rv_m_s
        if rvs.size and np.allclose(rvs, rvs[0]):
            rv_d = float(rvs[0])
            er_d = float(ers[0])
        else:
            w = np.where(np.isfinite(ers) & (ers > 0), 1.0 / (ers.astype(np.float64) ** 2), 0.0)
            if np.sum(w) <= 0:
                rv_d = float(np.nanmean(rvs))
                er_d = float("nan")
            else:
                rv_d = float(np.sum(w * rvs) / np.sum(w))
                er_d = float(1.0 / np.sqrt(np.sum(w)))
        bias = rv_d - inj
        if inl.size == 0:
            inl_s = "?"
        elif np.all(inl == inl[0]):
            inl_s = "1" if inl[0] != 0 else "0"
        else:
            inl_s = "mixed"
        lines.append(
            f"{int(o):6d}  {inj:14.3f}  {rv_d:14.3f}  {er_d:12.3f}  {bias:12.3f}  {inl_s:>6s}"
        )

    agg_rv, agg_er, _agg_order = _fiber_agg_rvs_from_rvccf_table(tbl_rvccf)
    (
        comb_rv,
        comb_err,
        comb_inlier,
        iv_se,
        scatter_se,
    ) = combine_rv_estimates_median_mad_weighted(agg_rv, agg_er, outlier_sigma)

    n_in = int(np.sum(comb_inlier))
    mad_inliers = float("nan")
    if n_in > 1:
        ins = agg_rv[comb_inlier]
        mad_inliers = float(mad_std(ins, ignore_nan=True))
    scatter_note = (
        f"{scatter_se:.3f}"
        if np.isfinite(scatter_se)
        else "n/a (need ≥2 inlier fiber_agg inputs)"
    )
    mad_note = (
        f"{mad_inliers:.3f}"
        if np.isfinite(mad_inliers)
        else "n/a (need ≥2 inlier fiber_agg RVs)"
    )

    lines.extend(
        [
            "",
            f"RVCCFAVG (written to L2):     RV = {avg_rv:.3f}   RV_ERR = {avg_err:.3f}",
            f"Total bias (RVCCFAVG − inj):  {avg_rv - injected_rv_m_s:.3f}",
            "",
            "Recomputed from RVCCF table (fiber_agg collapse → "
            "combine_rv_estimates_median_mad_weighted):",
            f"  RV (MAD/IV combined):     {comb_rv:.3f}",
            f"  RV_ERR (max IV_SE, MAD/√N): {comb_err:.3f}",
            f"  IV-only SE (1/√Σw):       {iv_se:.3f}",
            f"  MAD/√N component:         {scatter_note}",
            f"  n_inlier / n_fiber_agg:   {n_in} / {len(agg_rv)}",
            "  MAD among inlier fiber_agg RVs (dispersion of order/chunk inputs): "
            f"{mad_note}",
            "=" * 72,
        ]
    )
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
    window_size_um: float | None = None,
    l2_model: datamodels.L2Model | None = None,
    mask_kpf_parabolic_vertex_n_samples: int = _BTSETTL_MASK_KPF_PARABOLIC_VERTEX_N_SAMPLES,
) -> list[tuple[str, float, str, bool]]:
    """
    BSPEC L2 + ``ccf_mask_kpf`` with a packaged ``.mas`` mask (previews / mask-overlay HTML).

    By default this loads lte049 and builds an L2 via :func:`_build_bspec_l2_btsettl_doppler_poly`
    (Doppler-shifted BT-Settl × polynomial in observed µm). That model is **not** identical to the
    synthetic L2 stacked in :func:`test_fit_ccf_kpf_style_btsettl_bspec_all_chunks_parallel`
    (interp only, no polynomial). For that test, pass ``l2_model=…`` (e.g. the same
    :class:`~hispecdrp.datamodels.L2Model` used for the RVCCF validation ``apply``) so preview RVs
    and the printed recovery report refer to the **same** spectra.

    ``mask_path`` defaults to ``K2_all.mas`` (serial BT-Settl preview). Pass the template-derived
    ``lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.mas`` path for the parallel test (same as :data:`LTE049_BTSETTL_MASK_MAS`).

    ``window_size_um`` defaults to :data:`WINDOW_UM` (0.3 µm). Pass the same value as the
    main parallel test’s mask-KPF chunk width so preview RVs and RVCCF tables match the
    validation ``StellarRVPrimitive`` pass.

    ``do_orders`` is forwarded to :class:`StellarRVPrimitive` and passed again to
    :meth:`~hispecdrp.primitives.base_primitive.BasePrimitive.apply` as ``do_orders=…`` so the
    echelle subset matches the test selection (including ``HISPEC_DRP_BSPEC_BTSETTL_SINGLE_ORDER``).
    When the argument is ``None`` (= all echelle orders), uses :func:`_mask_kpf_preview_do_orders`
    (same selection as mask-KPF BT-Settl tests: default single order 139 unless
    ``HISPEC_DRP_BSPEC_BTSETTL_ALL_ORDERS=1``).

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

    w_um = float(window_size_um) if window_size_um is not None else float(WINDOW_UM)

    if mask_path is None:
        mask_path = os.path.join(MASK_DIR, "K2_all.mas")
    if not os.path.isfile(mask_path):
        pytest.skip(f"mask not found: {mask_path}")

    if l2_model is not None:
        model = l2_model
    else:
        lte049 = _lte049_btsettl_spec_h5_path()
        if not os.path.isfile(lte049):
            pytest.skip(f"lte049 template not found: {lte049}")

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
                window_size_um=w_um,
                v_step=50.0,
                ccf_mask_kpf_auto_hybrid_scout=False,
                mask_kpf_parabolic_vertex_n_samples=int(mask_kpf_parabolic_vertex_n_samples),
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
                out = prim.apply(model, do_orders=do_orders)
            apply_wall = time.perf_counter() - t0
        else:
            prim = StellarRVPrimitive(
                algorithm="ccf_mask_kpf",
                mask_path=mask_path,
                do_fibers=["1"],
                do_orders=do_orders,
                window_size_um=w_um,
                v_step=50.0,
                ccf_mask_kpf_auto_hybrid_scout=False,
                mask_kpf_parabolic_vertex_n_samples=int(mask_kpf_parabolic_vertex_n_samples),
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
                    out = prim.apply(model, do_orders=do_orders)
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
def lte049_btsettl_session_cache():
    """Load :data:`LTE049_BTSETTL_SPEC_H5` once per session (parallel mask-KPF integration test)."""
    path = _lte049_btsettl_spec_h5_path()
    if not os.path.isfile(path):
        pytest.skip(
            "BT-Settl template not found (required for test_fit_ccf_kpf_style_btsettl_bspec_all_chunks_parallel): "
            f"{path}"
        )
    wl_min, wl_max = _BTSETTL_WL_RANGE_UM
    w_rest, flux_rest = _load_btsettl_sorted_micron(path, wl_min, wl_max)
    return {
        "w_rest": w_rest,
        "flux_rest": flux_rest,
        "template_ln": np.log(w_rest),
    }


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


@pytest.mark.slow
def test_fit_ccf_kpf_style_btsettl_bspec_all_chunks_parallel(
    tmp_path, lte049_btsettl_session_cache
):
    """
    BT-Settl **lte049** (:data:`LTE049_BTSETTL_SPEC_H5`) + BSPEC echelle model: one call to
    :func:`process_orders_rvs_numba_parallel`
    with ``max_cores=6`` (clamped
    to Numba’s allowed thread count on the host, e.g. 2 on small CI runners) and
    ``ccf_v_step`` = :data:`_TEST_TEMPLATE_CCF_V_STEP_M_S` (50 m/s; same as ``ccf_utils._V_STEP``).
    Phase-B Gaussian fits run serially in Python.

    By default only **one** BSPEC echelle order is used (default **139**). For all echelle orders
    from ``HISPECConfig``, set ``HISPEC_DRP_BSPEC_BTSETTL_ALL_ORDERS=1`` or pass pytest
    ``--bspec-all-orders``. Use ``HISPEC_DRP_BSPEC_BTSETTL_SINGLE_ORDER=n`` to override the
    default single order (that order is passed to :class:`~hispecdrp.primitives.StellarRVPrimitive`
    as ``do_orders`` on construct and on ``apply``). Mask-KPF and template CCF chunking use
    :data:`WINDOW_UM` (0.3 µm); a typical echelle order span is smaller, so expect **one chunk
    per order** unless ``HISPEC_DRP_BSPEC_BTSETTL_ALL_ORDERS`` yields many orders.

    Mask-line CCF always uses :data:`LTE049_BTSETTL_MASK_MAS` (paired with the same lte049 spectrum), not
    ``M2_filt_allRV.mas``. Velocity sampling is the **broad uniform** grid only (50 m/s); there is no
    merged hybrid grid (``ccf_input_velocity`` unset; ``ccf_mask_kpf_auto_hybrid_scout=False`` in this test).

    Mask-KPF sub-pixel refinement uses :data:`_BTSETTL_MASK_KPF_PARABOLIC_VERTEX_N_SAMPLES` (31)
    for ``mask_kpf_parabolic_vertex_n_samples``; the test asserts |RVCCFAVG − injected RV| < 1 m/s.

    Assertions require at least 70% of eligible chunks to have finite template CCF fits and a
    minimum count within tolerance vs the injected RV (scaled when only one chunk is eligible).

    After building the synthetic spectra, zero-flux pixels (including forced order endpoints)
    are assigned ``DQ`` ``DO_NOT_USE`` so CCF and mask-KPF paths exclude them via the standard
    bad-pixel mask.

    **Printed RV summary:** run with ``pytest -s`` (disable output capture). The test prints the
    injected RV, per-order recovered RV/uncertainty, RVCCFAVG aggregate, and recombined MAD/IV
    statistics **after** the wall-time summary so they appear at the end of the log (not buried
    under preview / INFO lines).
    """
    timing_rows: list[tuple[str, float, str, bool]] = []
    mas_lte049 = _lte049_btsettl_paired_mask_path()
    if not os.path.isfile(mas_lte049):
        pytest.skip(
            "Mask file not found (required for this test): "
            f"{mas_lte049}"
        )
    input_rv_m_s = INPUT_RV_KM_S * 1000.0
    v_step = _TEST_TEMPLATE_CCF_V_STEP_M_S
    n_pix = _TEST_ECHELLE_N_PIX
    t0 = time.perf_counter()
    w_rest = lte049_btsettl_session_cache["w_rest"]
    flux_rest = lte049_btsettl_session_cache["flux_rest"]
    template_ln = lte049_btsettl_session_cache["template_ln"]
    assert len(w_rest) > 50
    orders = _bspec_btsettl_ccf_mask_kpf_order_ids()
    timing_rows.append(
        (
            "lte049: session cache (w_rest, flux_rest, template_ln) + BSPEC mask-KPF order selection",
            time.perf_counter() - t0,
            "lte049_btsettl_session_cache, _bspec_btsettl_ccf_mask_kpf_order_ids",
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

    window_um_eff = float(WINDOW_UM)  # 0.3 µm — same default as pipeline ``window_size_um``
    timing_rows.append(
        (
            "lte049: stack BSPEC 2-D model (interp Doppler-shifted BT-Settl per order)",
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
            f"lte049: process_orders_rvs_numba_parallel (numba_threads={nthr}, "
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
    assert n_attempt >= 1, "Expected at least one eligible template CCF chunk"

    sel_fit = eligible & np.isfinite(all_rvs) & np.isfinite(all_rv_errs) & (all_rv_errs > 0)
    n_fit_ok = int(np.sum(sel_fit))
    tol_vec = np.maximum(8000.0, 3.0 * all_rv_errs)
    in_tol = (
        sel_fit
        & (np.abs(all_rvs - input_rv_m_s) < tol_vec)
        & (all_rv_errs < 1e6)
    )
    n_in_tol = int(np.sum(in_tol))

    # lte049 template spectrum yields looser per-chunk agreement vs lte025; keep meaningful floors.
    assert n_fit_ok >= 0.70 * n_attempt, f"fit fraction too low: {n_fit_ok}/{n_attempt}"
    min_in_tol = max(1, min(n_attempt, max(2, int(0.15 * n_attempt))))
    assert n_in_tol >= min_in_tol, f"RV recovery fraction too low: {n_in_tol}/{n_attempt} (need >= {min_in_tol})"

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
        v_step=50.0,
        ccf_mask_kpf_auto_hybrid_scout=False,
        mask_kpf_parabolic_vertex_n_samples=_BTSETTL_MASK_KPF_PARABOLIC_VERTEX_N_SAMPLES,
        outlier_sigma=4.0,
        max_cores=6,
        save_preview=False,
        show_preview=False,
        output_dir=str(tmp_path),
        suffix="parallel_rvccf_hdu_validate",
    )
    out_rvccf = prim_rvccf.apply(rvccf_l2, do_orders=orders)
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

    avg_tbl = getattr(out_rvccf, "rvccfavg_fib1", None)
    assert avg_tbl is not None and len(avg_tbl) > 0
    rv_ccfavg = float(avg_tbl["RV"][0])
    assert abs(rv_ccfavg - input_rv_m_s) < 1.0, (
        "mask-KPF aggregate RV vs injected: expect |bias| < 1 m/s with "
        f"mask_kpf_parabolic_vertex_n_samples={_BTSETTL_MASK_KPF_PARABOLIC_VERTEX_N_SAMPLES} "
        f"(got RVCCFAVG={rv_ccfavg:.4f} m/s, injected={input_rv_m_s:.4f} m/s)"
    )

    # ------------------------------------------------------------------
    # ccf_mask_kpf chunk-parallel benchmark (rv_per_chunks_mask_kpf path)
    # ------------------------------------------------------------------
    # With WINDOW_UM=0.3, many single-order calls have only 1 chunk, which can
    # hide parallel scaling. Use a smaller benchmark window to force many chunks.
    #
    # Each synthetic echelle row has span ~147/order^2 µm (see _order_wavelength_grid).
    # A fixed 0.01 µm window can be *wider* than the order span, so
    # ``ccf_order_n_chunks_for_span`` yields 1 chunk and Numba ``prange`` over chunks
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
                f"lte049: ccf_mask_kpf chunk benchmark req_cores={req_cores} "
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
                window_size_um=window_um_eff,
                l2_model=out_rvccf,
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

    _print_btsettl_mask_kpf_recovery_report(
        out_rvccf,
        tbl_rvccf,
        injected_rv_m_s=input_rv_m_s,
        outlier_sigma=float(getattr(prim_rvccf, "outlier_sigma", 4.0)),
    )

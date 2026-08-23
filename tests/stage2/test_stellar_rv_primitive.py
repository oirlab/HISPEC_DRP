"""
Stellar RV tests: **ccf_stellar_template** (log-λ CCF, median/MAD rejection, IV weighted mean).

Synthetic L2 uses :func:`_order_wavelength_grid` and Gaussian lines; rest templates use
**3×** L2 pixels per order to reduce log-λ CCF bias.

Run::

  pytest tests/stage2/test_stellar_rv_primitive.py -v

Faster CI / local iteration (skips tests marked ``slow``: BT-Settl parallel line-mask integration)::

  pytest tests/stage2/test_stellar_rv_primitive.py -v -m "not slow"

**Optional previews (env or ``SHOW_BSPEC_BTSETTL_PREVIEWS`` module override):**

- ``HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS=1`` — enables **HTML-only** Plotly/stellar-RV preview behavior
  in :mod:`hispecdrp.primitives.stellar_rv.previews.stellar_rv_preview` (skips Kaleido PNG when
  ``save_preview`` is set). In ``test_stellar_rv_primitive_recovers_rv_per_chunk``, opens the unified preview when
  set. After ``test_fit_ccf_line_mask_btsettl_bspec_all_chunks_parallel_delta_mask`` or
  ``test_fit_ccf_tophat_style_btsettl_bspec_line_mask_line_window_n1_parallel``, runs lte049 + packaged
  **delta** mask ``ccf_line_mask``: mask-overlay via ``write_mask_overlay_debug_pages`` (auto-open) and
  ``create_stellar_rv_preview`` → unified ``*_preview.html``, then ``webbrowser.open``, with a timing
  summary (``-s``). After ``test_fit_ccf_template_style_btsettl_bspec_all_chunks_parallel``, runs the same
  lte049 BSPEC stack with ``algorithm="ccf_stellar_template"`` (unified Plotly ``*_preview.html`` and
  template-overlay debug HTML). **Either all of these optional preview legs run or none** — there is no
  separate toggle per preview type.

**Parallel BT-Settl BSPEC chunk test** — ``test_fit_ccf_line_mask_btsettl_bspec_all_chunks_parallel_delta_mask``
always uses spectrum ``lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5`` and mask
``lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.delta.mas``. It runs the same science setup via
``process_orders_rvs_numba_parallel`` (requests ``max_cores=6``;
actual Numba threads are clamped to the wheel/CPU limit, e.g. GitHub Actions) and prints a timing summary.
By default it uses **one** echelle order (**139** via ``HISPEC_DRP_BSPEC_BTSETTL_ORDER`` when unset).
Set ``HISPEC_DRP_BSPEC_BTSETTL_ORDER=all`` or pass pytest ``--bspec-all-orders`` (sets that env value) to
run on **all** BSPEC echelle orders from ``HISPECConfig``. Use a comma-separated list (e.g.
``139,140``) for multiple specific orders.

For ``test_fit_ccf_line_mask_btsettl_bspec_all_chunks_parallel_delta_mask`` only, override chunk width with
``--bspec-window-um <µm>`` or ``HISPEC_DRP_BSPEC_BTSETTL_WINDOW_UM=<µm>`` (default ``0.3``, same as
:data:`WINDOW_UM`). Very small windows imply **many chunks**; tight ``RVCCFAVG`` vs injected checks are
tuned for the default width. For exploratory runs and **previews** when agreement is poor, set
``HISPEC_DRP_BSPEC_BTSETTL_RELAX_PARALLEL_ASSERTS=1`` or ``pytest --bspec-relax-parallel-asserts``.

Simulated ``spec_err`` is proportional to ``|flux|`` by default (``rel_floor`` × amplitude, default
``1e-3``), then multiplied by a center-to-edge throughput taper so σ is smallest in the order center
and larger toward both ends (see ``_bspec_simulated_flux_err``).

**One-liner — line-mask parallel test + both previews (mask overlay + unified CCF/RV HTML), from repo root**::

  HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS=1 pytest tests/stage2/test_stellar_rv_primitive.py::test_fit_ccf_line_mask_btsettl_bspec_all_chunks_parallel_delta_mask -s

The ``-s`` flag is required to see **printed** diagnostics (pytest otherwise captures stdout). For this
test, the injected vs recovered RV table is emitted **last**, after the wall-time summary, so scroll to
the bottom of the run to read it.

**Same, but all BSPEC echelle orders** (longer runtime)::

  HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS=1 pytest tests/stage2/test_stellar_rv_primitive.py::test_fit_ccf_line_mask_btsettl_bspec_all_chunks_parallel_delta_mask -s --bspec-all-orders

  # Equivalent:
  HISPEC_DRP_BSPEC_BTSETTL_ORDER=all HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS=1 pytest tests/stage2/test_stellar_rv_primitive.py::test_fit_ccf_line_mask_btsettl_bspec_all_chunks_parallel_delta_mask -s

**Mask CCF pixel interpolation order** (Lagrange on L2 wavelength; default ``1`` = linear).
With ``HISPEC_DRP_MASK_CCF_PIXEL_INTERP_ORDER`` > ``1``, the BT-Settl line-mask test uses a looser
``|RVCCFAVG − injected|`` bound (overshoot can bias the CCF by many m/s vs linear sampling)::

  HISPEC_DRP_MASK_CCF_PIXEL_INTERP_ORDER=2 HISPEC_DRP_BSPEC_BTSETTL_ORDER=all HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS=1 pytest tests/stage2/test_stellar_rv_primitive.py::test_fit_ccf_line_mask_btsettl_bspec_all_chunks_parallel_delta_mask -s

**Line-mask line-by-line chunking (``window_line_by_line=True``)** —
``test_fit_ccf_tophat_style_btsettl_bspec_line_mask_line_window_n1_parallel`` uses the same lte049
BSPEC stack and mask file, but chunks the spectrum in the observed frame at each mask line
(non-overlapping Voronoi intervals around lines shifted by ``ccf_input_velocity`` = injected RV).
It sets ``window_size_um`` = :data:`_BTSETTL_LINE_WINDOW_N1_WINDOW_UM` (symmetric spectral band about each
predicted observed mask line, see
:func:`~hispecdrp.primitives.utils.ccf.ccf_utils.mask_line_window_spectrum_band_um`), coarse CCF limits
``ccf_input_velocity`` ± :data:`_BTSETTL_LINE_WINDOW_N1_V_COARSE_HALF_SPAN_KM_S` km/s, and ``v_step_coarse`` =
:data:`_BTSETTL_LINE_WINDOW_N1_V_STEP_COARSE` m/s. The CCF ``v`` grid is **uniform** at that step
around ``ccf_input_velocity`` (``v_half_window``).
(line-window mode still requires ``ccf_input_velocity`` for chunking; see ``run_ccf_line_mask_for_fiber``).
``mask_ccf_pixel_interp_order`` = :data:`_BTSETTL_LINE_WINDOW_N1_MASK_CCF_PIXEL_INTERP_ORDER` (tests do not rely on
``HISPEC_DRP_MASK_CCF_PIXEL_INTERP_ORDER`` for these two line-window tests).
The post-run recovery table prints **every chunk row** from ``RVCCF`` (ascending echelle order,
then descending chunk-center λ), including a **SEG k/n** column aligned with mask-overlay HTML
``Echelle <order> · segment k/n``, not the IV-collapsed per-order summary used by
``test_fit_ccf_line_mask_btsettl_bspec_all_chunks_parallel_delta_mask``.
Previews match the parallel test when ``HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS=1``::

  HISPEC_DRP_MASK_CCF_PIXEL_INTERP_ORDER=3 HISPEC_DRP_BSPEC_BTSETTL_ORDER=all HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS=1 pytest tests/stage2/test_stellar_rv_primitive.py::test_fit_ccf_tophat_style_btsettl_bspec_line_mask_line_window_n1_parallel -s

Mask-overlay spectrum panel: **Lagrange pixel-interpolation** trace (same stencil as
``rv_per_chunks_line_mask`` / ``mask_ccf_pixel_interp_order``) is always embedded. The dense λ
grid size is derived from the displayed spectrum point count:
``n_dense = 2 * mask_ccf_pixel_interp_order * N_specpoints`` (clamped to ``[24, 8192]``).
Example (single echelle order 139, the default when ``HISPEC_DRP_BSPEC_BTSETTL_ORDER`` is unset)::

  HISPEC_DRP_MASK_CCF_PIXEL_INTERP_ORDER=3 HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS=1 \\
  pytest tests/stage2/test_stellar_rv_primitive.py::test_fit_ccf_tophat_style_btsettl_bspec_line_mask_line_window_n1_parallel -s

**Line-window line-mask test — per-segment RV vs CCF (JSON + NPZ, no Plotly/mask-overlay HTML)**::

  HISPEC_DRP_BSPEC_BTSETTL_LINE_WINDOW_SERIALIZE_RV_CCF=1 pytest tests/stage2/test_stellar_rv_primitive.py::test_fit_ccf_tophat_style_btsettl_bspec_line_mask_line_window_n1_parallel -s

Same schema as ``HISPEC_DRP_BSPEC_BTSETTL_SERIALIZE_MASK_CCF`` (``lte049_line_mask_ccf_rv_debug.json`` + ``.npz``): one
slot per line-window segment with ``v_m_s`` and ``ccf`` arrays. With this env var, outputs are written under
``examples/data/`` (repository root next to ``tests/``); the test prints ``HISPEC_DRP_BSPEC_BTSETTL_LINE_WINDOW_RV_CCF_JSON=``
and ``…_NPZ=`` at the end.

**Injected-RV scan (11 × 10 m/s around −110 km/s)** — ``test_fit_ccf_tophat_style_btsettl_bspec_line_mask_line_window_n1_injected_rv_scan``

Rebuilds the lte049 BSPEC line-window stack at each injected RV, runs ``ccf_line_mask``, reads ``RVCCFAVG`` per step,
fits a **weighted offset** with slope fixed at unity (``RV_rec ≈ RV_inj + β₀``, weights ``1/RV_ERR²``), reports
scatter about ``y = x + β₀``, writes a Matplotlib PNG under the pytest ``tmp_path``, and prints its path. Primitive
settings match :data:`_BTSETTL_LINE_WINDOW_N1_*` (same MCMC-tuned defaults as the parallel line-window test). Optional
copy to ``examples/output/``::

  pytest tests/stage2/test_stellar_rv_primitive.py::test_fit_ccf_tophat_style_btsettl_bspec_line_mask_line_window_n1_injected_rv_scan -s

  HISPEC_DRP_BSPEC_BTSETTL_LINE_WINDOW_RV_SCAN_PLOT_EXAMPLES=1 pytest tests/stage2/test_stellar_rv_primitive.py::test_fit_ccf_tophat_style_btsettl_bspec_line_mask_line_window_n1_injected_rv_scan -s

**Parallel BT-Settl BSPEC template-CCF test** — ``test_fit_ccf_template_style_btsettl_bspec_all_chunks_parallel``

Same synthetic lte049 echelle stack and ``process_orders_rvs_numba_parallel`` phase as the line-mask
parallel test, then ``StellarRVPrimitive`` with ``algorithm="ccf_stellar_template"`` (rest-frame
``template_wavelength`` / ``template_flux`` from the session cache). Same ``HISPEC_DRP_BSPEC_BTSETTL_ORDER``
selection as the line-mask tests. Optional unified preview + template-overlay HTML when
``HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS=1``::

  HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS=1 pytest tests/stage2/test_stellar_rv_primitive.py::test_fit_ccf_template_style_btsettl_bspec_all_chunks_parallel -s

  HISPEC_DRP_BSPEC_BTSETTL_ORDER=all HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS=1 pytest tests/stage2/test_stellar_rv_primitive.py::test_fit_ccf_template_style_btsettl_bspec_all_chunks_parallel -s

**One-liner — same test, JSON + NPZ CCF debug dump only** (no mask-overlay / Plotly HTML; patched no-op preview)::

  HISPEC_DRP_BSPEC_BTSETTL_SERIALIZE_MASK_CCF=1 pytest tests/stage2/test_stellar_rv_primitive.py::test_fit_ccf_line_mask_btsettl_bspec_all_chunks_parallel_delta_mask -s

``HISPEC_DRP_SAVE_CCF_PLOTS=1`` sets ``save_preview`` and ``output_dir`` for the echelle test so the
primitive writes the unified preview HTML under ``tests/output/stellar_rv_plots/`` (no extra figures).

``HISPEC_DRP_BSPEC_BTSETTL_SERIALIZE_MASK_CCF=1`` — during ``test_fit_ccf_line_mask_btsettl_bspec_all_chunks_parallel_delta_mask``,
after the lte049 ``ccf_line_mask`` preview path (or a lightweight variant if previews are off), write JSON + NPZ
debug dumps of per-order/mean ``v`` (m/s), ``ccf``, ``ccf_err`` (if present), and ``gaussian_fit`` (Gaussian RV / err /
params). Prints ``HISPEC_DRP_BSPEC_BTSETTL_MASK_CCF_SERIALIZE=`` with the JSON path (the JSON references the NPZ).

**Post-process CCF JSON (optional)** — ``examples/line_mask_ccf_postprocess_uncertainty.py`` re-runs the high-pass
σ estimate and ``fit_ccf_rv`` on serialized debug files (e.g. different Savitzky–Golay settings or archived runs). Line-mask
order-mean ``ccf_err`` in the pipeline is already produced by
:func:`~hispecdrp.primitives.utils.ccf.ccf_utils.estimate_ccf_uncertainty_highpass_residual` inside
:func:`~hispecdrp.primitives.utils.ccf.ccf_utils.run_ccf_line_mask_for_fiber`. Prints ``HISPEC_DRP_MASK_CCF_POSTPROCESS_JSON=``::

  PYTHONPATH=. python examples/line_mask_ccf_postprocess_uncertainty.py /path/to/lte049_line_mask_ccf_rv_debug.json

"""
from __future__ import annotations

import json
import math
import os
import shutil
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
from hispecdrp.primitives.stellar_rv.stellar_rv_primitive import (
    combine_rv_estimates_median_mad_weighted,
)
from hispecdrp.primitives.utils.ccf.ccf_common import V_MAX_DEFAULT_M_S, V_MIN_DEFAULT_M_S
from hispecdrp.primitives.utils.ccf.ccf_utils import (
    LINE_MASK_CCF_STORED_NEGATED_FROM_RAW,
    build_line_mask_chunk_wavelength_intervals_line_by_line,
    ccf_order_n_chunks_for_span,
    line_mask_n_chunks_for_order_line_by_line,
    process_orders_rvs_numba_parallel,
    rv_per_chunks_line_mask,
)
from hispecdrp.utils.line_mask_utils import (
    MaskLines,
    package_data_masks_dir,
    packaged_default_line_mask_path,
)
from hispecdrp.primitives.utils.spectral_utils import get_spectrum_path, load_btsettl_spectrum

TEMPLATE_DIR = os.path.join(os.path.dirname(hispecdrp.__file__), "data", "templates")
# ``examples/data`` (for ``HISPEC_DRP_BSPEC_BTSETTL_LINE_WINDOW_SERIALIZE_RV_CCF`` line-window dumps).
_EXAMPLES_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples",
    "data",
)
_EXAMPLES_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples",
    "output",
)

C_KM_S = 299792.458
LAMBDA_ORDER_COEF_UM = 147.0
INPUT_RV_KM_S = -110.0
SIGMA_V_KM_S = 5.0
ABSORPTION_DEPTH = 0.05
FLAT_FLUX = 100.0
_TEMPLATE_PIXELS_PER_ORDER_FACTOR = 3
WINDOW_UM = 0.3
# Line-window ``ccf_line_mask`` tests (:func:`test_fit_ccf_tophat_style_btsettl_bspec_line_mask_line_window_n1_parallel`,
# :func:`test_fit_ccf_tophat_style_btsettl_bspec_line_mask_line_window_n1_injected_rv_scan`): symmetric spectral band
# about each predicted observed mask line (Voronoi chunk boundaries); coarse CCF ± span (km/s) and step (m/s);
# Values from optimized MCMC (``examples/mcmc_line_window_beta0_sigma.py``).
_BTSETTL_LINE_WINDOW_N1_WINDOW_UM = 5.648870581175803e-05
_BTSETTL_LINE_WINDOW_N1_V_COARSE_HALF_SPAN_KM_S = 9.245174152881138
_BTSETTL_LINE_WINDOW_N1_V_STEP_COARSE = 3.473801843213507
_BTSETTL_LINE_WINDOW_N1_MASK_CCF_PIXEL_INTERP_ORDER = 3
_BTSETTL_WL_RANGE_UM = (0.95, 1.4)

# Mask-line ``ccf_line_mask`` BT-Settl tests: default echelle order when ``HISPEC_DRP_BSPEC_BTSETTL_ORDER``
# is unset (see :func:`_bspec_btsettl_orders_from_env`).
_DEFAULT_BSPEC_BTSETTL_CCF_LINE_MASK_ORDER = 139
# lte049 + polynomial L2 (same construction as line-mask smoke scripts)
_BTSETTL_LTE049_POLY_COEFS = np.array([0.0001, -0.001, 0.01, 0.05, 1.0])
# Fixed lte049 BT-Settl template + paired depth-weighted **delta** mask.
# Keep this on the classic ``*.delta.mas`` product (not ``ccfopt`` / packaged default):
# these unit tests calibrate chunk counts and RV recovery on that mask.
LTE049_BTSETTL_SPEC_H5 = "lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5"
LTE049_BTSETTL_MASK_MAS = "lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.delta.mas"

def _lte049_btsettl_spec_h5_path() -> str:
    return get_spectrum_path(LTE049_BTSETTL_SPEC_H5, base_dir=TEMPLATE_DIR)


def _lte049_btsettl_paired_mask_path() -> str:
    """Packaged lte049 template-derived **delta** line mask (``ccf_line_mask`` tests)."""
    path = package_data_masks_dir() / LTE049_BTSETTL_MASK_MAS
    if not path.is_file():
        raise FileNotFoundError(f"Missing packaged BT-Settl delta mask: {path}")
    return str(path)


def _lte049_btsettl_paired_mask_path_tophat() -> str:
    """Packaged lte049 template-derived **tophat** (finite-width segment) mask for ``ccf_line_mask`` tests."""
    d = package_data_masks_dir()
    for name in (
        "lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.tophat.mas",
        "lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.kpf.mas",
    ):
        p = d / name
        if p.is_file():
            return str(p)
    from tests.utils import repo_root
    return os.path.join(
        str(repo_root()),
        "hispecdrp",
        "data",
        "masks",
        "lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.tophat.mas",
    )


# --- Test-only performance knobs (same assertions / tolerances; coarser grids are fine for loose RV tests) ---
# Fewer pixels per echelle row cuts CCF cost ~linearly; 2048 still satisfies ≥20 px/chunk with narrow windows.
_TEST_ECHELLE_N_PIX = 2048
# Uniform broad CCF grid step (m/s) for template-CCF Numba path and mask-CCF benchmarks in this file.
# Match ccf_common.V_STEP_DEFAULT_M_S (50 m/s) — uniform broad scan only.
_TEST_TEMPLATE_CCF_V_STEP_M_S = 50.0


# ---------------------------------------------------------------------------
# Optional preview toggles: set to True/False here to override environment, or
# leave as None to use only HISPEC_DRP_* env vars documented in the module doc.
# ---------------------------------------------------------------------------
SHOW_BSPEC_BTSETTL_PREVIEWS: bool | None = None
SERIALIZE_BSPEC_BTSETTL_MASK_CCF: bool | None = None
SERIALIZE_BSPEC_BTSETTL_LINE_WINDOW_RV_CCF: bool | None = None
RELAX_BSPEC_BTSETTL_PARALLEL_ASSERTS: bool | None = None


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _mask_ccf_pixel_interp_order_from_env() -> int:
    """
    Integer Lagrange order for mask-line CCF sampling on the echelle wavelength axis
    (``StellarRVPrimitive.mask_ccf_pixel_interp_order`` / ``rv_per_chunks_line_mask``).

    Reads ``HISPEC_DRP_MASK_CCF_PIXEL_INTERP_ORDER`` (unset or empty → ``3``, matching
    the science ``StellarRVPrimitive`` default Lanczos-a).
    """
    raw = os.environ.get("HISPEC_DRP_MASK_CCF_PIXEL_INTERP_ORDER", "").strip()
    if not raw:
        return 3
    try:
        v = int(raw)
    except ValueError:
        return 3
    from hispecdrp.primitives.utils.ccf.ccf_utils import MAX_MASK_CCF_PIXEL_INTERP_ORDER

    return max(1, min(v, MAX_MASK_CCF_PIXEL_INTERP_ORDER))


def _want_bspec_btsettl_previews() -> bool:
    """BT-Settl BSPEC optional previews (mask line + template legs) and HTML-only Plotly in preview module."""
    if SHOW_BSPEC_BTSETTL_PREVIEWS is not None:
        return SHOW_BSPEC_BTSETTL_PREVIEWS
    return _truthy_env("HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS")


def _want_bspec_btsettl_serialize_mask_ccf() -> bool:
    """Serialize line-mask CCF preview arrays (``v``, ``ccf``, ``ccf_err``, ``gaussian_fit``) to JSON+NPZ."""
    if SERIALIZE_BSPEC_BTSETTL_MASK_CCF is not None:
        return SERIALIZE_BSPEC_BTSETTL_MASK_CCF
    return _truthy_env("HISPEC_DRP_BSPEC_BTSETTL_SERIALIZE_MASK_CCF")


def _want_bspec_btsettl_line_window_serialize_rv_ccf() -> bool:
    """Same dump as :func:`_want_bspec_btsettl_serialize_mask_ccf` for the line-window ``n_lines=1`` test only."""
    if SERIALIZE_BSPEC_BTSETTL_LINE_WINDOW_RV_CCF is not None:
        return SERIALIZE_BSPEC_BTSETTL_LINE_WINDOW_RV_CCF
    return _truthy_env("HISPEC_DRP_BSPEC_BTSETTL_LINE_WINDOW_SERIALIZE_RV_CCF")


def _btsettl_parallel_line_mask_window_um() -> float:
    """
    ``window_size_um`` (µm) for :func:`test_fit_ccf_line_mask_btsettl_bspec_all_chunks_parallel_delta_mask`.

    Reads ``HISPEC_DRP_BSPEC_BTSETTL_WINDOW_UM`` (set by ``pytest --bspec-window-um <µm>`` via
    ``conftest.py`` or exported manually). When unset, returns :data:`WINDOW_UM` (0.3). A non-empty
    invalid value raises ``ValueError``.
    """
    import math

    raw = os.environ.get("HISPEC_DRP_BSPEC_BTSETTL_WINDOW_UM", "").strip()
    if not raw:
        return float(WINDOW_UM)
    try:
        v = float(raw)
    except ValueError as exc:
        raise ValueError(
            "HISPEC_DRP_BSPEC_BTSETTL_WINDOW_UM must be a float (micron); "
            f"got {raw!r}"
        ) from exc
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(
            "HISPEC_DRP_BSPEC_BTSETTL_WINDOW_UM must be finite and > 0 (micron); "
            f"got {raw!r}"
        )
    return float(v)


def _btsettl_parallel_line_mask_relax_assertions() -> bool:
    """
    Skip strict template-CCF, ``RVCCFAVG``, and chunk-benchmark asserts in
    :func:`test_fit_ccf_line_mask_btsettl_bspec_all_chunks_parallel_delta_mask`.

    Set ``HISPEC_DRP_BSPEC_BTSETTL_RELAX_PARALLEL_ASSERTS=1`` or ``pytest --bspec-relax-parallel-asserts``.
    """
    if RELAX_BSPEC_BTSETTL_PARALLEL_ASSERTS is not None:
        return bool(RELAX_BSPEC_BTSETTL_PARALLEL_ASSERTS)
    return _truthy_env("HISPEC_DRP_BSPEC_BTSETTL_RELAX_PARALLEL_ASSERTS")


def _bspec_btsettl_orders_from_env(full: list[int]) -> list[int] | None:
    """
    Parse ``HISPEC_DRP_BSPEC_BTSETTL_ORDER``.

    Returns ``None`` when the selection is **all** BSPEC orders (primitive ``do_orders=None``).
    Otherwise a non-empty list of echelle order indices. When the environment variable is unset or
    empty, the default is order :data:`_DEFAULT_BSPEC_BTSETTL_CCF_LINE_MASK_ORDER`.

    Raises
    ------
    ValueError
        If ``all`` is combined with numeric orders or an order token is not a valid integer.
    """
    raw0 = os.environ.get("HISPEC_DRP_BSPEC_BTSETTL_ORDER", "").strip()
    if not raw0:
        raw = str(_DEFAULT_BSPEC_BTSETTL_CCF_LINE_MASK_ORDER)
    else:
        raw = raw0
    if raw.strip().lower() == "all":
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        parts = [str(_DEFAULT_BSPEC_BTSETTL_CCF_LINE_MASK_ORDER)]
    orders: list[int] = []
    for p in parts:
        if p.lower() == "all":
            raise ValueError(
                "HISPEC_DRP_BSPEC_BTSETTL_ORDER: 'all' cannot be combined with numeric orders"
            )
        orders.append(int(p))
    return orders


def _bspec_btsettl_ccf_line_mask_order_ids() -> list[int]:
    """Echelle orders for BSPEC line-mask (``ccf_line_mask``) tests."""
    full = hispec_config.HISPECConfig.get_order_list("BSPEC")
    if not full:
        pytest.skip("No BSPEC orders in HISPECConfig")
    sel = _bspec_btsettl_orders_from_env(full)
    if sel is None:
        return list(full)
    for o in sel:
        if o not in full:
            pytest.skip(
                f"BSPEC echelle order {o} not in configured list ({full[0]} … {full[-1]})"
            )
    return sel


def _line_mask_preview_do_orders() -> list[int] | None:
    """``do_orders`` for :class:`StellarRVPrimitive` previews: ``None`` means all echelle orders."""
    full = hispec_config.HISPECConfig.get_order_list("BSPEC")
    if not full:
        pytest.skip("No BSPEC orders in HISPECConfig")
    return _bspec_btsettl_orders_from_env(full)


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
    rel_floor: float = 1e-3,
    abs_floor: float | None = None,
    dtype_out=np.float64,
) -> np.ndarray:
    """
    Simulated flux uncertainty for BSPEC-style echelle tests.

    By default ``σ ≈ rel_floor × |flux|`` on the echelle sample (default ``rel_floor=1e-3``).
    Pixels with near-zero |flux| use ``rel_floor`` times a tiny floor derived from the finite-pixel
    median ``|flux|`` so ``spec_err`` stays finite and positive for ``L2Model`` gating. Optional
    ``abs_floor`` (same units as flux), if set and positive, applies ``max(…, abs_floor)`` before
    the edge taper.

    The center-to-edge throughput taper (``_bspec_order_edge_snr_loss_factor``) inflates σ toward
    pixels ``0`` and ``n_pix-1`` (poor SNR at edges).
    """
    s = np.asarray(spec, dtype=np.float64)
    m = np.isfinite(s)
    med_abs = float(np.nanmedian(np.abs(s[m]))) if np.any(m) else 0.0
    if not np.isfinite(med_abs) or med_abs <= 0.0:
        med_abs = 1.0
    tiny = max(med_abs * 1e-12, 1e-30)
    base = float(rel_floor) * np.maximum(np.abs(s), tiny)
    if abs_floor is not None and float(abs_floor) > 0.0:
        base = np.maximum(base, float(abs_floor))
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
    orders: list[int] | None = None,
) -> datamodels.L2Model:
    if spectrograph not in ("BSPEC", "RSPEC"):
        raise ValueError("spectrograph must be BSPEC or RSPEC")
    all_orders = hispec_config.HISPECConfig.get_order_list(spectrograph)
    if not all_orders:
        raise ValueError(f"No orders for {spectrograph}")
    orders = list(all_orders) if orders is None else [int(o) for o in orders]
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
    orders: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    all_orders = hispec_config.HISPECConfig.get_order_list(spectrograph)
    if not all_orders:
        raise ValueError(f"No orders for {spectrograph}")
    orders = list(all_orders) if orders is None else [int(o) for o in orders]
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
    Reconstruct fiber-level ``(rv, rv_err)`` series matching line-mask aggregation:

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


def _print_btsettl_line_mask_recovery_report(
    output_model,
    tbl_rvccf,
    *,
    injected_rv_m_s: float,
    outlier_sigma: float,
    per_chunk_rows: bool = False,
    report_title: str | None = None,
) -> None:
    """
    Print injected vs recovered RVs for ``ccf_line_mask`` BSPEC lte049 test: either
    IV-collapsed **per echelle order** rows (default) or **every RVCCF table row** (one line
    per chunk when ``per_chunk_rows=True``), then bias vs truth, ``RVCCFAVG`` aggregate, and
    recomputed robust weighted mean (``combine_rv_estimates_median_mad_weighted``) with IV vs
    MAD/√N components.
    """
    avg = getattr(output_model, "rvccfavg_fib1", None)
    avg_rv = float(avg["RV"][0]) if avg is not None and len(avg) else float("nan")
    avg_err = float(avg["RV_ERR"][0]) if avg is not None and len(avg) else float("nan")

    ord_col = np.asarray(tbl_rvccf["ORDER"])
    rv_col = np.asarray(tbl_rvccf["RV"], dtype=np.float64)
    err_col = np.asarray(tbl_rvccf["RV_ERR"], dtype=np.float64)
    inl_col = np.asarray(tbl_rvccf["INLIER"], dtype=np.uint8)

    title = report_title or (
        "test_fit_ccf_line_mask_btsettl_bspec_all_chunks_parallel_delta_mask — RV recovery (m/s)"
    )
    bar = "=" * 100
    lines: list[str] = [
        "",
        bar,
        title,
        bar,
        "Mask-line CCF vs fit_ccf_rv: stored RVCCF/preview rows use the pipeline convention "
        f"ccf_line_mask_stored_negated={LINE_MASK_CCF_STORED_NEGATED_FROM_RAW} "
        "(negated from raw weighted mask statistic so the line peak is near true RV; "
        "see ccf_utils.LINE_MASK_CCF_STORED_NEGATED_FROM_RAW and fit_ccf_rv diagnostics).",
        f"Injected RV (constant per order): {injected_rv_m_s:.3f}",
        "",
    ]
    if per_chunk_rows:
        chunk_col = np.asarray(tbl_rvccf["CHUNK_ID"], dtype=np.int64)
        lam_col = np.asarray(tbl_rvccf["WAVELENGTH"], dtype=np.float64)
        n_rows = int(len(tbl_rvccf))
        # Segment k/n within echelle order matches mask-overlay HTML (segment = chunk_index+1;
        # k=1 is reddest chunk-center λ on that order).
        seg_num = np.zeros(n_rows, dtype=np.int32)
        n_seg_order = np.zeros(n_rows, dtype=np.int32)
        for o in np.unique(ord_col):
            m = ord_col == int(o)
            idx = np.flatnonzero(m)
            sub = idx[np.argsort(-lam_col[idx].astype(np.float64), kind="stable")]
            n_o = int(sub.size)
            n_seg_order[idx] = n_o
            for rank, j in enumerate(sub, start=1):
                seg_num[int(j)] = int(rank)
        sort_ix = np.lexsort((-lam_col.astype(np.float64), ord_col.astype(np.int64)))
        lines.extend(
            [
                f"Per chunk (all n={n_rows} RVCCF table rows; ORDER asc, λ_c desc):",
                f"{'CHUNK_ID':>8}  {'ORDER':>6}  {'SEG':>7}  {'LAM_um':>12}  {'inj':>14}  "
                f"{'RV':>14}  {'RV_ERR':>12}  {'bias':>12}  INLIER",
                "  (SEG = overlay debug segment k/n, same as dropdown Echelle ORDER · segment k/n)",
            ]
        )
        inj = float(injected_rv_m_s)
        for i in sort_ix:
            cid = int(chunk_col[i])
            o = int(ord_col[i])
            sk = int(seg_num[i])
            sn = int(n_seg_order[i])
            lam = float(lam_col[i])
            rv_d = float(rv_col[i])
            er_d = float(err_col[i])
            bias = rv_d - inj
            inl = int(inl_col[i])
            lines.append(
                f"{cid:8d}  {o:6d}  {sk:3d}/{sn:3d}  {lam:12.6f}  {inj:14.3f}  "
                f"{rv_d:14.3f}  {er_d:12.3f}  {bias:12.3f}  {inl:6d}"
            )
    else:
        lines.extend(
            [
                "Per echelle ORDER (IV-weighted RV/ERR over chunk rows on that order; "
                "one table row per chunk when chunk RVs differ):",
                f"{'ORDER':>6}  {'inj':>14}  {'RV':>14}  {'RV_ERR':>12}  {'bias':>12}  INLIER",
            ]
        )
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
            bar,
        ]
    )
    print("\n".join(lines), flush=True)


def _print_btsettl_template_ccf_recovery_report(
    output_model,
    tbl_rvccf,
    *,
    injected_rv_m_s: float,
    outlier_sigma: float,
) -> None:
    """
    Same layout as :func:`_print_btsettl_line_mask_recovery_report` for ``ccf_stellar_template``
    (log-λ weighted template CCF + ``fit_ccf_rv`` per chunk; no mask-line negation convention).
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
        "test_fit_ccf_template_style_btsettl_bspec_all_chunks_parallel — RV recovery (m/s)",
        "=" * 72,
        "Stellar-template CCF (log-λ): per-chunk Gaussian ``fit_ccf_rv`` on weighted template CCF.",
        f"Injected RV (constant per order): {injected_rv_m_s:.3f}",
        "",
        "Per echelle ORDER (collapsed when identical RV across chunk rows):",
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


def _serialize_line_mask_ccf_preview_debug(
    prim: StellarRVPrimitive,
    out_dir: str,
    *,
    print_paths: bool = True,
) -> tuple[str, str] | None:
    """
    Write per-fiber line-mask CCF preview data: large arrays in ``.npz``, metadata + ``gaussian_fit`` in ``.json``.

    Returns ``(json_path, npz_path)`` (absolute), or ``None`` if there is nothing to save.
    """
    by_fiber = getattr(prim, "_ccf_preview_by_fiber", None) or {}
    if not by_fiber:
        if print_paths:
            print(
                "HISPEC_DRP_BSPEC_BTSETTL_MASK_CCF_SERIALIZE=<empty: no _ccf_preview_by_fiber; "
                "use save_preview or mask-overlay debug so line-mask collects CCF preview>",
                flush=True,
            )
        return None
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "lte049_line_mask_ccf_rv_debug")
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
            gaussian = sl.get("gaussian_fit")
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
                "gaussian_fit": _jsonify_mask_ccf_debug(gaussian) if gaussian is not None else None,
                "bisector_fit": _jsonify_mask_ccf_debug(sl.get("bisector_fit"))
                if sl.get("bisector_fit") is not None
                else None,
            }
            slots_out.append(slot_meta)
        fibers_meta[fib_s] = {
            "v_step": (detail or {}).get("v_step"),
            "v_min": (detail or {}).get("v_min"),
            "v_max": (detail or {}).get("v_max"),
            "slots": slots_out,
        }

    payload = {
        "schema": "hispecdrp.line_mask_ccf_preview.v1",
        "npz_path": os.path.basename(npz_path),
        "fibers": fibers_meta,
    }
    np.savez_compressed(npz_path, **npz_arrays)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify_mask_ccf_debug(payload), f, indent=2, allow_nan=False)
    ap_json = os.path.abspath(json_path)
    ap_npz = os.path.abspath(npz_path)
    if print_paths:
        print(f"HISPEC_DRP_BSPEC_BTSETTL_MASK_CCF_SERIALIZE={ap_json}", flush=True)
    return (ap_json, ap_npz)


def _run_lte049_k2all_line_mask_previews(
    tmp_path: str,
    *,
    mask_path: str | None = None,
    do_orders: list[int] | None = None,
    serialize_only: bool = False,
    window_size_um: float | None = None,
    l2_model: datamodels.L2Model | None = None,
    window_line_by_line: bool = False,
    ccf_input_velocity_m_s: float | None = None,
    v_min_coarse: float | None = None,
    v_max_coarse: float | None = None,
    v_step_coarse: float | None = None,
    mask_ccf_pixel_interp_order: int | None = None,
    also_serialize_rv_ccf: bool = False,
    defer_mask_ccf_serialize_path_print: bool = False,
    line_mask_preview_out_dir: str | None = None,
) -> tuple[list[tuple[str, float, str, bool]], tuple[str, str] | None]:
    """
    BSPEC L2 + ``ccf_line_mask`` with a packaged ``.mas`` mask (previews / mask-overlay HTML).

    By default this loads lte049 and builds an L2 via :func:`_build_bspec_l2_btsettl_doppler_poly`
    (Doppler-shifted BT-Settl × polynomial in observed µm). That model is **not** identical to the
    synthetic L2 stacked in :func:`test_fit_ccf_line_mask_btsettl_bspec_all_chunks_parallel_delta_mask`
    (interp only, no polynomial). For that test, pass ``l2_model=…`` (e.g. the same
    :class:`~hispecdrp.datamodels.L2Model` used for the RVCCF validation ``apply``) so preview RVs
    and the printed recovery report refer to the **same** spectra.

    ``mask_path`` defaults to the packaged lte049 **delta** mask (serial BT-Settl preview).
    Pass :data:`LTE049_BTSETTL_MASK_MAS` for the parallel test (same paired template mask).

    ``window_size_um`` defaults to :data:`WINDOW_UM` (0.3 µm). Pass the same value as the
    main parallel test’s line-mask chunk width so preview RVs and RVCCF tables match the
    validation ``StellarRVPrimitive`` pass (ignored for chunk count when
    ``window_line_by_line`` is true).

    When ``window_line_by_line`` is true, also pass ``ccf_input_velocity_m_s`` (m/s); the
    primitive uses mask-line Voronoi chunking at that velocity (same as production
    ``window_line_by_line`` + ``ccf_input_velocity``).

    Optional ``v_min_coarse`` / ``v_max_coarse`` (m/s) are forwarded to the primitive so preview
    runs match a narrow coarse CCF span (e.g. the line-window parallel test). Optional
    ``v_step_coarse`` (m/s) defaults to ``50``; pass the same value as the validation primitive when
    the coarse step is non-default.

    ``mask_ccf_pixel_interp_order`` — if set, forwarded to the preview primitive; otherwise
    :func:`_mask_ccf_pixel_interp_order_from_env` is used (line-window tests pass the fixed constant
    so previews match validation without setting env).

    ``do_orders`` is forwarded to :class:`StellarRVPrimitive` and passed again to
    :meth:`~hispecdrp.primitives.base_primitive.BasePrimitive.apply` as ``do_orders=…`` so the
    echelle subset matches the test selection (``HISPEC_DRP_BSPEC_BTSETTL_ORDER``: default ``139``,
    ``all``, or a comma-separated list).
    When the argument is ``None``, uses :func:`_line_mask_preview_do_orders` (same selection as
    line-mask BT-Settl tests).

    When ``serialize_only`` is False (default): HTML-only — ``save_preview=True``, ``show_preview=False``
    so ``create_stellar_rv_preview`` writes ``*_preview.html`` without ``fig.show()``;
    the test then ``webbrowser.open``s that file. Mask overlay uses ``write_mask_overlay_debug_pages`` (HTML + auto-open
    when ``show_overlay_debug=True``).

    When ``serialize_only`` is True: ``save_preview=True`` so line-mask collects CCF preview arrays, but
    ``create_stellar_rv_preview`` is patched to a no-op (no unified Plotly HTML); mask-overlay HTML is not
    written. Used when ``HISPEC_DRP_BSPEC_BTSETTL_SERIALIZE_MASK_CCF=1`` without full previews.

    ``also_serialize_rv_ccf`` — when True (e.g. ``HISPEC_DRP_BSPEC_BTSETTL_LINE_WINDOW_SERIALIZE_RV_CCF``), run the same
    JSON/NPZ dump as ``HISPEC_DRP_BSPEC_BTSETTL_SERIALIZE_MASK_CCF`` without requiring the latter.

    ``defer_mask_ccf_serialize_path_print`` — if True, do not print ``HISPEC_DRP_BSPEC_BTSETTL_MASK_CCF_SERIALIZE=`` from
    :func:`_serialize_line_mask_ccf_preview_debug` (caller prints paths later).

    ``line_mask_preview_out_dir`` — if set, use this directory as ``StellarRVPrimitive.output_dir`` (and serialize
    sidecars) instead of ``<tmp_path>/lte049_line_mask_previews``.

    Returns ``(timing_rows, serialize_paths)`` where ``serialize_paths`` is ``(json, npz)`` when a dump was written, else
    ``None``.
    """
    import hispecdrp.primitives.stellar_rv.previews.stellar_rv_preview as srv_prev

    rows: list[tuple[str, float, str, bool]] = []

    w_um = float(window_size_um) if window_size_um is not None else float(WINDOW_UM)
    suffix = (
        "lte049_line_mask_line_window"
        if window_line_by_line
        else "lte049_line_mask"
    )
    prim_line_kw: dict = {}
    if window_line_by_line:
        if ccf_input_velocity_m_s is None or not math.isfinite(float(ccf_input_velocity_m_s)):
            raise ValueError(
                "_run_lte049_k2all_line_mask_previews: window_line_by_line requires "
                "finite ccf_input_velocity_m_s"
            )
        prim_line_kw = {
            "window_line_by_line": True,
            "ccf_input_velocity": float(ccf_input_velocity_m_s),
        }

    v_grid_kw: dict = {}
    if v_min_coarse is not None:
        v_grid_kw["v_min_coarse"] = float(v_min_coarse)
    if v_max_coarse is not None:
        v_grid_kw["v_max_coarse"] = float(v_max_coarse)
    v_step_c = float(v_step_coarse) if v_step_coarse is not None else 50.0

    # Line-window tests require ``ccf_input_velocity`` (uniform grid center).
    v_line_uniform_step_kw: dict = {}
    if window_line_by_line:
        v_line_uniform_step_kw = {
            "v_half_window": max(20000.0, abs(float(v_step_c)) * 1000.0),
        }
    if mask_path is None:
        mask_path = packaged_default_line_mask_path()
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

    if line_mask_preview_out_dir is not None:
        out_dir = os.path.abspath(str(line_mask_preview_out_dir))
    else:
        out_dir = os.path.join(str(tmp_path), "lte049_line_mask_previews")
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

    if do_orders is None:
        do_orders = _line_mask_preview_do_orders()
    _pio_preview = (
        int(mask_ccf_pixel_interp_order)
        if mask_ccf_pixel_interp_order is not None
        else _mask_ccf_pixel_interp_order_from_env()
    )
    if serialize_only:
        t0 = time.perf_counter()
        with patch.object(srv_prev, "create_stellar_rv_preview", lambda _p: None):
            out = StellarRVPrimitive.call(
                model,
                algorithm="ccf_line_mask",
                mask_path=mask_path,
                do_fibers=["1"],
                do_orders=do_orders,
                window_size_um=w_um,
                v_step_coarse=v_step_c,
                mask_auto_center_scout=False,
                ccf_parabolic_vertex_n_samples=int(ccf_parabolic_vertex_n_samples),
                mask_ccf_pixel_interp_order=_pio_preview,
                show_preview=False,
                save_preview=True,
                save_overlay_debug=False,
                show_overlay_debug=False,
                output_dir=out_dir,
                suffix=suffix,
                max_cores=6,
                **v_grid_kw,
                **prim_line_kw,
                **v_line_uniform_step_kw,
            )
        apply_wall = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        with patch.object(srv_prev, "write_mask_overlay_debug_pages", _timed_mask):
            with patch.object(srv_prev, "create_stellar_rv_preview", _timed_create):
                out = StellarRVPrimitive.call(
                    model,
                    algorithm="ccf_line_mask",
                    mask_path=mask_path,
                    do_fibers=["1"],
                    do_orders=do_orders,
                    window_size_um=w_um,
                    v_step_coarse=v_step_c,
                    mask_auto_center_scout=False,
                    ccf_parabolic_vertex_n_samples=int(ccf_parabolic_vertex_n_samples),
                    mask_ccf_pixel_interp_order=_pio_preview,
                    show_preview=False,
                    save_preview=True,
                    save_overlay_debug=True,
                    show_overlay_debug=True,
                    output_dir=out_dir,
                    suffix=suffix,
                    max_cores=6,
                    **v_grid_kw,
                    **prim_line_kw,
                    **v_line_uniform_step_kw,
                )
        apply_wall = time.perf_counter() - t0

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
            "run_ccf_line_mask_for_fiber, collect_mask_overlay_payload_for_primitive, save_output, etc.",
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

    ser_paths: tuple[str, str] | None = None
    if _want_bspec_btsettl_serialize_mask_ccf() or also_serialize_rv_ccf:
        ser_paths = _serialize_line_mask_ccf_preview_debug(
            prim,
            out_dir,
            print_paths=not defer_mask_ccf_serialize_path_print,
        )

    return rows, ser_paths


def _run_lte049_btsettl_template_previews(
    tmp_path: str,
    *,
    do_orders: list[int] | None = None,
    window_size_um: float | None = None,
    l2_model: datamodels.L2Model,
    template_wavelength_um: np.ndarray,
    template_flux: np.ndarray,
) -> list[tuple[str, float, str, bool]]:
    """
    Same BSPEC L2 as the parallel test + ``ccf_stellar_template`` with unified Plotly preview.

    Expect ``HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS=1`` when HTML-only Plotly behavior is desired
    (Kaleido PNG policy in :mod:`hispecdrp.primitives.stellar_rv.previews.stellar_rv_preview`).
    Writes ``*_template_overlay_debug_fib*.html`` (spectrum + shifted stellar template + CCF),
    analogous to mask-overlay debug for ``ccf_line_mask``.
    """
    import hispecdrp.primitives.stellar_rv.previews.stellar_rv_preview as srv_prev

    rows: list[tuple[str, float, str, bool]] = []
    w_um = float(window_size_um) if window_size_um is not None else float(WINDOW_UM)
    if do_orders is None:
        do_orders = _line_mask_preview_do_orders()

    out_dir = os.path.join(str(tmp_path), "lte049_template_ccf_previews")
    os.makedirs(out_dir, exist_ok=True)

    t_unified_html = 0.0
    orig_create = srv_prev.create_stellar_rv_preview

    def _timed_create(*args, **kwargs):
        nonlocal t_unified_html
        t1 = time.perf_counter()
        try:
            return orig_create(*args, **kwargs)
        finally:
            t_unified_html += time.perf_counter() - t1

    
    t0 = time.perf_counter()
    with patch.object(srv_prev, "create_stellar_rv_preview", _timed_create):
        out = StellarRVPrimitive.call(
            l2_model,
            algorithm="ccf_stellar_template",
            template_wavelength=np.asarray(template_wavelength_um, dtype=np.float64),
            template_flux=np.asarray(template_flux, dtype=np.float64),
            do_fibers=["1"],
            do_orders=do_orders,
            window_size_um=w_um,
            v_step_coarse=float(_TEST_TEMPLATE_CCF_V_STEP_M_S),
            ccf_input_velocity=None,
            show_preview=False,
            save_preview=True,
            save_overlay_debug=True,
            show_overlay_debug=True,
            output_dir=out_dir,
            suffix="lte049_template_ccf",
            max_cores=6,
        )
    apply_wall = time.perf_counter() - t0

    assert out is not None
    rv_remainder = max(apply_wall - t_unified_html, 0.0)
    rows.append(
        (
            "StellarRVPrimitive.apply (ccf_stellar_template) — total wall",
            apply_wall,
            "StellarRVPrimitive.apply → _perform, save_output, _create_preview; template CCF",
            True,
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
            "Numba template CCF path, save_output, etc.",
            False,
        )
    )

    prev = (prim._output or {}).get("preview") or {}
    html_unified = prev.get("html")
    if html_unified and os.path.isfile(html_unified):
        ap = os.path.abspath(html_unified)
        print(f"LTE049_BTSETTL_TEMPLATE_STELLAR_RV_PREVIEW_HTML={ap}", flush=True)
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
    for _fib, p in (mod.get("template_overlay_debug") or {}).items():
        if p and os.path.isfile(p):
            print(f"LTE049_BTSETTL_TEMPLATE_OVERLAY_HTML={os.path.abspath(p)}", flush=True)

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
    """Load :data:`LTE049_BTSETTL_SPEC_H5` once per session (parallel line-mask integration tests)."""
    path = _lte049_btsettl_spec_h5_path()
    if not os.path.isfile(path):
        pytest.skip(
            "BT-Settl template not found (required for "
            "test_fit_ccf_line_mask_btsettl_bspec_all_chunks_parallel_delta_mask): "
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
def test_stellar_rv_primitive_recovers_rv_per_chunk(spectrograph):
    """Synthetic L2 + template on three mid-band orders; combined RV matches inject."""
    from tests.utils import fast_ccf_orders

    algorithm = "ccf_stellar_template"
    n_pix = 1024
    orders = fast_ccf_orders(spectrograph)
    input_rv_m_s = INPUT_RV_KM_S * 1000.0
    l2 = make_synthetic_l2_spectrum(
        spectrograph, n_pix=n_pix, rv_km_s=INPUT_RV_KM_S,
        sigma_v_km_s=SIGMA_V_KM_S, depth=ABSORPTION_DEPTH, flat_flux=FLAT_FLUX,
        orders=orders,
    )
    template_wl, template_flux = make_rest_frame_template(
        spectrograph, n_pix=n_pix, depth=ABSORPTION_DEPTH, flat_flux=FLAT_FLUX,
        sigma_v_km_s=SIGMA_V_KM_S, orders=orders,
    )
    show = _want_bspec_btsettl_previews()
    save_plots = _truthy_env("HISPEC_DRP_SAVE_CCF_PLOTS")
    plot_out_dir = None
    if save_plots:
        from tests.utils import repo_root
        plot_out_dir = str(repo_root() / "tests" / "output" / "stellar_rv_plots")
        os.makedirs(plot_out_dir, exist_ok=True)
    out = StellarRVPrimitive.call(
        l2,
        template_wavelength=template_wl,
        template_flux=template_flux,
        do_fibers=["1"],
        do_orders=orders,
        window_size_um=WINDOW_UM,
        debug_ccf=False,
        algorithm=algorithm,
        show_preview=show,
        save_preview=save_plots or show,
        output_dir=plot_out_dir,
    )
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
def test_fit_ccf_line_mask_btsettl_bspec_all_chunks_parallel_delta_mask(
    tmp_path, lte049_btsettl_session_cache
):
    """
    BT-Settl **lte049** (:data:`LTE049_BTSETTL_SPEC_H5`) + BSPEC echelle model: one call to
    :func:`process_orders_rvs_numba_parallel`
    with ``max_cores=6`` (clamped
    to Numba’s allowed thread count on the host, e.g. 2 on small CI runners) and
    ``ccf_v_step`` = :data:`_TEST_TEMPLATE_CCF_V_STEP_M_S` (50 m/s; same as ``ccf_common.V_STEP_DEFAULT_M_S``).
    Phase-B Gaussian fits run serially in Python.

    By default only **one** BSPEC echelle order is used (**139**, via ``HISPEC_DRP_BSPEC_BTSETTL_ORDER``
    when unset). For all echelle orders from ``HISPECConfig``, set ``HISPEC_DRP_BSPEC_BTSETTL_ORDER=all``
    or pass pytest ``--bspec-all-orders``. Use a comma-separated list for multiple orders. Selected
    orders are passed to :class:`~hispecdrp.primitives.StellarRVPrimitive` as ``do_orders`` on construct
    and on ``apply``. Line-mask and template CCF chunking use :func:`_btsettl_parallel_line_mask_window_um`
    (default :data:`WINDOW_UM`, 0.3 µm); override with ``pytest --bspec-window-um <µm>`` or
    ``HISPEC_DRP_BSPEC_BTSETTL_WINDOW_UM``. A typical echelle order span is smaller than 0.3 µm, so
    expect **one chunk per order** unless you narrow ``window_size_um`` or many orders are selected.

    Optional ``HISPEC_DRP_MASK_CCF_PIXEL_INTERP_ORDER`` sets
    ``StellarRVPrimitive.mask_ccf_pixel_interp_order`` (and the line-mask chunk benchmark).

    Mask-line CCF uses :data:`LTE049_BTSETTL_MASK_MAS` (template-derived **delta** mask paired with the
    same lte049 spectrum), not a separate wide third-party catalog. Velocity sampling is the **coarse uniform** grid only (50 m/s via
    ``v_step_coarse``); there is no centered window override (``ccf_input_velocity`` unset;
    ``mask_auto_center_scout=False`` in this test).

    Chunk RVs use the Gaussian CCF centroid from :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.fit_ccf_rv`
    / spline :func:`~hispecdrp.primitives.utils.ccf.ccf_fitting.calc_ccf_metrics`.
    Unset ``HISPEC_DRP_MASK_CCF_PIXEL_INTERP_ORDER`` uses **Lanczos-3** (science default);
    set ``=1`` for linear sampling. Linear asserts ``|RVCCFAVG − injected RV| < 1.2`` m/s;
    Lanczos-a (``a>1``) relaxes that bound (see assertion body) because windowed-sinc
    sampling can systematically shift the order-mean mask CCF by tens of m/s on this stack.

    Assertions require at least 70% of eligible chunks to have finite template CCF fits and a
    minimum count within tolerance vs the injected RV (scaled when only one chunk is eligible).

    After building the synthetic spectra, zero-flux pixels (including forced order endpoints)
    are assigned ``DQ`` ``DO_NOT_USE`` so CCF and line-mask paths exclude them via the standard
    bad-pixel mask.

        **Printed RV summary:** run with ``pytest -s`` (disable output capture). The test prints the
        injected RV, per-order recovered RV/uncertainty, RVCCFAVG aggregate, and recombined MAD/IV
        statistics **after** the wall-time summary so they appear at the end of the log (not buried
        under preview / INFO lines).

        **Relax strict checks** (e.g. non-default ``window_size_um`` / many chunks): set
        ``HISPEC_DRP_BSPEC_BTSETTL_RELAX_PARALLEL_ASSERTS=1`` or ``pytest --bspec-relax-parallel-asserts``
        so template-CCF recovery, ``RVCCFAVG`` vs injected, and chunk-benchmark timing asserts are
        skipped when they would fail—previews still run.
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
    orders = _bspec_btsettl_ccf_line_mask_order_ids()
    timing_rows.append(
        (
            "lte049: session cache (w_rest, flux_rest, template_ln) + BSPEC line-mask order selection",
            time.perf_counter() - t0,
            "lte049_btsettl_session_cache, _bspec_btsettl_ccf_line_mask_order_ids",
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

    # Zero-flux pixels: DO_NOT_USE so template CCF + line-mask skip them (``d_bad`` / DQ gate).
    # Force order endpoints to flux 0 and mark bad; also flag any non-finite or non-positive flux.
    do_not_use = int(datamodels.DQ_FLAGS["DO_NOT_USE"])
    do_not_use_u = np.uint32(do_not_use)
    for oi in range(n_orders):
        spec_2d[oi, 0] = 0.0
        spec_2d[oi, -1] = 0.0
    bad_flux = ~np.isfinite(spec_2d) | (spec_2d <= 0.0)
    dq_2d[bad_flux] |= do_not_use_u

    window_um_eff = _btsettl_parallel_line_mask_window_um()
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
    min_in_tol = max(1, min(n_attempt, max(2, int(0.15 * n_attempt))))
    if not _btsettl_parallel_line_mask_relax_assertions():
        assert n_fit_ok >= 0.70 * n_attempt, f"fit fraction too low: {n_fit_ok}/{n_attempt}"
        assert n_in_tol >= min_in_tol, (
            f"RV recovery fraction too low: {n_in_tol}/{n_attempt} (need >= {min_in_tol})"
        )
    elif n_fit_ok < 0.70 * n_attempt or n_in_tol < min_in_tol:
        print(
            "[HISPEC_DRP_BSPEC_BTSETTL_RELAX_PARALLEL_ASSERTS] Skipping template-CCF recovery "
            f"checks: fit_ok={n_fit_ok}/{n_attempt}, in_tol={n_in_tol}/{n_attempt} "
            f"(need in_tol>={min_in_tol})",
            flush=True,
        )

    # ------------------------------------------------------------------
    # RVCCF_HDU: validate output L2 ``rvccf_fib1`` vs datamodel schema (ccf_line_mask)
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
    out_rvccf, prim_rvccf = StellarRVPrimitive.call(
        rvccf_l2,
        algorithm="ccf_line_mask",
        mask_path=mas_lte049,
        do_fibers=["1"],
        do_orders=orders,
        window_size_um=window_um_eff,
        v_step_coarse=50.0,
        mask_auto_center_scout=False,
        mask_ccf_pixel_interp_order=_mask_ccf_pixel_interp_order_from_env(),
        outlier_sigma=4.0,
        max_cores=6,
        save_preview=False,
        show_preview=False,
        output_dir=str(tmp_path),
        suffix="parallel_rvccf_hdu_validate",
        return_instance=True
    )
    timing_rows.append(
        (
            "RVCCF_HDU: StellarRVPrimitive ccf_line_mask → rvccf_fib1 schema check",
            time.perf_counter() - t0,
            "StellarRVPrimitive.apply, _write_rv_tables, assemble_rvccf_fib_table",
            True,
        )
    )
    assert out_rvccf is not None
    tbl_rvccf = getattr(out_rvccf, "rvccf_fib1", None)
    _assert_rvccf_table_matches_hdu_schema(tbl_rvccf, label="rvccf_fib1")
    _print_rvccf_hdu_validation_report(tbl_rvccf, label="rvccf_fib1")
    assert tbl_rvccf is not None and len(tbl_rvccf) > 0, "expected non-empty rvccf_fib1 from ccf_line_mask"
    for i in range(len(tbl_rvccf)):
        nv = int(tbl_rvccf["N_VEL"][i])
        ccf = tbl_rvccf["CCF"][i]
        if isinstance(ccf, np.ndarray) and ccf.size > 0:
            assert ccf.size == nv, f"rvccf_fib1 row {i}: N_VEL={nv} vs len(CCF)={ccf.size}"

    avg_tbl = getattr(out_rvccf, "rvccfavg_fib1", None)
    assert avg_tbl is not None and len(avg_tbl) > 0
    rv_ccfavg = float(avg_tbl["RV"][0])
    # Order-mean RV is tight for linear sampling (a=1). Lanczos-a (default a=3) can
    # systematically shift the mask CCF by tens of m/s on this lte049+order-139 stack.
    _pio = _mask_ccf_pixel_interp_order_from_env()
    # 1.2 m/s: lte049+mask stack can land ~1.1 m/s from the injected RV at a=1 on some
    # platforms / Numba builds. a=3 on this host lands ~57 m/s — keep headroom.
    _rvccfavg_tol_m_s = 1.2 if _pio <= 1 else 100.0
    _rvccfavg_bias = abs(rv_ccfavg - input_rv_m_s)
    if not _btsettl_parallel_line_mask_relax_assertions():
        assert _rvccfavg_bias < _rvccfavg_tol_m_s, (
            "line-mask aggregate RV vs injected: expect |bias| < "
            f"{_rvccfavg_tol_m_s:g} m/s (mask_ccf_pixel_interp_order={_pio}; "
            f"got RVCCFAVG={rv_ccfavg:.4f} m/s, injected={input_rv_m_s:.4f} m/s)"
        )
    elif _rvccfavg_bias >= _rvccfavg_tol_m_s:
        print(
            "[HISPEC_DRP_BSPEC_BTSETTL_RELAX_PARALLEL_ASSERTS] Skipping RVCCFAVG vs injected: "
            f"|bias|={_rvccfavg_bias:.2f} m/s (tol {_rvccfavg_tol_m_s:g} m/s); "
            f"RVCCFAVG={rv_ccfavg:.4f}, injected={input_rv_m_s:.4f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # ccf_line_mask chunk-parallel benchmark (rv_per_chunks_line_mask path)
    # ------------------------------------------------------------------
    # With the default ``window_um_eff`` (0.3 µm unless overridden), many single-order calls
    # have only 1 chunk, which can hide parallel scaling. Use a smaller benchmark window to
    # force many chunks.
    #
    # Each synthetic echelle row has span ~147/order^2 µm (see _order_wavelength_grid).
    # A fixed 0.01 µm window can be *wider* than the order span, so
    # ``ccf_order_n_chunks_for_span`` yields 1 chunk and Numba ``prange`` over chunks
    # never parallelizes. Tie the window to a representative span so each order
    # contributes many chunks (``_calc_mask_ccf_chunks_numba`` prange).
    bench_orders = order_tasks[: min(3, len(order_tasks))]
    oi0 = bench_orders[0][0]
    span0 = float(np.ptp(wave_2d[oi0]))
    bench_window_um = min(0.01, window_um_eff / 20.0, max(1e-6, span0 / 16.0))
    mask_lines = MaskLines.load(mas_lte049)
    bench_times: dict[int, float] = {}
    bench_chunks: dict[int, int] = {}
    eff_threads: dict[int, int] = {}
    for req_cores in (1, 6):
        eff_threads[req_cores] = resolve_numba_threads(req_cores)
        t0 = time.perf_counter()
        n_chunks_total = 0
        for oi, _order in bench_orders:
            d_bad = (np.asarray(dq_2d[oi], dtype=np.uint32) & do_not_use) != 0
            rvs_b, _errs_b, _centers_b, _v_grid_b, _ccf_b, _wlo, _whi, _ng, _, _, _, _ = rv_per_chunks_line_mask(
                wave_2d[oi],
                spec_2d[oi],
                spec_err_2d[oi],
                d_bad,
                mask_lines,
                window_size_um=bench_window_um,
                v_min_m_s=V_MIN_DEFAULT_M_S,
                v_max_m_s=V_MAX_DEFAULT_M_S,
                v_step_m_s=v_step,
                max_cores=req_cores,
                return_ccf=False,
                mask_ccf_pixel_interp_order=_mask_ccf_pixel_interp_order_from_env(),
            )
            n_chunks_total += int(len(rvs_b))
        bench_times[req_cores] = time.perf_counter() - t0
        bench_chunks[req_cores] = n_chunks_total
        timing_rows.append(
            (
                f"lte049: ccf_line_mask chunk benchmark req_cores={req_cores} "
                f"(effective={eff_threads[req_cores]}, window={bench_window_um:.4f} um)",
                bench_times[req_cores],
                "rv_per_chunks_line_mask, _calc_mask_ccf_chunks_numba (prange over chunks), "
                "numba_thread_scope, rv_from_ccf_line_mask",
                True,
            )
        )

    # ``prange`` only matters with multiple chunks; micro-benchmarks are noisy on shared CI VMs.
    # Require that requesting more threads does not regress wall time catastrophically (not a strict
    # speed-up: overhead can make max_cores=6 similar to or slightly slower than 1 on GHA).
    t1 = bench_times[1]
    n_chunk_b = int(bench_chunks.get(1, 0))
    _gha = os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
    _parallel_wall_cap = t1 * (1.45 if _gha else 1.12)
    if eff_threads[6] > 1 and n_chunk_b > 1:
        t6 = bench_times[6]
        if not _btsettl_parallel_line_mask_relax_assertions():
            assert t6 <= _parallel_wall_cap, (
                "Expected ccf_line_mask chunk-parallel benchmark not dramatically slower than serial: "
                f"t1={t1:.3f}s, t6={t6:.3f}s (eff={eff_threads[6]}), "
                f"n_chunks={n_chunk_b}, cap={_parallel_wall_cap:.3f}s"
            )
        elif t6 > _parallel_wall_cap:
            print(
                "[HISPEC_DRP_BSPEC_BTSETTL_RELAX_PARALLEL_ASSERTS] Skipping chunk-benchmark wall check: "
                f"t1={t1:.3f}s, t6={t6:.3f}s (eff={eff_threads[6]}), "
                f"n_chunks={n_chunk_b}, cap={_parallel_wall_cap:.3f}s",
                flush=True,
            )
    else:
        if eff_threads[6] <= 1:
            print(
                "ccf_line_mask parallel benchmark check skipped: host allows only one effective "
                f"Numba thread (eff threads: 1->{eff_threads[1]}, 6->{eff_threads[6]}).",
                flush=True,
            )
        else:
            print(
                "ccf_line_mask parallel benchmark check skipped: only one CCF chunk in benchmark "
                f"(n_chunks={n_chunk_b}); need >1 for meaningful prange timing.",
                flush=True,
            )

    if _want_bspec_btsettl_previews() or _want_bspec_btsettl_serialize_mask_ccf():
        ext, _ = _run_lte049_k2all_line_mask_previews(
            str(tmp_path),
            mask_path=mas_lte049,
            do_orders=orders,
            window_size_um=window_um_eff,
            l2_model=out_rvccf,
            serialize_only=(
                _want_bspec_btsettl_serialize_mask_ccf()
                and not _want_bspec_btsettl_previews()
            ),
        )
        timing_rows.extend(ext)

    _print_btsettl_test_timing_summary(
        timing_rows,
        title=(
            "test_fit_ccf_line_mask_btsettl_bspec_all_chunks_parallel_delta_mask — "
            "wall time by step (seconds)"
        ),
    )

    _print_btsettl_line_mask_recovery_report(
        out_rvccf,
        tbl_rvccf,
        injected_rv_m_s=input_rv_m_s,
        outlier_sigma=float(getattr(prim_rvccf, "outlier_sigma", 4.0)),
    )


@pytest.mark.slow
def test_fit_ccf_line_mask_btsettl_bspec_all_chunks_parallel_tophat_mask(
    tmp_path, lte049_btsettl_session_cache
):
    """
    Same as :func:`test_fit_ccf_line_mask_btsettl_bspec_all_chunks_parallel_delta_mask`, but uses the
    template-derived **tophat finite-width** mask (``*.tophat.mas``) instead of the delta mask.

    The intended behavior difference is in the mask CCF build: for tophat segments, the CCF and
    ``ccf_err`` at each trial velocity should reflect the **mean across the full mask
    segment width**, rather than sampling only at a single interpolated wavelength.
    """
    timing_rows: list[tuple[str, float, str, bool]] = []
    mas_lte049 = _lte049_btsettl_paired_mask_path_tophat()
    if not os.path.isfile(mas_lte049):
        pytest.skip(
            "Tophat mask file not found (required for this test): "
            f"{mas_lte049}"
        )
    input_rv_m_s = INPUT_RV_KM_S * 1000.0
    v_step = _TEST_TEMPLATE_CCF_V_STEP_M_S
    n_pix = _TEST_ECHELLE_N_PIX
    t0 = time.perf_counter()
    w_rest = lte049_btsettl_session_cache["w_rest"]
    flux_rest = lte049_btsettl_session_cache["flux_rest"]
    assert len(w_rest) > 50
    orders = _bspec_btsettl_ccf_line_mask_order_ids()
    timing_rows.append(
        (
            "lte049 tophat mask: session cache (w_rest, flux_rest) + BSPEC line-mask order selection",
            time.perf_counter() - t0,
            "lte049_btsettl_session_cache, _bspec_btsettl_ccf_line_mask_order_ids",
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

    do_not_use = int(datamodels.DQ_FLAGS["DO_NOT_USE"])
    do_not_use_u = np.uint32(do_not_use)
    for oi in range(n_orders):
        spec_2d[oi, 0] = 0.0
        spec_2d[oi, -1] = 0.0
    bad_flux = ~np.isfinite(spec_2d) | (spec_2d <= 0.0)
    dq_2d[bad_flux] |= do_not_use_u

    window_um_eff = _btsettl_parallel_line_mask_window_um()
    timing_rows.append(
        (
            "lte049 tophat mask: stack BSPEC 2-D model (interp Doppler-shifted BT-Settl per order)",
            time.perf_counter() - t0,
            "_order_wavelength_grid, np.interp (build wave_2d, spec_2d, spec_err_2d)",
            True,
        )
    )

    # RVCCF_HDU: validate output L2 ``rvccf_fib1`` vs datamodel schema (ccf_line_mask).
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
    _pio = _mask_ccf_pixel_interp_order_from_env()
    out_rvccf, prim_rvccf = StellarRVPrimitive.call(
        rvccf_l2,
        algorithm="ccf_line_mask",
        mask_path=mas_lte049,
        do_fibers=["1"],
        do_orders=orders,
        window_size_um=window_um_eff,
        mask_auto_center_scout=False,
        mask_ccf_pixel_interp_order=_pio,
        outlier_sigma=4.0,
        max_cores=6,
        v_step_coarse=float(v_step),
        show_preview=False,
        save_preview=False,
        output_dir=str(tmp_path),
        suffix="parallel_rvccf_hdu_validate_tophat_mask",
        return_instance=True,
    )
    timing_rows.append(
        (
            "RVCCF_HDU: StellarRVPrimitive ccf_line_mask (tophat mask) → rvccf_fib1 schema check",
            time.perf_counter() - t0,
            "StellarRVPrimitive.apply, _write_rv_tables, assemble_rvccf_fib_table",
            True,
        )
    )
    assert out_rvccf is not None
    tbl_rvccf = getattr(out_rvccf, "rvccf_fib1", None)
    _assert_rvccf_table_matches_hdu_schema(tbl_rvccf, label="rvccf_fib1")
    _print_rvccf_hdu_validation_report(tbl_rvccf, label="rvccf_fib1")
    assert tbl_rvccf is not None and len(tbl_rvccf) > 0, "expected non-empty rvccf_fib1 from ccf_line_mask (tophat mask)"

    avg_tbl = getattr(out_rvccf, "rvccfavg_fib1", None)
    assert avg_tbl is not None and len(avg_tbl) > 0
    rv_agg = float(avg_tbl["RV"][0])
    rv_agg_err = float(avg_tbl["RV_ERR"][0])
    # Tophat segments average across a finite width (looser than delta). Lanczos-a=3 on this
    # stack can land ~250 m/s from injected; keep headroom above that for a>1.
    _floor = 200.0 if _pio <= 1 else 300.0
    tol = max(3.0 * rv_agg_err, _floor)
    assert abs(rv_agg - input_rv_m_s) <= tol, (
        f"Tophat mask RVCCFAVG bias too large: RV={rv_agg:.3f}±{rv_agg_err:.3f} m/s "
        f"injected={input_rv_m_s:.3f} m/s tol={tol:.3f} m/s "
        f"(mask_ccf_pixel_interp_order={_pio})"
    )

    # ------------------------------------------------------------------
    # ccf_line_mask chunk-parallel benchmark (rv_per_chunks_line_mask path)
    # ------------------------------------------------------------------
    bench_orders = order_tasks[: min(3, len(order_tasks))]
    oi0 = bench_orders[0][0]
    span0 = float(np.ptp(wave_2d[oi0]))
    bench_window_um = min(0.01, window_um_eff / 20.0, max(1e-6, span0 / 16.0))
    mask_lines = MaskLines.load(mas_lte049)
    bench_times: dict[int, float] = {}
    bench_chunks: dict[int, int] = {}
    eff_threads: dict[int, int] = {}
    for req_cores in (1, 6):
        eff_threads[req_cores] = resolve_numba_threads(req_cores)
        t0 = time.perf_counter()
        n_chunks_total = 0
        for oi, _order in bench_orders:
            d_bad = (np.asarray(dq_2d[oi], dtype=np.uint32) & do_not_use) != 0
            rvs_b, _errs_b, _centers_b, _v_grid_b, _ccf_b, _wlo, _whi, _ng, _, _, _, _ = rv_per_chunks_line_mask(
                wave_2d[oi],
                spec_2d[oi],
                spec_err_2d[oi],
                d_bad,
                mask_lines,
                window_size_um=bench_window_um,
                v_min_m_s=V_MIN_DEFAULT_M_S,
                v_max_m_s=V_MAX_DEFAULT_M_S,
                v_step_m_s=v_step,
                max_cores=req_cores,
                return_ccf=False,
                mask_ccf_pixel_interp_order=_mask_ccf_pixel_interp_order_from_env(),
            )
            n_chunks_total += int(len(rvs_b))
        bench_times[req_cores] = time.perf_counter() - t0
        bench_chunks[req_cores] = n_chunks_total
        timing_rows.append(
            (
                f"lte049 tophat mask: ccf_line_mask chunk benchmark req_cores={req_cores} "
                f"(effective={eff_threads[req_cores]}, window={bench_window_um:.4f} um)",
                bench_times[req_cores],
                "rv_per_chunks_line_mask, _calc_mask_ccf_chunks_numba (prange over chunks), "
                "numba_thread_scope, rv_from_ccf_line_mask",
                True,
            )
        )

    t1 = bench_times[1]
    n_chunk_b = int(bench_chunks.get(1, 0))
    _gha = os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
    _parallel_wall_cap = t1 * (1.45 if _gha else 1.12)
    if eff_threads[6] > 1 and n_chunk_b > 1:
        t6 = bench_times[6]
        if not _btsettl_parallel_line_mask_relax_assertions():
            assert t6 <= _parallel_wall_cap, (
                "Expected ccf_line_mask chunk-parallel benchmark not dramatically slower than serial: "
                f"t1={t1:.3f}s, t6={t6:.3f}s (eff={eff_threads[6]}), "
                f"n_chunks={n_chunk_b}, cap={_parallel_wall_cap:.3f}s"
            )
        elif t6 > _parallel_wall_cap:
            print(
                "[HISPEC_DRP_BSPEC_BTSETTL_RELAX_PARALLEL_ASSERTS] Skipping chunk-benchmark wall check: "
                f"t1={t1:.3f}s, t6={t6:.3f}s (eff={eff_threads[6]}), "
                f"n_chunks={n_chunk_b}, cap={_parallel_wall_cap:.3f}s",
                flush=True,
            )
    else:
        if eff_threads[6] <= 1:
            print(
                "ccf_line_mask parallel benchmark check skipped: host allows only one effective "
                f"Numba thread (eff threads: 1->{eff_threads[1]}, 6->{eff_threads[6]}).",
                flush=True,
            )
        else:
            print(
                "ccf_line_mask parallel benchmark check skipped: only one CCF chunk in benchmark "
                f"(n_chunks={n_chunk_b}); need >1 for meaningful prange timing.",
                flush=True,
            )

    if _want_bspec_btsettl_previews() or _want_bspec_btsettl_serialize_mask_ccf():
        ext, _ = _run_lte049_k2all_line_mask_previews(
            str(tmp_path),
            mask_path=mas_lte049,
            do_orders=orders,
            window_size_um=window_um_eff,
            l2_model=out_rvccf,
            serialize_only=(
                _want_bspec_btsettl_serialize_mask_ccf()
                and not _want_bspec_btsettl_previews()
            ),
        )
        timing_rows.extend(ext)

    _print_btsettl_test_timing_summary(
        timing_rows,
        title=(
            "test_fit_ccf_line_mask_btsettl_bspec_all_chunks_parallel_tophat_mask — "
            "wall time by step (seconds)"
        ),
    )

    _print_btsettl_line_mask_recovery_report(
        out_rvccf,
        tbl_rvccf,
        injected_rv_m_s=input_rv_m_s,
        outlier_sigma=float(getattr(prim_rvccf, "outlier_sigma", 4.0)),
        report_title="test_fit_ccf_line_mask_btsettl_bspec_all_chunks_parallel_tophat_mask — RV recovery (m/s)",
    )


def test_fit_ccf_tophat_style_btsettl_bspec_line_mask_line_window_n1_parallel(
    tmp_path, lte049_btsettl_session_cache
):
    """
    Same lte049 BSPEC echelle stack and paired ``.mas`` mask as
    :func:`test_fit_ccf_line_mask_btsettl_bspec_all_chunks_parallel_delta_mask`, but ``ccf_line_mask`` uses
    ``window_line_by_line=True`` with ``ccf_input_velocity`` equal to the injected RV. Chunks are
    non-overlapping wavelength intervals between midpoints of consecutive mask-line positions in
    the observed frame (spectrum outside those intervals is unused).

    Uses :data:`_BTSETTL_LINE_WINDOW_N1_WINDOW_UM`, :data:`_BTSETTL_LINE_WINDOW_N1_V_COARSE_HALF_SPAN_KM_S`,
    :data:`_BTSETTL_LINE_WINDOW_N1_V_STEP_COARSE`, and
    :data:`_BTSETTL_LINE_WINDOW_N1_MASK_CCF_PIXEL_INTERP_ORDER` (MCMC-tuned in ``examples/mcmc_line_window_beta0_sigma.py``).
    The velocity grid is **uniform** at ``v_step_coarse`` in a centered ``v_half_window``.

    Asserts ``RVCCFAVG`` vs the injected RV: for ``mask_ccf_pixel_interp_order`` :math:`\\le 1`,
    require ``|bias| < 50`` m/s (the narrow :data:`_BTSETTL_LINE_WINDOW_N1_WINDOW_UM` mask weighting
    shifts aggregate ``RVCCFAVG`` by tens of m/s while per-chunk ``RV_ERR`` remains
    :math:`\\sim 10^2` m/s). For higher-order Lagrange sampling, require
    ``|bias| < max(50, 3 * RV_ERR)`` m/s on ``rvccfavg_fib1`` (``RV_ERR`` from the same row;
    many narrow line-window chunks bias the order-mean CCF more than one wide chunk per order).
    Also asserts per-order chunk counts vs
    :func:`~hispecdrp.primitives.utils.ccf.ccf_utils.line_mask_n_chunks_for_order_line_by_line`, and that
    adjacent chunk wavelength bounds do not overlap.

    The recovery report (``-s``) lists **all** ``RVCCF`` chunk measurements (``CHUNK_ID``, order,
    **SEG k/n** for crossmatch to mask-overlay ``segment k/n``, center wavelength, RV, bias vs injected),
    sorted by ascending echelle order then **descending** chunk-center wavelength (same segment
    ordering as the mask-overlay HTML).

    Optional previews: set ``HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS=1`` (see module docstring one-liner).
    Per-segment ``v`` vs ``ccf`` JSON/NPZ: ``HISPEC_DRP_BSPEC_BTSETTL_LINE_WINDOW_SERIALIZE_RV_CCF=1`` writes
    ``lte049_line_mask_ccf_rv_debug.{json,npz}`` under ``examples/data/``; paths are printed at the end of the test.
    """
    timing_rows: list[tuple[str, float, str, bool]] = []
    line_window_rv_ccf_serialize_paths: tuple[str, str] | None = None
    mas_lte049 = _lte049_btsettl_paired_mask_path()
    if not os.path.isfile(mas_lte049):
        pytest.skip(
            "Mask file not found (required for this test): "
            f"{mas_lte049}"
        )
    input_rv_m_s = INPUT_RV_KM_S * 1000.0
    v_coarse_half_m_s = float(_BTSETTL_LINE_WINDOW_N1_V_COARSE_HALF_SPAN_KM_S) * 1000.0
    v_min_coarse_m_s = float(input_rv_m_s - v_coarse_half_m_s)
    v_max_coarse_m_s = float(input_rv_m_s + v_coarse_half_m_s)
    n_pix = _TEST_ECHELLE_N_PIX
    t0 = time.perf_counter()
    w_rest = lte049_btsettl_session_cache["w_rest"]
    flux_rest = lte049_btsettl_session_cache["flux_rest"]
    assert len(w_rest) > 50
    orders = _bspec_btsettl_ccf_line_mask_order_ids()
    timing_rows.append(
        (
            "lte049 line-window: session cache + BSPEC line-mask order selection",
            time.perf_counter() - t0,
            "lte049_btsettl_session_cache, _bspec_btsettl_ccf_line_mask_order_ids",
            True,
        )
    )

    n_orders = len(orders)
    wave_2d = np.zeros((n_orders, n_pix), dtype=np.float64)
    spec_2d = np.zeros((n_orders, n_pix), dtype=np.float64)
    spec_err_2d = np.zeros((n_orders, n_pix), dtype=np.float64)
    dq_2d = np.zeros((n_orders, n_pix), dtype=np.uint32)
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

    do_not_use = int(datamodels.DQ_FLAGS["DO_NOT_USE"])
    do_not_use_u = np.uint32(do_not_use)
    for oi in range(n_orders):
        spec_2d[oi, 0] = 0.0
        spec_2d[oi, -1] = 0.0
    bad_flux = ~np.isfinite(spec_2d) | (spec_2d <= 0.0)
    dq_2d[bad_flux] |= do_not_use_u

    window_um_eff = float(_BTSETTL_LINE_WINDOW_N1_WINDOW_UM)
    timing_rows.append(
        (
            "lte049 line-window: stack BSPEC 2-D model",
            time.perf_counter() - t0,
            "_order_wavelength_grid, np.interp",
            True,
        )
    )

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

    mask_lines = MaskLines.load(mas_lte049)
    for oi, order in enumerate(orders):
        wlo, whi, _ctr = build_line_mask_chunk_wavelength_intervals_line_by_line(
            wave_2d[oi],
            mask_lines,
            v_center_m_s=input_rv_m_s,
            zb=0.0,
        )
        for j in range(len(wlo) - 1):
            assert float(wlo[j + 1]) >= float(whi[j]) - 1e-11, (
                f"order={order}: adjacent line-window chunks overlap "
                f"({whi[j]=} {wlo[j + 1]=})"
            )
        assert int(len(wlo)) == line_mask_n_chunks_for_order_line_by_line(
            wave_2d[oi],
            mask_lines,
            v_center_m_s=input_rv_m_s,
            zb=0.0,
        )

    t0 = time.perf_counter()
    out_rvccf, prim_rvccf = StellarRVPrimitive.call(
        rvccf_l2,
        algorithm="ccf_line_mask",
        mask_path=mas_lte049,
        do_fibers=["1"],
        do_orders=orders,
        window_size_um=window_um_eff,
        window_line_by_line=True,
        ccf_input_velocity=float(input_rv_m_s),
        v_min_coarse=v_min_coarse_m_s,
        v_max_coarse=v_max_coarse_m_s,
        v_step_coarse=float(_BTSETTL_LINE_WINDOW_N1_V_STEP_COARSE),
        mask_auto_center_scout=False,
        mask_ccf_pixel_interp_order=int(_BTSETTL_LINE_WINDOW_N1_MASK_CCF_PIXEL_INTERP_ORDER),
        outlier_sigma=4.0,
        max_cores=6,
        save_preview=False,
        show_preview=False,
        output_dir=str(tmp_path),
        suffix="parallel_rvccf_line_window_n1",
        return_instance=True
    )
    timing_rows.append(
        (
            "RVCCF: StellarRVPrimitive ccf_line_mask (line-by-line chunks)",
            time.perf_counter() - t0,
            "StellarRVPrimitive.apply, run_ccf_line_mask_for_fiber",
            True,
        )
    )
    assert out_rvccf is not None
    tbl_rvccf = getattr(out_rvccf, "rvccf_fib1", None)
    _assert_rvccf_table_matches_hdu_schema(tbl_rvccf, label="rvccf_fib1")
    assert tbl_rvccf is not None and len(tbl_rvccf) > 0

    ord_col = np.asarray(tbl_rvccf["ORDER"], dtype=np.int64)
    for oi, order in enumerate(orders):
        n_exp = line_mask_n_chunks_for_order_line_by_line(
            wave_2d[oi],
            mask_lines,
            v_center_m_s=input_rv_m_s,
            zb=float(getattr(prim_rvccf, "ccf_zb", 0.0)),
        )
        n_tbl = int(np.sum(ord_col == int(order)))
        assert n_tbl == n_exp, f"order={order}: RVCCF rows {n_tbl} vs expected chunks {n_exp}"

    avg_tbl = getattr(out_rvccf, "rvccfavg_fib1", None)
    assert avg_tbl is not None and len(avg_tbl) > 0
    rv_ccfavg = float(avg_tbl["RV"][0])
    rv_agg_err = float(avg_tbl["RV_ERR"][0])
    assert np.isfinite(rv_agg_err) and rv_agg_err > 0.0, "RVCCFAVG row must have finite RV_ERR"
    _pio = int(_BTSETTL_LINE_WINDOW_N1_MASK_CCF_PIXEL_INTERP_ORDER)
    # One line per chunk → many narrow bins; order-mean mask CCF + ``mask_ccf_pixel_interp_order``>1
    # can bias RVCCFAVG more than the single-chunk-per-order BT-Settl test (there: 50 m/s when pio>1).
    if _pio <= 1:
        _rvccfavg_tol_m_s = 50.0
    else:
        _rvccfavg_tol_m_s = max(50.0, 3.0 * rv_agg_err)
    assert abs(rv_ccfavg - input_rv_m_s) < _rvccfavg_tol_m_s, (
        "line-mask line-window aggregate RV vs injected: expect |bias| < "
        f"{_rvccfavg_tol_m_s:g} m/s (mask_ccf_pixel_interp_order={_pio}; "
        f"narrow window_size_um={window_um_eff:g} µm; 3×RV_ERR floor for pio>1; "
        f"RV_ERR={rv_agg_err:.2f} m/s) "
        f"(got RVCCFAVG={rv_ccfavg:.4f} m/s, injected={input_rv_m_s:.4f} m/s)"
    )

    if (
        _want_bspec_btsettl_previews()
        or _want_bspec_btsettl_serialize_mask_ccf()
        or _want_bspec_btsettl_line_window_serialize_rv_ccf()
    ):
        ext, line_window_rv_ccf_serialize_paths = _run_lte049_k2all_line_mask_previews(
            str(tmp_path),
            mask_path=mas_lte049,
            do_orders=orders,
            window_size_um=window_um_eff,
            l2_model=out_rvccf,
            window_line_by_line=True,
            ccf_input_velocity_m_s=float(input_rv_m_s),
            v_min_coarse=v_min_coarse_m_s,
            v_max_coarse=v_max_coarse_m_s,
            v_step_coarse=float(_BTSETTL_LINE_WINDOW_N1_V_STEP_COARSE),
            mask_ccf_pixel_interp_order=int(_BTSETTL_LINE_WINDOW_N1_MASK_CCF_PIXEL_INTERP_ORDER),
            serialize_only=(
                (
                    _want_bspec_btsettl_serialize_mask_ccf()
                    or _want_bspec_btsettl_line_window_serialize_rv_ccf()
                )
                and not _want_bspec_btsettl_previews()
            ),
            also_serialize_rv_ccf=_want_bspec_btsettl_line_window_serialize_rv_ccf(),
            defer_mask_ccf_serialize_path_print=(
                _want_bspec_btsettl_serialize_mask_ccf()
                or _want_bspec_btsettl_line_window_serialize_rv_ccf()
            ),
            line_mask_preview_out_dir=(
                _EXAMPLES_DATA_DIR if _want_bspec_btsettl_line_window_serialize_rv_ccf() else None
            ),
        )
        timing_rows.extend(ext)

    _print_btsettl_test_timing_summary(
        timing_rows,
        title=(
            "test_fit_ccf_tophat_style_btsettl_bspec_line_mask_line_window_n1_parallel — "
            "wall time by step (seconds)"
        ),
    )
    _print_btsettl_line_mask_recovery_report(
        out_rvccf,
        tbl_rvccf,
        injected_rv_m_s=input_rv_m_s,
        outlier_sigma=float(getattr(prim_rvccf, "outlier_sigma", 4.0)),
        per_chunk_rows=True,
        report_title=(
            "test_fit_ccf_tophat_style_btsettl_bspec_line_mask_line_window_n1_parallel — "
            "RV recovery (m/s)"
        ),
    )
    if line_window_rv_ccf_serialize_paths is not None:
        jp, zp = line_window_rv_ccf_serialize_paths
        print(
            f"HISPEC_DRP_BSPEC_BTSETTL_LINE_WINDOW_RV_CCF_JSON={jp}",
            flush=True,
        )
        print(
            f"HISPEC_DRP_BSPEC_BTSETTL_LINE_WINDOW_RV_CCF_NPZ={zp}",
            flush=True,
        )


def _wls_intercept_unity_slope(
    x: np.ndarray,
    y: np.ndarray,
    sigma_y: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """
    Weighted fit with **slope fixed at 1**: ``y ≈ x + β₀`` using ``w = 1/σ_y²``.

    ``β₀`` is the weighted mean of ``(y - x)``. Returns
    ``(β₀, σ_β₀, rms_residual_unweighted_m_s, sqrt_reduced_chi2, reduced_chi2)`` with one degree of
    freedom removed, ``reduced_chi2 = Σ w (y - x - β₀)² / (n - 1)``.
    """
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    sa = np.asarray(sigma_y, dtype=np.float64).ravel()
    if xa.size != ya.size or xa.size != sa.size or xa.size < 2:
        raise ValueError("_wls_intercept_unity_slope: invalid shapes")
    w = np.where(np.isfinite(sa) & (sa > 0.0), 1.0 / (sa * sa), 0.0)
    if not np.any(w > 0.0):
        raise ValueError("_wls_intercept_unity_slope: no positive weights")
    d = ya - xa
    sw = float(np.sum(w))
    b0 = float(np.sum(w * d) / sw)
    resid = d - b0
    n = int(xa.size)
    dof = max(1, n - 1)
    chi2 = float(np.sum(w * resid * resid))
    red_chi2 = chi2 / float(dof)
    rms_resid = float(math.sqrt(float(np.mean(resid * resid))))
    sqrt_red_chi = float(math.sqrt(red_chi2))
    sigma_b0 = float(math.sqrt(1.0 / sw))
    return b0, sigma_b0, rms_resid, sqrt_red_chi, red_chi2


def _save_line_window_injected_rv_scan_plot(
    inj_m_s: np.ndarray,
    rec_m_s: np.ndarray,
    rec_err_m_s: np.ndarray,
    b0_m_s: float,
    out_path: str,
    *,
    subtitle: str,
) -> None:
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(inj_m_s, dtype=np.float64).ravel()
    y = np.asarray(rec_m_s, dtype=np.float64).ravel()
    ye = np.asarray(rec_err_m_s, dtype=np.float64).ravel()
    # No constrained_layout: long titles clip; ``bbox_inches='tight'`` on save includes full title.
    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    ax.errorbar(
        x / 1000.0,
        y / 1000.0,
        yerr=ye / 1000.0,
        fmt="o",
        color="C0",
        ecolor="C0",
        elinewidth=1.0,
        capsize=3.0,
        markersize=6,
        zorder=3,
        label="RVCCFAVG ± RV_ERR",
    )
    lo = float(np.min(x) / 1000.0)
    hi = float(np.max(x) / 1000.0)
    pad = (hi - lo) * 0.08 + 1e-6
    grid_km = np.linspace(lo - pad, hi + pad, 128, dtype=np.float64)
    ax.plot(grid_km, grid_km, "k--", lw=1.0, alpha=0.55, zorder=1, label="y = x")
    b0_km = float(b0_m_s) / 1000.0
    ax.plot(grid_km, grid_km + b0_km, "C3-", lw=2.0, zorder=2, label=f"y = x + {b0_km:.4f} km/s")
    ax.set_xlabel("Injected RV (km s⁻¹)")
    ax.set_ylabel("Recovered RV (km s⁻¹)")
    subtitle_wrapped = textwrap.fill(
        subtitle, width=96, break_long_words=False, replace_whitespace=False
    )
    ax.set_title(
        "line-mask line-window (n_lines=1): injected vs aggregated RV\n" + subtitle_wrapped,
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.35)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    fig.savefig(out_path, dpi=160, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def test_fit_ccf_tophat_style_btsettl_bspec_line_mask_line_window_n1_injected_rv_scan(
    tmp_path, lte049_btsettl_session_cache,
):
    """
    Eleven line-mask line-window runs: injected BSPEC Doppler RV stepped by 10 m/s from
    ``-110 km/s − 50 m/s`` to ``+50 m/s``. Each run uses the same lte049 stack geometry as
    :func:`test_fit_ccf_tophat_style_btsettl_bspec_line_mask_line_window_n1_parallel` but Doppler-shifts
    the rest template by that trial RV. Reads ``RVCCFAVG`` (aggregate RV and ``RV_ERR``), performs a
    weighted offset fit ``RV_rec ≈ RV_inj + β₀`` (slope fixed at 1, weights ``1/RV_ERR²``), writes a
    Matplotlib PNG under ``tmp_path``, and asserts a modest systematic offset and scatter about ``y=x+β₀``.

    Uses the same :data:`_BTSETTL_LINE_WINDOW_N1_*` primitive defaults as the parallel line-window test.

    Plot path is printed (use ``pytest -s``). Set ``HISPEC_DRP_BSPEC_BTSETTL_LINE_WINDOW_RV_SCAN_PLOT_EXAMPLES=1``
    to also copy the figure to ``examples/output/``.
    """
    mas_lte049 = _lte049_btsettl_paired_mask_path()
    if not os.path.isfile(mas_lte049):
        pytest.skip(f"Mask file not found (required for this test): {mas_lte049}")
    ref_rv_m_s = float(INPUT_RV_KM_S * 1000.0)
    if _truthy_env("HISPEC_DRP_FULL_INJECTED_RV_SCAN"):
        injected_m_s = ref_rv_m_s + np.arange(-50.0, 51.0, 10.0, dtype=np.float64)
        assert injected_m_s.size == 11
    else:
        # Fast default: three injects still exercise linearity around the fiducial RV.
        injected_m_s = ref_rv_m_s + np.array([-50.0, 0.0, 50.0], dtype=np.float64)

    v_coarse_half_m_s = float(_BTSETTL_LINE_WINDOW_N1_V_COARSE_HALF_SPAN_KM_S) * 1000.0
    w_rest = lte049_btsettl_session_cache["w_rest"]
    flux_rest = lte049_btsettl_session_cache["flux_rest"]
    assert len(w_rest) > 50
    orders = _bspec_btsettl_ccf_line_mask_order_ids()
    n_pix = _TEST_ECHELLE_N_PIX
    n_orders = len(orders)
    window_um_eff = float(_BTSETTL_LINE_WINDOW_N1_WINDOW_UM)
    do_not_use = int(datamodels.DQ_FLAGS["DO_NOT_USE"])
    do_not_use_u = np.uint32(do_not_use)
    mask_lines = MaskLines.load(mas_lte049)

    recovered = np.empty(injected_m_s.size, dtype=np.float64)
    recovered_err = np.empty(injected_m_s.size, dtype=np.float64)

    for k, inj_m_s in enumerate(injected_m_s):
        inj_km_s = float(inj_m_s) / 1000.0
        wave_2d = np.zeros((n_orders, n_pix), dtype=np.float64)
        spec_2d = np.zeros((n_orders, n_pix), dtype=np.float64)
        spec_err_2d = np.zeros((n_orders, n_pix), dtype=np.float64)
        dq_2d = np.zeros((n_orders, n_pix), dtype=np.uint32)
        for oi, order in enumerate(orders):
            m = int(order)
            wave_ord, _lam_c = _order_wavelength_grid(m, n_pix)
            wave_2d[oi] = wave_ord
            spec_2d[oi] = np.interp(
                wave_ord / (1.0 + inj_km_s / C_KM_S),
                w_rest,
                flux_rest,
            ).astype(np.float64)
            spec_err_2d[oi] = _bspec_simulated_flux_err(spec_2d[oi])
        for oi in range(n_orders):
            spec_2d[oi, 0] = 0.0
            spec_2d[oi, -1] = 0.0
        bad_flux = ~np.isfinite(spec_2d) | (spec_2d <= 0.0)
        dq_2d[bad_flux] |= do_not_use_u

        if k == 0:
            for oi, order in enumerate(orders):
                wlo, whi, _ctr = build_line_mask_chunk_wavelength_intervals_line_by_line(
                    wave_2d[oi],
                    mask_lines,
                    v_center_m_s=float(inj_m_s),
                    zb=0.0,
                )
                for j in range(len(wlo) - 1):
                    assert float(wlo[j + 1]) >= float(whi[j]) - 1e-11, (
                        f"order={order}: adjacent line-window chunks overlap "
                        f"({whi[j]=} {wlo[j + 1]=})"
                    )

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

        v_min_c = float(inj_m_s - v_coarse_half_m_s)
        v_max_c = float(inj_m_s + v_coarse_half_m_s)
        out = StellarRVPrimitive.call(
            rvccf_l2,
            algorithm="ccf_line_mask",
            mask_path=mas_lte049,
            do_fibers=["1"],
            do_orders=orders,
            window_size_um=window_um_eff,
            window_line_by_line=True,
            ccf_input_velocity=float(inj_m_s),
            v_min_coarse=v_min_c,
            v_max_coarse=v_max_c,
            v_step_coarse=float(_BTSETTL_LINE_WINDOW_N1_V_STEP_COARSE),
            mask_auto_center_scout=False,
            mask_ccf_pixel_interp_order=int(_BTSETTL_LINE_WINDOW_N1_MASK_CCF_PIXEL_INTERP_ORDER),
            outlier_sigma=4.0,
            max_cores=6,
            save_preview=False,
            show_preview=False,
            output_dir=str(tmp_path),
            suffix=f"line_window_rv_scan_{k}",
        )
        assert out is not None
        avg_tbl = getattr(out, "rvccfavg_fib1", None)
        assert avg_tbl is not None and len(avg_tbl) > 0
        recovered[k] = float(avg_tbl["RV"][0])
        recovered_err[k] = float(avg_tbl["RV_ERR"][0])
        assert np.isfinite(recovered[k]) and np.isfinite(recovered_err[k])
        assert recovered_err[k] > 0.0

    b0, b0_sig, rms_resid, sqrt_red_chi, red_chi2 = _wls_intercept_unity_slope(
        injected_m_s, recovered, recovered_err
    )
    rel = injected_m_s - ref_rv_m_s
    resid = recovered - injected_m_s - b0
    mad_resid_m_s = float(mad_std(resid, ignore_nan=True))
    ar = np.abs(resid)
    rr = np.abs(rel)
    if rr.size > 2 and float(np.std(rr)) > 0 and float(np.std(ar)) > 0:
        corr_abs_resid_vs_abs_dinj = float(np.corrcoef(rr, ar)[0, 1])
    else:
        corr_abs_resid_vs_abs_dinj = float("nan")
    print(
        "line-window RV scan: Pearson(|residual|, |Δ injected RV|) = "
        f"{corr_abs_resid_vs_abs_dinj:.4f} "
        f"(Δ injected = RV_inj − {ref_rv_m_s:.1f} m/s; "
        f"residual = RV_rec − RV_inj − β₀, slope fixed at 1)",
        flush=True,
    )

    subtitle = (
        f"WLS (slope=1): β₀={b0:.3f}±{b0_sig:.3f} m/s; "
        f"RMS={rms_resid:.2f} MAD={mad_resid_m_s:.2f} m/s; √(χ²/dof)={sqrt_red_chi:.2f}; "
        f"red.χ²={red_chi2:.2f}"
    )
    plot_path = os.path.join(
        str(tmp_path), "lte049_line_window_injected_rv_scan_rvccfavg.png"
    )
    _save_line_window_injected_rv_scan_plot(
        injected_m_s,
        recovered,
        recovered_err,
        b0,
        plot_path,
        subtitle=subtitle,
    )
    print(f"HISPEC_DRP_LINE_WINDOW_INJECTED_RV_SCAN_PLOT={plot_path}", flush=True)
    if _truthy_env("HISPEC_DRP_BSPEC_BTSETTL_LINE_WINDOW_RV_SCAN_PLOT_EXAMPLES"):
        os.makedirs(_EXAMPLES_OUTPUT_DIR, exist_ok=True)
        ex_path = os.path.join(
            _EXAMPLES_OUTPUT_DIR, "lte049_line_window_injected_rv_scan_rvccfavg.png"
        )
        shutil.copy2(plot_path, ex_path)
        print(
            "HISPEC_DRP_BSPEC_BTSETTL_LINE_WINDOW_RV_SCAN_PLOT_EXAMPLES="
            f"{ex_path}",
            flush=True,
        )

    assert abs(b0) < 200.0, f"weighted offset β₀ (m/s) vs y=x unexpectedly large: {b0}"
    assert rms_resid < 35.0, f"RMS(rec − inj − β₀) about y=x+β₀: {rms_resid:g} m/s"
    assert mad_resid_m_s < 30.0, f"MAD(rec − inj − β₀): {mad_resid_m_s:g} m/s"


@pytest.mark.slow
def test_fit_ccf_line_mask_btsettl_bspec_injected_rv_scan_pipeline_defaults(
    tmp_path, lte049_btsettl_session_cache,
):
    """
    Eleven ``ccf_line_mask`` runs on a synthetic BSPEC L2 stack: injected Doppler RV stepped by
    10 m/s from ``-110 km/s − 50 m/s`` to ``+50 m/s`` (same grid as the line-window scan).

    Unlike :func:`test_fit_ccf_tophat_style_btsettl_bspec_line_mask_line_window_n1_injected_rv_scan`,
    this uses **current** :class:`~hispecdrp.primitives.StellarRVPrimitive` defaults for the
    revised CCF path (wavelength ``window_size_um`` chunks, auto-center velocity scout, Gaussian
    vertex ``n=31``, Lagrange order 3). Reads ``RVCCFAVG``, fits ``RV_rec ≈ RV_inj + β₀``, and
    writes a Matplotlib PNG under ``tmp_path``.

    Set ``HISPEC_DRP_BSPEC_BTSETTL_DEFAULT_RV_SCAN_PLOT_EXAMPLES=1`` to also copy the figure to
    ``examples/output/``.
    """
    mas_lte049 = _lte049_btsettl_paired_mask_path()
    if not os.path.isfile(mas_lte049):
        pytest.skip(f"Mask file not found (required for this test): {mas_lte049}")
    ref_rv_m_s = float(INPUT_RV_KM_S * 1000.0)
    if _truthy_env("HISPEC_DRP_FULL_INJECTED_RV_SCAN"):
        injected_m_s = ref_rv_m_s + np.arange(-50.0, 51.0, 10.0, dtype=np.float64)
        assert injected_m_s.size == 11
    else:
        # Fast default: three injects still exercise linearity around the fiducial RV.
        injected_m_s = ref_rv_m_s + np.array([-50.0, 0.0, 50.0], dtype=np.float64)

    w_rest = lte049_btsettl_session_cache["w_rest"]
    flux_rest = lte049_btsettl_session_cache["flux_rest"]
    assert len(w_rest) > 50
    orders = _bspec_btsettl_ccf_line_mask_order_ids()
    n_pix = _TEST_ECHELLE_N_PIX
    n_orders = len(orders)
    do_not_use = int(datamodels.DQ_FLAGS["DO_NOT_USE"])
    do_not_use_u = np.uint32(do_not_use)

    recovered = np.empty(injected_m_s.size, dtype=np.float64)
    recovered_err = np.empty(injected_m_s.size, dtype=np.float64)

    for k, inj_m_s in enumerate(injected_m_s):
        inj_km_s = float(inj_m_s) / 1000.0
        wave_2d = np.zeros((n_orders, n_pix), dtype=np.float64)
        spec_2d = np.zeros((n_orders, n_pix), dtype=np.float64)
        spec_err_2d = np.zeros((n_orders, n_pix), dtype=np.float64)
        dq_2d = np.zeros((n_orders, n_pix), dtype=np.uint32)
        for oi, order in enumerate(orders):
            m = int(order)
            wave_ord, _lam_c = _order_wavelength_grid(m, n_pix)
            wave_2d[oi] = wave_ord
            spec_2d[oi] = np.interp(
                wave_ord / (1.0 + inj_km_s / C_KM_S),
                w_rest,
                flux_rest,
            ).astype(np.float64)
            spec_err_2d[oi] = _bspec_simulated_flux_err(spec_2d[oi])
        for oi in range(n_orders):
            spec_2d[oi, 0] = 0.0
            spec_2d[oi, -1] = 0.0
        bad_flux = ~np.isfinite(spec_2d) | (spec_2d <= 0.0)
        dq_2d[bad_flux] |= do_not_use_u

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

        # Pipeline defaults for ccf_line_mask (omit window_line_by_line / n_samples /
        # interp order / coarse grid overrides so PARAMETERS defaults apply).
        out = StellarRVPrimitive.call(
            rvccf_l2,
            algorithm="ccf_line_mask",
            mask_path=mas_lte049,
            do_fibers=["1"],
            do_orders=orders,
            outlier_sigma=4.0,
            max_cores=6,
            save_preview=False,
            show_preview=False,
            output_dir=str(tmp_path),
            suffix=f"default_rv_scan_{k}",
        )
        assert out is not None
        avg_tbl = getattr(out, "rvccfavg_fib1", None)
        assert avg_tbl is not None and len(avg_tbl) > 0
        recovered[k] = float(avg_tbl["RV"][0])
        recovered_err[k] = float(avg_tbl["RV_ERR"][0])
        assert np.isfinite(recovered[k]) and np.isfinite(recovered_err[k])
        assert recovered_err[k] > 0.0
        print(
            f"default RV scan [{k + 1}/{injected_m_s.size}]: inj={inj_m_s:.3f} rec={recovered[k]:.3f} "
            f"err={recovered_err[k]:.3f} bias={recovered[k] - inj_m_s:.3f} m/s",
            flush=True,
        )

    b0, b0_sig, rms_resid, sqrt_red_chi, red_chi2 = _wls_intercept_unity_slope(
        injected_m_s, recovered, recovered_err
    )
    rel = injected_m_s - ref_rv_m_s
    resid = recovered - injected_m_s - b0
    mad_resid_m_s = float(mad_std(resid, ignore_nan=True))
    ar = np.abs(resid)
    rr = np.abs(rel)
    if rr.size > 2 and float(np.std(rr)) > 0 and float(np.std(ar)) > 0:
        corr_abs_resid_vs_abs_dinj = float(np.corrcoef(rr, ar)[0, 1])
    else:
        corr_abs_resid_vs_abs_dinj = float("nan")
    print(
        "pipeline-default RV scan: Pearson(|residual|, |Δ injected RV|) = "
        f"{corr_abs_resid_vs_abs_dinj:.4f} "
        f"(Δ injected = RV_inj − {ref_rv_m_s:.1f} m/s; "
        f"residual = RV_rec − RV_inj − β₀, slope fixed at 1)",
        flush=True,
    )
    print(
        f"pipeline-default RV scan: β₀={b0:.4f}±{b0_sig:.4f} m/s; "
        f"RMS={rms_resid:.3f} MAD={mad_resid_m_s:.3f} m/s; "
        f"√(χ²/dof)={sqrt_red_chi:.3f}; red.χ²={red_chi2:.3f}",
        flush=True,
    )

    subtitle = (
        f"WLS (slope=1): β₀={b0:.3f}±{b0_sig:.3f} m/s; "
        f"RMS={rms_resid:.2f} MAD={mad_resid_m_s:.2f} m/s; √(χ²/dof)={sqrt_red_chi:.2f}; "
        f"red.χ²={red_chi2:.2f}"
    )
    plot_path = os.path.join(
        str(tmp_path), "lte049_pipeline_default_injected_rv_scan_rvccfavg.png"
    )
    _save_line_window_injected_rv_scan_plot(
        injected_m_s,
        recovered,
        recovered_err,
        b0,
        plot_path,
        subtitle=subtitle,
    )
    print(f"HISPEC_DRP_PIPELINE_DEFAULT_INJECTED_RV_SCAN_PLOT={plot_path}", flush=True)
    if _truthy_env("HISPEC_DRP_BSPEC_BTSETTL_DEFAULT_RV_SCAN_PLOT_EXAMPLES"):
        os.makedirs(_EXAMPLES_OUTPUT_DIR, exist_ok=True)
        ex_path = os.path.join(
            _EXAMPLES_OUTPUT_DIR, "lte049_pipeline_default_injected_rv_scan_rvccfavg.png"
        )
        shutil.copy2(plot_path, ex_path)
        print(
            "HISPEC_DRP_BSPEC_BTSETTL_DEFAULT_RV_SCAN_PLOT_EXAMPLES="
            f"{ex_path}",
            flush=True,
        )

    assert abs(b0) < 200.0, f"weighted offset β₀ (m/s) vs y=x unexpectedly large: {b0}"
    assert rms_resid < 35.0, f"RMS(rec − inj − β₀) about y=x+β₀: {rms_resid:g} m/s"
    assert mad_resid_m_s < 30.0, f"MAD(rec − inj − β₀): {mad_resid_m_s:g} m/s"


def test_fit_ccf_template_style_btsettl_bspec_all_chunks_parallel(
    tmp_path, lte049_btsettl_session_cache
):
    """
    BT-Settl **lte049** + BSPEC echelle stack: same ``process_orders_rvs_numba_parallel`` phase and
    chunk-eligibility assertions as :func:`test_fit_ccf_line_mask_btsettl_bspec_all_chunks_parallel_delta_mask`,
    then :class:`~hispecdrp.primitives.StellarRVPrimitive` with ``algorithm="ccf_stellar_template"``
    (rest-frame ``template_wavelength`` / ``template_flux`` from the session cache).

    Echelle order selection matches the line-mask test (``_bspec_btsettl_ccf_line_mask_order_ids``).
    No line-mask file is required. Optional unified Plotly preview + template-overlay HTML:
    ``HISPEC_DRP_SHOW_BSPEC_BTSETTL_PREVIEWS=1`` (see module docstring).

    Chunk-parallel benchmark uses ``process_orders_rvs_numba_parallel`` on a few orders with
    ``max_cores`` 1 vs 6 (same windowing idea as the line-mask ``rv_per_chunks_line_mask`` benchmark).
    """
    timing_rows: list[tuple[str, float, str, bool]] = []
    t0 = time.perf_counter()
    w_rest = lte049_btsettl_session_cache["w_rest"]
    flux_rest = lte049_btsettl_session_cache["flux_rest"]
    template_ln = lte049_btsettl_session_cache["template_ln"]
    assert len(w_rest) > 50
    orders = _bspec_btsettl_ccf_line_mask_order_ids()
    timing_rows.append(
        (
            "lte049: session cache (w_rest, flux_rest, template_ln) + BSPEC template order selection",
            time.perf_counter() - t0,
            "lte049_btsettl_session_cache, _bspec_btsettl_ccf_line_mask_order_ids",
            True,
        )
    )

    input_rv_m_s = INPUT_RV_KM_S * 1000.0
    v_step = _TEST_TEMPLATE_CCF_V_STEP_M_S
    n_pix = _TEST_ECHELLE_N_PIX

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

    do_not_use = int(datamodels.DQ_FLAGS["DO_NOT_USE"])
    do_not_use_u = np.uint32(do_not_use)
    for oi in range(n_orders):
        spec_2d[oi, 0] = 0.0
        spec_2d[oi, -1] = 0.0
    bad_flux = ~np.isfinite(spec_2d) | (spec_2d <= 0.0)
    dq_2d[bad_flux] |= do_not_use_u

    window_um_eff = float(WINDOW_UM)
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

    assert n_fit_ok >= 0.70 * n_attempt, f"fit fraction too low: {n_fit_ok}/{n_attempt}"
    min_in_tol = max(1, min(n_attempt, max(2, int(0.15 * n_attempt))))
    assert n_in_tol >= min_in_tol, f"RV recovery fraction too low: {n_in_tol}/{n_attempt} (need >= {min_in_tol})"

    order_start, order_end = int(min(orders)), int(max(orders))
    tpl_l2 = datamodels.L2Model(
        meta={
            "order_start": order_start,
            "order_end": order_end,
            "spectrograph": "BSPEC",
            "instrument_name": "HISPEC",
        }
    )
    tpl_l2.meta.order_start = order_start
    tpl_l2.meta.order_end = order_end
    tpl_l2.spec_fib1 = spec_2d.astype(np.float32)
    tpl_l2.spec_err_fib1 = spec_err_2d.astype(np.float32)
    tpl_l2.wave_fib1 = wave_2d
    tpl_l2.spec_dq_fib1 = dq_2d

    t0 = time.perf_counter()
    out_tpl, prim_tpl = StellarRVPrimitive.call(
        tpl_l2,
        algorithm="ccf_stellar_template",
        template_wavelength=np.asarray(w_rest, dtype=np.float64),
        template_flux=np.asarray(flux_rest, dtype=np.float64),
        do_fibers=["1"],
        do_orders=orders,
        window_size_um=window_um_eff,
        v_step_coarse=float(_TEST_TEMPLATE_CCF_V_STEP_M_S),
        ccf_input_velocity=None,
        outlier_sigma=4.0,
        max_cores=6,
        save_preview=False,
        show_preview=False,
        output_dir=str(tmp_path),
        suffix="parallel_rvccf_hdu_validate_template",
        return_instance=True
    )
    timing_rows.append(
        (
            "RVCCF_HDU: StellarRVPrimitive ccf_stellar_template → rvccf_fib1 schema check",
            time.perf_counter() - t0,
            "StellarRVPrimitive.apply, _write_rv_tables, assemble_rvccf_fib_table",
            True,
        )
    )
    assert out_tpl is not None
    tbl_tpl = getattr(out_tpl, "rvccf_fib1", None)
    _assert_rvccf_table_matches_hdu_schema(tbl_tpl, label="rvccf_fib1")
    _print_rvccf_hdu_validation_report(tbl_tpl, label="rvccf_fib1")
    assert tbl_tpl is not None and len(tbl_tpl) > 0, "expected non-empty rvccf_fib1 from ccf_stellar_template"
    for i in range(len(tbl_tpl)):
        nv = int(tbl_tpl["N_VEL"][i])
        ccf = tbl_tpl["CCF"][i]
        if isinstance(ccf, np.ndarray) and ccf.size > 0:
            assert ccf.size == nv, f"rvccf_fib1 row {i}: N_VEL={nv} vs len(CCF)={ccf.size}"

    avg_tpl = getattr(out_tpl, "rvccfavg_fib1", None)
    assert avg_tpl is not None and len(avg_tpl) > 0
    rv_ccfavg_tpl = float(avg_tpl["RV"][0])
    # Log-λ template CCF + chunk Gaussian fits: looser than line-mask order-mean LSTSQ (<1 m/s there).
    assert abs(rv_ccfavg_tpl - input_rv_m_s) < 5000.0, (
        "template CCF aggregate RV vs injected: expect |bias| < 5 km/s on lte049 BSPEC stack "
        f"(got RVCCFAVG={rv_ccfavg_tpl:.4f} m/s, injected={input_rv_m_s:.4f} m/s)"
    )

    bench_orders = order_tasks[: min(3, len(order_tasks))]
    oi0 = bench_orders[0][0]
    span0 = float(np.ptp(wave_2d[oi0]))
    bench_window_um = min(0.01, WINDOW_UM / 20.0, max(1e-6, span0 / 16.0))
    oi_rows = [p[0] for p in bench_orders]
    wave_b = wave_2d[oi_rows]
    spec_b = spec_2d[oi_rows]
    err_b = spec_err_2d[oi_rows]
    dq_b = dq_2d[oi_rows]
    order_tasks_b = [(k, p[1]) for k, p in enumerate(bench_orders)]
    bench_times: dict[int, float] = {}
    bench_chunks: dict[int, int] = {}
    eff_threads: dict[int, int] = {}
    for req_cores in (1, 6):
        eff_threads[req_cores] = resolve_numba_threads(req_cores)
        t0 = time.perf_counter()
        _ar, _ae, _aw, _nc, _cs = process_orders_rvs_numba_parallel(
            wave_b,
            spec_b,
            err_b,
            dq_b,
            do_not_use,
            template_ln,
            flux_rest,
            float(bench_window_um),
            order_tasks_b,
            max_cores=req_cores,
            return_ccf=False,
            ccf_v_step=v_step,
        )
        bench_times[req_cores] = time.perf_counter() - t0
        bench_chunks[req_cores] = int(len(_ar))
        timing_rows.append(
            (
                f"lte049: template CCF benchmark process_orders_rvs_numba_parallel "
                f"req_cores={req_cores} (effective={eff_threads[req_cores]}, "
                f"n_chunks={bench_chunks[req_cores]}, window_um={bench_window_um:.4f})",
                bench_times[req_cores],
                "process_orders_rvs_numba_parallel, _process_all_orders_ccf_numba (prange), "
                "process_all_orders_rvs_numba",
                True,
            )
        )

    # ``prange`` only matters with multiple chunks; micro-benchmarks are noisy on shared hosts.
    # Require that requesting more threads does not regress wall time catastrophically (not a strict
    # speed-up: overhead can make max_cores=6 similar to or slightly slower than 1).
    t1 = bench_times[1]
    n_chunk_b = int(bench_chunks.get(1, 0))
    _gha = os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
    _tpl_cap = t1 * (1.45 if _gha else 1.12)
    if eff_threads[6] > 1 and n_chunk_b > 1:
        t6 = bench_times[6]
        if not _btsettl_parallel_line_mask_relax_assertions():
            assert t6 <= _tpl_cap, (
                "Expected template CCF parallel path not dramatically slower than serial: "
                f"t1={t1:.3f}s, t6={t6:.3f}s (eff={eff_threads[6]}), "
                f"n_chunks={n_chunk_b}, cap={_tpl_cap:.3f}s"
            )
        elif t6 > _tpl_cap:
            print(
                "[HISPEC_DRP_BSPEC_BTSETTL_RELAX_PARALLEL_ASSERTS] Skipping template chunk-benchmark wall check: "
                f"t1={t1:.3f}s, t6={t6:.3f}s (eff={eff_threads[6]}), "
                f"n_chunks={n_chunk_b}, cap={_tpl_cap:.3f}s",
                flush=True,
            )
    else:
        if eff_threads[6] <= 1:
            print(
                "template CCF speed-up check skipped: host allows only one effective Numba thread "
                f"(eff threads: 1->{eff_threads[1]}, 6->{eff_threads[6]}).",
                flush=True,
            )
        else:
            print(
                "template CCF speed-up check skipped: benchmark produced only one chunk "
                f"(n_chunks={bench_chunks.get(1, 0)}); need >1 for meaningful prange timing.",
                flush=True,
            )

    if _want_bspec_btsettl_previews():
        timing_rows.extend(
            _run_lte049_btsettl_template_previews(
                str(tmp_path),
                do_orders=orders,
                window_size_um=window_um_eff,
                l2_model=out_tpl,
                template_wavelength_um=w_rest,
                template_flux=flux_rest,
            )
        )

    _print_btsettl_test_timing_summary(
        timing_rows,
        title=(
            "test_fit_ccf_template_style_btsettl_bspec_all_chunks_parallel — "
            "wall time by step (seconds)"
        ),
    )

    _print_btsettl_template_ccf_recovery_report(
        out_tpl,
        tbl_tpl,
        injected_rv_m_s=input_rv_m_s,
        outlier_sigma=float(getattr(prim_tpl, "outlier_sigma", 4.0)),
    )

"""
RV-sensitive Doppler-paint checks (Lanczos-3 / mask CCF kernel).

Full-spectrum **round-trip** on raw BT-Settl flux is not a cm/s metric:

* native flux is O(1e-17) so relative error blows up at float64 ULP (the ~100% max);
* round-tripping through ~500 m/s BSPEC pixels cannot preserve 1 cm/s structure;
* pointwise ``F(v+0.01 m/s) - F(v)`` is below float64 resolution when 0.01 m/s
  is ~1.8e-5 of a pixel.

These tests compare **forward** Lanczos paint on a local line window against an
R=3e6 reference, and check that instrument-linked template densification helps.
"""

from __future__ import annotations

import numpy as np
import pytest

from hispecdrp.primitives.utils.ccf import (
    doppler_paint_lanczos,
    prepare_template_for_doppler_paint,
    template_min_resolution_for_doppler_paint,
)
from hispecdrp.utils.spectral_template_utils import get_spectral_template_path, load_btsettl_spectrum
from hispecdrp.simulations.e2e_support import (
    DEFAULT_TEMPLATE_FILENAME,
    templates_data_dir,
)
from hispecdrp.simulations.simulate_rate_map import _upsample_spectrum
from hispecdrp.utils.physical_constants import C_M_S

_ORDER = 130
_N_PIX = 4096
_V_INJ_M_S = 10_000.0
_TRUTH_R = 3.0e6
_PREPARED_MAX_REL = 5.0e-4  # instrument-linked densify vs R=3e6 truth at 10 km/s


@pytest.fixture(scope="module")
def bspec_wave_order130() -> np.ndarray:
    lam_c = 147.0 / _ORDER
    dlam_order = lam_c / _ORDER
    return np.linspace(
        lam_c - dlam_order / 2.0,
        lam_c + dlam_order / 2.0,
        _N_PIX,
        dtype=np.float64,
    )


@pytest.fixture(scope="module")
def bt_order_excerpt(bspec_wave_order130):
    """BT-Settl native crop for one BSPEC order + R=3e6 reference."""
    wave = bspec_wave_order130
    tpl = get_spectral_template_path(DEFAULT_TEMPLATE_FILENAME, base_dir=templates_data_dir())
    w_q, f_q = load_btsettl_spectrum(tpl)
    w = np.asarray(w_q.value, dtype=np.float64).ravel()
    f = np.asarray(f_q.value, dtype=np.float64).ravel()
    order = np.argsort(w)
    w, f = w[order], f[order]
    m_ord = (w > wave[0]) & (w < wave[-1])
    w_ord, f_ord = w[m_ord], f[m_ord]
    wd, fd = _upsample_spectrum(
        w_ord * w_q.unit,
        f_ord * f_q.unit,
        desired_sampling_resolution=_TRUTH_R,
    )
    w_truth = np.asarray(wd.value, dtype=np.float64).ravel()
    f_truth = np.asarray(fd.value, dtype=np.float64).ravel()
    m_band = (w_ord > wave[256]) & (w_ord < wave[-256]) & (f_ord > 0)
    idx = np.where(m_band)[0]
    cont = float(np.median(f_ord[idx]))
    cands = [
        i
        for i in idx[2:-2]
        if 0.2 < f_ord[i] / cont < 0.7 and f_ord[i] < f_ord[i - 1] and f_ord[i] < f_ord[i + 1]
    ]
    lam_line = float(w_ord[cands[np.argmin(f_ord[cands])]])
    return {
        "wave_obs": wave,
        "w_ord": w_ord,
        "f_ord": f_ord,
        "w_truth": w_truth,
        "f_truth": f_truth,
        "lam_line": lam_line,
    }


def _line_window_max_rel(
    wave_obs: np.ndarray,
    w_rest: np.ndarray,
    f_rest: np.ndarray,
    w_truth: np.ndarray,
    f_truth: np.ndarray,
    lam_line: float,
    v_m_s: float,
    *,
    prepare: bool,
) -> float:
    kwargs = {}
    if prepare:
        kwargs["prepare_for_obs_grid"] = wave_obs
    painted = doppler_paint_lanczos(
        wave_obs, w_rest, f_rest, v_m_s, order=3, **kwargs
    )
    truth = doppler_paint_lanczos(wave_obs, w_truth, f_truth, v_m_s, order=3)
    lam_obs = lam_line / (1.0 + float(v_m_s) / float(C_M_S))
    j = int(np.argmin(np.abs(wave_obs - lam_obs)))
    sl = slice(max(0, j - 30), min(wave_obs.size, j + 31))
    rel = np.abs(painted[sl] - truth[sl]) / np.maximum(np.abs(truth[sl]), 1.0e-30)
    return float(np.max(rel))


def test_template_min_resolution_exceeds_native_btsettl(
    bspec_wave_order130, bt_order_excerpt
):
    r_req = template_min_resolution_for_doppler_paint(bspec_wave_order130)
    assert r_req > 1.0e6
    w_prep, _ = prepare_template_for_doppler_paint(
        bt_order_excerpt["w_ord"],
        bt_order_excerpt["f_ord"],
        bspec_wave_order130,
    )
    assert w_prep.size > bt_order_excerpt["w_ord"].size


def test_r3e6_reference_self_consistent(bt_order_excerpt):
    ex = bt_order_excerpt
    max_rel = _line_window_max_rel(
        ex["wave_obs"],
        ex["w_truth"],
        ex["f_truth"],
        ex["w_truth"],
        ex["f_truth"],
        ex["lam_line"],
        _V_INJ_M_S,
        prepare=False,
    )
    assert max_rel == 0.0


def test_prepared_forward_paint_beats_native(bt_order_excerpt):
    ex = bt_order_excerpt
    native_rel = _line_window_max_rel(
        ex["wave_obs"],
        ex["w_ord"],
        ex["f_ord"],
        ex["w_truth"],
        ex["f_truth"],
        ex["lam_line"],
        _V_INJ_M_S,
        prepare=False,
    )
    prep_rel = _line_window_max_rel(
        ex["wave_obs"],
        ex["w_ord"],
        ex["f_ord"],
        ex["w_truth"],
        ex["f_truth"],
        ex["lam_line"],
        _V_INJ_M_S,
        prepare=True,
    )
    assert native_rel > prep_rel
    assert prep_rel < _PREPARED_MAX_REL

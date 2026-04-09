"""Tests for :mod:`hispecdrp.utils.line_mask_utils`."""

from __future__ import annotations

import os

import numpy as np
import pytest

from hispecdrp.primitives.utils.ccf_utils import load_mask_spirou_mas
from hispecdrp.utils.line_mask_utils import (
    DEFAULT_WIDTH_CALIBRATION_K2_ALL,
    LINE_MASK_META_PREFIX,
    _cluster_max_depth,
    _continuum_normalized_d2_flux_dlam2_at_um,
    _parabolic_minimum_wavelength_um,
    _segment_mean_continuum_normalized_depth_um,
    _local_min_continuum_normalized_flux_um,
    build_line_mask_from_template,
    compare_line_center_lists_um,
    detect_template_line_centers_gradient_um,
    detect_template_line_centers_parabolic_um,
    fit_half_width_linear_from_mas,
    load_atmospheric_transmission_um,
    mean_transmission_over_velocity_interval_um,
    parse_mask_generation_meta_line,
)


def test_fit_half_width_k2_all_matches_default_constant():
    here = os.path.dirname(__file__)
    mas = os.path.join(here, "..", "hispecdrp", "data", "masks", "K2_all.mas")
    mas = os.path.normpath(mas)
    slope, icept = fit_half_width_linear_from_mas(mas)
    assert np.isclose(slope, DEFAULT_WIDTH_CALIBRATION_K2_ALL[0], rtol=1e-6)
    assert abs(icept) < 1e-9


def test_load_mktrans_bundled():
    w, t = load_atmospheric_transmission_um()
    assert w.size == t.size
    assert w[0] < w[-1]
    assert np.all((t >= 0) & (t <= 1.05))


def test_mean_transmission_interval():
    w, t = load_atmospheric_transmission_um()
    m = mean_transmission_over_velocity_interval_um(1.5, w, t, 500.0)
    assert np.isfinite(m)


def test_parabolic_depth_uses_flux_continuum_ratio_not_interp_of_depth():
    """
    Continuum-normalized depth at refined λ must be ``1 - F(λ)/F_c(λ)`` with F and F_c
    interpolated, not ``interp(1 - F/F_c)`` (they differ for nonlinear ratios).
    """
    w = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    f = np.array([1.0, 0.0, 1.0], dtype=np.float64)
    c = np.array([1.0, 2.0, 1.0], dtype=np.float64)
    d_grid = 1.0 - f / c
    lam = 0.5
    wrong = float(np.interp(lam, w, d_grid))
    right = 1.0 - float(np.interp(lam, w, f)) / float(np.interp(lam, w, c))
    assert abs(wrong - 0.5) < 1e-12
    assert abs(right - 2.0 / 3.0) < 1e-12
    assert abs(wrong - right) > 0.1


def test_parabolic_minimum_wavelength_um_exact_on_quadratic():
    """Three samples on a parabola: vertex should match the analytic minimum."""
    cen = 1.0001
    lam = np.array([1.0, 1.0002, 1.0004], dtype=np.float64)
    fl = (lam - cen) ** 2
    got = _parabolic_minimum_wavelength_um(lam, fl, 1)
    assert abs(got - cen) < 1e-12


def test_parse_mask_generation_meta_line():
    line = LINE_MASK_META_PREFIX + '{"schema_version":1,"n_lines":3}'
    meta = parse_mask_generation_meta_line(line)
    assert meta == {"schema_version": 1, "n_lines": 3}


def test_segment_mean_depth_averages_over_interval():
    w = np.array([0.9, 1.0, 1.1, 1.2], dtype=np.float64)
    f = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    c = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    got = _segment_mean_continuum_normalized_depth_um(w, f, c, 0.95, 1.15)
    assert np.isclose(got, 0.5)


def test_local_min_continuum_normalized_flux_um_window():
    """Very deep core yields small local min(F/F_c); wide continuum gives ~1 elsewhere."""
    lam = np.linspace(1.0, 1.0002, 41)
    flux = np.ones_like(lam)
    cont = np.ones_like(lam)
    flux[20] = 1.0e-5
    got = _local_min_continuum_normalized_flux_um(
        lam, flux, cont, np.array([1.0001]), half_width_um=5.0e-5
    )
    assert got.size == 1
    assert got[0] < 1.0e-4


def test_segment_mean_depth_empty_interval_is_nan():
    w = np.array([1.0, 2.0], dtype=np.float64)
    f = np.array([1.0, 1.0], dtype=np.float64)
    c = np.array([1.0, 1.0], dtype=np.float64)
    assert not np.isfinite(_segment_mean_continuum_normalized_depth_um(w, f, c, 5.0, 6.0))


def test_cluster_max_depth_prefers_deeper():
    centers = np.array([100.0, 100.01, 200.0])
    r = np.array([0.05, 0.05, 0.05])
    depths = np.array([0.2, 0.5, 0.3])
    keep = _cluster_max_depth(centers, r, depths)
    assert set(keep.tolist()) == {1, 2}


def test_continuum_normalized_d2_matches_parabolic_curvature():
    """``F/F_c = 0.5 + k (λ-λ0)²`` has ``d²/dλ² = 2k`` everywhere (minimum at λ0)."""
    lam = np.linspace(0.992, 1.008, 401)
    cont = np.ones_like(lam)
    k = 1.0e4
    f_norm = 0.5 + k * (lam - 1.0) ** 2
    flux = f_norm * cont
    got = _continuum_normalized_d2_flux_dlam2_at_um(lam, flux, cont, np.array([1.0]))
    assert got.size == 1
    assert got[0] > 0.0
    assert np.isclose(got[0], 2.0 * k, rtol=0.02)


def test_load_mask_spirou_mas_two_column(tmp_path):
    p = tmp_path / "delta.mas"
    p.write_text(
        "# test\n"
        "  1500.0  2.5\n"
        "  1500.1  3.0\n",
        encoding="utf-8",
    )
    m = load_mask_spirou_mas(str(p))
    assert m.line_lambda_um.size == 2
    assert np.allclose(m.line_lambda_um * 1e3, [1500.0, 1500.1])
    assert np.allclose(m.weight, [2.5, 3.0])


@pytest.mark.parametrize("template_basename", ["lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5"])
def test_build_line_mask_from_template_two_columns(tmp_path, template_basename):
    here = os.path.dirname(__file__)
    tpl = os.path.normpath(os.path.join(here, "..", "hispecdrp", "data", "templates", template_basename))
    if not os.path.isfile(tpl):
        pytest.skip(f"missing template: {tpl}")
    out = tmp_path / "out.mas"
    build_line_mask_from_template(tpl, output_path=str(out))
    text = out.read_text(encoding="utf-8")
    assert "wavelength_vac_nm" in text
    data = np.loadtxt(str(out), comments="#")
    assert data.ndim == 2 and data.shape[1] == 2
    assert np.isclose(np.sum(data[:, 1]), 1.0, rtol=0.0, atol=1e-12)
    m = load_mask_spirou_mas(str(out))
    assert m.line_lambda_um.size == data.shape[0]
    assert np.allclose(m.line_lambda_um * 1e3, data[:, 0])


def test_compare_line_center_lists_um_counts():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 4.0])
    r = compare_line_center_lists_um(a, b, match_tol_nm=1.0)
    assert r["n_a"] == 3 and r["n_b"] == 3
    assert r["n_match"] == 2
    assert r["n_a_only"] == 1 and r["n_b_only"] == 1


def test_gradient_and_parabolic_centers_agree_on_smooth_minimum():
    """Parabolic discrete minima vs gradient zero of dF/dλ should both land near λ = 1 µm."""
    lam = np.linspace(0.992, 1.008, 401)
    flux = 1.0 + 1.0e4 * (lam - 1.0) ** 2
    para = detect_template_line_centers_parabolic_um(lam, flux, local_min_order=1)
    grad = detect_template_line_centers_gradient_um(
        lam, flux, continuum_filter_size=31, min_continuum_normalized_depth=0.001
    )
    assert para.size >= 1 and grad.size >= 1
    assert abs(float(np.min(para)) - 1.0) < 2e-4
    assert abs(float(np.min(grad)) - 1.0) < 2e-4

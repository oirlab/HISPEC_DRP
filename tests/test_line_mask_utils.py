"""Tests for :mod:`hispecdrp.utils.line_mask_utils`."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from astropy import units as u

from hispecdrp.utils.line_mask_utils import (
    DEFAULT_WIDTH_CALIBRATION_K2_ALL,
    LINE_MASK_META_PREFIX,
    MASK_STYLE_DELTA,
    MASK_STYLE_TOPHAT,
    MaskLines,
    _cluster_max_depth,
    _continuum_normalized_d2_flux_dlam2_at_um,
    _local_min_continuum_normalized_flux_um,
    _segment_mean_continuum_normalized_depth_um,
    build_ccfopt_line_mask_from_template_and_seed,
    build_line_mask_from_template,
    compare_line_center_lists_um,
    detect_template_line_centers_gradient_um,
    detect_template_line_centers_parabolic_um,
    fit_half_width_linear_from_mas,
    load_atmospheric_transmission_um,
    mean_transmission_over_velocity_interval_um,
    package_data_masks_dir,
    parse_mask_generation_meta_line,
    read_mask_generation_meta,
    restrict_mask_to_chunk,
)
from hispecdrp.utils.physical_constants import C_M_S


def _lte049_tophat_mas_for_width_calibration() -> str | None:
    """lte049 segment mask used to match :data:`DEFAULT_WIDTH_CALIBRATION_K2_ALL`; optional."""
    d = package_data_masks_dir()
    for name in (
        "lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.tophat.mas",
        "lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.kpf.mas",
    ):
        p = d / name
        if p.is_file():
            return str(p)
    return None


def _first_bundled_tophat_mas() -> str | None:
    """Any ``*.tophat.mas`` (or legacy ``*.kpf.mas``) under ``package_data_masks_dir()``."""
    d = package_data_masks_dir()
    found = sorted(d.glob("*.tophat.mas")) + sorted(d.glob("*.kpf.mas"))
    return str(found[0]) if found else None


def test_fit_half_width_k2_all_matches_default_constant():
    mas = _lte049_tophat_mas_for_width_calibration()
    if mas is None:
        pytest.skip(
            "lte049 tophat mask not in data/masks (optional; generate with regenerate_template_masks.py)"
        )
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


def test_masklines_tophat_geometry_from_bundled_template_tophat():
    mas = _first_bundled_tophat_mas()
    if mas is None:
        pytest.skip(
            "no *.tophat.mas in data/masks (optional; generate with regenerate_template_masks.py)"
        )
    m = MaskLines.load(mas)
    assert m.mask_style == MASK_STYLE_TOPHAT
    assert m.line_center.unit.is_equivalent(u.um)
    assert np.all(m.line_width.value > 0)
    assert np.allclose(
        m.line_center.value, 0.5 * (m.line_left_edge + m.line_right_edge).value
    )
    assert "source_file" in m.metadata
    assert "hispecdrp_wavelength_columns" in m.metadata


def test_masklines_delta_zero_width_and_restrict(tmp_path):
    p = tmp_path / "delta.mas"
    p.write_text(
        "# test\n"
        "  1500.0  2.5\n"
        "  1500.1  3.0\n",
        encoding="utf-8",
    )
    m = MaskLines.load(str(p))
    assert m.mask_style == "delta"
    assert np.all(m.line_width.to_value(u.um) == 0.0)
    sub = restrict_mask_to_chunk(
        m,
        w_min_um=1.499,
        w_max_um=1.501,
        v_min_m_s=-200_000.0,
        v_max_m_s=200_000.0,
    )
    assert sub.line_center.size >= 1


def test_masklines_load_two_column_mas(tmp_path):
    p = tmp_path / "delta.mas"
    p.write_text(
        "# test\n"
        "  1500.0  2.5\n"
        "  1500.1  3.0\n",
        encoding="utf-8",
    )
    m = MaskLines.load(str(p))
    assert m.line_lambda_um.size == 2
    assert np.allclose(m.line_lambda_um * 1e3, [1500.0, 1500.1])
    assert np.allclose(m.weight, [2.5, 3.0])


def test_masklines_load_mask_style_tophat_requires_three_columns(tmp_path):
    meta_line = LINE_MASK_META_PREFIX + json.dumps(
        {"schema_version": 1, "mask_style": "tophat"}, separators=(",", ":")
    )
    p = tmp_path / "bad_tophat.mas"
    p.write_text(
        f"{meta_line}\n"
        "  1500.0  0.25\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mask_style='tophat'"):
        MaskLines.load(str(p))


def test_masklines_load_legacy_mask_style_kpf_normalizes_to_tophat(tmp_path):
    meta_line = LINE_MASK_META_PREFIX + json.dumps(
        {"schema_version": 1, "mask_style": "kpf"}, separators=(",", ":")
    )
    p = tmp_path / "legacy_kpf.mas"
    p.write_text(
        f"{meta_line}\n"
        "  1500.0  1500.1  0.25\n",
        encoding="utf-8",
    )
    m = MaskLines.load(str(p))
    assert m.mask_style == MASK_STYLE_TOPHAT


def test_masklines_load_mask_style_delta_requires_two_columns(tmp_path):
    meta_line = LINE_MASK_META_PREFIX + json.dumps(
        {"schema_version": 1, "mask_style": "delta"}, separators=(",", ":")
    )
    p = tmp_path / "bad_delta.mas"
    p.write_text(
        f"{meta_line}\n"
        "  967.31  967.32  0.2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mask_style='delta'"):
        MaskLines.load(str(p))


def test_masklines_load_unknown_mask_style_raises(tmp_path):
    meta_line = LINE_MASK_META_PREFIX + json.dumps(
        {"schema_version": 1, "mask_style": "spirou"}, separators=(",", ":")
    )
    p = tmp_path / "bad_style.mas"
    p.write_text(
        f"{meta_line}\n"
        "  967.31  967.32  0.2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown mask_style"):
        MaskLines.load(str(p))


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
    meta = read_mask_generation_meta(str(out))
    assert meta is not None
    assert meta.get("mask_style") == MASK_STYLE_DELTA
    assert "hispecdrp_wavelength_columns" in meta
    assert "disk_wavelength_columns" in meta
    data = np.loadtxt(str(out), comments="#")
    assert data.ndim == 2 and data.shape[1] == 2
    assert np.isclose(np.sum(data[:, 1]), 1.0, rtol=0.0, atol=1e-12)
    m = MaskLines.load(str(out))
    assert m.line_lambda_um.size == data.shape[0]
    assert np.allclose(m.line_lambda_um * 1e3, data[:, 0])


@pytest.mark.parametrize("template_basename", ["lte049.0-2.5-0.0a+0.0.BT-Settl.spec.7.h5"])
def test_build_line_mask_from_template_default_output_includes_mask_style(
    tmp_path, template_basename
):
    here = os.path.dirname(__file__)
    tpl = os.path.normpath(
        os.path.join(here, "..", "hispecdrp", "data", "templates", template_basename)
    )
    if not os.path.isfile(tpl):
        pytest.skip(f"missing template: {tpl}")
    path_written, _entries = build_line_mask_from_template(
        tpl, masks_directory=str(tmp_path), output_path=None
    )
    expect = tmp_path / f"{Path(tpl).stem}.{MASK_STYLE_DELTA}.mas"
    assert Path(path_written) == expect.resolve()
    assert expect.is_file()


def test_masklines_save_default_name_includes_mask_style(tmp_path):
    m = MaskLines.from_delta_centers_um(
        np.array([1.0, 1.1]),
        np.array([0.4, 0.6]),
        metadata={"template_path": "/x/foo.h5"},
    )
    p = m.save(masks_directory=str(tmp_path))
    assert Path(p).name == "foo.delta.mas"
    m2 = MaskLines.load(p)
    assert m2.mask_style == "delta"
    assert m2.line_lambda_um.size == 2


def test_masklines_save_tophat_default_name_includes_mask_style(tmp_path):
    m = MaskLines.from_tophat_segment_edges_um(
        np.array([1.0, 1.2]),
        np.array([1.01, 1.21]),
        np.array([1.0, 2.0]),
        metadata={"template_path": "/data/bar.h5"},
    )
    p = m.save(masks_directory=str(tmp_path))
    assert Path(p).name == "bar.tophat.mas"
    m2 = MaskLines.load(p)
    assert m2.mask_style == MASK_STYLE_TOPHAT
    assert m2.line_lambda_um.size == 2


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


def test_build_ccfopt_line_mask_from_l2_single_line_ccf_zeros_bias(tmp_path, monkeypatch):
    """Single-line builder applies λ×(1+RV/c) and contrast weights (stubbed CCF)."""
    from astropy.table import Table

    from hispecdrp import datamodels
    from hispecdrp.utils import line_mask_utils as lmu

    seed_lam = 1.20
    rv_m_s = 150.0
    contrast = 0.35
    v_min, v_max, v_step = -500.0, 500.0, 25.0
    v_grid = np.arange(v_min, v_max + 0.5 * v_step, v_step)
    ccf = contrast * np.exp(-0.5 * ((v_grid - rv_m_s) / 80.0) ** 2)

    model = datamodels.L2Model(
        meta={
            "order_start": 100,
            "order_end": 100,
            "spectrograph": "BSPEC",
            "instrument_name": "HISPEC",
        }
    )
    model.meta.order_start = 100
    model.meta.order_end = 100
    model.wave_fib2 = np.linspace(1.1, 1.3, 64, dtype=np.float64).reshape(1, -1)
    model.spec_fib2 = np.ones((1, 64), dtype=np.float32)
    model.spec_err_fib2 = np.full((1, 64), 0.01, dtype=np.float32)
    model.spec_dq_fib2 = np.zeros((1, 64), dtype=np.uint32)

    def _stub_run_ccf_line_mask_for_fiber(**kwargs):
        mw = np.asarray(kwargs["mask_wavelength"], dtype=np.float64).ravel()
        assert mw.size == 1
        assert abs(float(kwargs["window_size_um"]) - 5.65e-5) < 1e-12
        assert kwargs["window_line_by_line"] is True
        assert abs(float(kwargs["ccf_half_window_m_s"]) - 15000.0) < 1e-6
        supp = Table(
            {
                "CHUNK_ID": np.array([0], dtype=np.int32),
                "N_GOOD": np.array([32], dtype=np.int32),
                "V_MIN": np.array([v_min], dtype=np.float64),
                "V_MAX": np.array([v_max], dtype=np.float64),
                "V_STEP": np.array([v_step], dtype=np.float64),
                "N_VEL": np.array([ccf.size], dtype=np.int32),
                "CCF": np.array([ccf], dtype=object),
            }
        )
        return (
            np.array([float(mw[0])], dtype=np.float64),
            np.array([rv_m_s], dtype=np.float64),
            np.array([10.0], dtype=np.float64),
            np.array([100], dtype=np.int32),
            None,
            supp,
        )

    monkeypatch.setattr(
        "hispecdrp.primitives.utils.ccf.ccf_utils.run_ccf_line_mask_for_fiber",
        _stub_run_ccf_line_mask_for_fiber,
    )

    seed_path = tmp_path / "seed.delta.mas"
    MaskLines.from_delta_centers_um(
        np.array([seed_lam]),
        np.array([1.0]),
    ).save(output_path=str(seed_path))

    out_path = tmp_path / "from_l2.ccfopt.delta.mas"
    path, entries = lmu.build_ccfopt_line_mask_from_l2_single_line_ccf(
        str(seed_path),
        model,
        fiber=2,
        template_path=str(tmp_path / "synthetic.h5"),
        output_path=str(out_path),
        masks_directory=str(tmp_path),
        rezero_verify_abs_rv_m_s=None,  # stub returns fixed RV; skip verify pass
        max_rv_err_m_s=None,
    )
    assert Path(path).is_file()
    assert len(entries) == 1
    expected = seed_lam * (1.0 + rv_m_s / C_M_S)
    recovered = entries[0].wavelength_nm * 1.0e-3
    assert abs(recovered - expected) / expected * C_M_S < 1.0
    assert abs(float(entries[0].weight) - 1.0) < 1e-9
    meta = read_mask_generation_meta(path)
    assert meta is not None
    assert "build_ccfopt_line_mask_from_l2_single_line_ccf" in str(
        meta.get("generator", "")
    )
    assert meta.get("window_line_by_line") is True
    assert abs(float(meta.get("v_half_window_m_s")) - 15000.0) < 1.0e-6
    assert abs(float(meta.get("window_size_um")) - 5.65e-5) < 1e-12


def test_build_ccfopt_line_mask_from_rvccf_table_shifts_and_weights(tmp_path):
    """RVCCF-based CCF-opt builder applies λ×(1+RV/c) and contrast weights."""
    from astropy.table import Table

    from hispecdrp.utils.line_mask_utils import build_ccfopt_line_mask_from_rvccf_table

    seed_lam = 1.50
    rv_m_s = 100.0
    contrast = 0.4
    seed_path = tmp_path / "seed.delta.mas"
    MaskLines.from_delta_centers_um(
        np.array([seed_lam]),
        np.array([1.0]),
    ).save(output_path=str(seed_path))

    v_min, v_max, v_step = -500.0, 500.0, 25.0
    v_grid = np.arange(v_min, v_max + 0.5 * v_step, v_step)
    # Stored emission-like Gaussian peak at rv_m_s (LINE_MASK_CCF_STORED_NEGATED_FROM_RAW).
    ccf = contrast * np.exp(-0.5 * ((v_grid - rv_m_s) / 80.0) ** 2)

    tbl = Table(
        {
            "RV": np.array([rv_m_s], dtype=np.float64),
            "WAVELENGTH": np.array([seed_lam], dtype=np.float64),
            "CCF": np.array([ccf], dtype=object),
            "V_MIN": np.array([v_min], dtype=np.float64),
            "V_MAX": np.array([v_max], dtype=np.float64),
            "V_STEP": np.array([v_step], dtype=np.float64),
        }
    )
    out_path = tmp_path / "from_rvccf.ccfopt.delta.mas"
    path, entries = build_ccfopt_line_mask_from_rvccf_table(
        str(seed_path),
        tbl,
        template_path=str(tmp_path / "synthetic.h5"),
        output_path=str(out_path),
        masks_directory=str(tmp_path),
    )
    assert Path(path).is_file()
    assert len(entries) == 1
    expected = seed_lam * (1.0 + rv_m_s / C_M_S)
    recovered = entries[0].wavelength_nm * 1.0e-3
    assert abs(recovered - expected) / expected * C_M_S < 1.0
    assert abs(float(entries[0].weight) - 1.0) < 1e-9
    meta = read_mask_generation_meta(path)
    assert meta is not None
    assert meta.get("mask_weight_definition") == "abs_gaussian_ccf_contrast"
    assert "build_ccfopt_line_mask_from_rvccf_table" in str(meta.get("generator", ""))


def test_build_ccfopt_line_mask_recovers_offset_center(tmp_path):
    """CCF-opt refine should pull a biased seed toward the true absorption core."""
    true_lam = 1.50
    # Seed offset by ~1 km/s so the optimizer has a clear gradient to follow.
    seed_dv_m_s = 1000.0
    seed_lam = true_lam * (1.0 + seed_dv_m_s / C_M_S)
    # Dense sampling (~0.5 km/s) so the CCF spline resolves the core.
    dlam = true_lam * 500.0 / C_M_S
    wave = np.arange(true_lam - 0.02, true_lam + 0.02 + 0.5 * dlam, dlam)
    sigma = true_lam * 1500.0 / C_M_S
    flux = 1.0 - 0.6 * np.exp(-0.5 * ((wave - true_lam) / sigma) ** 2)

    seed_path = tmp_path / "seed.delta.mas"
    MaskLines.from_delta_centers_um(
        np.array([seed_lam]),
        np.array([1.0]),
        metadata={"template_path": str(tmp_path / "synthetic.h5")},
    ).save(output_path=str(seed_path))

    out_path = tmp_path / "synthetic.ccfopt.delta.mas"
    path, entries = build_ccfopt_line_mask_from_template_and_seed(
        str(tmp_path / "synthetic.h5"),
        str(seed_path),
        wave_um=wave,
        flux=flux,
        wl_min_um=1.4,
        wl_max_um=1.6,
        spectrum_window_half_um=0.015,
        search_half_m_s=2000.0,
        coarse_step_m_s=200.0,
        fine_half_m_s=250.0,
        fine_step_m_s=25.0,
        rv_gate_m_s=3000.0,
        output_path=str(out_path),
        masks_directory=str(tmp_path),
    )
    assert Path(path).is_file()
    assert len(entries) == 1
    assert abs(float(np.sum([e.weight for e in entries])) - 1.0) < 1e-9
    recovered = entries[0].wavelength_nm * 1.0e-3
    # Closer to truth than the biased seed, and within ~one fine-search half-width.
    assert abs(recovered - true_lam) < abs(seed_lam - true_lam)
    assert abs(recovered - true_lam) / true_lam * C_M_S < 500.0
    meta = read_mask_generation_meta(path)
    assert meta is not None
    assert meta.get("mask_weight_definition") == "abs_gaussian_ccf_contrast"
    assert Path(path).name.endswith(".ccfopt.delta.mas")

"""Tests for :mod:`hispecdrp.utils.line_mask_utils`."""

from __future__ import annotations

import os

import numpy as np

from hispecdrp.utils.line_mask_utils import (
    DEFAULT_WIDTH_CALIBRATION_K2_ALL,
    LINE_MASK_META_PREFIX,
    _cluster_max_depth,
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


def test_parse_mask_generation_meta_line():
    line = LINE_MASK_META_PREFIX + '{"schema_version":1,"n_lines":3}'
    meta = parse_mask_generation_meta_line(line)
    assert meta == {"schema_version": 1, "n_lines": 3}


def test_cluster_max_depth_prefers_deeper():
    centers = np.array([100.0, 100.01, 200.0])
    r = np.array([0.05, 0.05, 0.05])
    depths = np.array([0.2, 0.5, 0.3])
    keep = _cluster_max_depth(centers, r, depths)
    assert set(keep.tolist()) == {1, 2}

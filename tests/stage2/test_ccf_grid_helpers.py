"""Fast unit tests for CCF velocity-grid helpers."""

from __future__ import annotations

import numpy as np
import pytest

from hispecdrp.primitives.utils.ccf.ccf_common import (
    V_STEP_DEFAULT_M_S,
    effective_ccf_v_step_for_fit,
    resolve_ccf_broad_grid_m_s,
)
from hispecdrp.primitives.utils.ccf.ccf_utils import (
    build_centered_uniform_velocity_grid,
    dv_per_pixel_m_s,
)


def test_resolve_ccf_broad_grid_defaults_and_validation():
    vm, vx, vs = resolve_ccf_broad_grid_m_s()
    assert vs == V_STEP_DEFAULT_M_S
    assert vx > vm

    custom = resolve_ccf_broad_grid_m_s(v_min=-100.0, v_max=100.0, v_step=10.0)
    assert custom == (-100.0, 100.0, 10.0)

    with pytest.raises(ValueError, match="v_step"):
        resolve_ccf_broad_grid_m_s(v_step=0.0)
    with pytest.raises(ValueError, match="v_max"):
        resolve_ccf_broad_grid_m_s(v_min=10.0, v_max=5.0)


def test_effective_ccf_v_step_for_fit():
    assert effective_ccf_v_step_for_fit(np.array([0.0])) == 1.0
    grid = np.array([-100.0, -50.0, 0.0, 25.0, 100.0])
    assert effective_ccf_v_step_for_fit(grid) == pytest.approx(25.0)


def test_build_centered_uniform_velocity_grid():
    grid = build_centered_uniform_velocity_grid(0.0, half_width_m_s=100.0, step_m_s=50.0)
    np.testing.assert_allclose(grid, [-100.0, -50.0, 0.0, 50.0, 100.0])

    with pytest.raises(ValueError, match="half-width"):
        build_centered_uniform_velocity_grid(0.0, half_width_m_s=0.0, step_m_s=10.0)
    with pytest.raises(ValueError, match="step"):
        build_centered_uniform_velocity_grid(0.0, half_width_m_s=10.0, step_m_s=-1.0)


def test_dv_per_pixel_m_s_scale_and_invalid():
    # λ=1.0 µm → order≈147 → Δv = c / (147 * 4096)
    c = 299_792_458.0
    expected = c / (147.0 * 4096.0)
    assert dv_per_pixel_m_s(1.0) == pytest.approx(expected, rel=1e-12)
    assert np.isnan(dv_per_pixel_m_s(0.0))
    assert np.isnan(dv_per_pixel_m_s(np.nan))

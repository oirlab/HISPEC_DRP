"""Guard rails for spectral extraction windows (non-finite / OOB trace positions)."""

from __future__ import annotations

import numpy as np
import pytest

from hispecdrp.primitives.spectral_extraction.extract_box import extract_spectrum_box
from hispecdrp.primitives.spectral_extraction.extraction_utils import get_extraction_window


def test_get_extraction_window_non_finite_returns_empty():
    for yc in (np.nan, np.inf, -np.inf):
        idx, w = get_extraction_window(float(yc), 10.0, 4160)
        assert idx.size == 0
        assert w.size == 0


def test_get_extraction_window_clips_to_detector():
    idx, w = get_extraction_window(1.0e9, 10.0, 4160)
    assert idx.size >= 1
    assert idx.min() >= 0
    assert idx.max() < 4160


def test_extract_box_survives_extrapolated_trace_positions():
    """Regression: wild trace yc must not segfault Numba box extraction."""
    ny, nx = 128, 160
    data = np.ones((ny, nx), dtype=np.float32)
    err = np.full((ny, nx), 0.1, dtype=np.float32)
    dq = np.zeros((ny, nx), dtype=np.uint32)
    var_r = np.full((ny, nx), 0.01, dtype=np.float32)
    var_p = np.full((ny, nx), 0.01, dtype=np.float32)
    mid = np.zeros((ny, nx), dtype=np.float64)

    trace = np.linspace(10.0, float(nx) + 500.0, ny, dtype=np.float64)
    trace[20:40] = np.nan

    result = extract_spectrum_box(
        input_data=data,
        input_err=err,
        input_dq=dq,
        input_var_rnoise=var_r,
        input_var_poisson=var_p,
        input_mid_time=mid,
        trace_positions=trace,
        extraction_aperture=10.0,
        orders_are_vertical=True,
    )
    assert result["spec"].shape == (ny,)
    assert np.all(np.isfinite(result["spec"][:20]))
    assert np.all(np.isnan(result["spec"][20:40]))


def test_extract_box_excludes_nonfinite_rate_and_err():
    """NaN rate is excluded; NaN err with finite rate is kept (var→0, no renorm spike)."""
    ny, nx = 8, 32
    data = np.ones((ny, nx), dtype=np.float32)
    err = np.full((ny, nx), 0.1, dtype=np.float32)
    dq = np.zeros((ny, nx), dtype=np.uint32)
    var_r = np.full((ny, nx), 0.01, dtype=np.float32)
    var_p = np.full((ny, nx), 0.01, dtype=np.float32)
    mid = np.zeros((ny, nx), dtype=np.float64)
    trace = np.full(ny, 15.0, dtype=np.float64)

    # Poison one cross-dispersion pixel in every row; extraction must stay finite.
    data[:, 15] = np.nan
    err[:, 16] = np.nan

    result = extract_spectrum_box(
        input_data=data,
        input_err=err,
        input_dq=dq,
        input_var_rnoise=var_r,
        input_var_poisson=var_p,
        input_mid_time=mid,
        trace_positions=trace,
        extraction_aperture=5.0,
        orders_are_vertical=True,
    )
    assert np.all(np.isfinite(result["spec"]))
    assert np.all(np.isfinite(result["err"]))
    # NaN-rate pixel is dropped without renormalizing empty-flux loss; NaN-err
    # pixel at col 16 is kept. Aperture weight ≈5; one unit of flux lost → ~4.
    assert np.allclose(result["spec"], 4.0, rtol=0, atol=1e-5)


def test_extract_box_nan_err_on_zero_flux_does_not_spike():
    """Regression: empty aperture-edge pixels with NaN err must not boost flux."""
    ny, nx = 16, 40
    data = np.zeros((ny, nx), dtype=np.float32)
    # Compact PSF-like profile around col 20
    data[:, 19:22] = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    err = np.full((ny, nx), np.nan, dtype=np.float32)
    err[:, 19:22] = 0.1
    dq = np.zeros((ny, nx), dtype=np.uint32)
    var_r = np.zeros((ny, nx), dtype=np.float32)
    var_p = np.zeros((ny, nx), dtype=np.float32)
    mid = np.zeros((ny, nx), dtype=np.float64)
    trace = np.full(ny, 20.0, dtype=np.float64)

    result = extract_spectrum_box(
        input_data=data,
        input_err=err,
        input_dq=dq,
        input_var_rnoise=var_r,
        input_var_poisson=var_p,
        input_mid_time=mid,
        trace_positions=trace,
        extraction_aperture=8.0,
        orders_are_vertical=True,
    )
    # True aperture sum of the three illuminated pixels (full weights) == 4
    assert np.allclose(result["spec"], 4.0, rtol=0, atol=1e-4)
    # Must not look like renormalized boost (~4 / (3/8) ≈ 10.7)
    assert np.all(result["spec"] < 5.0)

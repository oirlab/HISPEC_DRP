"""Unit tests for SpectralExtractionPrimitive and extraction on data models."""

import numpy as np
import pytest

from hispecdrp import datamodels
from hispecdrp.primitives.spectral_extraction.spectral_extraction_primitive import (
    SpectralExtractionPrimitive,
)


def _make_l0_model(ny=50, nx=100, n_orders=3, order_start=110):
    """Minimal L0Model for extraction tests (orders_are_vertical=True, BSPEC-like)."""
    config = datamodels.HISPECConfig
    model = datamodels.L0Model(config=config)
    model.config.set_meta_defaults(model)
    model.meta.order_start = order_start
    model.meta.order_end = order_start + n_orders - 1
    model.meta.orders_are_vertical = True
    model.meta.spectrograph = "BSPEC"

    shape = (ny, nx)
    model.rate = np.ones(shape, dtype=np.float32) * 0.5
    model.rate_err = np.ones(shape, dtype=np.float32) * 0.1
    model.rate_var_rnoise = np.ones(shape, dtype=np.float32) * 0.01
    model.rate_var_poisson = np.ones(shape, dtype=np.float32) * 0.01
    model.dq = np.zeros(shape, dtype=np.uint32)
    model.mid_time = np.ones(shape, dtype=np.float64)
    model.generate_filename()
    return model


def _make_tracepos_model(n_orders, npix, fiber=1, center_row=25.0):
    """TracePosModel with POS_FIB{fiber} shape (n_orders, npix)."""
    model = datamodels.TracePosModel(config=datamodels.HISPECConfig)
    pos = np.full((n_orders, npix), center_row, dtype=np.float64)
    setattr(model, f"POS_FIB{fiber}", pos)
    return model


def _make_traceprof_model(ny, nx, fiber=1):
    """TraceProfModel with PROF_FIB{fiber} shape (ny, nx) - simple top-hat."""
    model = datamodels.TraceProfModel(config=datamodels.HISPECConfig)
    prof = np.zeros((ny, nx), dtype=np.float64)
    prof[23:28, :] = 1.0 / 5.0  # 5-pixel profile
    setattr(model, f"PROF_FIB{fiber}", prof)
    return model


@pytest.fixture
def l0_model():
    """L0Model with small dimensions and 3 orders."""
    return _make_l0_model(ny=50, nx=100, n_orders=3, order_start=110)


@pytest.fixture
def tracepos_model(l0_model):
    """TracePosModel matching l0_model (3 orders, 50 spectral bins)."""
    order_vec = l0_model.make_order_vec()
    ny = l0_model.rate.shape[0]
    return _make_tracepos_model(len(order_vec), ny, fiber=1, center_row=25.0)


@pytest.fixture
def traceprof_model(l0_model):
    """TraceProfModel matching l0_model shape."""
    ny, nx = l0_model.rate.shape
    return _make_traceprof_model(ny, nx, fiber=1)


# --- Helper methods: uppercase vs lowercase attribute names ---


def test_get_trace_positions_uppercase(tracepos_model):
    """_get_trace_positions returns data when model has POS_FIB1 (in-memory convention)."""
    out = SpectralExtractionPrimitive._get_trace_positions(tracepos_model, 1)
    assert out is not None
    assert out.ndim == 2
    np.testing.assert_array_equal(out, 25.0)


def test_get_trace_positions_lowercase():
    """_get_trace_positions returns data when model has pos_fib1 (FITS-loaded convention)."""
    model = datamodels.TracePosModel(config=datamodels.HISPECConfig)
    arr = np.ones((3, 50), dtype=np.float64) * 10.0
    setattr(model, "pos_fib1", arr)
    out = SpectralExtractionPrimitive._get_trace_positions(model, 1)
    assert out is not None
    np.testing.assert_array_equal(out, 10.0)


def test_get_trace_positions_missing_raises():
    """_get_trace_positions raises ValueError when fiber has no data."""
    model = datamodels.TracePosModel(config=datamodels.HISPECConfig)
    with pytest.raises(ValueError, match="no trace positions for fiber 1"):
        SpectralExtractionPrimitive._get_trace_positions(model, 1)


def test_get_trace_profile_uppercase(traceprof_model):
    """_get_trace_profile returns data when model has PROF_FIB1."""
    out = SpectralExtractionPrimitive._get_trace_profile(traceprof_model, 1)
    assert out is not None
    assert out.shape == traceprof_model.PROF_FIB1.shape


def test_get_trace_profile_lowercase():
    """_get_trace_profile returns data when model has prof_fib1 (FITS-loaded)."""
    model = datamodels.TraceProfModel(config=datamodels.HISPECConfig)
    arr = np.ones((50, 100), dtype=np.float64) * 0.2
    setattr(model, "prof_fib1", arr)
    out = SpectralExtractionPrimitive._get_trace_profile(model, 1)
    assert out is not None
    np.testing.assert_array_equal(out, 0.2)


def test_get_trace_profile_missing_raises():
    """_get_trace_profile raises ValueError when fiber has no profile."""
    model = datamodels.TraceProfModel(config=datamodels.HISPECConfig)
    with pytest.raises(ValueError, match="no profile for fiber 1"):
        SpectralExtractionPrimitive._get_trace_profile(model, 1)


# --- Box extraction on data model ---


def test_box_extraction_single_fiber(l0_model, tracepos_model, traceprof_model):
    """Box extraction produces finite 1D spectra per order."""
    prim = SpectralExtractionPrimitive(
        method="box",
        extraction_aperture=10,
        do_fibers=[1],
        do_orders=l0_model.make_order_vec(),
        tracepos=tracepos_model,
        traceprof=traceprof_model,
    )
    result = prim.apply(l0_model)
    out = result if not isinstance(result, list) else result[0]
    assert isinstance(out, datamodels.L1Model)

    spec = getattr(out, "SPEC_UNCAL_FIB1")
    assert spec is not None
    assert spec.shape[0] == len(l0_model.make_order_vec())
    assert spec.shape[1] == l0_model.rate.shape[0]  # npix = ny when orders_are_vertical
    assert np.any(np.isfinite(spec))
    assert np.all(np.isfinite(spec) | np.isnan(spec))


def test_box_extraction_list_input(l0_model, tracepos_model, traceprof_model):
    """Box extraction accepts list of one L0 model."""
    prim = SpectralExtractionPrimitive(
        method="box",
        extraction_aperture=10,
        do_fibers=[1],
        do_orders=l0_model.make_order_vec(),
        tracepos=tracepos_model,
        traceprof=traceprof_model,
    )
    result = prim.apply([l0_model])
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], datamodels.L1Model)


# --- Optimal extraction ---


def test_optimal_extraction_single_fiber(l0_model, tracepos_model, traceprof_model):
    """Optimal extraction runs and produces finite spectra."""
    prim = SpectralExtractionPrimitive(
        method="optimal",
        extraction_aperture=10,
        do_fibers=[1],
        do_orders=l0_model.make_order_vec(),
        tracepos=tracepos_model,
        traceprof=traceprof_model,
    )
    result = prim.apply(l0_model)
    out = result if not isinstance(result, list) else result[0]
    assert isinstance(out, datamodels.L1Model)
    spec = getattr(out, "SPEC_UNCAL_FIB1")
    assert spec is not None
    assert np.any(np.isfinite(spec))


# --- Error when trace data missing ---


def test_extraction_fails_when_tracepos_has_no_data(l0_model, traceprof_model):
    """Applying extraction with empty TracePosModel (no POS_FIB1/pos_fib1) raises ValueError."""
    empty_tracepos = datamodels.TracePosModel(config=datamodels.HISPECConfig)
    # Leave POS_FIB1 / pos_fib1 as None (init_data sets lowercase to None; we set nothing)
    prim = SpectralExtractionPrimitive(
        method="box",
        extraction_aperture=10,
        do_fibers=[1],
        do_orders=l0_model.make_order_vec(),
        tracepos=empty_tracepos,
        traceprof=traceprof_model,
    )
    with pytest.raises(ValueError, match="no trace positions for fiber 1"):
        prim.apply(l0_model)

import numpy as np
import astropy.units as u
from hispecdrp import datamodels, BlazeCorrectionPrimitive
from hispecdrp.datamodels.dqflags import DQ_FLAGS
import time
import os


def test_blaze_correction():
    start_time = time.time()

    science_model = datamodels.L1Model()
    for fib in range(1,4):
        setattr(science_model, f"SPEC_UNCAL_FIB{fib}", np.ones((39,4096)))
        setattr(science_model, f"SPEC_UNCAL_ERR_FIB{fib}", np.ones((39,4096)) / 100)
        setattr(science_model, f"SPEC_UNCAL_VAR_POISSON_FIB{fib}", np.zeros((39,4096)))
        setattr(science_model, f"SPEC_UNCAL_VAR_RNOISE_FIB{fib}", np.zeros((39,4096)))
        setattr(science_model, f"SPEC_DQ_FIB{fib}", np.zeros((39, 4096), dtype=np.uint32))


    fiberflat_model = datamodels.FiberflatModel()
    for fib in range(1,4):

        setattr(fiberflat_model, f"SPEC_FIB{fib}", np.ones((39,4096)))
        setattr(fiberflat_model, f"SPEC_ERR_FIB{fib}", np.ones((39,4096))/100)
        setattr(fiberflat_model, f"SPEC_DQ_FIB{fib}", np.zeros((39, 4096), dtype=np.uint32))

    blaze_corrected = BlazeCorrectionPrimitive.call(
        science_model,
        fiberflat=fiberflat_model
    )

    step_time = time.time()
    print(f"Native rate map, elapsed: {step_time - start_time:.2f}s")

    assert getattr(blaze_corrected, "SPEC_FIB1").all() == 1.

    do_not_use = DQ_FLAGS["DO_NOT_USE"]
    dq = getattr(blaze_corrected, "SPEC_DQ_FIB1")
    edge = 16 * 10
    flagged = (dq & do_not_use) != 0
    assert flagged.mean() < 0.2, "blaze DQ should not flag the entire detector"
    assert flagged[:, :edge].all() and flagged[:, -edge:].all()
    assert not flagged[:, edge:-edge].any()


def test_blaze_correction_error_propagation_after_normalize():
    """Normalized blaze and blaze uncertainty must share the same per-order scale factor."""
    n_orders, n_pix = 39, 4096
    science_model = datamodels.L1Model()
    fiberflat_model = datamodels.FiberflatModel()
    for fib in range(1, 4):
        spec = np.full((n_orders, n_pix), 300.0)
        err = np.full((n_orders, n_pix), 4.0)
        flat = np.linspace(500.0, 3000.0, n_pix)
        flat = np.broadcast_to(flat, (n_orders, n_pix)).copy()
        flat_err = np.full((n_orders, n_pix), 40.0)
        setattr(science_model, f"SPEC_UNCAL_FIB{fib}", spec)
        setattr(science_model, f"SPEC_UNCAL_ERR_FIB{fib}", err)
        setattr(science_model, f"SPEC_UNCAL_VAR_POISSON_FIB{fib}", np.zeros((n_orders, n_pix)))
        setattr(science_model, f"SPEC_UNCAL_VAR_RNOISE_FIB{fib}", np.zeros((n_orders, n_pix)))
        setattr(science_model, f"SPEC_DQ_FIB{fib}", np.zeros((n_orders, n_pix), dtype=np.uint32))
        setattr(fiberflat_model, f"SPEC_FIB{fib}", flat)
        setattr(fiberflat_model, f"SPEC_ERR_FIB{fib}", flat_err)
        setattr(fiberflat_model, f"SPEC_DQ_FIB{fib}", np.zeros((n_orders, n_pix), dtype=np.uint32))

    out = BlazeCorrectionPrimitive.call(science_model, fiberflat=fiberflat_model)
    corr_err = getattr(out, "SPEC_ERR_FIB1")
    corr_spec = getattr(out, "SPEC_FIB1")
    ok = np.isfinite(corr_spec) & np.isfinite(corr_err) & (corr_spec > 0) & (corr_err > 0)
    snr = corr_spec[ok] / corr_err[ok]
    assert np.nanmedian(snr) > 10.0
    assert np.nanmedian(corr_err[ok]) < 0.5 * np.nanmedian(spec[ok])


def test_blaze_correction_gappy_fiberflat():
    """NaN holes from failed trace extraction must not break blaze spline fitting."""
    from hispecdrp.primitives.blaze_correction.blaze_correction_primitive import (
        estimate_blaze_spline,
        get_smooth_blaze,
    )

    n_pix = 4096
    flat = np.ones(n_pix)
    flat_err = np.ones(n_pix) / 100.0
    dq = np.zeros(n_pix, dtype=np.uint32)
    # Simulate trace-edge NaN gaps plus an internal hole (like real fiberflats).
    flat[:400] = np.nan
    flat[1200:1800] = np.nan
    flat[-350:] = np.nan

    smooth = estimate_blaze_spline(flat, flat_err, dq)
    assert smooth.shape == (n_pix,)
    assert np.isfinite(smooth[400:1200]).all()
    assert np.isnan(smooth[:400]).all()
    assert np.isnan(smooth[1200:1800]).all()

    order_vec = np.array([110, 111, 113])
    do_orders = order_vec
    flux = np.stack([flat, flat.copy(), flat.copy()])
    err = np.stack([flat_err, flat_err, flat_err])
    dq3 = np.stack([dq, dq, dq])
    blaze = get_smooth_blaze(flux, err, dq3, order_vec, do_orders)
    assert blaze.shape == (3, n_pix)
    assert np.all(np.isfinite(blaze[:, 400:1200]))


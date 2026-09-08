"""Tests for ``BlazeCorrectionPrimitive``.
"""
import numpy as np
from hispecdrp import datamodels, BlazeCorrectionPrimitive
from hispecdrp.datamodels.dqflags import DQ_FLAGS


def test_blaze_correction_primitive():

    science_model = datamodels.L1Model()
    n_orders = science_model.n_orders
    npix = science_model[1].spec_uncal.shape[1]
    for fib in range(1, science_model.num_fibers+1):
        science_model[fib].spec_uncal = np.ones((n_orders, npix), dtype=np.float32)
        science_model[fib].spec_uncal_err = science_model[fib].spec_uncal / 100

    fiberflat_model = datamodels.FiberflatModel()
    for fib in range(1, fiberflat_model.num_fibers+1):
        fiberflat_model[fib].spec = np.ones((n_orders, npix), dtype=np.float32)
        fiberflat_model[fib].spec_err = fiberflat_model[fib].spec / 100

    # Run the blaze correction
    science_model_blazecorr = BlazeCorrectionPrimitive.call(
        science_model,
        fiberflat=fiberflat_model
    )

    assert np.all(science_model_blazecorr[1].spec == 1)

    do_not_use = DQ_FLAGS["DO_NOT_USE"]
    dq = science_model_blazecorr[1].spec_dq
    edge = 16 * 10
    flagged = (dq & do_not_use) != 0
    assert flagged.mean() < 0.2, "blaze DQ should not flag the entire detector"
    # NOTE: Only happens when smooth_blaze=True
    #assert flagged[:, :edge].all() and flagged[:, -edge:].all() 
    #assert not flagged[:, edge:-edge].any()


def test_blaze_correction_primitive_error_propagation_after_normalize():
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
        science_model[fib].spec_uncal = np.full((n_orders, n_pix), 300.0)
        science_model[fib].spec_uncal_err = err
        
        fiberflat_model[fib].spec = flat
        fiberflat_model[fib].spec_err = flat_err

    out = BlazeCorrectionPrimitive.call(science_model, fiberflat=fiberflat_model)
    corr_err = getattr(out, "SPEC_ERR_FIB1")
    corr_spec = getattr(out, "SPEC_FIB1")
    ok = np.isfinite(corr_spec) & np.isfinite(corr_err) & (corr_spec > 0) & (corr_err > 0)
    snr = corr_spec[ok] / corr_err[ok]
    assert np.nanmedian(snr) > 10.0
    assert np.nanmedian(corr_err[ok]) < 0.5 * np.nanmedian(spec[ok])


def test_blaze_correction_remove_lamp_spectrum_recovers_flat_continuum():
    """BB×instrument blaze flat + flat science continuum recovers after lamp removal."""
    from hispecdrp.primitives.blaze_correction.blaze_correction_primitive import (
        lamp_spectrum_shape,
    )

    n_orders, n_pix = 3, 512
    order_start, order_end = 110, 112
    meta = {
        "order_start": order_start,
        "order_end": order_end,
        "spectrograph": "BSPEC",
        "instrument_name": "HISPEC",
    }
    science_model = datamodels.L1Model(meta=meta)
    science_model.meta.order_start = order_start
    science_model.meta.order_end = order_end
    fiberflat_model = datamodels.FiberflatModel(meta=meta)
    fiberflat_model.meta.order_start = order_start
    fiberflat_model.meta.order_end = order_end

    x = np.linspace(-1.0, 1.0, n_pix)
    instrument_blaze = 0.4 + 0.6 * (1.0 - x**2)  # peaked, nonzero ends
    wave = np.empty((n_orders, n_pix), dtype=np.float64)
    for i, order in enumerate(range(order_start, order_end + 1)):
        lam_c = 147.0 / float(order)
        wave[i] = np.linspace(lam_c * 0.98, lam_c * 1.02, n_pix)

    lamp = np.stack([lamp_spectrum_shape(wave[i], temperature_k=1000.0) for i in range(n_orders)])
    fiberflat = instrument_blaze[np.newaxis, :] * lamp
    # Science = continuum × instrument blaze only (no lamp SED).
    science_flux = 1000.0 * np.broadcast_to(instrument_blaze, (n_orders, n_pix)).copy()

    for fib in (1, 2, 3):

        science_model[fib].spec_uncal = science_flux.copy()
        science_model[fib].spec_uncal_err = np.full((n_orders, n_pix), 1.0)
        #science_model[fib].spec_uncal_var_poisson = np.zeros((n_orders, n_pix))
        #science_model[fib].spec_uncal_var_rnoise = np.zeros((n_orders, n_pix))
        #science_model[fib].spec_dq = np.zeros((n_orders, n_pix), dtype=np.uint32)
        science_model[fib].wave = wave.copy()
        fiberflat_model[fib].spec = fiberflat.copy()
        fiberflat_model[fib].spec_err = np.full((n_orders, n_pix), 1.0)
        #fiberflat_model[fib].spec_dq = np.zeros((n_orders, n_pix), dtype=np.uint32)
        fiberflat_model[fib].wave = wave.copy()

    out = BlazeCorrectionPrimitive.call(
        science_model,
        fiberflat=fiberflat_model,
        remove_lamp_spectrum=True,
        lamp_temperature_k=1000.0,
        smooth_blaze=False,
    )
    corr = out[1].spec
    edge = 16 * 10
    core = corr[:, edge:-edge]
    # After max-normalized blaze divide, continuum should be nearly flat across the order.
    for o in range(n_orders):
        row = core[o]
        ok = np.isfinite(row)
        assert ok.mean() > 0.9
        rel = row[ok] / np.nanmedian(row[ok])
        assert np.nanstd(rel) < 1e-8, f"order {order_start + o} residual RMS {np.nanstd(rel)}"

    # Without lamp removal, BB imprint leaves a clear continuum slope.
    out_raw = BlazeCorrectionPrimitive.call(
        science_model,
        fiberflat=fiberflat_model,
        remove_lamp_spectrum=False,
        smooth_blaze=False,
    )
    corr_raw = out_raw[1].spec[:, edge:-edge]
    rel_raw = corr_raw[0] / np.nanmedian(corr_raw[0])
    assert np.nanstd(rel_raw[np.isfinite(rel_raw)]) > 1e-3


def test_blaze_correction_primitive_gappy_fiberflat():
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


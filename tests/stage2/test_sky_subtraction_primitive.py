"""Tests for ``SkySubtractionPrimitive``.
"""
import numpy as np
from hispecdrp import datamodels, SkySubtractionPrimitive
import time


def test_sky_subtraction_primitive():
    """Exercise SkySubtractionPrimitive.call on a compact L1 + fiberflat stack."""
    start_time = time.time()

    n_orders, n_pix = 4, 512
    order_start, order_end = 120, 123
    science_model = datamodels.L1Model(
        meta={
            "order_start": order_start,
            "order_end": order_end,
            "spectrograph": "BSPEC",
            "instrument_name": "HISPEC",
        }
    )
    science_model.meta.order_start = order_start
    science_model.meta.order_end = order_end

    fiberflat_model = datamodels.FiberflatModel(
        meta={
            "order_start": order_start,
            "order_end": order_end,
            "spectrograph": "BSPEC",
            "instrument_name": "HISPEC",
        }
    )
    fiberflat_model.meta.order_start = order_start
    fiberflat_model.meta.order_end = order_end

    for fib in range(1, 4):
        setattr(science_model, f"SPEC_UNCAL_FIB{fib}", np.ones((n_orders, n_pix)))
        setattr(science_model, f"SPEC_UNCAL_ERR_FIB{fib}", np.ones((n_orders, n_pix)) / 100)
        setattr(science_model, f"SPEC_UNCAL_VAR_POISSON_FIB{fib}", np.zeros((n_orders, n_pix)))
        setattr(science_model, f"SPEC_UNCAL_VAR_RNOISE_FIB{fib}", np.zeros((n_orders, n_pix)))
        setattr(science_model, f"SPEC_UNCAL_DQ_FIB{fib}", np.zeros((n_orders, n_pix), dtype=np.uint32))
        setattr(fiberflat_model, f"SPEC_FIB{fib}", np.ones((n_orders, n_pix)))
        setattr(fiberflat_model, f"SPEC_ERR_FIB{fib}", np.ones((n_orders, n_pix)) / 100)
        setattr(fiberflat_model, f"SPEC_DQ_FIB{fib}", np.zeros((n_orders, n_pix), dtype=np.uint32))

    # Synthetic WAVE grids (avoid SimulateRateMap HDF load).
    for fiber in range(1, 4):
        wv = np.empty((n_orders, n_pix), dtype=np.float32)
        for i, order in enumerate(range(order_start, order_end + 1)):
            lam_c = 147.0 / float(order)
            wv[i] = np.linspace(lam_c * 0.99, lam_c * 1.01, n_pix, dtype=np.float32)
        setattr(science_model, f"WAVE_FIB{fiber}", wv)
        setattr(fiberflat_model, f"WAVE_FIB{fiber}", wv)

    science_model.meta.object_fiber1 = "SCIENCE"
    science_model.meta.object_fiber2 = "SPECKLE"
    science_model.meta.object_fiber3 = "SKY"
    science_model.meta.distance_fiber1 = 0.0
    science_model.meta.distance_fiber2 = 0.7
    science_model.meta.distance_fiber3 = 2.07

    sky_subtracted = SkySubtractionPrimitive.call(
        science_model,
        fiberflat=fiberflat_model,
        show_preview=False,
        save_preview=False,
    )

    step_time = time.time()
    print(f"Sky subtraction, elapsed: {step_time - start_time:.2f}s")
    assert getattr(sky_subtracted, "SPEC_FIB1").all() == 0.0

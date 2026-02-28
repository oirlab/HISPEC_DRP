import numpy as np
import astropy.units as u
from hispecdrp import datamodels, BlazeCorrectionPrimitive
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
        setattr(science_model, f"SPEC_DQ_FIB{fib}", np.zeros((39,4096)))


    fiberflat_model = datamodels.FiberflatModel()
    for fib in range(1,4):

        setattr(fiberflat_model, f"SPEC_FIB{fib}", np.ones((39,4096)))
        setattr(fiberflat_model, f"SPEC_ERR_FIB{fib}", np.ones((39,4096))/100)
        setattr(fiberflat_model, f"SPEC_DQ_FIB{fib}", np.zeros((39,4096)))

    prim = BlazeCorrectionPrimitive()
    blaze_corrected = prim.apply(science_model, fiberflat=fiberflat_model)

    step_time = time.time()
    print(f"Native rate map, elapsed: {step_time - start_time:.2f}s")

    assert getattr(blaze_corrected, "SPEC_FIB1").all() == 1.


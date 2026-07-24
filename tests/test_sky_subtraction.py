import numpy as np
from hispecdrp import datamodels, SkySubtractionPrimitive
import hispecdrp.simulations as simulations
import time
import os

def test_sky_subtraction():
    start_time = time.time()

    science_model = datamodels.L1Model()
    for fib in range(1,4):
        setattr(science_model, f"SPEC_UNCAL_FIB{fib}", np.ones((39,4096)))
        setattr(science_model, f"SPEC_UNCAL_ERR_FIB{fib}", np.ones((39,4096)) / 100)
        setattr(science_model, f"SPEC_UNCAL_VAR_POISSON_FIB{fib}", np.zeros((39,4096)))
        setattr(science_model, f"SPEC_UNCAL_VAR_RNOISE_FIB{fib}", np.zeros((39,4096)))
        setattr(science_model, f"SPEC_DQ_FIB{fib}", np.zeros((39,4096)).astype(np.uint32))


    fiberflat_model = datamodels.FiberflatModel()
    for fib in range(1,4):

        setattr(fiberflat_model, f"SPEC_FIB{fib}", np.ones((39,4096)))
        setattr(fiberflat_model, f"SPEC_ERR_FIB{fib}", np.ones((39,4096))/100)
        setattr(fiberflat_model, f"SPEC_DQ_FIB{fib}", np.zeros((39,4096)).astype(np.uint32))

    sim = simulations.SimulateRateMap()
    wave_soln = sim.get_wavelength_solution()

    # wavelength solutions
    order_start = 110
    order_end = 148
    for fiber in range(1,4):
        wv_soln_array = np.empty((39, 4096), dtype=np.float32)
        for i, o in enumerate(np.arange(order_start, order_end+1,1)):
            wv_soln_array[i] = wave_soln[int(fiber)][o]['wavelength']
            print(o,np.nanmax(wv_soln_array[i]))
        setattr(science_model, f'WAVE_FIB{fiber}', wv_soln_array)
        setattr(fiberflat_model, f'WAVE_FIB{fiber}', wv_soln_array)

    science_model.meta.object_fiber1 = 'SCIENCE'
    science_model.meta.object_fiber2 = 'SPECKLE'
    science_model.meta.object_fiber3 = 'SKY'

    science_model.meta.distance_fiber1 = 0.0
    science_model.meta.distance_fiber2 = 0.7
    science_model.meta.distance_fiber3 = 2.07

    prim = SkySubtractionPrimitive()
    sky_subtracted = prim.apply(science_model, fiberflat=fiberflat_model, save_preview=False, show_preview=False)

    step_time = time.time()
    print(f"Sky subtraction, elapsed: {step_time - start_time:.2f}s")

    assert getattr(sky_subtracted, "SPEC_FIB1").all() == 0.
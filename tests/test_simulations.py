import numpy as np
import astropy.units as u
from hispecdrp import simulations
import time
import os


def test_simulate_rate_map():
    start_time = time.time()

    srm = simulations.SimulateRateMap()
    fiber_1_wavelengths = np.arange(0.9,2.5,2.5e-6) * u.micron
    fiber_1_flux = np.ones_like(fiber_1_wavelengths.value)*1e-14 * u.erg/u.s/u.cm**2/u.micron

    fiber_23_wavelengths = np.arange(0.9,2.5,2.5e-3) * u.micron
    fiber_23_flux = np.ones_like(fiber_23_wavelengths.value)*1e-15 * u.erg/u.s/u.cm**2/u.micron

    srm.set_fiber_spectrum(fiber_number=1,
                        spectrum_wavelengths=fiber_1_wavelengths,
                        spectrum_flux=fiber_1_flux,
                        continuous=True)
    srm.set_fiber_spectrum(fiber_number=2,
                        spectrum_wavelengths=fiber_23_wavelengths,
                        spectrum_flux=fiber_23_flux,
                        continuous=False)
    srm.set_fiber_spectrum(fiber_number=3,
                        spectrum_wavelengths=fiber_23_wavelengths,
                        spectrum_flux=fiber_23_flux,
                        continuous=False)
    
    spectrograph = 'BSPEC'
    native_rate_map = srm.create_native_rate_map(spectrograph=spectrograph, method='prf_convolution', save=False)

    step_time = time.time()
    print(f"Native rate map, elapsed: {step_time - start_time:.2f}s")

    assert isinstance(srm.rate_map, np.ndarray)


import numpy as np
from hispecdrp import datamodels, TelluricCorrectionPrimitive
from hispecdrp.primitives.telluric_correction.telluric_models import _convolve_and_resample
import hispecdrp.simulations as simulations
import time
from pathlib import Path

def test_telluric_correction():
    start_time = time.time()

    # load telluric model
    data_dir = Path(__file__).parent.parent / "hispecdrp" / "data" / "atmosphere"
    telluric_basemodel_filepath = data_dir / "mktrans_zm_10_10.dat"

    telluric_data = np.loadtxt(telluric_basemodel_filepath)
    telluric_wavelength = telluric_data[:,0] # units assumed in microns
    telluric_trans = telluric_data[:,1]

    # create science model
    science_model = datamodels.L1Model()
    sci_flux = 100.
    for fib in range(1,4):
        setattr(science_model, f"SPEC_UNCAL_FIB{fib}", np.ones((39,4096))*sci_flux)
        setattr(science_model, f"SPEC_UNCAL_ERR_FIB{fib}", np.ones((39,4096)) *np.sqrt(sci_flux))
        setattr(science_model, f"SPEC_UNCAL_VAR_POISSON_FIB{fib}", np.zeros((39,4096)))
        setattr(science_model, f"SPEC_UNCAL_VAR_RNOISE_FIB{fib}", np.zeros((39,4096)))
        setattr(science_model, f"SPEC_DQ_FIB{fib}", np.zeros((39,4096)))

    science_model.meta.object_fiber1 = 'SCIENCE'
    science_model.meta.airmass_start = 1.1
    science_model.meta.airmass_end = 1.11

    avg_airmass = np.nanmean([science_model.meta.airmass_start, science_model.meta.airmass_end])

    sim = simulations.SimulateRateMap()
    wave_soln = sim.get_wavelength_solution()

    variable_lsf = np.linspace(1.5/9e4 / 2.355, 1.5/1.1e5 / 2.355, 4096) # not meant to be very realistic, just to test

    order_start = 110
    order_end = 148
    for fiber in range(1,4):
        wv_soln_array = np.empty((39, 4096), dtype=np.float32)
        interp_flux = np.ones((39,4096), dtype=np.float32)*sci_flux
        for i, o in enumerate(np.arange(order_start, order_end+1,1)):
            wv_soln_array[i] = wave_soln[int(fiber)][o]['wavelength']
            tel_conv= _convolve_and_resample(telluric_wavelength, telluric_trans**avg_airmass, wv_soln_array[i], variable_lsf)
            interp_flux[i] *= tel_conv
        setattr(science_model, f'WAVE_FIB{fiber}', wv_soln_array)
        setattr(science_model, f"SPEC_UNCAL_FIB{fiber}", interp_flux)

    # create dummy LSF model
    lsf_model = datamodels.GaussianLSFModel()

    lsf_mu = np.zeros((39,4096))
    lsf_sigma = np.ones((39,4096)) * 1.5/1e5 / 2.355 
    lsf_sigma = variable_lsf
    lsf_offset = np.zeros((39,4096))
    lsf_params = np.empty((39,4096,3))

    lsf_params[:,:,0] = lsf_mu
    lsf_params[:,:,1] = lsf_sigma
    lsf_params[:,:,2] = lsf_offset

    setattr(lsf_model, "LSF_PARAMS_FIB1", lsf_params)

    # run primitive
    telluric_corrected = TelluricCorrectionPrimitive.call(science_model, lsf=lsf_model, show_preview=False, save_preview=False)

    step_time = time.time()
    print(f"Telluric correction, elapsed: {step_time - start_time:.2f}s")

    assert ((getattr(telluric_corrected, "SPEC_FIB1")-sci_flux)/sci_flux).all() < 1e-4
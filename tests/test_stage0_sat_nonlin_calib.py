from hispecdrp import ProcessSatFlatsPipeline
from hispecdrp import datamodels
import hispecdrp.simulations as simu
import numpy as np
import os
from hispecdrp.datamodels.dqflags import DQ_FLAGS
import time

def test_stage0_sat_nonlin_calib(tmp_path):
    np.random.seed(12345)
    start_time = time.time()

    # --- Define parameters for the simulated ramp ---
    max_cores = 2
    max_slope = 1000  # e-/s
    readtime = 1.7  # seconds
    n_reads = 60
    N_files = 10  # number of simulated files to create
    ny, nx = 2,2  # size of the detector

    step_time = time.time();
    print(f"Overheads, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Non-linearity and its inverse polynomial coefficients ---

    # Create an ideal ramp (1D array for demonstration)
    ideal_ramp = np.linspace(0, 1.5 * np.iinfo(np.uint16).max, 1001, endpoint=True)
    nonlin_coeffs = [-1e-6, 1, 0]  # Example coefficients for a cubic non-linearity
    # nonlin_coeffs = [1, 0]  # Example coefficients for a cubic non-linearity
    nonlinear_ramp = np.polyval(nonlin_coeffs, ideal_ramp)

    # Fit inverse polynomial (correction) coefficients
    nonlin_correction_coefs = np.polyfit(nonlinear_ramp, ideal_ramp, deg=5)
    print(nonlin_correction_coefs)
    # nonlin_correction_coefs = np.array([0.00000000e+00, 9.93497729e-01,  9.46247810e-07,  4.09586662e-12, -4.29914583e-17,  3.92020728e-22])[::-1]
    nonlin_correction_coefs = np.array([0.00000000e+00, 9.93166438e-01, 9.93459922e-07, 1.99687217e-12, 4.28159139e-18, 2.47963102e-23])[::-1]
    print(nonlin_correction_coefs)
    print("Fitted correction coefficients:", nonlin_correction_coefs)

    # Apply correction and check accuracy
    nonlin_corrected_ramp = np.polyval(nonlin_correction_coefs, nonlinear_ramp)
    # nonlin_corrected_from_linear = np.polyval(nonlin_correction_coefs, ideal_ramp)


    coeffs = np.broadcast_to(nonlin_correction_coefs[::-1, None, None], (np.size(nonlin_correction_coefs), ny, nx))
    thresh = np.zeros(shape=(ny, nx), dtype=np.uint16)
    max = np.full(shape=(ny, nx), fill_value=0.9 * np.iinfo(np.uint16).max, dtype=np.uint16)
    nonlin_dq = np.zeros(shape=(ny, nx), dtype=np.uint32)
    nonlin_model = datamodels.NonlinearCorrectionModel(data=dict(coeffs=coeffs,
                                                                 nonlin_thresh=thresh,
                                                                 nonlin_max=max,
                                                                 dq=nonlin_dq))

    step_time = time.time();
    print(f"Step: defining nonlin coefficients, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Create DQ init map ---
    dqinit_map = np.zeros((ny, nx), dtype=np.uint32)
    # set row to bad
    # dqinit_map[5, :] |= DQ_FLAGS["DO_NOT_USE"]
    dqinit_model = datamodels.DQModel(data={"DQ": dqinit_map})

    step_time = time.time();
    print(f"Step: defining DQ init calibration, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Create saturation threshold map ---
    saturation_threshold = np.iinfo(np.uint16).max  # DN
    saturation_map = np.full((ny, nx), saturation_threshold, dtype=np.float32)
    saturation_model = datamodels.SaturationModel(data={'SAT_THRESH': saturation_map,
                                                        "DQ": np.zeros_like(saturation_map, dtype=np.uint32), })

    step_time = time.time();
    print(f"Step: defining saturation calibration, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Create random gain map ---
    gain = 1  # e-/ADU
    gain_map = np.full((ny, nx), gain, dtype=np.float32)
    gain_model = datamodels.GainModel(data={'GAIN': gain_map,
                                           "RN": np.full_like(gain_map, 0.0),
                                           "COV": np.full((2, 2, ny, nx), 0, dtype=np.float32),
                                           "DQ": np.zeros_like(gain_map, dtype=np.uint32), })

    step_time = time.time();
    print(f"Step: defining gain calibration, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Create random dark map ---
    # dark_current = 1  # e-/s
    # dark_map = np.random.normal(loc=dark_current, scale=dark_current / 10, size=(ny, nx)).astype(np.float32)
    dark_map = np.zeros((ny, nx), dtype=np.float32)
    dark_model = datamodels.DarkModel(data={'RATE': dark_map / gain_map,  # dark in DN/s
                                            "RATE_ERR": np.zeros_like(dark_map),
                                            "RATE_VAR_RNOISE": np.zeros_like(dark_map),
                                            "RATE_VAR_POISSON": np.zeros_like(dark_map),
                                            "DQ": np.zeros_like(dark_map, dtype=np.uint32), })

    step_time = time.time();
    print(f"Step: defining dark calibration, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Create bias map ---
    bias = 1000  # e-/s
    bias_map = np.full((ny, nx), bias, dtype=np.float32)
    kTC_noise = 0  # e-/s
    kTC_noise_map = np.full((ny, nx), kTC_noise, dtype=np.float32)
    bias_model = datamodels.BiasModel(data={'BIAS': bias_map / gain_map,  # bias in DN
                                            "kTC": kTC_noise_map / gain_map,  # kTC in DN
                                            "DQ": np.zeros_like(bias_map, dtype=np.uint32), })

    step_time = time.time();
    print(f"Step: defining gain calibration, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Create random flat map ---
    # detflat_map = np.random.normal(loc=1, scale=0.1, size=(ny, nx)).astype(np.float32)
    detflat_map = np.ones((ny, nx), dtype=np.float32)
    detflat_model = datamodels.DetectorFlatModel(data={'FLAT': detflat_map,
                                                       "FLAT_ERR": np.full_like(gain_map, 0.03),
                                                       "FLAT_VAR_RNOISE": np.full_like(gain_map, 0.01),
                                                       "FLAT_VAR_POISSON": np.full_like(gain_map, 0.02),
                                                       "DQ": np.zeros_like(detflat_map, dtype=np.uint32), })

    step_time = time.time();
    print(f"Step: defining flat calibration, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Create random read noise map ---
    RN = 0.1  # e-
    rn_map = np.full((ny, nx), RN, dtype=np.float32)
    rn_model = datamodels.ReadNoiseModel(data={'RN': rn_map / gain_map,
                                               "RN_ERR": np.ones_like(rn_map),
                                               "DQ": np.zeros_like(gain_map, dtype=np.uint32), })

    # --- Create a fake UTR cube ---
    electron_rate_map = np.full((ny, nx), max_slope, dtype=np.float32)
    start_time = time.time()
    ramp_model_sequence = []
    for k in range(N_files):
        ramp_data = simu.create_ramp(electron_rate_map,  # in e-/s
                                           readtime=readtime, n_reads=n_reads,
                                           nonlin_coeffs=nonlin_coeffs,
                                           gain=gain_map,  # in e-/s
                                           flat=detflat_map,
                                           dark=dark_map,  # in e-/s
                                           bias=bias_map,  # in e-/s
                                           kTC_noise=kTC_noise,  # in e-/s
                                           poisson_noise=False, read_noise=rn_map,  # in e-/s
                                           convert_to_uint16=True, clip_ramps=True,
                                           max_cores=max_cores)
        step_time = time.time();
        print(f"Step: creating ramp (new), elapsed: {step_time - start_time:.2f}s")
        start_time = time.time()

        ramp_model = datamodels.RampModel(data=dict(data=ramp_data['data'], dq_raw=ramp_data['dq_raw']),
                                          meta=ramp_data['meta'])
        ramp_model.meta.filename = "simulated_ramp.fits"
        print(ramp_model.meta.clock_rate)

        ramp_model_sequence.append(ramp_model)

    step_time = time.time();
    print(f"Step: creating ramp, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    pipe_max_cores = max_cores
    # --- Specify the primitive arguments here ---
    prim_args = {
        'dq_init': {
            'run': True,
            'max_cores': pipe_max_cores,
        },
        'make_saturation': {
            'run': True,
            'max_cores': pipe_max_cores,
        },
        'register_saturation_cal':{
            'run': False,
        },
        'saturation_check': {
            'run': True,
            'max_cores': pipe_max_cores,
        },
        'make_nonlin': {
            'run': True,
            'mode': 'full_detector',
            'use_legendre': True,
            'order': 5,
            'readstart':0,
            'satval_frac': 0.95,
            'criterion_for_nonlin_thresh':0.01,
            'criterion_for_nonlin_max':0.05,
            'max_cores': pipe_max_cores,
        },
        'register_nonlin_cal':{
            'run': False,
        },
        'bias_sub': {
            'run': False,
            'max_cores': pipe_max_cores,
        },
        'nonlin_corr': {
            'run': False,
            'max_cores': pipe_max_cores,
        },
        'jump_detec' : {
            'run': False,
            'ouput_model' : 'input',
            'method': 'tpd',
            'max_cores': pipe_max_cores,
            'rejection_threshold': 4.0,
        },
        'ramp_fit' : {
            'run': False,
            'method': 'cython_likely', # 'ols', 'mcds', 'cds', 'jwst_ols', 'jwst_likely', 'cython_likely'
            'max_cores': pipe_max_cores,
            'jwst_jump_detection': False,
            'jwst_jump_rejection_threshold': 4.0,
        },
        'dark_sub': {
            'run': False,
            'max_cores': pipe_max_cores,
        },
        'gain_corr': {
            'run': False,
            'max_cores': pipe_max_cores,
        },
        'detflat_corr': {
            'run': False,
            'max_cores': pipe_max_cores,
        },
    }

    # Setup and call the stage 0 pipeline
    satflats_pipe = ProcessSatFlatsPipeline()

    # Set any primitive args

    satflats_pipe.dq_init.dq = dqinit_model
    satflats_pipe.saturation_check.saturation = saturation_model
    satflats_pipe.bias_sub.bias = bias_model
    satflats_pipe.make_nonlin.gain = gain_model
    satflats_pipe.make_nonlin.rn = rn_model
    satflats_pipe.make_nonlin.saturation = saturation_model
    satflats_pipe.make_nonlin.bias = bias_model

    step_time = time.time();
    print(f"Step: defining pipeline, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # Run pipeline
    model_result = satflats_pipe.apply(ramp_model_sequence, prim_args=prim_args, output_dir=tmp_path)

    np.testing.assert_allclose(model_result['saturation'].sat_thresh, 65435, rtol=1e-6, atol=1)
    assert model_result['saturation'].sat_thresh.shape[0] == ny
    assert model_result['saturation'].sat_thresh.shape[1] == nx

    # print("nonlin full detector")
    # print(model_result['nonlin'].coeffs[:,0])
    # print(model_result['nonlin'].nonlin_thresh[0])
    # print(model_result['nonlin'].nonlin_max[0])
    # print(model_result['nonlin'].dq)

    nonlin_corrected_ramp_4testing = np.polyval(model_result['nonlin'].coeffs[::-1,0], nonlinear_ramp)
    rel_err_ideal = (nonlin_corrected_ramp[1::] - ideal_ramp[1::])/ideal_ramp[1::]
    rel_err_4testing = (nonlin_corrected_ramp_4testing[1::] - ideal_ramp[1::]) / ideal_ramp[1::]
    # print(np.abs(rel_err_ideal[1::] - rel_err_4testing[1::]))
    assert np.sum(np.abs(rel_err_ideal - rel_err_4testing) > 1e-2) == 0
    assert abs(model_result['nonlin'].nonlin_thresh[0]-18000) < 500
    assert abs(model_result['nonlin'].nonlin_max[0]-52600) < 500

    # Run pipeline
    prim_args['make_nonlin']['mode'] = 'per_pixel'
    model_result = satflats_pipe.apply(ramp_model_sequence, prim_args=prim_args, output_dir=tmp_path)

    # print("nonlin per_pixel")
    # print(model_result['nonlin'].coeffs.shape)
    # print(model_result['nonlin'].nonlin_thresh)
    # print(model_result['nonlin'].nonlin_max)
    # print(model_result['nonlin'].dq)

    nonlin_corrected_ramp_4testing = np.polyval(model_result['nonlin'].coeffs[::-1,0,0], nonlinear_ramp)
    rel_err_ideal = (nonlin_corrected_ramp[1::] - ideal_ramp[1::])/ideal_ramp[1::]
    rel_err_4testing = (nonlin_corrected_ramp_4testing[1::] - ideal_ramp[1::]) / ideal_ramp[1::]
    # print(np.abs(rel_err_ideal[1::] - rel_err_4testing[1::]))
    assert np.sum(np.abs(rel_err_ideal - rel_err_4testing) > 1e-2) == 0
    assert abs(model_result['nonlin'].nonlin_thresh[0,0]-18000) < 500
    assert abs(model_result['nonlin'].nonlin_max[0,0]-52600) < 500
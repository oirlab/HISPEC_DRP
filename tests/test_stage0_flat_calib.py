from hispecdrp import ProcessDetFlatsPipeline
from hispecdrp import datamodels
import numpy as np
import os
import time
from hispecdrp.datamodels.dqflags import DQ_FLAGS

def test_stage0_gain_calib(tmp_path):
    start_time = time.time()

    # --- Define parameters for the simulated ramp ---
    max_cores = 1
    max_slope = 200  # e-/s
    readtime = 1.7  # seconds
    n_reads = 30
    N_files = 100  # number of simulated files to create
    ny, nx = 2, 2  # size of the detector

    step_time = time.time();
    print(f"Overheads, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Define parameters for the simulated ramp ---
    max_cores = 1
    max_slope = 200  # e-/s
    readtime = 1.7  # seconds
    n_reads = 30
    N_files = 10  # number of simulated files to create
    ny, nx = 10,10  # size of the detector

    step_time = time.time();
    print(f"Overheads, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Non-linearity and its inverse polynomial coefficients ---

    # Create an ideal ramp (1D array for demonstration)
    ideal_ramp = np.linspace(0, 1.5 * np.iinfo(np.uint16).max, 1001, endpoint=True)
    # nonlin_coeffs = [-1e-6, 1, 0]  # Example coefficients for a cubic non-linearity
    nonlin_coeffs = [1, 0]  # Example coefficients for a cubic non-linearity
    nonlinear_ramp = np.polyval(nonlin_coeffs, ideal_ramp)

    # Fit inverse polynomial (correction) coefficients
    nonlin_correction_coefs = np.polyfit(nonlinear_ramp, ideal_ramp, deg=5)
    print("Fitted correction coefficients:", nonlin_correction_coefs)

    # Apply correction and check accuracy
    nonlin_corrected_ramp = np.polyval(nonlin_correction_coefs, nonlinear_ramp)

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
    gain = 3  # e-/ADU
    gain_map = np.full((ny, nx), gain, dtype=np.float32)
    detgain_model = datamodels.DetectorGainModel(data={'GAIN': gain_map,
                                                       "RN": np.full_like(gain_map, 0.0),
                                                       "COV": np.full((2, 2, ny, nx), 0, dtype=np.float32),
                                                       "DQ": np.zeros_like(gain_map, dtype=np.uint32), })

    step_time = time.time();
    print(f"Step: defining gain calibration, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Create random dark map ---
    # dark_current = 1  # e-/s
    # dark_map = np.random.normal(loc=dark_current, scale=dark_current / 10, size=(ny, nx)).astype(np.float32)
    dark_map = np.ones((ny, nx), dtype=np.float32)
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
    bias_model = datamodels.BiasModel(data={'BIAS': bias_map / gain_map,  # bias in DN
                                            "BIAS_ERR": np.zeros_like(bias_map),
                                            "DQ": np.zeros_like(bias_map, dtype=np.uint32), })

    step_time = time.time();
    print(f"Step: defining gain calibration, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Create random flat map ---
    # detflat_map = np.random.normal(loc=1, scale=0.1, size=(ny, nx)).astype(np.float32)
    detflat_map = np.ones((ny, nx), dtype=np.float32)
    detflat_map[0,0] = 0.7  # cold pixel
    detflat_map[0,1] = 0.95
    detflat_map[0,2] = 1.1
    detflat_map[0,3] = 1.4
    detflat_model = datamodels.DetectorFlatModel(data={'FLAT': detflat_map,
                                                       "FLAT_ERR": np.full_like(gain_map, 0.03),
                                                       "FLAT_VAR_RNOISE": np.full_like(gain_map, 0.01),
                                                       "FLAT_VAR_POISSON": np.full_like(gain_map, 0.02),
                                                       "DQ": np.zeros_like(detflat_map, dtype=np.uint32), })

    step_time = time.time();
    print(f"Step: defining flat calibration, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Create random read noise map ---
    RN = 8.5  # e-
    rn_map = np.full((ny, nx), RN, dtype=np.float32)
    rn_model = datamodels.ReadNoiseModel(data={'RN': rn_map / gain_map,
                                               "RN_ERR": np.ones_like(rn_map),
                                               "DQ": np.zeros_like(gain_map, dtype=np.uint32), })

    # --- Create a fake UTR cube ---
    electron_rate_map = np.full((ny, nx), max_slope, dtype=np.float32)
    start_time = time.time()
    ramp_model_sequence = []
    for k in range(N_files):
        ramp_data = datamodels.create_ramp(electron_rate_map,  # in e-/s
                                           readtime=readtime, n_reads=n_reads,
                                           nonlin_coeffs=nonlin_coeffs,
                                           gain=gain_map,  # in e-/s
                                           flat=detflat_map,
                                           dark=dark_map,  # in e-/s
                                           bias=bias_map,  # in e-/s
                                           kTC_noise=50,  # in e-/s
                                           poisson_noise=True, read_noise=rn_map,  # in e-/s
                                           convert_to_uint16=True, clip_ramps=True,
                                           max_cores=max_cores)
        # adding a jump
        # ramp_data['data'][4::,:,:] += 1000
        # ramp_data['data'][10::,:,:] += 1000
        # ramp_data['data'][15::,:,:] += 1000
        # ramp_data['data'][20::,:,:] += 1000
        # ramp_data['data'][4::,5,5] += 3000

        # print(ramp_data['data'][:,0,0])
        # print(gain_map)
        # exit()

        step_time = time.time();
        print(f"Step: creating ramp (new), elapsed: {step_time - start_time:.2f}s")
        start_time = time.time()

        ramp_model = datamodels.RampModel(data=dict(data=ramp_data['data'], dq_raw=ramp_data['dq_raw']),
                                          meta=ramp_data['meta'])
        ramp_model.meta.filename = "simulated_ramp.fits"
        print(ramp_model.meta.clock_rate)

        ramp_model_sequence.append(ramp_model)
    # exit()

    step_time = time.time();
    print(f"Step: creating ramp, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    pipe_max_cores = max_cores
    # --- Specify the primitive arguments here ---
    prim_args = {
        'dq_init': {
            'suffix': 'dqinit',
            'max_cores': pipe_max_cores,
        },
        'saturation': {
            'suffix': 'checksat',
            'max_cores': pipe_max_cores,
        },
        'bias': {
            'suffix': 'bias',
            'run': True,
            'max_cores': pipe_max_cores,
        },
        'nonlin_corr': {
            'suffix': 'nonlin',
            'run': True,
            'max_cores': pipe_max_cores,
        },
        'jump_detec': {
            'run': True,
            'suffix': 'jumpdetec',
            'ouput_model': 'input',
            'method': 'tpd',
            'max_cores': pipe_max_cores,
            'rejection_threshold': 40.0,
        },
        'fit_det_gain': {
            'run': False,
            'suffix': 'detgain',
            'max_cores': pipe_max_cores,
            'mode': 'per_pixel',  # 'full_detector', 'per_pixel'
            'method': 'quad',
            'fix_RN': True,
            'init_delta_gains': np.linspace(-0.5, 0.5, 100),  # Example delta gain values to try
            'init_delta_RNs': np.linspace(-1, 1, 100),
            'quad_delta_gain': 0.2,
            'quad_delta_RN': 0.5
        },
        'ramp_fit' : {
            'run': True,
            'suffix': 'rampfit',
            'method': 'jwst_ols', # 'ols', 'mcds', 'cds', 'jwst_ols', 'jwst_likely'
            'max_cores': pipe_max_cores,
            'jwst_jump_detection': False,
            'jwst_jump_rejection_threshold': 4.0,
        },
        'dark': {
            'run': True,
            'suffix': 'dark',
            'max_cores': pipe_max_cores,
        },
        'det_gain': {
            'run': True,
            'suffix': 'detgain',
            'max_cores': pipe_max_cores,
        },
        'coadd': {
            'run': True,
        },
        'fit_det_flat': {
            'run': True,
        },
        'det_flat': {
            'run': False,
            'suffix': 'detflat',
            'max_cores': pipe_max_cores,
        },
    }

    # Setup and call the stage 0 pipeline
    detflat_pipe = ProcessDetFlatsPipeline()

    # Set any primitive args

    detflat_pipe.dq_init.dq = dqinit_model
    detflat_pipe.saturation.saturation = saturation_model
    detflat_pipe.bias.bias = bias_model
    detflat_pipe.nonlin_corr.nonlin = nonlin_model
    detflat_pipe.jump_detec.detgain = detgain_model
    detflat_pipe.jump_detec.rn = rn_model
    detflat_pipe.fit_det_gain.detgain_init = detgain_model
    detflat_pipe.fit_det_gain.rn_init = rn_model
    detflat_pipe.ramp_fit.detgain = detgain_model
    detflat_pipe.ramp_fit.rn = rn_model
    detflat_pipe.dark.dark = dark_model
    detflat_pipe.det_gain.detgain = detgain_model
    detflat_pipe.det_flat.detflat = detflat_model

    step_time = time.time();
    print(f"Step: defining pipeline, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # Run pipeline
    model_result = detflat_pipe.apply(ramp_model_sequence, prim_args=prim_args)
    # print("Flat")
    # print(model_result['detflat'].flat)
    # print("flat error")
    # print(model_result['detflat'].flat_err)

    assert model_result['detflat'].dq[0,0] & DQ_FLAGS['COLD']
    assert not (model_result['detflat'].dq[0,1] & DQ_FLAGS['COLD'])
    assert not (model_result['detflat'].dq[0,2] & DQ_FLAGS['COLD'])
    assert not (model_result['detflat'].dq[0,3] & DQ_FLAGS['COLD'])

    rel_residuals = np.abs(model_result['detflat'].flat - detflat_model.flat) / model_result['detflat'].flat_err
    print("Relative residuals")
    print(rel_residuals)
    assert np.sum(rel_residuals > 5) == 0

    # Check 1 sigma statistics of the flat field residuals
    confidence_level = np.sum(rel_residuals < 1) / np.size(rel_residuals)
    assert abs(confidence_level-0.683)<0.1
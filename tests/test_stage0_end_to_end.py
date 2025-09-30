from hispecdrp import Stage0Pipeline
from hispecdrp import datamodels
import numpy as np
import os
from hispecdrp.datamodels.dqflags import DQ_FLAGS
from hispecdrp.primitives.ramp_fitting.utils import get_mid_delta_times_ramp,get_ramp_chunks
import time

def test_stage0(tmp_path):
    start_time = time.time()

    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + '/KOA_CALIBRATION_CACHE/'

    # --- Define parameters for the simulated ramp ---
    max_cores = 2
    max_slope = 50  # e-/s
    readtime = 1.7  # seconds
    n_reads = 10
    ny, nx = 10, 10 # size of the detector

    step_time = time.time();
    print(f"Overheads, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Non-linearity and its inverse polynomial coefficients ---

    # Create an ideal ramp (1D array for demonstration)
    ideal_ramp = np.linspace(0, np.iinfo(np.uint16).max, 1001, endpoint=True)
    nonlin_coeffs = [-1e-6, 1, 0]  # Example coefficients for a cubic non-linearity
    # nonlin_coeffs = [ 1, 0]  # Example coefficients for a cubic non-linearity
    nonlinear_ramp = np.polyval(nonlin_coeffs, ideal_ramp)

    # Fit inverse polynomial (correction) coefficients
    nonlin_correction_coefs = np.polyfit(nonlinear_ramp, ideal_ramp, deg=5)
    print("Fitted correction coefficients:", nonlin_correction_coefs)

    # Apply correction and check accuracy
    nonlin_corrected_ramp = np.polyval(nonlin_correction_coefs, nonlinear_ramp)

    coeffs = np.broadcast_to(nonlin_correction_coefs[::-1, None, None], (np.size(nonlin_correction_coefs), ny, nx))
    thresh = np.zeros(shape=(ny, nx), dtype=np.uint16)
    max = np.zeros(shape=(ny, nx), dtype=np.uint16)
    nonlin_dq = np.zeros(shape=(ny,nx), dtype=np.uint32)
    nonlin_model = datamodels.NonlinearCorrectionModel(data=dict(coeffs=coeffs,
                                                                 nonlin_thresh=thresh,
                                                                 nonlin_max=max,
                                                                 dq=nonlin_dq))
    datamodels.set_default_meta(nonlin_model)

    step_time = time.time();
    print(f"Step: defining nonlin coefficients, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Create DQ init map ---
    dqinit_map = np.zeros((ny, nx), dtype=np.uint32)
    # set row to bad
    dqinit_map[5, :] |= DQ_FLAGS["DO_NOT_USE"]
    dqinit_model = datamodels.DQModel(data={"DQ": dqinit_map})
    datamodels.set_default_meta(dqinit_model)

    step_time = time.time();
    print(f"Step: defining DQ init calibration, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Create saturation threshold map ---
    saturation_threshold = np.iinfo(np.uint16).max  # DN
    saturation_map = np.full((ny, nx), saturation_threshold, dtype=np.float32)
    saturation_model = datamodels.SaturationModel(data={'SAT_THRESH': saturation_map,
                                                        "DQ": np.zeros_like(saturation_map, dtype=np.uint32), })
    datamodels.set_default_meta(saturation_model)

    step_time = time.time();
    print(f"Step: defining saturation calibration, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Create random gain map ---
    gain = 1.0  # e-/ADU
    gain_map = np.full((ny, nx), gain, dtype=np.float32)
    detgain_model = datamodels.DetectorGainModel(data={'GAIN': gain_map,
                                                       "GAIN_ERR": np.zeros_like(gain_map),
                                                       "GAIN_VAR_RNOISE": np.zeros_like(gain_map),
                                                       "GAIN_VAR_POISSON": np.zeros_like(gain_map),
                                                       "DQ": np.zeros_like(gain_map, dtype=np.uint32), })
    datamodels.set_default_meta(detgain_model)

    step_time = time.time();
    print(f"Step: defining gain calibration, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Create random dark map ---
    dark_current = 10  # e-/s
    dark_map = np.random.normal(loc=dark_current, scale=dark_current / 10, size=(ny, nx)).astype(np.float32)
    dark_model = datamodels.DarkModel(data={'RATE': dark_map / gain_map,  # dark in DN/s
                                            "RATE_ERR": np.zeros_like(dark_map),
                                            "RATE_VAR_RNOISE": np.zeros_like(dark_map),
                                            "RATE_VAR_POISSON": np.zeros_like(dark_map),
                                            "DQ": np.zeros_like(dark_map, dtype=np.uint32), })
    datamodels.set_default_meta(dark_model)

    step_time = time.time();
    print(f"Step: defining dark calibration, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Create bias map ---
    bias = 1000  # e-/s
    bias_map = np.full((ny, nx), bias, dtype=np.float32)
    bias_model = datamodels.BiasModel(data={'BIAS': bias_map / gain_map,  # bias in DN
                                            "BIAS_ERR": np.zeros_like(bias_map),
                                            "DQ": np.zeros_like(bias_map, dtype=np.uint32), })
    datamodels.set_default_meta(bias_model)

    step_time = time.time();
    print(f"Step: defining gain calibration, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Create random flat map ---
    detflat_map = np.random.normal(loc=1, scale=0.1, size=(ny, nx)).astype(np.float32)
    detflat_model = datamodels.DetectorFlatModel(data={'FLAT': detflat_map,
                                                       "FLAT_ERR": np.zeros_like(detflat_map),
                                                       "FLAT_VAR_RNOISE": np.zeros_like(detflat_map),
                                                       "FLAT_VAR_POISSON": np.zeros_like(detflat_map),
                                                       "DQ": np.zeros_like(detflat_map, dtype=np.uint32), })
    datamodels.set_default_meta(detflat_model)

    step_time = time.time();
    print(f"Step: defining flat calibration, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # --- Create random read noise map ---
    RN = 20.0  # e-
    rn_map = np.full((ny, nx), RN, dtype=np.float32)
    rn_model = datamodels.ReadNoiseModel(data={'RN': rn_map / gain_map,
                                               "RN_ERR": np.zeros_like(rn_map),
                                               "DQ": np.zeros_like(gain_map, dtype=np.uint32), })
    datamodels.set_default_meta(rn_model)

    # --- Create a fake UTR cube ---
    electron_rate_map = np.linspace(0, max_slope, nx, dtype=np.float32)[None, :] * np.ones((ny, 1), dtype=np.float32)
    start_time = time.time()
    ramp_data = datamodels.create_ramp(electron_rate_map,  # in e-/s
                                       readtime=readtime, n_reads=n_reads,
                                       nonlin_coeffs=nonlin_coeffs,
                                       gain=gain_map,  # in e-/s
                                       flat=detflat_map,
                                       dark=dark_map,  # in e-/s
                                       bias=bias_map,  # in e-/s
                                       kTC_noise=50,  # in e-/s
                                       poisson_noise=False, read_noise=0,  # in e-/s
                                       convert_to_uint16=False, clip_ramps=True,
                                       max_cores=max_cores)

    #adding a jump
    ramp_data['data'][4::,:,0:nx//2] += 1000

    step_time = time.time();
    print(f"Step: creating ramp (new), elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    ramp_model = datamodels.RampModel(data=dict(data=ramp_data['data'], dq_raw=ramp_data['dq_raw']),
                                      meta=ramp_data['meta'])
    datamodels.set_default_meta(ramp_model)
    ramp_model.meta.filename = "simulated_ramp.fits"
    print(ramp_model.meta.clock_rate)

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
            'run': False,
            # 'save_result':True,
            # # 'output_path': os.path.join(tmp_path,"bias_subtracted2.fits"),
            # 'output_dir': tmp_path,
            'max_cores': pipe_max_cores,
        },
        'nonlin_corr': {
            'suffix': 'nonlin',
            'max_cores': pipe_max_cores,
        },
        'jump_detec' : {
            'run': False,
            'suffix' : 'jumpdetec',
            'ouput_model' : 'input',
            'method': 'tpd',
            'max_cores': pipe_max_cores,
            'rejection_threshold': 4.0,
        },
        'ramp_fit' : {
            'suffix': 'rampfit',
            'method': 'jwst_ols', # 'ols', 'mcds', 'cds', 'jwst_ols', 'jwst_likely'
            'max_cores': pipe_max_cores,
            'jwst_jump_detection': True,
            'jwst_jump_rejection_threshold': 4.0,
        },
        'dark': {
            'suffix': 'dark',
            'max_cores': pipe_max_cores,
        },
        'det_gain': {
            'suffix': 'detgain',
            'max_cores': pipe_max_cores,
        },
        'det_flat': {
            'suffix': 'detflat',
            'max_cores': pipe_max_cores,
        },
    }

    # Setup and call the stage 0 pipeline
    stage0_pipe = Stage0Pipeline()

    # Set any primitive args

    stage0_pipe.dq_init.dq = dqinit_model
    stage0_pipe.saturation.saturation = saturation_model
    stage0_pipe.bias.bias = bias_model
    stage0_pipe.nonlin_corr.nonlin = nonlin_model
    stage0_pipe.ramp_fit.detgain = detgain_model
    stage0_pipe.ramp_fit.rn = rn_model
    stage0_pipe.dark.dark = dark_model
    stage0_pipe.det_gain.detgain = detgain_model
    stage0_pipe.det_flat.detflat = detflat_model

    step_time = time.time();
    print(f"Step: defining pipeline, elapsed: {step_time - start_time:.2f}s")
    start_time = time.time()

    # Run pipeline
    model_result = stage0_pipe.apply(ramp_model, prim_args=prim_args, output_dir=tmp_path)  #

    where_good = np.where(~(model_result.dq & np.full((ny, nx), DQ_FLAGS["DO_NOT_USE"])).astype(bool))
    np.testing.assert_allclose(model_result.rate[where_good], electron_rate_map[where_good], rtol=1e-6, atol=1)

from hispecdrp import Stage0Pipeline
from hispecdrp import datamodels
import numpy as np
import os
from hispecdrp.datamodels.dqflags import DQ_FLAGS


def test_stage0(tmp_path):

    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + '/KOA_CALIBRATION_CACHE/'

    max_slope = 50000  # e-/s
    readtime = 1.7  # seconds
    n_reads = 2
    ny, nx = 10, 10

    # --- Non-linearity and its inverse polynomial coefficients ---

    # Create an ideal ramp (1D array for demonstration)
    ideal_ramp = np.linspace(0, np.iinfo(np.uint16).max, 1001, endpoint=True)
    nonlin_coeffs = [-1e-6, 1, 0]  # Example coefficients for a cubic non-linearity
    # nonlin_coeffs = [ 1, 0]  # Example coefficients for a cubic non-linearity
    nonlinear_ramp = np.polyval(nonlin_coeffs, ideal_ramp)

    # Fit inverse polynomial (correction) coefficients
    nonlin_correction_coefs = np.polyfit(nonlinear_ramp, ideal_ramp, deg=5)

    # Apply correction and check accuracy
    nonlin_corrected_ramp = np.polyval(nonlin_correction_coefs, nonlinear_ramp)

    coeffs = np.broadcast_to(nonlin_correction_coefs[::-1, None, None], (np.size(nonlin_correction_coefs), ny, nx))
    thresh = np.zeros(shape=(ny, nx), dtype=np.uint16)
    max = np.zeros(shape=(ny, nx), dtype=np.uint16)
    nonlin_model = datamodels.NonlinearCorrectionModel(data=dict(coeffs=coeffs, nonlin_thresh=thresh, nonlin_max=max))
    datamodels.set_default_meta(nonlin_model)

    # --- Create DQ init map ---
    dqinit_map = np.zeros((ny, nx), dtype=np.uint32)
    # set row to bad
    dqinit_map[5, :] |= DQ_FLAGS["DO_NOT_USE"]
    dqinit_model = datamodels.DQModel(data={"DQ": dqinit_map})
    datamodels.set_default_meta(dqinit_model)

    # --- Create saturation threshold map ---
    saturation_threshold = np.iinfo(np.uint16).max  # DN
    saturation_map = np.full((ny, nx), saturation_threshold, dtype=np.float32)
    saturation_model = datamodels.SaturationModel(data={'SAT_THRESH': saturation_map,
                                                        "DQ": np.zeros_like(saturation_map, dtype=np.uint32), })
    datamodels.set_default_meta(saturation_model)

    # --- Create random gain map ---
    gain = 1.0  # e-/ADU
    gain_map = np.full((ny, nx), gain, dtype=np.float32)
    detgain_model = datamodels.DetectorGainModel(data={'GAIN': gain_map,
                                                       "GAIN_ERR": np.zeros_like(gain_map),
                                                       "GAIN_VAR_RNOISE": np.zeros_like(gain_map),
                                                       "GAIN_VAR_POISSON": np.zeros_like(gain_map),
                                                       "DQ": np.zeros_like(gain_map, dtype=np.uint32), })
    datamodels.set_default_meta(detgain_model)

    # --- Create random dark map ---
    dark_current = 10  # e-/s
    dark_map = np.random.normal(loc=dark_current, scale=dark_current / 10, size=(ny, nx)).astype(np.float32)
    # dark_map0 = np.zeros((ny,nx), dtype=np.float32)  # For testing, use zero dark
    dark_model = datamodels.DarkModel(data={'RATE': dark_map / gain_map,  # dark in DN/s
                                            "RATE_ERR": np.zeros_like(dark_map),
                                            "RATE_VAR_RNOISE": np.zeros_like(dark_map),
                                            "RATE_VAR_POISSON": np.zeros_like(dark_map),
                                            "DQ": np.zeros_like(dark_map, dtype=np.uint32), })
    datamodels.set_default_meta(dark_model)

    # --- Create bias map ---
    bias = 1000  # e-/s
    bias_map = np.full((ny, nx), bias, dtype=np.float32)
    bias_model = datamodels.BiasModel(data={'BIAS': bias_map / gain_map,  # bias in DN
                                            "BIAS_ERR": np.zeros_like(bias_map),
                                            "BIAS_VAR_RNOISE": np.zeros_like(bias_map),
                                            "BIAS_VAR_POISSON": np.zeros_like(bias_map),
                                            "DQ": np.zeros_like(bias_map, dtype=np.uint32), })
    datamodels.set_default_meta(bias_model)

    # --- Create random flat map ---
    detflat_map = np.random.normal(loc=1, scale=0.1, size=(ny, nx)).astype(np.float32)
    detflat_model = datamodels.DetectorFlatModel(data={'FLAT': detflat_map,
                                                       "FLAT_ERR": np.zeros_like(detflat_map),
                                                       "FLAT_VAR_RNOISE": np.zeros_like(detflat_map),
                                                       "FLAT_VAR_POISSON": np.zeros_like(detflat_map),
                                                       "DQ": np.zeros_like(detflat_map, dtype=np.uint32), })
    datamodels.set_default_meta(detflat_model)

    # --- Create a fake UTR cube ---
    electron_rate_map = np.linspace(0, max_slope, nx, dtype=np.float32)[None, :] * np.ones((ny, 1), dtype=np.float32)
    ramp_data = datamodels.create_ramp(electron_rate_map,  # in e-/s
                                       readtime=readtime, n_reads=n_reads,
                                       nonlin_coeffs=nonlin_coeffs,
                                       gain=gain_map,  # in e-/s
                                       flat=detflat_map,
                                       dark=dark_map,  # in e-/s
                                       bias=bias_map,  # in e-/s
                                       kTC_noise=50,  # in e-/s
                                       poisson_noise=False, read_noise=0,
                                       convert_to_uint16=False, clip_ramps=True)
    ramp_model = datamodels.RampModel(data=dict(data=ramp_data['data'], dq_raw=ramp_data['dq_raw']),
                                      meta=ramp_data['meta'])
    datamodels.set_default_meta(ramp_model)

    # --- Specify the primitive arguments here ---
    prim_args = {
        'dq_init': {
            'suffix': 'dqinit',
        },
        'saturation': {
            'suffix': 'checksat',
        },
        'bias': {
            'suffix': 'bias',
        },
        'nonlin_corr': {
            'suffix': 'nonlin',
        },
        'ramp_fit': {
            'suffix': 'rampfit',
            'method': 'cds',
            'num_coadd': 1,
            'max_cores': 1,
        },
        'dark': {
            'suffix': 'dark',
        },
        'det_gain': {
            'suffix': 'detgain',
        },
        'det_flat': {
            'suffix': 'detflat',
        },
    }

    # Setup and call the stage 0 pipeline
    stage0_pipe = Stage0Pipeline()

    # Set any primitive args

    stage0_pipe.dq_init.dq = dqinit_model
    stage0_pipe.saturation.saturation = saturation_model
    stage0_pipe.bias.bias = bias_model
    stage0_pipe.nonlin_corr.nonlin = nonlin_model
    stage0_pipe.dark.dark = dark_model
    stage0_pipe.det_gain.detgain = detgain_model
    stage0_pipe.det_flat.detflat = detflat_model

    # Run pipeline
    model_result = stage0_pipe.apply(ramp_model, prim_args=prim_args, output_dir=tmp_path)  #

    where_good = np.where(~(model_result.dq & np.full((ny, nx), DQ_FLAGS["DO_NOT_USE"])).astype(bool))
    np.testing.assert_allclose(model_result.rate[where_good], electron_rate_map[where_good], rtol=1e-6, atol=1)

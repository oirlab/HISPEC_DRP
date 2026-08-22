from hispecdrp import ProcessDetFlatsPipeline
from hispecdrp import datamodels
from hispecdrp.datamodels.dqflags import DQ_FLAGS
from .utils import make_stage0_cals, make_ramp_model
import numpy as np
import os

def test_stage0_flat_calib(tmp_path):
    np.random.seed(12345)

    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + 'KOA_CALIBRATION_CACHE'

    # --- Define parameters for the simulated ramp ---
    max_cores = 1
    max_slope = 200  # e-/s
    readtime = 1.7  # seconds
    n_reads = 30
    n_files = 10  # number of simulated files to create
    shape = (10, 10)  # size of the detector
    read_noise = 8.5
    gain = 3.0
    bias = 1000.0
    kTC_noise = 50.0
    dark_current = 1.0

    # --- Non-linearity and its inverse polynomial coefficients ---

    # Create an ideal ramp
    ideal_ramp = np.linspace(0, 1.5*np.iinfo(np.uint16).max, 1001, endpoint=True)

    # Polynomial coeffs such that read_out = P(read_true)
    nonlin_coeffs_inject = np.array([1, 0]) # c2, c1, c0
    ramp_nonlinear = np.polyval(nonlin_coeffs_inject, ideal_ramp)

    # Polynomial coeffs such that read_true = P(read_out)
    nonlin_coeffs_correct = np.polyfit(ramp_nonlinear, ideal_ramp, 5)

    cals = make_stage0_cals(
        shape=shape,
        nonlin_coeffs=nonlin_coeffs_correct,
        read_noise=read_noise,
        gain=gain,
        bias=bias,
        kTC_noise=kTC_noise,
        dark_current=dark_current,
        flat_level=1.0,
    )

    # --- Inject a few non-uniform pixels into the flat, incl. a cold pixel ---
    detflat_map = cals['detflat'].flat.copy()
    detflat_map[0, 0] = 0.7  # cold pixel
    detflat_map[0, 1] = 0.95
    detflat_map[0, 2] = 1.1
    detflat_map[0, 3] = 1.4
    cals['detflat'].flat = detflat_map

    # --- Create a fake UTR cube ---
    electron_rate_map = np.full(shape, max_slope, dtype=np.float32)
    ramp_model_sequence = []
    for k in range(n_files):

        ramp_model = make_ramp_model(
            source=electron_rate_map,
            gain=cals['gain'].gain,
            bias=cals['bias'].bias,
            kTC_noise=cals['bias'].ktc,
            nonlin_coeffs=nonlin_coeffs_inject,
            dark=cals['dark'].rate,
            flat=cals['detflat'].flat,
            n_reads=n_reads,
            readtime=readtime,
            poisson_noise=True,
            read_noise=cals['rn'].rn,
            convert_to_uint16=True,
            clip_ramps=True,
            max_cores=max_cores,
        )

        ramp_model.meta.filename = f"simulated_ramp_{k}.fits"

        ramp_model_sequence.append(ramp_model)

    # --- Specify the primitive arguments here ---
    primitives = {
        'dq_init': {
            'dq': cals['dq'],
            'max_cores': max_cores,
        },
        'saturation_check': {
            'saturation': cals['saturation'],
            'max_cores': max_cores,
        },
        'bias_sub': {
            'bias': cals['bias'],
            'max_cores': max_cores,
        },
        'nonlin_corr': {
            'nonlin': cals['nonlin'],
            'max_cores': max_cores,
        },
        'jump_detec': {
            'run': False,
        },
        'make_gain': {
            'run': False,
        },
        'ramp_fit': {
            'gain': cals['gain'],
            'rn': cals['rn'],
            'method': 'cython_likely', # 'ols', 'mcds', 'cds', 'jwst_ols', 'jwst_likely', 'cython_likely'
            'max_cores': max_cores,
            'jwst_jump_detection': False,
            'jwst_jump_rejection_threshold': 4.0,
        },
        'dark_sub': {
            'dark': cals['dark'],
            'max_cores': max_cores,
        },
        'gain_corr': {
            'gain': cals['gain'],
            'max_cores': max_cores,
        },
        'detflat_corr': {
            'run': False,
        },
    }

    # Setup and call the stage 0 pipeline
    model_result = ProcessDetFlatsPipeline.call(ramp_model_sequence, primitives=primitives)

    assert model_result['detflat'].dq[0, 0] & DQ_FLAGS['COLD']
    assert not (model_result['detflat'].dq[0, 1] & DQ_FLAGS['COLD'])
    assert not (model_result['detflat'].dq[0, 2] & DQ_FLAGS['COLD'])
    assert not (model_result['detflat'].dq[0, 3] & DQ_FLAGS['COLD'])

    # The measured flat is normalized by its own median, so no gain scaling
    # is needed to compare against the truth (unlike bias/dark/rn residuals).
    rel_residuals = np.abs(model_result['detflat'].flat - detflat_map) / model_result['detflat'].flat_err
    assert np.sum(rel_residuals > 5) == 0

    # Check 1 sigma statistics of the flat field residuals
    confidence_level = np.sum(rel_residuals < 1) / np.size(rel_residuals)
    assert abs(confidence_level-0.683)<0.1

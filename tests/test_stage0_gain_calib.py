from hispecdrp import ProcessDetFlatsPipeline
from .utils import make_stage0_cals, make_ramp_model
import numpy as np
import os

def test_stage0_gain_calib(tmp_path):
    np.random.seed(12345)

    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + 'KOA_CALIBRATION_CACHE'

    # --- Define parameters for the simulated ramp ---
    max_cores = 1
    max_slope = 200  # e-/s
    readtime = 1.7  # seconds
    n_reads = 30
    n_files = 100  # number of simulated files to create
    shape = (2, 2)  # size of the detector
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

    # --- Create a fake UTR cube ---
    electron_rate_map = np.full(shape, max_slope, dtype=np.float32)
    ramp_model_sequence = []
    for k in range(n_files):

        ramp_model = make_ramp_model(
            source=electron_rate_map,
            gain=cals['gain'].gain,
            bias=bias,
            kTC_noise=kTC_noise,
            nonlin_coeffs=nonlin_coeffs_inject,
            dark=dark_current,
            flat=cals['detflat'].flat,
            n_reads=n_reads,
            readtime=readtime,
            poisson_noise=True,
            read_noise=read_noise,
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
            'run': False,
        },
        'nonlin_corr': {
            'run': False,
        },
        'jump_detec': {
            'gain': cals['gain'],
            'rn': cals['rn'],
            'run': True,
            'method': 'tpd',
            'max_cores': max_cores,
            'rejection_threshold': 40.0,
        },
        'make_gain': {
            'gain_init': cals['gain'],
            'rn_init': cals['rn'],
            'max_cores': max_cores,
            'mode': 'per_pixel',  # 'full_detector', 'per_pixel'
            'method': 'quad',
            'fix_RN': True,
            'init_delta_gains': np.linspace(-0.5, 0.5, 100),  # Example delta gain values to try
            'init_delta_RNs': np.linspace(-1, 1, 100),
            'quad_delta_gain': 0.2,
            'quad_delta_RN': 0.5,
        },
        'ramp_fit': {
            'run': False,
        },
        'dark_sub': {
            'run': False,
        },
        'gain_corr': {
            'run': False,
        },
        'coadd': {
            'run': False,
        },
        'make_detflat': {
            'run': False,
        },
        'detflat_corr': {
            'run': False,
        },
    }

    # Setup and call the stage 0 pipeline
    model_result = ProcessDetFlatsPipeline.call(ramp_model_sequence, primitives=primitives)
    rel_residuals = np.abs(model_result['gain'].gain - cals['gain'].gain) / np.sqrt(
        model_result['gain'].cov[0, 0, :, :])
    assert np.sum(rel_residuals > 5) == 0

    primitives['make_gain']['fix_RN'] = False
    model_result = ProcessDetFlatsPipeline.call(ramp_model_sequence, primitives=primitives)
    rel_residuals = np.abs(model_result['gain'].gain - cals['gain'].gain) / np.sqrt(
        model_result['gain'].cov[0, 0, :, :])
    assert np.sum(rel_residuals > 5) == 0

    primitives['make_gain']['fix_RN'] = False
    primitives['make_gain']['mode'] = 'full_detector'
    primitives['make_gain']['method'] = 'quad'
    model_result = ProcessDetFlatsPipeline.call(ramp_model_sequence, primitives=primitives)
    rel_residuals = np.abs(model_result['gain'].gain - cals['gain'].gain) / np.sqrt(
        model_result['gain'].cov[0, 0])
    assert np.sum(rel_residuals > 5) == 0

    primitives['make_gain']['fix_RN'] = False
    primitives['make_gain']['mode'] = 'full_detector'
    primitives['make_gain']['method'] = 'grid'
    model_result = ProcessDetFlatsPipeline.call(ramp_model_sequence, primitives=primitives)
    rel_residuals = np.abs(model_result['gain'].gain - cals['gain'].gain) / np.sqrt(
        model_result['gain'].cov[0, 0])
    assert np.sum(rel_residuals > 5) == 0

    primitives['make_gain']['fix_RN'] = False
    primitives['make_gain']['mode'] = 'full_detector'
    primitives['make_gain']['method'] = 'both'
    model_result = ProcessDetFlatsPipeline.call(ramp_model_sequence, primitives=primitives)
    rel_residuals = np.abs(model_result['gain'].gain - cals['gain'].gain) / np.sqrt(
        model_result['gain'].cov[0, 0])
    assert np.sum(rel_residuals > 5) == 0

    primitives['make_gain']['fix_RN'] = False
    primitives['make_gain']['mode'] = 'per_pixel'
    primitives['make_gain']['method'] = 'both'
    primitives['make_gain']['max_cores'] = 2
    model_result = ProcessDetFlatsPipeline.call(ramp_model_sequence, primitives=primitives)
    rel_residuals = np.abs(model_result['gain'].gain - cals['gain'].gain) / np.sqrt(
        model_result['gain'].cov[0, 0, :, :])
    assert np.sum(rel_residuals > 5) == 0


def test_stage0_gain_calib_stats(tmp_path):
    np.random.seed(12345)

    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + 'KOA_CALIBRATION_CACHE'

    # --- Define parameters for the simulated ramp ---
    max_cores = 1
    max_slope = 200  # e-/s
    readtime = 1.7  # seconds
    n_reads = 30
    n_files = 100  # number of simulated files to create
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

    # --- Create a fake UTR cube ---
    electron_rate_map = np.full(shape, max_slope, dtype=np.float32)
    ramp_model_sequence = []
    for k in range(n_files):

        ramp_model = make_ramp_model(
            source=electron_rate_map,
            gain=cals['gain'].gain,
            bias=bias,
            kTC_noise=kTC_noise,
            nonlin_coeffs=nonlin_coeffs_inject,
            dark=dark_current,
            flat=cals['detflat'].flat,
            n_reads=n_reads,
            readtime=readtime,
            poisson_noise=True,
            read_noise=read_noise,
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
            'run': False,
        },
        'nonlin_corr': {
            'run': False,
        },
        'jump_detec': {
            'run': False,
        },
        'make_gain': {
            'gain_init': cals['gain'],
            'rn_init': cals['rn'],
            'max_cores': max_cores,
            'mode': 'per_pixel',  # 'full_detector', 'per_pixel'
            'method': 'both',
            'fix_RN': False,
            'init_delta_gains': np.linspace(-0.5, 0.5, 100),  # Example delta gain values to try
            'init_delta_RNs': np.linspace(-1, 1, 100),
            'quad_delta_gain': 0.2,
            'quad_delta_RN': 0.5,
        },
        'ramp_fit': {
            'run': False,
        },
        'dark_sub': {
            'run': False,
        },
        'gain_corr': {
            'run': False,
        },
        'coadd': {
            'run': False,
        },
        'make_detflat': {
            'run': False,
        },
        'detflat_corr': {
            'run': False,
        },
    }

    # Setup and call the stage 0 pipeline
    model_result = ProcessDetFlatsPipeline.call(ramp_model_sequence, primitives=primitives)
    rel_residuals = np.abs(model_result['gain'].gain - cals['gain'].gain) / np.sqrt(
        model_result['gain'].cov[0, 0, :, :])
    assert np.sum(rel_residuals > 5) == 0

    # Check 1 sigma statistics of the flat field residuals
    confidence_level = np.sum(rel_residuals < 1) / np.size(rel_residuals)
    print(confidence_level)
    assert abs(confidence_level-0.683)<0.1

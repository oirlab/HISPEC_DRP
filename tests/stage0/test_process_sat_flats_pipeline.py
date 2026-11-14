"""Tests for ``ProcessSatFlatsPipeline`` (saturation / non-linearity).
"""
from hispecdrp import ProcessSatFlatsPipeline
from tests.utils import make_ramp_model
from tests.sim_cals import make_stage0_cals
import numpy as np
import os

def test_process_sat_flats_pipeline(tmp_path):
    np.random.seed(12345)

    # --- Define parameters for the simulated ramp ---
    # max_cores=1: spawn pools dominate wall time on a 2x2 toy detector (~20s → ~1s).
    max_cores = 1
    max_slope = 1000  # e-/s
    readtime = 1.7  # seconds
    n_reads = 60  # must reach near saturation for sat_thresh asserts
    n_files = 4  # keep 60 reads for sat_thresh; fewer files for wall time
    shape = (2, 2)  # size of the detector
    read_noise = 0.1
    gain = 1.0
    bias = 1000.0
    kTC_noise = 0.0

    # --- Non-linearity and its inverse polynomial coefficients ---

    # Create an ideal ramp
    ideal_ramp = np.linspace(0, 1.5*np.iinfo(np.uint16).max, 1001, endpoint=True)
    nonlin_coeffs_inject = [-1e-6, 1, 0]  # Example coefficients for a cubic non-linearity
    ramp_nonlinear = np.polyval(nonlin_coeffs_inject, ideal_ramp)

    # Fit inverse polynomial (correction) coefficients
    nonlin_correction_coefs = np.polyfit(ramp_nonlinear, ideal_ramp, deg=5)
    print("Fitted correction coefficients:", nonlin_correction_coefs)

    # Use a pre-validated, more precise correction than the fit above for
    # comparison against the pipeline's own fitted correction below.
    nonlin_correction_coefs = np.array([0.00000000e+00, 9.93166438e-01, 9.93459922e-07, 1.99687217e-12, 4.28159139e-18, 2.47963102e-23])[::-1]

    # Apply correction and check accuracy
    nonlin_corrected_ramp = np.polyval(nonlin_correction_coefs, ramp_nonlinear)

    cals = make_stage0_cals(
        register=False,
        shape=shape,
        nonlin_coeffs=None,
        read_noise=read_noise,
        gain=gain,
        bias=bias,
        kTC_noise=kTC_noise,
        dark_current=None,
        flat_level=None,
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
            dark=0,
            n_reads=n_reads,
            readtime=readtime,
            poisson_noise=False,
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
        'make_saturation': {
            'max_cores': max_cores,
        },
        'saturation_check': {
            'saturation': cals['saturation'],
            'max_cores': max_cores,
        },
        'bias_sub': {
            'run': False,
        },
        'make_nonlin': {
            'gain': cals['gain'],
            'rn': cals['rn'],
            'saturation': cals['saturation'],
            'bias': cals['bias'],
            'mode': 'full_detector',
            'use_legendre': True,
            'order': 5,
            'readstart': 0,
            'satval_frac': 0.95,
            'criterion_for_nonlin_thresh': 0.01,
            'criterion_for_nonlin_max': 0.05,
            'max_cores': max_cores,
        },
        'nonlin_corr': {
            'run': False,
        },
        'jump_detec': {
            'run': False,
            'method': 'tpd',
            'max_cores': max_cores,
            'rejection_threshold': 4.0,
        },
        'ramp_fit': {
            'run': False,
            'method': 'cython_likely', # 'ols', 'mcds', 'cds', 'jwst_ols', 'jwst_likely', 'cython_likely'
            'max_cores': max_cores,
            'jwst_jump_detection': False,
            'jwst_jump_rejection_threshold': 4.0,
        },
        'dark_sub': {
            'run': False,
        },
        'gain_corr': {
            'run': False,
        },
        'detflat_corr': {
            'run': False,
        },
    }

    # Setup and call the stage 0 pipeline
    model_result = ProcessSatFlatsPipeline.call(ramp_model_sequence, primitives=primitives)

    np.testing.assert_allclose(model_result['saturation'].sat_thresh, 65435, rtol=1e-6, atol=1)
    assert model_result['saturation'].sat_thresh.shape[0] == shape[0]
    assert model_result['saturation'].sat_thresh.shape[1] == shape[1]

    nonlin_corrected_ramp_4testing = np.polyval(model_result['nonlin'].coeffs[::-1, 0], ramp_nonlinear)
    rel_err_ideal = (nonlin_corrected_ramp[1::] - ideal_ramp[1::]) / ideal_ramp[1::]
    rel_err_4testing = (nonlin_corrected_ramp_4testing[1::] - ideal_ramp[1::]) / ideal_ramp[1::]
    assert np.sum(np.abs(rel_err_ideal - rel_err_4testing) > 1e-2) == 0
    assert abs(model_result['nonlin'].nonlin_thresh[0]-18000) < 500
    assert abs(model_result['nonlin'].nonlin_max[0]-52600) < 500

    # Run pipeline again in per-pixel mode
    primitives['make_nonlin']['mode'] = 'per_pixel'
    model_result = ProcessSatFlatsPipeline.call(ramp_model_sequence, primitives=primitives)

    nonlin_corrected_ramp_4testing = np.polyval(model_result['nonlin'].coeffs[::-1, 0, 0], ramp_nonlinear)
    rel_err_ideal = (nonlin_corrected_ramp[1::] - ideal_ramp[1::]) / ideal_ramp[1::]
    rel_err_4testing = (nonlin_corrected_ramp_4testing[1::] - ideal_ramp[1::]) / ideal_ramp[1::]
    assert np.sum(np.abs(rel_err_ideal - rel_err_4testing) > 1e-2) == 0
    assert abs(model_result['nonlin'].nonlin_thresh[0, 0]-18000) < 500
    assert abs(model_result['nonlin'].nonlin_max[0, 0]-52600) < 500

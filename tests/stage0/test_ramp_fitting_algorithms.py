"""Tests for Stage0 ramp-fitting algorithms via ``Stage0Pipeline``.
"""
from hispecdrp import Stage0Pipeline
import numpy as np
import os
from hispecdrp.datamodels.dqflags import DQ_FLAGS
from hispecdrp.primitives.ramp_fitting.utils import get_mid_delta_times_ramp,get_ramp_chunks
from tests.utils import make_ramp_model, make_stage0_cals

def test_ramp_fitting_readnoise_dominated(tmp_path):
    np.random.seed(12345)

    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + '/KOA_CALIBRATION_CACHE/'

    # --- Define parameters for the simulated ramp ---
    max_slope = 0  # e-/s
    readtime = 1.7  # seconds
    n_reads = 30
    gain = 3.0
    dark_current = 0.0
    read_noise = 8.5
    bias = 1000.0
    kTC_noise = 50.0
    shape = (100, 100) # size of the detector
    # --- Non-linearity and its inverse polynomial coefficients ---

    # Create an ideal ramp
    ideal_ramp = np.linspace(0, 1.5*np.iinfo(np.uint16).max, 1001, endpoint=True)

    # Polynomial coeffs such that read_out = P(read_true)
    nonlin_coeffs_inject = np.array([-1e-6, 1, 0]) # c2, c1, c0
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

    methods = ['ols', 'mcds', 'cds', 'jwst_likely', 'cython_likely'] #'jwst_ols'
    for method in methods:
        # --- Create a fake UTR cube ---
        electron_rate_map = np.full(shape, max_slope, dtype=np.float32)

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
            poisson_noise=False,
            read_noise=read_noise,
            convert_to_uint16=False,
            clip_ramps=True,
        )

        #adding a jump
        # ramp_data['data'][4::,:,0:nx//2] += 1000

        # --- Specify the primitive arguments here ---
        primitives = {
            'dq_init': {
                'dq': cals['dq'],
            },
            'saturation_check': {
                'saturation': cals['saturation'],
            },
            'bias_sub': {
                'bias': cals['bias'],
            },
            'nonlin_corr': {
                'nonlin': cals['nonlin'],
            },
            'jump_detec' : {
                'run': False,
                'method': 'jwst',
                'rejection_threshold': 4.0,
            },
            'ramp_fit' : {
                # 'method': method, # 'ols', 'mcds', 'cds', 'jwst_ols', 'jwst_likely', 'cython_likely'
                'method': method,
                'jwst_jump_detection': False,
                'jwst_jump_rejection_threshold': 4.0,
                'gain': cals['gain'],
                'rn': cals['rn'],
            },
            'dark_sub': {
                'dark': cals['dark'],
            },
            'gain_corr': {
                'gain': cals['gain'],
            },
            'detflat_corr': {
                'detflat': cals['detflat'],
            },
        }

        # Setup and call the stage 0 pipeline
        model_result = Stage0Pipeline.call(
            dataset=ramp_model,
            primitives=primitives,
            output_dir=tmp_path
        )

        where_good = np.where(~(model_result.dq & np.full(shape, DQ_FLAGS["DO_NOT_USE"])).astype(bool))

        rel_residuals = np.abs(model_result.rate[where_good] - electron_rate_map[where_good]) / model_result.rate_err[where_good]

        assert np.sum(rel_residuals > 10) == 0

        # Check 1 sigma statistics
        confidence_level = np.sum(rel_residuals < 1) / np.size(rel_residuals)

        assert abs(confidence_level-0.683)<0.15



def test_ramp_fitting_photon_dominated(tmp_path):
    np.random.seed(12345)

    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + '/KOA_CALIBRATION_CACHE/'

    # --- Define parameters for the simulated ramp ---
    max_slope = 2000  # e-/s
    readtime = 1.7  # seconds
    n_reads = 30
    gain = 3.0
    dark_current = 0.0
    read_noise = 8.5
    bias = 1000.0
    kTC_noise = 50.0
    shape = (100, 100) # size of the detector
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

    methods = ['ols', 'mcds', 'cds', 'jwst_fixsen', 'jwst_likely', 'cython_likely']
    for method in methods:
        # --- Create a fake UTR cube ---
        electron_rate_map = np.full(shape, max_slope, dtype=np.float32)

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
            read_noise=read_noise,
            convert_to_uint16=False,
            clip_ramps=True,
        )

        #adding a jump
        # ramp_data['data'][4::,:,0:nx//2] += 1000

        # --- Specify the primitive arguments here ---
        primitives = {
            'dq_init': {
                'dq': cals['dq'],
            },
            'saturation_check': {
                'saturation': cals['saturation'],
            },
            'bias_sub': {
                'bias': cals['bias'],
            },
            'nonlin_corr': {
                'nonlin': cals['nonlin'],
            },
            'jump_detec' : {
                'run': False,
                'method': 'jwst',
                'rejection_threshold': 4.0,
            },
            'ramp_fit' : {
                # 'method': method, # 'ols', 'mcds', 'cds', 'jwst_ols', 'jwst_likely', 'cython_likely'
                'method': method,
                'jwst_jump_detection': False,
                'jwst_jump_rejection_threshold': 4.0,
                'gain': cals['gain'],
                'rn': cals['rn'],
            },
            'dark_sub': {
                'dark': cals['dark'],
            },
            'gain_corr': {
                'gain': cals['gain'],
            },
            'detflat_corr': {
                'detflat': cals['detflat'],
            },
        }

        # Setup and call the stage 0 pipeline
        model_result = Stage0Pipeline.call(
            dataset=ramp_model,
            primitives=primitives,
            output_dir=tmp_path
        )

        where_good = np.where(~(model_result.dq & np.full(shape, DQ_FLAGS["DO_NOT_USE"])).astype(bool))

        rel_residuals = np.abs(model_result.rate[where_good] - electron_rate_map[where_good]) / model_result.rate_err[where_good]

        assert np.sum(rel_residuals > 10) == 0

        # Check 1 sigma statistics
        confidence_level = np.sum(rel_residuals < 1) / np.size(rel_residuals)

        assert abs(confidence_level-0.683)<0.05
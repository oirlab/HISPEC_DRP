"""Tests for ``Stage0Pipeline`` end-to-end on synthetic ramps.
"""
from hispecdrp import Stage0Pipeline
import numpy as np
from hispecdrp.datamodels.dqflags import DQ_FLAGS

from tests.utils import make_stage0_cals, make_ramp_model

def test_stage0_pipeline(tmp_path):
    np.random.seed(12345)

    # --- Define parameters for the simulated ramp ---
    max_slope = 50  # e-/s
    readtime = 1.7  # seconds
    n_reads = 10
    read_noise = 20.0
    gain = 1.0
    bias = 1000.0
    dark_current = 10.0  # e-/s
    kTC_noise = 1.0
    shape = (10, 10) # size of the detector

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

    # Set one row to bad
    cals['dq'].dq[5, :] |= DQ_FLAGS["DO_NOT_USE"]

    # --- Create a fake UTR cube ---
    electron_rate_map = np.linspace(0, max_slope, shape[1], dtype=np.float32)[None, :] * np.ones((shape[0], 1), dtype=np.float32)

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
        read_noise=0,
        convert_to_uint16=False,
        clip_ramps=True,
    )

    #adding a jump
    ramp_model.data[4::, :, 0:shape[1]//2] += 1000
    ramp_model.meta.filename = "simulated_ramp.fits"

    # Setup and call the stage 0 pipeline
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
        'jump_detec': {
            'rn': cals['rn'],
            'gain': cals['gain'],
        },
        'ramp_fit': {
            'rn': cals['rn'],
            'gain': cals['gain'],
        },
        'dark_sub' : {
            'dark': cals['dark'],
        },
        'gain_corr' : {
            'gain': cals['gain'],
        },
        'detflat_corr' : {
            'detflat': cals['detflat'],
        }
    }
    model_result = Stage0Pipeline.call(ramp_model, primitives=primitives, output_dir=tmp_path)
    where_good = np.where(~(model_result.dq & np.full(shape, DQ_FLAGS["DO_NOT_USE"])).astype(bool))
    np.testing.assert_allclose(model_result.rate[where_good], electron_rate_map[where_good], rtol=1e-6, atol=1)
"""Tests for ``RefPixelCorrectionPrimitive``.
"""
import numpy as np
import os
import time

from hispecdrp import datamodels
import hispecdrp.simulations as simu
from hispecdrp.primitives.refpix.refpix_primitive import RefPixelCorrectionPrimitive
from hispecdrp.datamodels.dqflags import DQ_FLAGS

from tests.utils import make_ramp_model
from tests.sim_cals import make_stage0_cals

def test_refpix_primitive(tmp_path):
    np.random.seed(12345)
    start_time = time.time()

    # --- Define parameters for the simulated ramp ---
    max_cores = 1
    max_slope = 0  # e-/s
    readtime = 1.7  # seconds
    n_reads = 2
    read_noise = 8.5
    gain = 1.0
    kTC_noise = 0.0
    dark_current = 0.0
    bias = 1000.0
    shape = (2048, 2048) # size of the detector
    n_channels = 4  # number of readout channels

    # Create an ideal ramp
    ideal_ramp = np.linspace(0, 1.5*np.iinfo(np.uint16).max, 1001, endpoint=True)

    # Polynomial coeffs such that read_out = P(read_true)
    nonlin_coeffs_inject = np.array([1, 0]) # c2, c1, c0
    ramp_nonlinear = np.polyval(nonlin_coeffs_inject, ideal_ramp)

    # Polynomial coeffs such that read_true = P(read_out)
    nonlin_coeffs_correct = np.polyfit(ramp_nonlinear, ideal_ramp, 5)

    cals = make_stage0_cals(
        register=False,
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
        read_noise=None,
        convert_to_uint16=False,
        std_1overf=read_noise*10/gain, # DN
        clip_ramps=True,
        n_channels=n_channels,
        vertical=True,
    )

    #adding a jump
    # ramp_data['data'][4::,:,0:nx//2] += 1000
    
    ramp_model.meta.filename = "simulated_ramp.fits"

    model_result = RefPixelCorrectionPrimitive.call(
        ramp_model.copy(),
        method='scales_drp',
        max_cores=1,
        n_channels=n_channels,
        n_ref_left=4,
        n_ref_right=4,
        run=True,
    )

    for i in range(n_reads):
        assert np.nanstd(model_result.data[i,:,:]) < 0.9*np.nanstd(ramp_model.data[i,:,:])

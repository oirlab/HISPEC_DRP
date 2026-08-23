from hispecdrp import ProcessBiasPipeline
from hispecdrp import datamodels
from .utils import make_stage0_cals, make_ramp_model
import numpy as np
import os
from hispecdrp.primitives.bias_subtraction.master_bias_primitive import MakeMasterBiasPrimitive

def test_stage0_bias_calib(tmp_path):
    np.random.seed(12345)

    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + 'KOA_CALIBRATION_CACHE'

    # --- Define parameters for the simulated ramp ---
    max_cores = 1
    max_slope = 200  # e-/s
    readtime = 1.7  # seconds
    n_reads = 1
    n_files = 100  # number of simulated files to create
    shape = (10, 10)  # size of the detector
    read_noise = 8.5
    gain = 3.0
    bias = 1000.0
    kTC_noise = 50.0
    dark_current = 5.0

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
            bias=cals['bias'].bias,
            kTC_noise=cals['bias'].ktc,
            nonlin_coeffs=nonlin_coeffs_inject,
            dark=cals['dark'].rate,
            flat=cals['detflat'].flat,
            n_reads=n_reads,
            readtime=readtime,
            poisson_noise=False,
            read_noise=cals['rn'].rn,
            convert_to_uint16=False,
            clip_ramps=True,
        )

        # adding a jump
        # ramp_data['data'][4::,:,:] += 1000
        # ramp_data['data'][10::,:,:] += 1000
        # ramp_data['data'][15::,:,:] += 1000
        # ramp_data['data'][20::,:,:] += 1000
        # ramp_data['data'][4::,5,5] += 3000
        ramp_model.meta.filename = f"simulated_ramp_{k}.fits"

        ramp_model_sequence.append(ramp_model)

    # --- Specify the primitive arguments here ---
    primitives = {
        'dq_init': {
            'dq' : cals['dq'],
        },
    }

    # Setup and call the stage 0 pipeline
    model_result = ProcessBiasPipeline.call(ramp_model_sequence, primitives=primitives)

    # Residuals (bias - true bias)
    residuals_bias = np.abs(model_result['bias'].bias - cals['bias'].bias / cals['gain'].gain) # in DN
    assert np.sum(residuals_bias > 10) == 0

    # Residuals (kTC - true kTC)
    residuals_kTC = np.abs(model_result['bias'].ktc - cals['bias'].ktc / cals['gain'].gain) # in DN
    assert np.sum(residuals_kTC > 5) == 0


def test_stage0_master_bias(tmp_path):
    tmp_path = str(tmp_path)
    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + 'KOA_CALIBRATION_CACHE'

    # --- Create bias map ---
    shape = (10, 10)  # size of the detector
    gain = 1  # e-/ADU
    gain_map = np.full(shape, gain, dtype=np.float32)
    bias = 1000  # e-/s
    bias_map = np.full(shape, bias, dtype=np.float32)
    kTC_noise = 50  # e-/s
    kTC_noise_map = np.full(shape, kTC_noise, dtype=np.float32)
    bias_model1 = datamodels.BiasModel(
        bias=bias_map / gain_map,  # bias in DN
        ktc=kTC_noise_map / gain_map,  # kTC in DN
        dq=np.zeros_like(bias_map, dtype=np.uint32)
    )
    bias_model2 = bias_model1.copy()
    bias_model3 = bias_model1.copy()
    bias_model4 = bias_model1.copy()

    bias_list = [bias_model1, bias_model2, bias_model3, bias_model4]
    output_master_bias = MakeMasterBiasPrimitive.call(bias_list)

    assert output_master_bias.meta.master_cal
    np.testing.assert_array_equal(output_master_bias.bias, bias_map / gain_map)
    np.testing.assert_array_equal(output_master_bias.kTC, kTC_noise_map)  # error should reduce by sqrt(4)
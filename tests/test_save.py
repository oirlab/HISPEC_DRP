from hispecdrp import datamodels
from hispecdrp.simulations.utr import create_ramp
from hispecdrp.pipelines import Stage0Pipeline
from hispecdrp.primitives import DarkSubtractionPrimitive
import os
import copy
import numpy as np

from .utils import make_stage0_cals, make_ramp_model

def test_save_result_primitive(tmp_path, datadir):
    
    shape = (100, 100)

    os.environ['KOA_SKIP_CALIBRATION_VERSION'] = '1'
    
    # Create input science model
    input_model = datamodels.L0Model()
    input_model.rate = np.abs(np.random.randn(*shape) * 0.01 + 1).astype(np.float32)
    input_model.rate_err = np.full(shape, 0.01, dtype=np.float32)
    input_model.rate_var_poisson = np.abs(np.random.randn(*shape).astype(np.float32))
    input_model.rate_var_rnoise = np.abs(np.random.randn(*shape).astype(np.float32))
    input_model.dq = np.zeros(shape, dtype=np.uint32)
    
    # Create dark calibration model
    dark_model = datamodels.DarkModel()
    dark_model.rate = (np.random.randn(*shape) * 0.01 + 1).astype(np.float32)
    dark_model.rate_err = np.full(shape, 0.01, dtype=np.float32)
    dark_model.rate_var_poisson = np.abs(np.random.randn(*shape).astype(np.float32))
    dark_model.rate_var_rnoise = np.abs(np.random.randn(*shape).astype(np.float32))
    dark_model.dq = np.zeros(shape, dtype=np.uint32)
    dark_filepath = dark_model.save(output_dir=tmp_path)

    # Test passing output_dir
    suffix = 'my-test-file'
    result_model = DarkSubtractionPrimitive.call(
        input_model.copy(),
        dark=dark_filepath,
        save_result=True,
        output_dir=tmp_path,
        suffix=suffix
    )
    assert os.path.isfile(result_model._filepath)

    # Test that the suffix was applied
    assert result_model._filepath.endswith(f'{suffix}.fits')
    assert result_model.meta.filename.endswith(f'{suffix}.fits')

    # Test output_dir with in memory data model (model._filepath = model.meta.filename = None)
    input_model._filepath = None
    input_model.meta.filename = None
    result_model = DarkSubtractionPrimitive.call(
        input_model.copy(),
        dark=dark_filepath,
        save_result=True,
        output_dir=tmp_path,
        suffix=suffix
    )
    output_path = str(tmp_path / result_model.meta.filename)
    assert result_model._filepath == output_path
    assert os.path.isfile(output_path)

    # Test explicitly passing output_file and output_dir
    # Test passing output_dir
    output_file = 'myfile.fits'
    result_model = DarkSubtractionPrimitive.call(
        input_model.copy(),
        dark=dark_filepath,
        save_result=True,
        output_dir=tmp_path,
        output_file=output_file
    )
    output_path = str(tmp_path / output_file)
    assert os.path.isfile(result_model._filepath)
    assert result_model._filepath == output_path

    # Test that the file is NOT saved if save_result=False
    output_file = 'myfile2.fits'
    result_model = DarkSubtractionPrimitive.call(
        input_model.copy(),
        dark=dark_filepath,
        save_result=False,
        output_dir=tmp_path,
        output_file=output_file
    )
    output_path = str(tmp_path / output_file)
    assert result_model._filepath is None


def test_save_result_pipeline(tmp_path, datadir):
    
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
    cals['dq'].dq[5, :] |= datamodels.DQ_FLAGS["DO_NOT_USE"]

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

    # Test that pipeline saves automatically with nothing explicity specified.
    model_result = Stage0Pipeline.call(ramp_model, primitives=primitives)
    output_path = os.path.join(os.getcwd(), model_result.generate_filename())
    assert os.path.isfile(output_path)
    assert model_result._filepath == output_path

    # Test with explicit output_dir
    model_result = Stage0Pipeline.call(ramp_model, primitives=primitives, output_dir=tmp_path)
    output_path = os.path.join(tmp_path, model_result.generate_filename())
    assert os.path.isfile(output_path)
    assert model_result._filepath == output_path

    # Test that a primitive saves the result
    primitives2 = copy.deepcopy(primitives)
    primitives2['dark_sub']['save_result'] = True
    primitives2['dark_sub']['output_dir'] = tmp_path
    suffix = 'dark_subtracted_test'
    primitives2['dark_sub']['suffix'] = suffix
    
    model_result = Stage0Pipeline.call(ramp_model, primitives=primitives2)
    
    output_path_darksub = os.path.join(tmp_path, model_result.generate_filename(suffix=suffix))
    assert os.path.isfile(output_path_darksub)

    # Test that the pipeline's final output is saved with the correct name and path
    output_path_pipe = os.path.join(tmp_path, model_result.generate_filename())
    assert os.path.isfile(output_path_pipe)
    assert model_result._filepath == output_path_pipe
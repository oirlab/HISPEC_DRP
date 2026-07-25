from hispecdrp import datamodels
from hispecdrp.simulations.utr import create_ramp
from hispecdrp.pipelines import Stage0Pipeline
from hispecdrp.primitives import DarkSubtractionPrimitive
import os
import copy
import numpy as np

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
    prim = DarkSubtractionPrimitive()
    suffix = 'my-test-file'
    result_model = prim.apply(
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
    prim = DarkSubtractionPrimitive()
    result_model = prim.apply(
        input_model.copy(),
        dark=dark_filepath,
        save_result=True,
        output_dir=tmp_path,
    )
    output_path = str(tmp_path / result_model.meta.filename)
    assert result_model._filepath == output_path
    assert os.path.isfile(output_path)

    # Test explicitly passing output_file and output_dir
    # Test passing output_dir
    prim = DarkSubtractionPrimitive()
    output_file = 'myfile.fits'
    result_model = prim.apply(
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
    prim = DarkSubtractionPrimitive()
    output_file = 'myfile.fits'
    result_model = prim.apply(
        input_model.copy(),
        dark=dark_filepath,
        #save_result=True,
        output_dir=tmp_path,
        output_file=output_file
    )
    output_path = str(tmp_path / output_file)
    assert os.path.isfile(result_model._filepath)
    assert result_model._filepath == output_path


def test_save_result_pipeline(tmp_path, datadir):
    
    max_slope = 1000  # e-/s
    ny, nx = 100, 100

    # Create an ideal ramp (1D array for demonstration)
    ideal_ramp = np.linspace(0, 1.5*np.iinfo(np.uint16).max, 1001, endpoint=True)
    nonlin_coeffs = [-1e-6, 1, 0]
    nonlinear_ramp = np.polyval(nonlin_coeffs, ideal_ramp)
    nonlin_correction_coefs = np.polyfit(nonlinear_ramp, ideal_ramp, deg=5)

    # Nonlinear model
    coeffs = np.broadcast_to(nonlin_correction_coefs[::-1, None, None], (np.size(nonlin_correction_coefs), ny, nx))
    thresh = np.zeros(shape=(ny, nx), dtype=np.uint16)
    max = np.full(shape=(ny, nx), fill_value=0.9 * np.iinfo(np.uint16).max, dtype=np.uint16)
    nonlin_dq = np.zeros(shape=(ny,nx), dtype=np.uint32)
    nonlin_model = datamodels.NonlinearCorrectionModel(
        coeffs=coeffs,
        nonlin_thresh=thresh,
        nonlin_max=max,
        dq=nonlin_dq
    )
    
    # --- Create DQ init map ---
    dq_map = np.zeros((ny, nx), dtype=np.uint32)
    dqinit_model = datamodels.DQModel(dq=dq_map)

    # --- Create saturation threshold map ---
    saturation_threshold = np.iinfo(np.uint16).max  # DN
    saturation_map = np.full((ny, nx), saturation_threshold, dtype=np.float32)
    saturation_model = datamodels.SaturationModel(
        sat_thresh=saturation_map,
        dq=np.zeros_like(saturation_map, dtype=np.uint32)
    )

    # --- Create random gain map ---
    gain = 1.0  # e-/ADU
    gain_map = np.full((ny, nx), gain, dtype=np.float32)
    gain_model = datamodels.GainModel(
        gain=gain_map,
        rn=np.full_like(gain_map, 0.0),
        cov=np.full((2, 2, ny, nx), 0, dtype=np.float32),
        dq=np.zeros_like(gain_map, dtype=np.uint32)
    )

    # --- Create random dark map ---
    dark_current = 10  # e-/s
    dark_map = np.random.normal(loc=dark_current, scale=dark_current / 10, size=(ny, nx)).astype(np.float32)
    dark_model = datamodels.DarkModel(
        rate=dark_map / gain_map,  # dark in DN/s
        rate_err=np.zeros_like(dark_map),
        rate_var_rnoise=np.zeros_like(dark_map),
        rate_var_poisson=np.zeros_like(dark_map),
        dq=np.zeros_like(dark_map, dtype=np.uint32)
    )

    # --- Create bias map ---
    bias = 1000  # e-/s
    bias_map = np.full((ny, nx), bias, dtype=np.float32)
    bias_model = datamodels.BiasModel(
        bias=bias_map / gain_map,  # bias in DN
        ktc=np.zeros_like(bias_map),  # kTC in DN
        dq=np.zeros_like(bias_map, dtype=np.uint32)
    )

    # --- Create random flat map ---
    detflat_map = np.random.normal(loc=1, scale=0.1, size=(ny, nx)).astype(np.float32)
    detflat_model = datamodels.DetectorFlatModel(
        flat=detflat_map,
        flat_err=np.zeros_like(detflat_map),
        flat_var_rnoise=np.zeros_like(detflat_map),
        flat_var_poisson=np.zeros_like(detflat_map),
        dq=np.zeros_like(detflat_map, dtype=np.uint32)
    )

    # --- Create random read noise map ---
    RN = 20.0  # e-
    rn_map = np.full((ny, nx), RN, dtype=np.float32)
    rn_model = datamodels.ReadNoiseModel(
        rn=rn_map / gain_map,
        rn_err=np.zeros_like(rn_map),
        dq=np.zeros_like(gain_map, dtype=np.uint32)
    )

    # --- Create a fake UTR cube ---
    electron_rate_map = np.linspace(0, max_slope, nx, dtype=np.float32)[None, :] * np.ones((ny, 1), dtype=np.float32)
    readtime = 1.5
    n_reads = 10
    max_cores = 1
    ramp_data = create_ramp(
        electron_rate_map,  # in e-/s
        readtime=readtime, n_reads=n_reads,
        nonlin_coeffs=nonlin_coeffs,
        gain=gain_map,  # in e-/s
        flat=detflat_map,
        dark=dark_map,  # in e-/s
        bias=bias_map,  # in e-/s
        kTC_noise=50,  # in e-/s
        poisson_noise=False, read_noise=0,  # in e-/s
        convert_to_uint16=False, clip_ramps=True,
        max_cores=max_cores
    )
    ramp_model = datamodels.RampModel(
        data=ramp_data['data'],
        dq_raw=ramp_data['dq_raw'],
        meta=ramp_data['meta']
    )
    ramp_model.generate_filename()

    # --- Specify the primitive arguments here ---
    prim_args = {
        'dq_init': {
            'dq': dqinit_model,
        },
        'saturation_check': {
            'saturation': saturation_model,
        },
        'bias_sub': {
            'run': False,
            'bias': bias_model,
        },
        'nonlin_corr': {
            'nonlin': nonlin_model,
        },
        'jump_detec' : {
            'run': False,
            'method': 'tpd',
            'rejection_threshold': 4.0,
            'gain': gain_model,
            'rn': rn_model,
        },
        'ramp_fit' : {
            'method': 'jwst_likely', # 'ols', 'mcds', 'cds', 'jwst_ols', 'jwst_likely'
            'jwst_jump_detection': False,
            'jwst_jump_rejection_threshold': 4.0,
            'gain': gain_model,
            'rn': rn_model,
        },
        'dark_sub': {
            'dark': dark_model,
        },
        'gain_corr': {
            'gain': gain_model,
        },
        'detflat_corr': {
            'detflat': detflat_model,
        },
    }

    # Test that pipeline saves automatically with nothing explicity specified.
    pipe = Stage0Pipeline()
    model_result = pipe.apply(ramp_model, prim_args=prim_args)
    output_path = os.path.join(os.getcwd(), model_result.generate_filename())
    assert os.path.isfile(output_path)
    assert model_result._filepath == output_path

    # Test with explicit output_dir
    pipe = Stage0Pipeline()
    model_result = pipe.apply(ramp_model, prim_args=prim_args, output_dir=tmp_path)
    output_path = os.path.join(tmp_path, model_result.generate_filename())
    assert os.path.isfile(output_path)
    assert model_result._filepath == output_path

    # Test that a primitive saves the result
    prim_args2 = copy.deepcopy(prim_args)
    prim_args2['dark_sub']['save_result'] = True
    prim_args2['dark_sub']['output_dir'] = tmp_path
    suffix = 'dark_subtracted_test'
    prim_args2['dark_sub']['suffix'] = suffix
    
    pipe = Stage0Pipeline()
    model_result = pipe.apply(ramp_model, prim_args=prim_args2)
    
    output_path_darksub = os.path.join(tmp_path, model_result.generate_filename(suffix=suffix))
    assert os.path.isfile(output_path_darksub)

    # Test that the pipeline's final output is saved with the correct name and path
    output_path_pipe = os.path.join(tmp_path, model_result.generate_filename())
    assert os.path.isfile(output_path_pipe)
    assert model_result._filepath == output_path_pipe
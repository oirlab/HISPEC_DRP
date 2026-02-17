from hispecdrp import datamodels
from hispecdrp import DarkSubtractionPrimitive, ProcessDarksPipeline
import os
import numpy as np

def test_save_result_primitive(tmp_path, datadir):
    
    shape = (64, 128)

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

    # Test passing output_path
    output_path = str(tmp_path / 'my-test-file.fits')
    prim = DarkSubtractionPrimitive()
    result_model = prim.apply(
        input_model.copy(),
        dark=dark_filepath,
        save_result=True,
        output_path=output_path,
    )
    assert result_model._filepath == output_path
    assert result_model.meta.filename == os.path.basename(output_path)
    assert os.path.isfile(output_path)

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


def test_save_result_pipeline(tmp_path, datadir):
    
    #ny, nx = 4096, 4096 + 64
    ny, nx = 64, 128
    n_reads = 10
    N_files = 3
    
    # Create calibration models
    gain = 3.0
    gain_map = np.full((ny, nx), gain, dtype=np.float32)
    gain_model = datamodels.GainModel(data={
        'GAIN': gain_map,
        'RN': np.full_like(gain_map, 8.5),
        'COV': np.full((2, 2, ny, nx), 0, dtype=np.float32),
        'DQ': np.zeros_like(gain_map, dtype=np.uint32),
    })
    
    rn_map = np.full((ny, nx), 8.5, dtype=np.float32)
    rn_model = datamodels.ReadNoiseModel(data={
        'RN': rn_map / gain_map,
        'RN_ERR': np.ones_like(rn_map),
        'DQ': np.zeros_like(gain_map, dtype=np.uint32),
    })
    
    bias_map = np.full((ny, nx), 1000, dtype=np.float32)
    bias_model = datamodels.BiasModel(data={
        'BIAS': bias_map / gain_map,
        'kTC': np.full_like(bias_map, 50) / gain_map,
        'DQ': np.zeros_like(bias_map, dtype=np.uint32),
    })
    
    dqinit_map = np.zeros((ny, nx), dtype=np.uint32)
    dqinit_model = datamodels.DQModel(data={'DQ': dqinit_map})
    
    saturation_threshold = np.iinfo(np.uint16).max
    saturation_map = np.full((ny, nx), saturation_threshold, dtype=np.float32)
    saturation_model = datamodels.SaturationModel(data={
        'SAT_THRESH': saturation_map,
        'DQ': np.zeros_like(saturation_map, dtype=np.uint32),
    })
    
    # Create simulated ramp data
    ramp_model_sequence = []
    for k in range(N_files):
        ramp_data = np.random.randn(n_reads, ny, nx).astype(np.uint16) + 1000
        ramp_model = datamodels.RampModel(data=dict(
            data=ramp_data,
            dq_raw=np.zeros_like(ramp_data, dtype=np.uint32)
        ))
        ramp_model_sequence.append(ramp_model)
    
    # Configure pipeline
    prim_args = {
        'dq_init': {'run': True},
        'make_bias': {'run': False},
        'register_bias_cal': {'run': False},
        'saturation_check': {'run': True},
        'bias_sub': {'run': True},
        'nonlin_corr': {'run': False},
        'jump_detec': {'run': False},
        'make_rn': {'run': False},
        'register_rn_cal': {'run': False},
        'ramp_fit': {'run': True, 'method': 'ols'},
        'coadd': {'run': True, 'save_result': True, 'output_dir': tmp_path},
        'make_dark': {'run': True},
        'register_dark_cal': {'run': False},
        'dark_sub': {'run': False},
        'gain_corr': {'run': False},
        'detflat_corr': {'run': False},
    }
    
    # Test saving a particular primitive's result
    pipe = ProcessDarksPipeline()
    pipe.dq_init.dq = dqinit_model
    pipe.saturation_check.saturation = saturation_model
    pipe.bias_sub.bias = bias_model
    pipe.ramp_fit.gain = gain_model
    pipe.ramp_fit.rn = rn_model
    
    result = pipe.apply(ramp_model_sequence, prim_args=prim_args, save_result=True, output_dir=tmp_path)
    dark_model = result['dark']
    assert os.path.isfile(dark_model._filepath)
    
    # Test pipeline level saving with suffix
    prim_args['coadd']['save_result'] = False
    prim_args['make_dark']['save_result'] = True
    prim_args['make_dark']['output_dir'] = None  # Make sure the pipeline sets this
    suffix = 'saved-from-pipeline'
    
    pipe = ProcessDarksPipeline()
    pipe.dq_init.dq = dqinit_model
    pipe.saturation_check.saturation = saturation_model
    pipe.bias_sub.bias = bias_model
    pipe.ramp_fit.gain = gain_model
    pipe.ramp_fit.rn = rn_model
    
    result = pipe.apply(
        ramp_model_sequence,
        prim_args=prim_args,
        save_result=True,
        output_dir=tmp_path,
        suffix=suffix
    )
    dark_model = result['dark']
    assert dark_model._filepath == str(tmp_path / dark_model.generate_filename(suffix=suffix))
    assert os.path.isfile(dark_model._filepath)
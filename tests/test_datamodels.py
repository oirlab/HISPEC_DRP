from hispecdrp import datamodels
import numpy as np
from .utils import download_gdrive_file

def test_hispec_level0(tmp_path, datadir):
    
    # Science file (just tellurics)
    filepath = download_gdrive_file('HISPEC/L0/HB.20240924.0000.00_RATE-TELLURICS-FIB1.fits', use_cached=True, output_dir=datadir)

    def _test_model_attrs(model):
        assert model.meta.instrument.name == 'HISPEC'
        assert model.meta.exposure.exposure_type == "SCI"
        assert model.meta.instrument.spectrograph == 'BSPEC'
        assert model.meta.model_type == 'HISPECL0Model'
        assert model.meta.instrument.mode == 'YJHK'
        shape = (4096, 4096)
        assert model.data.shape == shape
        assert model.err.shape == shape
        assert model.var_poisson.shape == shape
        assert model.var_rnoise.shape == shape
        assert model.dq.shape == shape
        assert np.issubdtype(model.data.dtype, np.float32)
        assert np.issubdtype(model.err.dtype, np.float32)
        assert np.issubdtype(model.var_poisson.dtype, np.float32)
        assert np.issubdtype(model.var_rnoise.dtype, np.float32)
        assert np.issubdtype(model.dq.dtype, np.uint32)

    # Open directly as HISPECL0
    model = datamodels.HISPECL0Model(filepath)

    # Set arbitrary metadata
    model.meta.exposure.exposure_time = 10.0 # valid

    # Set invalid metadata
    try:
        model.meta.exposure.exposure_time = '10.0'  # invalid
        raise RuntimeError("Invalid metadata was set.")
    except ValueError:
        pass
    
    # Print model metadata, shown with pytest -s
    model.meta.pprint(show_none=True)

    # Save model
    model.save(output_dir=tmp_path)

    # Test model
    _test_model_attrs(model)

    # Test datamodels.open
    model = datamodels.open(filepath)
    _test_model_attrs(model)


def test_parvi_level0(tmp_path, datadir):
    
    # Science file (just tellurics)
    filepath = download_gdrive_file('PARVI/L0/PR.20250407.39453.59_rate-GamDra.fits', use_cached=True, output_dir=datadir)

    def _test_model_attrs(model):
        assert model.meta.instrument.name == 'PARVI'
        assert model.meta.exposure.exposure_type == "SCI"
        assert model.meta.model_type == 'PARVIL0Model'
        shape = (2048, 2048)
        assert model.data.shape == shape
        assert model.err.shape == shape
        assert model.var_poisson.shape == shape
        assert model.var_rnoise.shape == shape
        assert model.dq.shape == shape
        assert np.issubdtype(model.data.dtype, np.float32)
        assert np.issubdtype(model.err.dtype, np.float32)
        assert np.issubdtype(model.var_poisson.dtype, np.float32)
        assert np.issubdtype(model.var_rnoise.dtype, np.float32)
        assert np.issubdtype(model.dq.dtype, np.uint32)

    # Open directly as PARVIL0
    model = datamodels.PARVIL0Model(filepath)

    # Save model
    model.save(output_dir=tmp_path)

    # Test model
    _test_model_attrs(model)

    # Test datamodels.open
    model = datamodels.open(filepath)
    _test_model_attrs(model)
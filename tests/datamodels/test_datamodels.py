"""Tests for HISPEC / PARVI datamodel I/O.
"""
from hispecdrp import datamodels
import numpy as np
import os
from hispecdrp.utils import download_gdrive_file

def test_hispec_level0(tmp_path, datadir):
    
    # Science file (just tellurics)
    filepath = download_gdrive_file('HB.20231006.00000.00_rate-TELLURICS-FIB1.fits', output_dir=datadir)

    def _test_model_attrs(model):
        assert model.meta.instrument_name == 'HISPEC'
        assert model.meta.exposure_type == "SPEC"
        assert model.meta.spectrograph == 'BSPEC'
        assert model.meta.model_type == 'L0Model'
        assert model.meta.instrument_mode == 'YJHK'
        shape = (4096, 4096 + 64)
        assert model.rate.shape == shape
        assert model.rate_err.shape == shape
        assert model.rate_var_poisson.shape == shape
        assert model.rate_var_rnoise.shape == shape
        assert model.dq.shape == shape
        assert np.issubdtype(model.rate.dtype, np.float32)
        assert np.issubdtype(model.rate_err.dtype, np.float32)
        assert np.issubdtype(model.rate_var_poisson.dtype, np.float32)
        assert np.issubdtype(model.rate_var_rnoise.dtype, np.float32)
        assert np.issubdtype(model.dq.dtype, np.uint32)

    # Open directly as HISPECL0
    model = datamodels.L0Model(filepath)

    # Set arbitrary metadata
    model.meta.exposure_time = 10.0 # valid

    # Set invalid metadata
    try:
        model.meta.exposure_time = 'abc'
    except ValueError:
        pass
    else:
        raise RuntimeError("Invalid metadata was set.")
    
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

    import os
    os.environ['HISPECDRP_INSTRUMENT_NAME'] = 'PARVI'
    
    # Science file (just tellurics)
    filepath = download_gdrive_file('PR.20260131.80205.77_rate-SOLAR.fits', output_dir=datadir)

    def _test_model_attrs(model):
        assert model.meta.instrument_name == 'PARVI'
        assert model.meta.exposure_type == "SPEC"
        assert model.meta.model_type == 'L0Model'
        shape = (2048, 2048)
        assert model.rate.shape == shape
        assert model.rate_err.shape == shape
        assert model.rate_var_poisson.shape == shape
        assert model.rate_var_rnoise.shape == shape
        assert model.dq.shape == shape
        assert np.issubdtype(model.rate.dtype, np.float32)
        assert np.issubdtype(model.rate_err.dtype, np.float32)
        assert np.issubdtype(model.rate_var_poisson.dtype, np.float32)
        assert np.issubdtype(model.rate_var_rnoise.dtype, np.float32)
        assert np.issubdtype(model.dq.dtype, np.uint32)

    # Open directly as PARVIL0
    model = datamodels.L0Model(filepath)

    # Save model
    model.save(output_dir=tmp_path)

    # Test model
    _test_model_attrs(model)

    # Test datamodels.open
    model = datamodels.open(filepath)
    _test_model_attrs(model)

    del os.environ['HISPECDRP_INSTRUMENT_NAME']
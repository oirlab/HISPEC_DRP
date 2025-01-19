from hispecdrp import datamodels
import numpy as np
from tests.utils import download_github_lfs_file
import astropy.utils.data

def test_load_hispec_file(tmp_path):
    
    # Science file (just tellurics)
    git_file_path = 'HISPEC/L0/HS_BSPEC_20240924T000000_LVL0_RATE-TELLURICS-FIB1.fits'
    cache_dir = astropy.utils.data._get_download_cache_loc()
    file_path = download_github_lfs_file(git_file_path, output_dir=cache_dir, preserve_github_path=True, use_cached=True)

    def _test_model_attrs(model):
        assert model.meta.exposure.exposure_type == "SCI"
        assert model.meta.instrument.spectrograph == 'BSPEC'
        assert model.meta.model_type == 'HISPECL0'
        assert model.meta.instrument.mode == 'YJHK'
        shape = (4096, 4096)
        assert model.data.shape == shape
        assert model.err_all.shape == shape
        assert model.err_floor.shape == shape
        assert model.dq.shape == shape
        assert model.data.dtype == np.float32
        assert model.err_all.dtype == np.float32
        assert model.err_floor.dtype == np.float32
        assert model.dq.dtype == np.uint32

    # Open directly as HISPECL0
    model = datamodels.HISPECL0(file_path)

    # Set arbitrary metadata
    model.meta.exposure.integration_time = 10.0 # valid

    # Set invalid metadata
    try:
        model.meta.exposure.integration_time = '10.0' # invalid
    except ValueError:
        pass
    
    # Print model metadata, shown with pytest -s
    model.meta.pprint()

    # Save model
    model.save()

    # Test model
    _test_model_attrs(model)

    # Test datamodels.open
    model = datamodels.open(file_path)
    _test_model_attrs(model)

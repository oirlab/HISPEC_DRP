from hispecdrp import FlatFieldPrimitive
from tests.utils import download_github_lfs_file
import astropy.utils.data
import textwrap
from hispecdrp import datamodels
import numpy as np

def create_flatfield_config():
    conf = """
    %YAML 1.2
    ---

    logging:
      enable: True
      level: info
      verbose: True
      log_path: None
    
    arguments:
      save_results: True
      suffix: 'flat-sub'
    """
    return conf


def test_flatfield(tmp_path):
    
    # Science file (just tellurics)
    #git_file_path = 'HISPEC/L0/HS_BSPEC_20240924T000000_LVL0_RATE-TELLURICS-FIB1.fits'
    #cache_dir = astropy.utils.data._get_download_cache_loc()
    #file_path = download_github_lfs_file(git_file_path, output_dir=cache_dir, preserve_github_path=True, use_cached=True)
    file_path = '/Users/cale/Desktop/HS_BSPEC_20240924T000000_LVL0_RATE-TELLURICS-FIB1.fits'
    flat_path = '/Users/cale/Desktop/HS_BSPEC_20240924T000000_master-flat.fits'

    # Config file
    flat_config_file = str(tmp_path / 'flat_config.yaml')
    flat_config_raw = create_flatfield_config()
    with open(flat_config_file, 'w') as f:
        f.write(textwrap.dedent(flat_config_raw))

    # Setup and call the flat primitive
    flat_prim = FlatFieldPrimitive(config_file=flat_config_file, flat=flat_path)
    model_result = flat_prim.apply(file_path, output_dir=tmp_path)

    # Open models
    input_model = datamodels.open(file_path)
    flat_model = datamodels.HISPECFlatModel(flat_path)
    
    # Test against manual correction
    z = input_model.data / flat_model.data
    np.testing.assert_array_equal(
        model_result.data,
        z
    )
    np.testing.assert_array_equal(
        model_result.err_all,
        z * np.sqrt((input_model.err_all / input_model.data)**2 + (flat_model.err_all / flat_model.data)**2)
    )
    np.testing.assert_array_equal(
        model_result.err_floor,
        z * np.sqrt((input_model.err_floor / input_model.data)**2 + (flat_model.err_floor / flat_model.data)**2)
    )
    np.testing.assert_array_equal(
        model_result.dq,
        input_model.dq | flat_model.dq
    )
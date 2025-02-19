from hispecdrp import DarkSubtractionPrimitive
from tests.utils import download_github_lfs_file
import astropy.utils.data
import textwrap
from hispecdrp import datamodels
import numpy as np

def create_dark_config():
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
      suffix: 'dark-sub'
    """
    return conf


def test_dark_level0(tmp_path):
    
    # Science file (just tellurics)
    cache_dir = astropy.utils.data._get_download_cache_loc()
    git_file_path_sci = 'HISPEC/L0/HS_BSPEC_20240924T000000_LVL0_RATE-TELLURICS-FIB1.fits'
    git_file_path_dark = 'HISPEC/Cals/HS_BSPEC_20240924T000000_master-dark.fits'
    sci_path = download_github_lfs_file(git_file_path_sci, output_dir=cache_dir, preserve_github_path=True, use_cached=True)
    dark_path = download_github_lfs_file(git_file_path_dark, output_dir=cache_dir, preserve_github_path=True, use_cached=True)

    # Config file
    dark_config_file = str(tmp_path / 'dark_config.yaml')
    dark_config_raw = create_dark_config()
    with open(dark_config_file, 'w') as f:
        f.write(textwrap.dedent(dark_config_raw))

    # Setup and call the dark primitive
    dark_prim = DarkSubtractionPrimitive(config_file=dark_config_file, dark=dark_path)
    model_result = dark_prim.apply(sci_path, output_dir=tmp_path)

    # Open models
    input_model = datamodels.open(sci_path)
    dark_model = datamodels.HISPECDarkModel(dark_path)
    
    # Test against manual correction
    np.testing.assert_array_equal(model_result.data, input_model.data - dark_model.data)
    np.testing.assert_array_equal(model_result.err_all, np.sqrt(input_model.err_all**2 + dark_model.err_all**2))
    np.testing.assert_array_equal(model_result.err_floor, np.sqrt(input_model.err_floor**2 + dark_model.err_floor**2))
    np.testing.assert_array_equal(model_result.dq,  input_model.dq | dark_model.dq)


def test_dark_ramp(tmp_path):
    
    # Grab files
    cache_dir = astropy.utils.data._get_download_cache_loc()
    git_file_path_sci = 'HISPEC/Raw/HS_BSPEC_20240924T000000_RAW-DETFLAT.fits'
    git_file_path_dark = 'HISPEC/Cals/HS_BSPEC_20240924T000000_master-dark.fits'
    sci_path = download_github_lfs_file(git_file_path_sci, output_dir=cache_dir, preserve_github_path=True, use_cached=True)
    dark_path = download_github_lfs_file(git_file_path_dark, output_dir=cache_dir, preserve_github_path=True, use_cached=True)

    # Config file
    dark_config_file = str(tmp_path / 'dark_config.yaml')
    dark_config_raw = create_dark_config()
    with open(dark_config_file, 'w') as f:
        f.write(textwrap.dedent(dark_config_raw))

    # Setup and call the dark primitive
    dark_prim = DarkSubtractionPrimitive(config_file=dark_config_file, dark=dark_path)
    model_result = dark_prim.apply(sci_path, output_dir=tmp_path)

    # Manual subtraction
    input_model = datamodels.open(sci_path)
    dark_model = datamodels.HISPECDarkModel(dark_path)
    
    # Test
    np.testing.assert_array_equal(model_result.data, input_model.data - dark_model.data[:, :, None])
    
    # TODO: Sort this out. Currently ramp data has no error or DQ arrays.
    # np.testing.assert_array_equal(model_result.err_all, np.sqrt(input_model.err_all**2 + dark_model.err_all**2))
    # np.testing.assert_array_equal(model_result.err_floor, np.sqrt(input_model.err_floor**2 + dark_model.err_floor**2))
    # np.testing.assert_array_equal(model_result.dq,  input_model.dq | dark_model.dq)
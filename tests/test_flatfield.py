from hispecdrp import FlatFieldPrimitive
from tests.utils import download_github_lfs_file
import astropy.utils.data
import textwrap
from hispecdrp import datamodels
import numpy as np


def test_flatfield(tmp_path):
    
    # Grab files
    cache_dir = astropy.utils.data._get_download_cache_loc()
    git_file_path_sci = 'HISPEC/L0/HS_BSPEC_20240924T000000_LVL0_RATE-TELLURICS-FIB1.fits'
    git_file_path_dark = 'HISPEC/Cals/HS_BSPEC_20240924T000000_master-flat.fits'
    sci_path = download_github_lfs_file(git_file_path_sci, output_dir=cache_dir, preserve_github_path=True, use_cached=True)
    flat_path = download_github_lfs_file(git_file_path_dark, output_dir=cache_dir, preserve_github_path=True, use_cached=True)

    # Config file
    flat_config_file = str(tmp_path / 'flat_config.yaml')
    flat_config_raw = """
    %YAML 1.2
    ---

    logging:
      enable: True
      level: info
      verbose: True
      log_path: None
    
    arguments:
      save_results: False
      suffix: 'flat-sub'
    """
    with open(flat_config_file, 'w') as f:
        f.write(textwrap.dedent(flat_config_raw))

    # Setup and call the flat primitive
    flat_prim = FlatFieldPrimitive(config_file=flat_config_file, flat=flat_path)
    model_result = flat_prim.apply(sci_path, output_dir=tmp_path)

    # Open models
    input_model = datamodels.open(sci_path)
    flat_model = datamodels.HISPECFlatModel(flat_path)
    
    # Test against manual correction
    z = input_model.data / flat_model.data
    np.testing.assert_array_equal(
        model_result.data,
        z
    )
    np.testing.assert_array_equal(
        model_result.err,
        z * np.sqrt((input_model.err / input_model.data)**2 + (flat_model.err / flat_model.data)**2)
    )
    np.testing.assert_array_equal(
        model_result.dq,
        input_model.dq | flat_model.dq
    )
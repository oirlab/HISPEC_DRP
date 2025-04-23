from hispecdrp import Stage0Pipeline
from tests.utils import download_github_lfs_file
import textwrap
from hispecdrp import datamodels
import numpy as np

def make_nonlin_coeffs():
    nonlin_coeffs = np.array([0, 1], dtype=np.float32)
    coeffs = np.broadcast_to(nonlin_coeffs, (4096, 4096, len(nonlin_coeffs)))
    thresh = np.zeros(shape=(4096, 4096), dtype=np.uint16)
    max = np.zeros(shape=(4096, 4096), dtype=np.uint16)
    nonlin_model = datamodels.HISPECNonlinearCoeffModel(data=dict(coeffs=coeffs, thresh=thresh, max=max))
    return nonlin_model


def test_stage0(tmp_path):
    
    # Get science file
    git_file_path_sci = 'HISPEC/Raw/HS_BSPEC_20240924T000000_RAW-DETFLAT.fits'
    sci_path = download_github_lfs_file(git_file_path_sci, use_cached=True)

    # Create nonlin coeffs
    nonlin_model = make_nonlin_coeffs()

    # Config file
    config_raw = """
    %YAML 1.2
    ---

    logging:
      enable: True
      level: info
      verbose: True
      log_path: None
    
    arguments:
      save_results: False
      suffix: 'dark-sub'
    """
    config_file = str(tmp_path / 'stage0_config.yaml')
    with open(config_file, 'w') as f:
        f.write(textwrap.dedent(config_raw))

    # Setup and call the stage 0 pipeline
    stage0_pipe = Stage0Pipeline(config_file=config_file)

    # Set any primitive args
    stage0_pipe.nonlin_corr.args.coeffs = nonlin_model

    # Run pipeline
    result = stage0_pipe.apply(sci_path, output_dir=tmp_path)
    
    # Test against manual correction? (True for now)
    assert result is not None
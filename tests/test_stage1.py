from hispecdrp import Stage1Pipeline
from tests.utils import download_github_lfs_file
import astropy.utils.data
import textwrap

def create_dark_config():
    conf = """
    %YAML 1.2
    ---
    class: 'DarkSubtraction'

    logger:
      enable: True
      level: info
      verbose: True
      log_path: None
    
    arguments:
      skip: False
      save_results: True
      output_dir: None
      suffix: 'examplesuffix'
    """
    return conf

def create_stage1_config():
    conf = """
    %YAML 1.2
    ---
    class: 'Stage1'

    logger:
      enable: True
      level: info
      verbose: True
      log_path: None
    
    arguments:
      skip: False
      save_results: 'final'
      output_dir: None
      suffix: 'stage1-suffix'

    primitives:
      dark_sub:
        arguments:
          config_file: 'dark_config.yaml'    
    """
    return conf


def test_primitive(tmp_path):
    
    # Science file (just tellurics)
    git_file_path = 'HISPEC/L0/HS_BSPEC_20240924T000000_LVL0_RATE-TELLURICS-FIB1.fits'
    cache_dir = astropy.utils.data._get_download_cache_loc()
    file_path = download_github_lfs_file(git_file_path, output_dir=cache_dir, preserve_github_path=True, use_cached=True)

    # Config
    dark_config_file = str(tmp_path / 'dark_config.yaml')
    dark_config_raw = create_dark_config()
    with open(dark_config_file, 'w') as f:
        f.write(textwrap.dedent(dark_config_raw))
    stage1_config_file = str(tmp_path / 'stage1_config.yaml')
    stage1_config_raw = create_stage1_config()
    with open(stage1_config_file, 'w') as f:
        f.write(textwrap.dedent(stage1_config_raw))

    # Pipeline
    pipeline = Stage1Pipeline(output_dir=tmp_path, config=stage1_config_file)
    result = pipeline.apply(file_path, output_dir=tmp_path)
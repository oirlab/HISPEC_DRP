from hispecdrp import datamodels
import numpy as np
from tests.utils import download_github_lfs_file
import astropy.utils.data

def test_primitive(tmp_path):
    
    # Science file (just tellurics)
    git_file_path = 'HISPEC/L0/HS_BSPEC_20240924T000000_LVL0_RATE-TELLURICS-FIB1.fits'
    cache_dir = astropy.utils.data._get_download_cache_loc()
    file_path = download_github_lfs_file(git_file_path, output_dir=cache_dir, preserve_github_path=True, use_cached=True)

    # Open directly as HISPECL0
    model = datamodels.HISPECL0(file_path)

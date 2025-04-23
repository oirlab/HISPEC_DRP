from hispecdrp import DarkSubtractionPrimitive
from tests.utils import download_github_lfs_file
from hispecdrp import get_calibration_store
from hispecdrp import datamodels
import numpy as np
import os
from tests.utils import add_default_metadata


def test_local_db(tmp_path):
    
    # Science file (just tellurics)
    git_file_path_sci = 'HISPEC/L0/HS_BSPEC_20240924T000000_LVL0_RATE-TELLURICS-FIB1.fits'
    git_file_path_dark = 'HISPEC/Cals/HS_BSPEC_20240924T000000_master-dark.fits'
    sci_path = download_github_lfs_file(git_file_path_sci, use_cached=True)
    dark_path = download_github_lfs_file(git_file_path_dark, use_cached=True)

    # Open the science file to get the metadata
    sci_model = datamodels.open(sci_path, meta_only=True) # For speed

    # Store 5 files in the local cache
    N = 5
    cal_cache_dir = str(tmp_path) + '/'
    os.makedirs(cal_cache_dir + 'calibrations/', exist_ok=True)
    for i in range(N):
        dark_model = datamodels.open(dark_path, meta_only=False)
        dark_model.meta.exposure.jd_start += 365 * i
        add_default_metadata(dark_model)
        filename = dark_model.generate_filename()
        filepath = cal_cache_dir + 'calibrations/' + filename
        dark_model.meta.filename = filename
        dark_model.filepath = filepath
        dark_model.save()

    # Connect to DB
    store = get_calibration_store(cache_dir=cal_cache_dir)

    # Register the local files
    store.update_from_cache()

    # Test if the local files are registered
    results = store.local_db.query()
    assert len(results) == N, f"Expected {N} files in the local DB, but found {len(results)}"

    # Test retrieval of dark
    dark_prim = DarkSubtractionPrimitive()
    dark1 = dark_prim.get_calibration(sci_model, use_cached=True)
    assert dark1 is not None, "No dark model not found in local DB"

    # Test retrieval of second dark
    sci_model.meta.exposure.jd_start += 365
    dark_prim = DarkSubtractionPrimitive()
    dark2 = dark_prim.get_calibration(sci_model, use_cached=True)
    store.close()
    assert dark2 is not None, "No dark model not found in local DB"

    # Test that the two darks are different
    assert dark1 != dark2, "Dark models are the same"
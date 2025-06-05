from hispecdrp import FlatFieldPrimitive
from hispecdrp.utils import download_gdrive_file
from hispecdrp import datamodels
import numpy as np


def test_dark_level0(tmp_path):
    
    # Files
    sci_path = download_gdrive_file('HISPEC/L0/HB.20240924.0000.00_RATE-TELLURICS-FIB1.fits', use_cached=True)
    flat_path = download_gdrive_file('HISPEC/Cals/HB.20240924.0000.00.master-detflat.fits', use_cached=True)

    # Setup and call the dark primitive
    dark_prim = FlatFieldPrimitive()
    model_result = dark_prim.apply(sci_path, output_dir=tmp_path, flat=flat_path)

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
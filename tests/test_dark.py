from hispecdrp import DarkSubtractionPrimitive
from .utils import download_gdrive_file
from hispecdrp import datamodels
import numpy as np
import os


def test_dark_level0(tmp_path, datadir):

    # Files
    sci_path = download_gdrive_file('HISPEC/L0/HB.20240924.0000.00_RATE-TELLURICS-FIB1.fits', use_cached=True, output_dir=datadir)
    dark_path = download_gdrive_file('HISPEC/Cals/HB.20240924.0000.00.master-dark.fits', use_cached=True, output_dir=datadir)

    # Calibration cache
    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + '/KOA_CALIBRATION_CACHE/'

    # Setup and call the dark primitive
    dark_prim = DarkSubtractionPrimitive()
    model_result = dark_prim.apply(sci_path, output_dir=tmp_path, dark=dark_path)

    # Open models
    input_model = datamodels.open(sci_path)
    dark_model = datamodels.HISPECDarkModel(dark_path)
    
    # Test against manual correction
    np.testing.assert_array_equal(model_result.data, input_model.data - dark_model.data)
    np.testing.assert_array_equal(model_result.err, np.sqrt(input_model.err**2 + dark_model.err**2))
    np.testing.assert_array_equal(model_result.var_poisson, input_model.var_poisson + dark_model.var_poisson)
    np.testing.assert_array_equal(model_result.var_rnoise, input_model.var_rnoise + dark_model.var_rnoise)
    np.testing.assert_array_equal(model_result.dq,  input_model.dq | dark_model.dq)

    # Check the calibrations used
    index = np.where(model_result.calibrations['CAL_TYPE'] == 'dark')[0]
    assert len(index) == 1, "Expected one dark calibration to be used."
    index = index[0]
    assert model_result.calibrations[index]['FILENAME'] == os.path.basename(dark_path), "Dark calibration path does not match expected value."

    # Setup and call the dark primitive again to ensure the calibrations table works as expected.
    dark_prim = DarkSubtractionPrimitive()
    model_result = dark_prim.apply(input_model, output_dir=tmp_path, dark=dark_path)

    # Check the calibrations used
    dark_cal_row = model_result.get_calibration_from_role('dark')
    assert dark_cal_row['FILENAME'] == os.path.basename(dark_path), "Dark calibration path does not match expected value."
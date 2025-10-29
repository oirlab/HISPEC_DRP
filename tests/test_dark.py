from hispecdrp import DarkSubtractionPrimitive
from hispecdrp import datamodels
import numpy as np
import os

import matplotlib.pyplot as plt


def test_dark_level0(tmp_path, datadir):

    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + '/KOA_CALIBRATION_CACHE/'

    # Science
    input_model = datamodels.HISPECL0Model()
    input_model.rate = np.abs(np.random.randn(4096, 4096) * 0.01 + 1).astype(np.float32)
    input_model.rate_err = np.full((4096, 4096), 0.01, dtype=np.float32)
    input_model.rate_var_poisson = np.abs(np.random.randn(4096, 4096).astype(np.float32))
    input_model.rate_var_rnoise = np.abs(np.random.randn(4096, 4096).astype(np.float32))
    input_model.dq = np.zeros((4096, 4096), dtype=np.uint32)

    # Dark
    dark_model = datamodels.HISPECDarkModel()
    dark_model.rate = (np.random.randn(4096, 4096) * 0.01 + 1).astype(np.float32)
    dark_model.rate_err = np.full((4096, 4096), 0.01, dtype=np.float32)
    dark_model.rate_var_poisson = np.abs(np.random.randn(4096, 4096).astype(np.float32))
    dark_model.rate_var_rnoise = np.abs(np.random.randn(4096, 4096).astype(np.float32))
    dark_model.dq = np.zeros((4096, 4096), dtype=np.uint32)
    dark_model.generate_calibration_id()
    dark_filepath = dark_model.save(output_dir=tmp_path)

    # Setup and call the dark primitive
    dark_prim = DarkSubtractionPrimitive()
    model_result = dark_prim.apply(input_model.copy(), output_dir=tmp_path, dark=dark_filepath)
    
    # Test against manual correction
    np.testing.assert_array_equal(model_result.rate, input_model.rate - dark_model.rate)
    np.testing.assert_array_equal(model_result.rate_err, np.sqrt(input_model.rate_err**2 + dark_model.rate_err**2))
    np.testing.assert_array_equal(model_result.dq,  input_model.dq | dark_model.dq)

    # Check the calibrations used
    index = np.where(model_result.calibrations['CAL_TYPE'] == 'dark')[0]
    assert len(index) == 1, "Expected one dark calibration to be used."
    index = index[0]
    assert model_result.calibrations[index]['FILENAME'] == os.path.basename(dark_filepath), "Dark calibration path does not match expected value."

    # Setup and call the dark primitive again to ensure the calibrations table works as expected.
    dark_prim = DarkSubtractionPrimitive()
    model_result = dark_prim.apply(input_model.copy(), output_dir=tmp_path, dark=dark_filepath)

    # Check the calibrations used
    dark_cal_row = model_result.get_calibration_from_role('dark')
    assert dark_cal_row['FILENAME'] == os.path.basename(dark_filepath), "Dark calibration path does not match expected value."

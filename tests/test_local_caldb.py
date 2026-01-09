#from hispecdrp.utils import download_gdrive_file
from koa_middleware import CalibrationStore
from hispecdrp import datamodels
from hispecdrp import DarkSubtractionPrimitive, RegisterCalibrationPrimitive
from hispecdrp.calibrations import DarkSelector

import astropy.time
import numpy as np
import os
from datetime import datetime


def test_local_caldb(tmp_path, datadir):

    # Science
    input_model = datamodels.L0Model()
    input_model.rate = np.abs(np.random.randn(4096, 4096) * 0.01 + 1).astype(np.float32)
    input_model.rate_err = np.full((4096, 4096), 0.01, dtype=np.float32)
    input_model.rate_var_poisson = np.abs(np.random.randn(4096, 4096).astype(np.float32))
    input_model.rate_var_rnoise = np.abs(np.random.randn(4096, 4096).astype(np.float32))
    input_model.dq = np.zeros((4096, 4096), dtype=np.uint32)

    # Dark
    dark_model = datamodels.DarkModel()
    dark_model.rate = (np.random.randn(4096, 4096) * 0.01 + 1).astype(np.float32)
    dark_model.rate_err = np.full((4096, 4096), 0.01, dtype=np.float32)
    dark_model.rate_var_poisson = np.abs(np.random.randn(4096, 4096).astype(np.float32))
    dark_model.rate_var_rnoise = np.abs(np.random.randn(4096, 4096).astype(np.float32))
    dark_model.dq = np.zeros((4096, 4096), dtype=np.uint32)
    dark_model.generate_calibration_id()
    dark_filepath = dark_model.save(output_dir=tmp_path)

    # Open the science file to get the metadata
    #input_model = datamodels.open(sci_path, meta_only=True)

    # Calibration cache
    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + '/KOA_CALIBRATION_CACHE/'

    # Store 5 files in the local cache
    N = 5
    dark_models = []
    for i in range(N):
        dark_model = datamodels.open(dark_filepath)
        dark_model.meta.mjd_start += i * 365
        dark_model.meta.datetime_obs = astropy.time.Time(dark_model.meta.mjd_start, format='mjd').isot
        dark_model.meta.last_updated = datetime.now().isoformat(timespec="microseconds")
        dark_model.generate_calibration_id()
        prim = RegisterCalibrationPrimitive()
        prim.apply(dark_model)
        dark_models.append(dark_model)

    # Connect to DB
    with CalibrationStore(orm_class=dark_models[0].get_calibration_orm_class(), use_cached=True) as store:

        # Test if the local files are registered
        results = store.local_db.query()
        assert len(results) == N, f"Expected {N} files in the local DB, but found {len(results)}"

        # Test retrieval of dark
        dark1, _ = store.get_calibration(input_model.meta.model_dump(), DarkSelector())
        assert dark1 is not None, "No dark model not found in local DB"

        # Test retrieval of second dark
        input_model.meta.mjd_start += 365
        dark2, _ = store.get_calibration(input_model.meta.model_dump(), DarkSelector())
        assert dark2 is not None, "No dark model not found in local DB"

        # Test that the two darks are different
        assert dark1 != dark2, "Dark models are the same"

        # Test retrieval from DarkPrimitive
        dark_prim = DarkSubtractionPrimitive()
        dark3, _ = dark_prim.get_calibration(input_model, 'dark')
        assert dark3 is not None, "No dark model not found in local DB"

        # Check calibration retrieval by ID
        for dark_model in dark_models:
            dark_path, dark_orm = store.get_calibration_by_id(dark_model.meta.cal_id)
            dark_cal = datamodels.open(dark_path)
            assert os.path.basename(dark_path) == dark_cal.meta.filename, "Calibration filename mismatch"
            assert dark_orm.id == dark_model.meta.cal_id, "Calibration ID mismatch"
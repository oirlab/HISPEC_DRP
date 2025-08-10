from hispecdrp.utils import download_gdrive_file
from koa_middleware import CalibrationStore
from hispecdrp import datamodels
from hispecdrp import DarkSubtractionPrimitive, RegisterCalibrationPrimitive
from hispecdrp.calibrations import HISPECCalibrationORM
from hispecdrp.calibrations import DarkSelector

import os


def test_local_caldb(tmp_path):
    
    # Files
    sci_path = download_gdrive_file('HISPEC/L0/HB.20240924.0000.00_RATE-TELLURICS-FIB1.fits', use_cached=True)
    dark_path = download_gdrive_file('HISPEC/Cals/HB.20240924.0000.00.master-dark.fits', use_cached=True)

    #sci_path = '/Users/cale/Desktop/HISPEC_PARVI_Test_Data/HISPEC/L0/HB.20240924.0000.00_RATE-TELLURICS-FIB1.fits'
    #dark_path = '/Users/cale/Desktop/HISPEC_PARVI_Test_Data/HISPEC/Cals/HB.20240924.0000.00.master-dark.fits'

    # Open the science file to get the metadata
    sci_model = datamodels.open(sci_path, meta_only=True)

    # Store 5 files in the local cache
    N = 5
    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + '/KOA_CALIBRATION_CACHE/'
    #os.makedirs(os.environ['KOA_CALIBRATION_CACHE'] + 'calibrations/', exist_ok=True)
    dark_models = []
    for i in range(N):
        dark_model = datamodels.open(dark_path, meta_only=False)
        dark_model.meta.exposure.datetime_start = None
        dark_model.meta.exposure.mjd_start += 365 * i
        datamodels.set_hispec_meta(dark_model)
        prim = RegisterCalibrationPrimitive()
        prim.apply(dark_model)
        #filepath = os.environ['KOA_CALIBRATION_CACHE'] + 'calibrations/' + dark_model.generate_filename()
        #dark_model.save(filepath=filepath)
        dark_models.append(dark_model)

    # Connect to DB
    with CalibrationStore(orm_class=HISPECCalibrationORM, use_cached=True) as store:

        # Register the local files
        for dark_model in dark_models:
            store.register_local_calibration(dark_model)

        # Test if the local files are registered
        results = store.local_db.query()
        assert len(results) == N, f"Expected {N} files in the local DB, but found {len(results)}"

        # Test retrieval of dark
        dark1, dark1_orm = store.get_calibration(sci_model.meta.to_flat_dict(), DarkSelector())
        assert dark1 is not None, "No dark model not found in local DB"

        # Test retrieval of second dark
        sci_model.meta.exposure.mjd_start += 365
        dark2, dark2_orm = store.get_calibration(sci_model.meta.to_flat_dict(), DarkSelector())
        assert dark2 is not None, "No dark model not found in local DB"

        # Test that the two darks are different
        assert dark1 != dark2, "Dark models are the same"

        # Test retrieval from DarkPrimitive
        dark_prim = DarkSubtractionPrimitive()
        dark3 = dark_prim.get_calibration(sci_model, 'dark')
        assert dark3 is not None, "No dark model not found in local DB"

        darks = [dark1, dark2, dark3]
        dark_models = [datamodels.open(d) for d in darks]

        # Check calibration retrieval by ID
        for dark_model in dark_models:
            dark_path, dark_orm = store.get_calibration_by_id(dark_model.meta.cal_id)
            dark_cal = datamodels.open(dark_path)
            assert os.path.basename(dark_path) == dark_cal.meta.filename, "Calibration filename mismatch"
            assert dark_orm.id == dark_model.meta.cal_id, "Calibration ID mismatch"
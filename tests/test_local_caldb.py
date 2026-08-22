#from hispecdrp.utils import download_gdrive_file
from hispecdrp import datamodels
from hispecdrp import DarkSubtractionPrimitive, RegisterCalibrationPrimitive
from hispecdrp.calibrations import HISPECCalibrationStore, DarkSelector
from koa_middleware.utils import generate_md5_file

import astropy.time
import numpy as np
import os

# Fixed epoch for cal-db ordering tests: spread over ~4 years without entering
# ERFA "dubious year" territory (IERS leap-second tables are unreliable past ~2028).
_CALDB_TEST_BASE_MJD = 60200.0  # 2023-10-27 UTC


def test_local_caldb(tmp_path, datadir, monkeypatch):

    # Calibration cache
    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + '/KOA_CALIBRATION_CACHE/'

    # Store 5 files in the local cache
    N = 5
    dark_models = []
    mjds = _CALDB_TEST_BASE_MJD + np.arange(N) * 365
    for i in range(N):
        dark_model = datamodels.DarkModel(meta={'mjd_start': mjds[i]})
        dark_model = RegisterCalibrationPrimitive.call(dark_model)
        dark_models.append(dark_model)

    # Connect to DB
    with HISPECCalibrationStore(connect_remote=False) as store:

        # Test if the local files are registered
        results = store.query()
        assert len(results) == N, f"Expected {N} files in the local DB, but found {len(results)}"

        # Ensure we select the first dark
        mjd0 = _CALDB_TEST_BASE_MJD + 1
        input_model = datamodels.L0Model(meta={'mjd_start': mjd0})
        dark1_filepath, _ = store.select_and_get_calibration(
            input_model.meta,
            DarkSelector()
        )
        assert dark1_filepath == dark_models[0]._filepath, "Wrong dark model retrieved from local DB"

        # Test retrieval of second dark
        input_model = datamodels.L0Model(meta={'mjd_start': mjd0 + 365})
        dark2_filepath, _ = store.select_and_get_calibration(
            input_model.meta,
            DarkSelector()
        )
        assert dark2_filepath == dark_models[1]._filepath, "Wrong dark model retrieved from local DB"

        # Test retrieval of third dark
        input_model = datamodels.L0Model(meta={'mjd_start': mjd0 + 2 * 365})
        dark3_filepath, _ = store.select_and_get_calibration(
            input_model.meta,
            DarkSelector()
        )
        assert dark3_filepath == dark_models[2]._filepath, "Wrong dark model retrieved from local DB"
        # Test retrieval from DarkPrimitive (local cache only; no remote KOA sync).
        _store_init = HISPECCalibrationStore.__init__

        def _local_only_store_init(self, *args, **kwargs):
            kwargs["connect_remote"] = False
            _store_init(self, *args, **kwargs)

        monkeypatch.setattr(HISPECCalibrationStore, "__init__", _local_only_store_init)
        dark_prim = DarkSubtractionPrimitive(input_model)
        dark4_filepath, _ = dark_prim.get_calibration(dark_prim.dataset, 'dark')
        assert dark4_filepath == dark_models[2]._filepath, "Wrong dark model retrieved via DarkSubtractionPrimitive"

        # Check calibration retrieval by ID
        for dark_model in dark_models:
            dark_record = store.query(cal_id=dark_model.meta.cal_id)
            assert dark_record['id'] == dark_model.meta.cal_id, "Calibration ID mismatch"

        # Check MD5 checksum in two ways
        for dark_model in dark_models:
            dark_record = store.query(cal_id=dark_model.meta.cal_id)
            file_md5_from_record = dark_record['file_md5']
            assert file_md5_from_record is not None, "MD5 checksum is None in the DB record"
            assert file_md5_from_record == generate_md5_file(dark_model._filepath), "MD5 checksum mismatch between record and live file"


def test_sync_db_from_existing_files(tmp_path, datadir):

    # Calibration cache
    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + '/KOA_CALIBRATION_CACHE/'

    # Store 5 files in the local cache without registering them.
    N = 5
    dark_models = []
    mjds = _CALDB_TEST_BASE_MJD + np.arange(N) * 365
    for i in range(N):
        dark_model = datamodels.DarkModel(meta={'mjd_start': mjds[i], 'origin' : 'KECK'
        ''})
        dark_models.append(dark_model)

    # Init the store to create the directores and local DB.
    # The local DB file will be created, but this is irrelevant
    with HISPECCalibrationStore(connect_remote=False) as store:

        # Save to cache dir manually without registering in DB
        for dark_model in dark_models:
            dark_model.save(
                output_dir=store.data_dir
            )

    # Connect to DB
    with HISPECCalibrationStore(connect_remote=False) as store:

        # Ensure the local DB is empty
        results = store.query()
        assert len(results) == 0, f"Expected 0 files in the local DB, but found {len(results)}"

        # Sync from local files
        cals_added = store.sync_records_from_cached_files()

        # Test if the local files are registered
        results = store.query()
        assert len(results) == N, f"Expected {N} files in the local DB, but found {len(results)}"


def test_calibration_versioning_with_datamodels(tmp_path):
    N = 3
    dark_models = []

    os.environ['KOA_CALIBRATION_CACHE'] = str(tmp_path) + '/KOA_CALIBRATION_CACHE/'

    # Create N dark models
    mjd = astropy.time.Time.now().mjd
    for i in range(N):
        dark_model = datamodels.DarkModel(
            meta={
                'mjd_start': mjd,
                'master_cal': True
            }
        )
        dark_models.append(dark_model)

    # Initialize store
    with HISPECCalibrationStore(connect_remote=False) as store:

        # Register each dark and check version increases since same dark
        for idx, dark_model in enumerate(dark_models):
            version = store._get_next_calibration_version(dark_model, origin='LOCAL')
            print(f"Dark model {idx}: version {version}")
            expected_version = f"{idx+1:03d}"  # "001", "002", ...
            assert version == expected_version, f"Expected {expected_version}, got {version}"

            # Register to DB to increment versions
            _, _ = store.register_calibration(dark_model, origin='LOCAL', new_version=True)

        # Change family version MJD values and ensure version starts over at "001"
        new_dark = datamodels.DarkModel(
            meta={
                'mjd_start': mjd + 1,
                'master_cal': True
            }
        )
        next_version = store._get_next_calibration_version(new_dark, origin='LOCAL')
        assert next_version == "001", f"Expected version to reset to '001' for new family, but got {next_version}"

        # Change family version master_cal and ensure version starts over at "001"
        new_dark = datamodels.DarkModel(
            meta={
                'mjd_start': mjd,
                'master_cal': False
            }
        )
        next_version = store.generate_calibration_version(new_dark, origin='LOCAL')
        assert next_version == "001", f"Expected version to reset to '001' for new family, but got {next_version}"

        # Change family version spectrograph values and ensure version starts over at "001"
        new_dark = datamodels.DarkModel(
            meta={
                'mjd_start': mjd,
                'master_cal': True,
                'spectrograph': 'RSPEC'
            }
        )
        next_version = store.generate_calibration_version(new_dark, origin='LOCAL')
        assert next_version == "001", f"Expected version to reset to '001' for new family, but got {next_version}"
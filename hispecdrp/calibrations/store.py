import os
from .metadata_db import Calibration, LocalCalibrationDB, RemoteCalibrationDB
from .selectors import CalibrationSelector
import datetime

from .. import datamodels
from .koa import download_koa

import logging
logger = logging.getLogger("HISPECDRP")

__all__ = ['get_calibration_store', 'CalibrationStore']

from threading import Lock


# TODO: MAKE SURE THIS WORKS WITH MULTIPROCESSING
_cal_store_instance = None
_lock = Lock()
def get_calibration_store(
        database_url : str | None = None,
        calibration_url : str | None = None,
        cache_dir : str | None = None
    ) -> "CalibrationStore":
    global _cal_store_instance
    with _lock:  # Ensures only one thread initializes it
        if _cal_store_instance is None \
            or (database_url is not None and database_url != _cal_store_instance.database_url) \
            or (calibration_url is not None and calibration_url != _cal_store_instance.calibration_url) \
            or (cache_dir is not None and cache_dir != _cal_store_instance.cache_dir):
            _cal_store_instance = CalibrationStore(database_url=database_url, calibration_url=calibration_url, cache_dir=cache_dir)
    return _cal_store_instance


class CalibrationStore:
    """
    A CalibrationStore is used to manage calibrations, including storing, retrieving, and downloading.

    Attributes:
    - database_url: URL for the database connection.
    - calibration_url: URL for the calibration storage.
    - cache_dir: Local directory for caching downloaded calibrations.
    """

    _DEFAULT_DATABASE_URL = None
    _DEFAULT_CALIBRATION_URL = "https://koa.ipac.caltech.edu/cgi-bin/getKOA/nph-getKOA?return_mode=json&"

    def __init__(
        self,
        database_url : str | None = None,
        calibration_url : str | None = None,
        cache_dir : str | None = None
    ):

        # POSTGRES DB URL (KOA)
        self.database_url = database_url if database_url is not None else os.environ.get('HISPEC_CALIBRATION_DATABASE_URL', self._DEFAULT_DATABASE_URL)

        # URL where actual calibrations (FITS files) are stored
        self.calibration_url = calibration_url if calibration_url is not None else os.environ.get('HISPEC_CALIBRATION_URL', self._DEFAULT_CALIBRATION_URL)

        # Init the local cache
        self.init_cache(cache_dir)

        # Initialize the database
        self.init_remote_db()


    def init_cache(self, cache_dir : str | None = None):
        self.cache_dir = cache_dir if cache_dir is not None else os.environ['HISPEC_CALIBRATION_CACHE']
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, 'calibrations'), exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, 'database'), exist_ok=True)
        self.local_db = LocalCalibrationDB(db_path=os.path.join(self.cache_dir, 'database', 'hispec_calibrations.db'))

    
    def init_remote_db(self):
        if self.database_url is not None:
            self.remote_db = RemoteCalibrationDB(url=self.database_url)
        else:
            self.remote_db = None


    def _get_calibration(self, calibration : Calibration, use_cached : bool = True) -> str:
        filepath_local = self.calibration_in_cache(calibration)
        if filepath_local is not None and use_cached:
            return filepath_local
        else:
            return self.download_calibration(calibration)
    

    def get_calibration(
        self,
        input : str | datamodels.HISPECDataModel,
        selector : CalibrationSelector,
        use_cached : bool = True,
        **kwargs
    ) -> str:
        """
        Select the best calibration(s) based on the input data and a selection rule, download if not already cached.
        
        Args:
            input: Input data product.
            **kwargs: Additional filtering parameters.
        
        Returns:
            Selected calibration file(s).
        """
        # NOTE: Change how this works if result is not a string, or implement in _get_calibration
        input = datamodels.open(input, copy=False, meta_only=True)
        result = selector.select(input, self.local_db, **kwargs)
        return self._get_calibration(result, use_cached=use_cached)

    
    def download_calibration(self, calibration : Calibration):
       # NOTE: Implement this once we are set up at KOA.
       raise NotImplementedError("Download calibration not implemented yet.")
    

    def calibration_in_cache(self, calibration : Calibration) -> str | None:
        filepath_local = self.get_local_filename(calibration)
        if os.path.exists(filepath_local):
            return filepath_local
        else:
            return None
    
    
    def get_local_filename(self, calibration : Calibration) -> str:
        return os.path.join(self.cache_dir, 'calibrations', calibration.filename)
    

    def close(self):
        """
        Close both databases.
        """
        if self.remote_db is not None:
            self.remote_db.close()
        if self.local_db is not None:
            self.local_db.close()
    

    def get_deltas(
        self,
        date_time_start : datetime.datetime = datetime.datetime.min,
        date_time_end : datetime.datetime = datetime.datetime.max,
        diff_mode : str = "remote-local"
    ) -> list[Calibration]:
        """
        Get the filenames of all calibrations that were added or updated in the remote DB between start_time and end_time.
        """
        if diff_mode.lower() == "remote-local":
            remote_calibrations = self.remote_db.query(date_time_start=date_time_start, date_time_end=date_time_end)
            missing = self.local_db.get_deltas(remote_calibrations)
        elif diff_mode.lower() == "local-remote":
            local_calibrations = self.local_db.query(date_time_start=date_time_start, date_time_end=date_time_end)
            missing = self.remote_db.get_deltas(local_calibrations)
        else:
            raise ValueError(f"Unknown diff mode: {diff_mode}")
        return missing

    
    def register_local_database(self, calibration : datamodels.HISPECCalibrationModel | str, note : str = None):
        """
        Register a calibration that is now stored in the appropriate calibrations directory and add to the local SQLLite DB.
        """
        return self.local_db.add(calibration, note=note)
    

    def sync(self, direction : str = 'remote->local') -> list[Calibration]:
        """
        Sync the local SQLite DB with the remote PostgreSQL DB.
        """
        if direction.lower() == 'remote->local':
            calibrations = self.get_deltas(diff_mode='remote-local')
            calibrations = [cal for cal in calibrations if cal is not None]
            if calibrations:
                self.local_db.add(calibrations)
        elif direction.lower() == 'local->remote':
            calibrations = self.get_deltas(diff_mode='local-remote')
            calibrations = [cal for cal in calibrations if cal is not None]
            if calibrations:
                self.remote_db.add(calibrations)
        else:
            raise ValueError(f"Unknown sync direction: {direction}")
        return calibrations
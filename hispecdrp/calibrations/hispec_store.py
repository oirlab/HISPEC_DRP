from __future__ import annotations
from pyexpat import model
from koa_middleware import CalibrationStore
import glob
import os
from .. import datamodels
from koa_middleware.utils import generate_md5_file, is_valid_uuid

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from datamodels.base_model import DataModel, CalibrationModel

import logging
logger = logging.getLogger(__name__)

__all__ = ['HISPECCalibrationStore']

class HISPECCalibrationStore(CalibrationStore):

    def __init__(
        self, *args,
        instrument_name : str | None = None,
        **kwargs
    ):
        if instrument_name is None:
            instrument_name = os.getenv(
                'HISPECDRP_INSTRUMENT_NAME',
                datamodels.DataModel._DEFAULT_INSTRUMENT
            )
        super().__init__(
            *args,
            instrument_name=instrument_name,
            **kwargs
        )

    def sync_local_db_from_local_files(self, origin : str | None = None) -> list[dict]:
        """
        Scans the local cache directory for calibration files and populates
        the local database with their metadata if they are missing.

        This is a utility function to assist during development.

        **May change in future versions**

        Parameters
        ----------
        origin : str, optional
            The origin to use for the calibrations being synced. If None, defaults to the store's origin.

        Returns
        -------
        list[dict]
            A list of calibration metadata dictionaries that were added to the local database.
        """
        if origin is None:
            origin = self.origin
        cals_added = []
        local_files = glob.glob(os.path.join(self.data_dir, '*.fits'))
        for filepath in local_files:
            model = datamodels.open(filepath, meta_only=True)
            cal_id = model.meta.cal_id
            if not self.calibration_record_in_cache(cal_id, mode='id'):
                _, cal_record_added = self.register_calibration(model, origin=origin)
                if cal_record_added is not None:
                    cals_added.append(cal_record_added)
                    logger.info(f"Added calibration to local DB: {cal_record_added['filename']}")

        return cals_added

    def generate_calibration_version(
        self,
        cal : DataModel | str,
        origin : str | None = None
    ) -> str:
        """
        Generate the next calibration version ("001", "002", ...), scoped to
        the calibration's version family and origin.

        Parameters
        ----------
        cal : dict | DataModel | str
            The calibration for which to generate the version.
            Metadata records, datamodel, or filepath.
        origin : str | None, optional
            The origin to use for generating the version. If None, the origin
            from the calibration metadata will be used.

        Returns
        -------
        str
            The calibration version string
        """

        model = datamodels.open(cal)
        record = model.to_record()
        return super().generate_calibration_version(record, origin=origin)
    
    def register_calibration(
        self,
        cal : str | DataModel,
        origin : str | None = None,
        new_version : bool = False
    ) -> tuple[str, dict]:
        """
        Registers a calibration to the local cache and metadata database.

        Parameters
        ----------
        calibration : str | DataModel
            A filepath or datamodel.
        origin : str, optional
            The origin to register the calibration under.

        Returns
        -------
        tuple[str, dict]
            A tuple containing:
                - `str`: The local file path where the calibration was saved.
                - `dict`: The calibration metadata dictionary as added to the database.

        TODO: Consider refactoring. This is largely redundant with the parent class method. It sets additional metadata fields before converting to a record, but needs to perform the same prelimiary checks as the parent method before converting.
        """

        # Open model
        model = datamodels.open(cal)

        if self.calibration_record_in_cache(model, mode='id'):
            msg = f"Calibration already exists in cache: {cal}. Skipping registration."
            logger.warning(msg)
            return None, None
        
        if not new_version and self.calibration_record_in_cache(cal, mode='version-family'):
            msg = f"Calibration already exists in cache: {cal}! Skipping registration."
            logger.warning(msg)
            return None, None

        # Add origin to metadata
        if origin is not None:
            model.meta.origin = origin.upper()
        if model.meta.origin is None:
            model.meta.origin = self.origin
        assert model.meta.origin is not None, "origin must be specified either in the input calibration metadata or as an argument to this function."

        # Generate version and add to metadata
        model.meta.cal_version = self.generate_calibration_version(model)

        # Ensure filename is up to date with version and other metadata
        model.generate_filename()

        # Convert to record
        cal_record = model.to_record()
        
        # Save to local cache
        local_filepath = model.save(output_dir=self.data_dir)

        # Generate MD5 and add to record
        cal_record['file_md5'] = generate_md5_file(local_filepath)

        # Add new record to local DB
        cal_record_added = self.local_db.add(cal_record)

        return local_filepath, cal_record_added

    def calibration_record_in_cache(
        self,
        cal: dict | str | 'CalibrationModel',
        mode: str = 'id'
    ) -> dict | None:
        """
        Checks if a calibration is already present in the local cache.

        Parameters
        ----------
        calibration : dict | str | DataModel
            Can be one of:
                - `str` : A calibration ID string or filepath.
                - `dict` : A calibration metadata dict.
                - `DataModel` : A calibration data model instance.

        mode : str
            The mode to check the cache. Can be one of:
                - 'id' : Check by calibration ID (cal_id), the primary key in the database.
                - 'version-family' : Check by the version family (cal_type, datetime_obs, master_cal, spectrograph) + version (cal_version).
                - 'md5' : Check by the MD5 checksum of the calibration file (*Not Implemented yet!*).

        Returns
        -------
        dict | None
            The calibration metadata record if found, otherwise None.
        """
        if isinstance(cal, str) and os.path.isfile(cal):
            model = datamodels.open(cal, meta_only=True)
            return super().calibration_record_in_cache(model, mode=mode)
        
        return super().calibration_record_in_cache(cal, mode=mode)
    
    def get_version_family_column_names(self, cal_type : str):
        """
        Retrieves the column names for the version family attributes.
        This takes in cal_type in case we want to have different version
        family fields for different calibration types in the future.

        Parameters
        ----------
        cal_type : str
            The type of calibration (e.g. 'dark', 'flat', etc.) to get the version family fields for.
        """
        return ['cal_type', 'datetime_obs', 'master_cal', 'spectrograph']

    def _convert_origin(self, cal : str | 'CalibrationModel', origin : str) -> str:
        """
        Convert an existing calibration file's origin (record and file) to a new origin.
        This creates a new file and record.
        This does not modify the input file or record.

        TODO: Need to generate new claibration ID.

        Parameters
        ----------
        cal : str | CalibrationModel
            The calibration file or model to copy the origin from.
        origin : str
            The new origin string to set.

        Returns
        -------
        str
            The standardized origin string.
        """
        if isinstance(cal, str):
            if os.path.isfile(cal):
                model = datamodels.open(cal)
            elif is_valid_uuid(cal):
                record = self.query(cal_id=cal)
                if record is None:
                    raise ValueError(f"No calibration found with ID: {cal}")
                local_filepath = self._get_local_filepath(record)
                model = datamodels.open(local_filepath)
        elif isinstance(cal, datamodels.CalibrationModel):
            model = cal

        # New model with updated origin
        origin = origin.upper()
        model_out = model.copy()
        model_out.meta.origin = origin

        # Register
        new_local_filepath, new_record = self.register_calibration(model_out)

        return new_local_filepath, new_record
    
    #### Misc. ####
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  instrument_name={self.instrument_name!r},\n"
            f"  origin={self.origin!r},\n"
            f"  local_db={self.local_db!r},\n"
            f"  remote_db={self.remote_db!r}\n"
            f")"
        )
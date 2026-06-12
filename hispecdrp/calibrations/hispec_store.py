from __future__ import annotations
import glob
from koa_middleware import CalibrationStore, CalibrationSelector
import os
from .. import datamodels
from koa_middleware.utils import is_valid_uuid

from typing import TYPE_CHECKING, Sequence
if TYPE_CHECKING:
    from datamodels.base_model import CalibrationModel

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

    def register_calibration(
        self,
        cal : CalibrationModel,
        origin : str | None = None,
        new_version : bool = False,
    ) -> tuple[str, dict]:
        """
        Registers a calibration to the local cache and metadata database.

        Parameters
        ----------
        cal : CalibrationModel | str
            The filepath or Calibration Datamodel object to register.
        origin : str, optional
            The origin to register the calibration under.
        new_version : bool, optional
            Whether to generate a new version for this calibration. If False, the method will check if a calibration with the same version family already exists in the cache and skip registration if so. Defaults to False.

        Returns
        -------
        tuple[str, dict]
            A tuple containing:
                - ``str``: The local file path where the calibration was saved.
                - ``dict``: The calibration metadata dictionary as added to the database.
        """
        cal = datamodels.open(cal)
        return super().register_calibration(cal, origin=origin, new_version=new_version)

    # Select and get calibration
    def select_and_get_calibration(
        self,
        input,
        selector : CalibrationSelector,
    ) -> tuple[str, dict]:
        """
        Selects the best calibration based on input data and a selection rule, then retrieves it.

        This method uses a `CalibrationSelector` to identify the most appropriate calibration
        for the given input data. Once selected, it retrieves the calibration file,
        downloading it if it's not already cached locally.

        Parameters
        ----------
        input
            The input data product for which a calibration is needed.
        selector : CalibrationSelector
            An instance of a ``CalibrationSelector`` class.

        Returns
        -------
        tuple[str, dict]
            - ``str``: The local file path of the retrieved calibration file.
            - ``dict``: The record of the selected calibration from the local database.

        Example:
            >>> # Assuming ``my_input_data`` and ``my_selector`` are defined
            >>> local_filepath, calibration_record = store.select_and_get_calibration(my_input_data, my_selector)
            >>> print(f"Calibration file: {local_filepath}")
            >>> print(f"Calibration ID: {calibration_record['id']}")
        """
        if isinstance(input, datamodels.DataModel):
            input = input.meta
        elif isinstance(input, str):
            if os.path.isfile(input):
                model = datamodels.open(input, meta_only=True)
                input = model.meta
            else:
                raise ValueError(f"Input string is not a valid file path: {input}")

        return super().select_and_get_calibration(input, selector)

    def save_calibration_file(self, cal: CalibrationModel, cal_record : dict | None = None) -> str:
        """
        Saves a calibration file to the local cache directory.

        Parameters
        ----------
        cal : CalibrationModel
            The calibration data model instance to save.
        cal_record : dict | None
            The corresponding record.

        Returns
        -------
        str
            The absolute local file path where the calibration file was saved.
        """

        # Add cal_version and origin to cal file
        if cal_record is not None:
            cal.meta.cal_version = cal_record.get('cal_version')
            cal.meta.origin = cal_record.get('origin')
        
        # Generate filename
        cal.generate_filename()

        # Call super to save file and return filename
        return super().save_calibration_file(cal, cal_record)

    def sync_records_from_cached_files(
        self,
        cals : str | CalibrationModel | Sequence[str | CalibrationModel] | None = None,
    ) -> None:
        """
        Populates the local database from existing cached calibration files.

        Parameters
        ----------
        cals : str | CalibrationModel | Sequence[str | CalibrationModel]
            A single calibration metadata dictionary or a data model instance,
            or a list of these.

        Notes
        -----
        This method may be removed in the future if not found useful.
        """
        if cals is None:
            cals = [
                f for f in glob.glob(os.path.join(self.data_dir, "*"))
                if os.path.isfile(f) and not os.path.basename(f).startswith(".")
            ]
        if isinstance(cals, (str, datamodels.CalibrationModel)):
            cals = [cals]
        cal_records_added = []
        for cal in cals:
            # Add new record (one at a time in case memory issues for many files)
            if isinstance(cal, str):
                cal_model = datamodels.open(cal, meta_only=True)
            else:
                cal_model = cal

            cal_record = self._prepare_cal_record(cal_model, origin='LOCAL')
            cal_record_added = self.local_db.add(cal_record)
            cal_records_added.append(cal_record_added)

        # Return new new records
        return cal_records_added

    def sync_records_from_remote(
        self,
        cals=None,
        mode: str = "id",
    ) -> list[dict]:
        """
        Synchronize the local DB with the remote KOA catalog.

        ``cals`` is accepted for compatibility with callers that pass an explicit
        remote query result; when omitted, records are discovered via
        ``get_missing_records`` (same as ``koa_middleware``).
        """
        return super().sync_records_from_remote(cals, mode=mode)

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
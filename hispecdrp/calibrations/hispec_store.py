from __future__ import annotations
import glob
from koa_middleware import CalibrationStore, CalibrationSelector
import os
from .. import datamodels
from koa_middleware.utils import generate_koa_filehandle, is_valid_uuid

from typing import TYPE_CHECKING, Sequence
if TYPE_CHECKING:
    from datamodels.base_model import CalibrationModel

import logging
logger = logging.getLogger(__name__)

from .download_verify import (
    ensure_calibration_on_disk,
    normalize_downloaded_calibration_path,
)

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

        Notes
        -----
        Upstream ``koa_middleware.CalibrationStore.register_calibration`` snapshots
        ``cal.to_record()`` *before* ``cal.save()`` and discards the return value of
        ``generate_calibration_version``, so ``filename`` / ``koa_id`` / ``cal_version``
        can be stored as ``None`` even though the FITS file is written. This override
        assigns those fields on the model and record before insert, and returns any
        existing version-family match instead of ``(None, None)``.
        """
        from koa_middleware.utils import generate_md5_file

        cal = datamodels.open(cal)

        if origin is not None:
            cal.meta.origin = origin
        elif getattr(cal.meta, "origin", None) is None:
            cal.meta.origin = os.getenv("KOA_CALIBRATION_ORIGIN", "LOCAL")

        if self.calibration_record_in_cache(cal, mode="id"):
            existing = self.calibration_record_in_cache(cal, mode="id")
            logger.warning("Calibration already exists in cache: %s! Skipping registration.", cal)
            return self._existing_calibration_path_and_record(existing)

        if not new_version and self.calibration_record_in_cache(cal, mode="version-family"):
            existing = self.calibration_record_in_cache(cal, mode="version-family")
            logger.warning("Calibration already exists in cache: %s! Skipping registration.", cal)
            return self._existing_calibration_path_and_record(existing)

        cal_record = cal.to_record()

        if new_version:
            # Middleware's generate_calibration_version sets origin on the dict and
            # *returns* the version string but does not assign cal_version itself.
            version = self.generate_calibration_version(cal_record, origin=cal.meta.origin)
            cal_record["cal_version"] = version
            cal.meta.cal_version = version
            cal.meta.origin = cal_record.get("origin") or cal.meta.origin

        if not getattr(cal.meta, "filename", None):
            cal.generate_filename(origin=cal.meta.origin, auto_version=False)

        cal_record["filename"] = cal.meta.filename
        cal_record["koa_id"] = cal_record.get("koa_id") or cal.meta.filename
        cal_record["origin"] = cal_record.get("origin") or cal.meta.origin
        if getattr(cal.meta, "koa_filepath", None):
            cal_record["koa_filepath"] = cal.meta.koa_filepath
        if getattr(cal.meta, "cal_version", None) and not cal_record.get("cal_version"):
            cal_record["cal_version"] = cal.meta.cal_version

        local_filepath = cal.save(output_dir=self.data_dir)
        fname = os.path.basename(local_filepath)
        cal_record["filename"] = getattr(cal.meta, "filename", None) or fname
        cal_record["koa_id"] = cal_record.get("koa_id") or cal_record["filename"]
        if getattr(cal.meta, "koa_filepath", None):
            cal_record["koa_filepath"] = cal.meta.koa_filepath

        cal_record["file_md5"] = generate_md5_file(local_filepath)
        cal_record_added = self.local_db.add(cal_record)
        return local_filepath, cal_record_added

    def _existing_calibration_path_and_record(
        self, existing: dict | list | None
    ) -> tuple[str | None, dict | None]:
        """Return ``(path, record)`` for a pre-existing cache row, or ``(None, None)``.

        Upstream ``calibration_record_in_cache(..., mode="version-family")`` returns a
        ``list[dict]`` when multiple version-family matches exist (default
        ``include_version=False``). Normalize to a single record before path lookup.
        """
        if not existing:
            return None, None
        if isinstance(existing, list):
            chosen = None
            for row in existing:
                if isinstance(row, dict) and row.get("filename"):
                    chosen = row
                    break
            existing = chosen or (existing[0] if existing else None)
            if not isinstance(existing, dict):
                return None, None
        try:
            path = self.calibration_file_in_cache(existing)
        except Exception:  # noqa: BLE001 - missing/null filename should not crash callers
            logger.warning(
                "Existing calibration record is not usable (filename=%r).",
                existing.get("filename"),
            )
            return None, existing
        if path and existing.get("filename"):
            return path, existing
        return None, existing

    def _prepare_cal_record(
        self,
        cal_model: CalibrationModel,
        origin: str = "LOCAL",
    ) -> dict:
        """Build a DB record from an on-disk / in-memory calibration model."""
        from koa_middleware.utils import generate_md5_file

        if origin and not getattr(cal_model.meta, "origin", None):
            cal_model.meta.origin = origin

        filepath = getattr(cal_model, "_filepath", None)
        if not getattr(cal_model.meta, "filename", None) and filepath:
            cal_model.meta.filename = os.path.basename(filepath)

        record = cal_model.to_record()
        if not record.get("filename") and filepath:
            record["filename"] = os.path.basename(filepath)
        if not record.get("origin"):
            record["origin"] = origin
        if not record.get("koa_id"):
            record["koa_id"] = record.get("filename")
        if filepath and os.path.isfile(filepath) and not record.get("file_md5"):
            record["file_md5"] = generate_md5_file(filepath)
        return record

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

        # koa_middleware logs result['filename'] without checking for None when
        # select_best finds no candidates. Guard here and raise a clear error.
        logger.info(
            "Selecting calibration for input=%s using %s",
            input.get("filename") if hasattr(input, "get") else None,
            selector,
        )
        result = selector.select(input, self.local_db)
        if result is None:
            raise ValueError(
                f"No matching calibration found for selector {type(selector).__name__} "
                f"(instrument_era={input.get('instrument_era')!r}, "
                f"spectrograph={input.get('spectrograph')!r}, "
                f"mjd_start={input.get('mjd_start')!r})."
            )
        logger.info(
            "Selected calibration filename=%s ID=%s",
            result.get("filename"),
            result.get("id"),
        )
        return self.get_calibration(result)

    def download_calibration_file(
        self,
        calibration: dict | str,
    ) -> str:
        """
        Download a remote calibration into ``data_dir`` under the DB filename.

        Wraps ``koa_middleware`` so a zip that uses a different member name is
        moved to the expected cache path, then verifies the file exists and is
        non-trivial before returning.
        """
        if isinstance(calibration, dict):
            cal_record = calibration
        elif isinstance(calibration, str):
            cal_record = self.calibration_record_in_cache(calibration, mode="id") or {
                "id": calibration
            }
        else:
            raise TypeError("Calibration must be a dict or str.")

        expected = self._get_local_filepath(cal_record)
        if expected is None:
            # Fall back to middleware behavior when filename is absent.
            path = super().download_calibration_file(calibration)
            return ensure_calibration_on_disk(path, cal_record=cal_record)

        path = super().download_calibration_file(calibration)
        path = normalize_downloaded_calibration_path(path, expected)
        return ensure_calibration_on_disk(
            path,
            expected_filename=os.path.basename(expected),
            cal_record=cal_record if isinstance(cal_record, dict) else None,
        )

    def get_calibration(self, cal: dict | str) -> tuple[str, dict]:
        """Retrieve a calibration file and verify it is present on disk."""
        path, record = super().get_calibration(cal)
        expected = self._get_local_filepath(record)
        if expected is not None:
            path = normalize_downloaded_calibration_path(path, expected)
        try:
            path = ensure_calibration_on_disk(
                path,
                expected_filename=(record or {}).get("filename"),
                cal_record=record if isinstance(record, dict) else None,
            )
            return path, record
        except (FileNotFoundError, RuntimeError) as first_err:
            # Stale empty/truncated cache hits: remove and force a fresh download.
            if path and os.path.isfile(path):
                try:
                    os.remove(path)
                    logger.warning(
                        "Removed invalid cached calibration %s (%s); re-downloading.",
                        path,
                        first_err,
                    )
                except OSError:
                    raise first_err from None
            if self.remote_db is None:
                raise
            path = self.download_calibration_file(record)
            path = ensure_calibration_on_disk(
                path,
                expected_filename=(record or {}).get("filename"),
                cal_record=record if isinstance(record, dict) else None,
            )
            return path, record

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
        if cals is not None:
            mode = str(mode).lower()
            if mode == "id":
                ids_local = {cal["id"] for cal in self.local_db.query()}
                missing = [cal for cal in cals if cal["id"] not in ids_local]
            elif mode == "last_updated":
                last_updated_local = self.get_last_updated(source="local")
                missing = [
                    cal
                    for cal in cals
                    if last_updated_local is None
                    or cal.get("last_updated", "") > last_updated_local
                ]
            else:
                raise ValueError(
                    f"Invalid mode {mode!r} for sync_records_from_remote()."
                )
            if missing:
                logger.info(
                    "Adding %d explicit remote record(s) to local DB (mode=%r).",
                    len(missing),
                    mode,
                )
                return self.local_db.add(missing)
            logger.info(
                "Local DB already contains explicit remote record(s) (mode=%r).",
                mode,
            )
            return []
        return super().sync_records_from_remote(mode=mode)

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
    
    def generate_calibration_version(
        self,
        calibration,
        origin: str | None = None,
    ) -> str:
        """Generate the next version string; accept a model or a metadata dict.

        Upstream ``koa_middleware`` assigns ``calibration['origin']`` and therefore
        only works with dicts. Datamodel callers (and
        :meth:`~hispecdrp.datamodels.base_model.CalibrationModel.generate_calibration_version`)
        pass the model itself, so convert to a record first and write the version
        back onto the model when applicable.
        """
        model = None
        if isinstance(calibration, datamodels.CalibrationModel):
            model = calibration
            if origin is not None:
                model.meta.origin = origin
            elif getattr(model.meta, "origin", None) is None:
                model.meta.origin = os.getenv("KOA_CALIBRATION_ORIGIN", "LOCAL")
            calibration = model.to_record()

        version = super().generate_calibration_version(calibration, origin=origin)
        # Upstream sets origin on the dict but does not assign the returned version.
        if isinstance(calibration, dict):
            calibration["cal_version"] = version
            if origin is not None:
                calibration["origin"] = origin.upper()
        if model is not None:
            model.meta.cal_version = version
            if calibration.get("origin"):
                model.meta.origin = calibration["origin"]
        return version

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

    def _finalize_cal_record(
        self,
        cal,
        cal_record: dict | str | None = None,
        local_filepath: str | None = None,
    ) -> dict:
        """
        Finalize a calibration record after the file has been saved to disk.

        Adds filename/MD5 via the base store, then fills KOA identifiers that are
        set on the saved file during :meth:`~datamodels.CalibrationModel.generate_filename`
        but are not yet present on the pre-save metadata record.

        Compatible with both koa_middleware call signatures:

        * ``_finalize_cal_record(cal, cal_record, local_filepath)`` (current git)
        * ``_finalize_cal_record(cal_record, local_filepath)`` (older releases)
        """
        if isinstance(cal_record, str):
            # Older koa_middleware: (cal_record, local_filepath)
            record = cal
            path = cal_record
            cal_record = super()._finalize_cal_record(record, path)
        else:
            cal_record = super()._finalize_cal_record(cal, cal_record, local_filepath)

        if cal_record.get("koa_id") is None:
            cal_record["koa_id"] = cal_record.get("filename")
        if cal_record.get("koa_filepath") is None and cal_record.get("koa_id") is not None:
            cal_record["koa_filepath"] = generate_koa_filehandle(
                cal_record.get("instrument_name"),
                cal_record.get("datetime_obs"),
                cal_record["koa_id"],
            )
        return cal_record

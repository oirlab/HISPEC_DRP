from ..base_model import DataModel
from ...calibrations import PARVICalibrationORM
from .dqflags import DQ_FLAGS

import astropy.time
from datetime import datetime

import uuid
import numpy as np

__all__ = ['PARVIDataModel', 'PARVICalibrationModel']

class PARVIDataModel(DataModel):

    HEADER_SCHEMAS = [
        'parvi_core.yaml',
    ]

    CAL_ORM_CLASS = PARVICalibrationORM

    DQ_FLAGS = DQ_FLAGS

    @staticmethod
    def _generate_filename(
        date_time : str,
        suffix : str | None = None
    ) -> str:
        """
        Create a filename for a PARVI data product.

        Format: PR.{datetime}_{suffix}.fits
            - datetime : YYYYMMDDSSSSS.SS
            - suffix : optional suffix

        Args:
            date_time (str): The date and time of the observation.
            suffix (str, optional): An optional suffix to append to the filename. Defaults to None.

        Returns:
            str : The filename
        """
        dt = astropy.time.Time(date_time, precision=3, format='isot')
        dt = dt.datetime
        midnight = datetime.combine(dt.date(), datetime.min.time())
        seconds = (dt - midnight).total_seconds()
        date_time = dt.strftime("%Y%m%d") + '.' + f"{seconds:07.2f}"
        filename = f"PR.{date_time}.fits"
        if suffix:
            filename = "".join([filename[0:-5], "_", suffix, ".fits"])
        return filename
    
    
    def generate_filename(self, suffix : str | None = None) -> str:
        """
        See _generate_filename for details.
        """
        date_time = self.meta.exposure.datetime_start
        if suffix is None:
            if self.data_level_suffix is not None:
                suffix = self.data_level_suffix
        else:
            if self.data_level_suffix is not None:
                suffix = f"{self.data_level_suffix}-{suffix}"
        return self._generate_filename(
            date_time=date_time,
            suffix=suffix
        )
    

class PARVICalibrationModel(PARVIDataModel):

    HEADER_SCHEMAS = ['parvi_calibration.yaml']

    CAL_TYPE = None # String: The type of calibration data.

    def on_init(self):
        super().on_init()
        if self.meta.cal_type is None and self.CAL_TYPE is not None:
            self.meta.cal_type = self.CAL_TYPE

    def on_save(self):
        """
        Override the save method to ensure that the cal_type is set.
        """
        super().on_save()
        if self.meta.cal_type is None and self.CAL_TYPE is not None:
            self.meta.cal_type = self.CAL_TYPE


    @staticmethod
    def _generate_filename(
        date_time : str,
        cal_type : str,
        master : bool = False,
        suffix : str | None = None
    ) -> str:
        """
        Create a filename for a PARVI calibration data product.

        Format: PR_{date}_[master-]{cal_type}_{suffix}.fits
            datetime : YYYYMMDDTHHMMSS.
            cal_type : The type of calibration data.
            master : Insert 'master-' before the cal_type if `master=True`.
            suffix : optional suffix

        Args:
            datetime (str): The date and time of the observation.
            cal_type (str): The type of calibration data.
            master (bool, optional): If True, insert 'master-' before the cal_type. Defaults to False.
            suffix (str, optional): An optional suffix to append to the filename. Defaults to None.

        Returns:
            str : The filename
        """
        dt = astropy.time.Time(date_time, precision=3, format='isot')
        dt = dt.datetime
        midnight = datetime.combine(dt.date(), datetime.min.time())
        seconds = (dt - midnight).total_seconds()
        date_time = dt.strftime("%Y%m%d") + '.' + f"{seconds:07.2f}"
        if master:
            cal_type = f"master-{cal_type}"
        filename = f"PR.{date_time}.{cal_type}.fits"
        if suffix is not None:
            filename = "".join([filename[0:-5], "_", suffix, ".fits"])
        return filename
    

    def generate_filename(self, suffix : str | None = None) -> str:
        """
        See _generate_filename for additional arguments.
        """
        date_time = self.meta.exposure.datetime_start
        cal_type = self.CAL_TYPE
        master = self.meta.master_cal
        return self._generate_filename(
            date_time=date_time,
            cal_type=cal_type,
            master=master,
            suffix=suffix
        )

    def get_calibration_id(self, regenerate_id: bool = False, last_updated: str | None = None) -> str:
        """
        Generate and store the calibration ID in meta.cal_id.
        
        Args:
            regenerate_id (bool): If True, regenerate the ID even if one already exists.
            last_updated (str | None): Timestamp to use for ID generation. If None, generates current timestamp.
            
        Returns:
            str: The calibration ID
        """
        if hasattr(self.meta, 'cal_id') and self.meta.cal_id is not None and not regenerate_id:
            return self.meta.cal_id
        else:
            if last_updated is None:
                last_updated = datetime.now().isoformat()
            self.meta.cal_id = self.generate_calibration_id(last_updated=last_updated)
            return self.meta.cal_id

    def generate_calibration_id(self, last_updated: str) -> str:
        """
        Generate the calibration ID.
        
        Args:
            last_updated (str): Timestamp to use for ID generation
            
        Returns:
            str: The calibration ID
        """
        filename = self.meta.filename or 'unknown'
        drp_version = self.meta.drp_version or 'unknown'
        UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "HISPEC_DRP")
        _id = str(uuid.uuid5(UUID_NAMESPACE, filename + drp_version + last_updated))
        return _id

    def to_orm(
        self, orm_class : type[PARVICalibrationORM] | None = None,
        note : str | None = None,
        regenerate_id : bool = False
    ) -> PARVICalibrationORM:
        
        if orm_class is None:
            orm_class = self.CAL_ORM_CLASS
        if orm_class is None:
            raise ValueError("ORM class must be provided or set as CAL_ORM_CLASS.")
        
        # Generate timestamp once for consistency
        last_updated = datetime.now().isoformat()
        
        # Get data from the datamodel
        datetime_obs = self.meta.exposure.datetime_start
        filename = self.meta.filename
        drp_version = self.meta.drp_version

        # Generate or use existing UUID using the dedicated method with consistent timestamp
        _id = self.get_calibration_id(regenerate_id=regenerate_id, last_updated=last_updated)
        
        # Create and populate the ORM instance
        orm = orm_class(
            id=_id,
            filename=filename,
            cal_type=self.meta.cal_type.lower(),
            instera=self.meta.instrument.era,
            read_mode=self.meta.exposure.read_mode,
            target_name=self.meta.target.name,
            object_fiber1=self.meta.target.object_fiber1,
            object_fiber2=self.meta.target.object_fiber2,
            object_fiber3=self.meta.target.object_fiber3,
            object_fiber4=self.meta.target.object_fiber4,
            mjd_start=self.meta.exposure.mjd_start,
            datetime_obs=datetime_obs,
            drp_version=drp_version,
            note=note,
            last_updated=last_updated,
        )
        return orm
from __future__ import annotations
from ..base_model import DataModel
from ... import datamodels
from ...calibrations import HISPECCalibrationORM
from .dqflags import DQ_FLAGS

from koa_middleware.file_utils import generate_koa_filepath

import uuid
from datetime import datetime
import astropy.time

__all__ = ['HISPECDataModel', 'HISPECCalibrationModel']

class HISPECDataModel(DataModel):

    HEADER_SCHEMAS = [
        'hispec_core.yaml',
    ]

    CAL_ORM_CLASS = HISPECCalibrationORM

    SPECTROGRAPH_ALIASES = {'BSPEC': 'HB', 'RSPEC': 'HR'}

    DQ_FLAGS = DQ_FLAGS

    @staticmethod
    def _generate_filename(
        spectrograph : str,
        date_time : str,
        suffix : str | None = None
    ) -> str:
        """
        Create a filename for a HISPEC data product.
        Format: {spectrograph}.{YYYYMMDD}.{SSSSS.SS}_{suffix}.fits

        Args:
            spectrograph (str): The spectrograph name.
            date_time (str): The properly formated datetime of the observation in .
            suffix (str, optional): An optional suffix to append to the filename. Defaults to None.

        Returns:
            str : The filename
        """
        spectrographs = {'BSPEC': 'HB', 'RSPEC': 'HR'}
        spectrograph = spectrograph.upper()
        if spectrograph in spectrographs:
            s = spectrographs[spectrograph]
        else:
            raise ValueError(f"Spectrpgrah must be BSPEC or RSPEC, got {spectrograph}")
        dt = astropy.time.Time(date_time, precision=3, format='isot')
        dt = dt.datetime
        midnight = datetime.combine(dt.date(), datetime.min.time())
        seconds = (dt - midnight).total_seconds()
        date_time = dt.strftime("%Y%m%d") + '.' + f"{seconds:07.2f}"
        filename = f"{s}.{date_time}.fits"
        if suffix:
            filename = "".join([filename[0:-5], "_", suffix, ".fits"])
        return filename
    
    def generate_filename(self, suffix : str | None = None) -> str:
        """
        See _generate_filename for details.
        """
        date_time = self.meta.exposure.datetime_start
        spectrograph = self.meta.instrument.spectrograph
        if suffix is None:
            if self.data_level_suffix is not None:
                suffix = self.data_level_suffix
        else:
            if self.data_level_suffix is not None:
                suffix = f"{self.data_level_suffix}-{suffix}"
        return self._generate_filename(
            spectrograph=spectrograph,
            date_time=date_time,
            suffix=suffix
        )

    def update_calibration_table(self, calibration : HISPECCalibrationModel | str, note : str | None = None):
        cal_model = datamodels.open(calibration, meta_only=True)
        cal_orm = cal_model.to_orm(note=note)
        row = {}
        for cal_column in self.calibrations.colnames:
            if cal_column in cal_orm.__table__.columns:
                val = getattr(cal_orm, cal_column)
                if val is None:
                    val = ''
                row[cal_column] = val
        self.calibrations.add_row(row)
    

class HISPECCalibrationModel(HISPECDataModel):

    HEADER_SCHEMAS = ['hispec_calibration.yaml']

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
        spectrograph : str,
        date_time : str,
        cal_type : str,
        master : bool = False,
        suffix : str | None = None
    ) -> str:
        """
        Create a filename for a HISPEC calibration data product.

        Format: HS_{spectrograph}.{date}.[master-]{cal_type}-{suffix}.fits
            datetime : YYYYMMDDTHHMMSS.
            cal_type : The type of calibration data.
            master : Insert 'master-' before the cal_type if `master=True`.
            suffix : optional suffix

        Args:
            spectrograph (str): The spectrograph name.
            datetime (str): The date and time of the observation.
            cal_type (str): The type of calibration data.
            master (bool, optional): If True, insert 'master-' before the cal_type. Defaults to False.
            suffix (str, optional): An optional suffix to append to the filename. Defaults to None.

        Returns:
            str : The filename
        """
        spectrograph = spectrograph.upper()
        if spectrograph in HISPECDataModel.SPECTROGRAPH_ALIASES:
            s = HISPECDataModel.SPECTROGRAPH_ALIASES[spectrograph]
        else:
            raise ValueError(f"Spectrograph must be BSPEC or RSPEC, got {spectrograph}")
        if master:
            cal_type = f"master-{cal_type}"
        dt = astropy.time.Time(date_time, precision=3, format='isot')
        dt = dt.datetime
        midnight = datetime.combine(dt.date(), datetime.min.time())
        seconds = (dt - midnight).total_seconds()
        date_time = dt.strftime("%Y%m%d") + '.' + f"{seconds:07.2f}"
        filename = f"{s}.{date_time}.{cal_type}.fits"
        if suffix is not None:
            filename = "".join([filename[0:-5], "_", suffix, ".fits"])
        return filename
    
    def generate_filename(self, suffix : str | None = None) -> str:
        """
        See _generate_filename for additional arguments.
        """
        spectrograph = self.meta.instrument.spectrograph
        date_time = self.meta.exposure.datetime_start
        cal_type = self.CAL_TYPE
        master = self.meta.master_cal
        return self._generate_filename(
            spectrograph=spectrograph,
            date_time=date_time,
            cal_type=cal_type,
            master=master,
            suffix=suffix
        )

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
        self, orm_class : type[HISPECCalibrationORM] | None = None,
        note : str | None = None,
        regenerate_id : bool = False,
        fill_koa : bool = False
    ) -> HISPECCalibrationORM:
        
        if orm_class is None:
            orm_class = self.CAL_ORM_CLASS
        if orm_class is None:
            raise ValueError("ORM class must be provided or set as CAL_ORM_CLASS.")
        
        # Generate timestamp once for consistency
        last_updated = datetime.now().isoformat(timespec="milliseconds")
        
        # Get data from the datamodel
        datetime_obs = self.meta.exposure.datetime_start
        filename = self.meta.filename
        drp_version = self.meta.drp_version
        
        # Conditionally generate KOA paths based on parameter
        if fill_koa:
            spectrograph = spectrograph.upper()
            if spectrograph in HISPECDataModel.SPECTROGRAPH_ALIASES:
                instrument_prefix = HISPECDataModel.SPECTROGRAPH_ALIASES[spectrograph]
            else:
                raise ValueError(f"Spectrograph must be BSPEC or RSPEC, got {spectrograph}")
            koa_filepath = generate_koa_filepath(
                instrument=self.meta.instrument.name,
                instrument_prefix=instrument_prefix,
                data_level='cal',
                date_obs=self.meta.date_obs,
                utc_obs=self.meta.utc
            )
            koa_id = koa_filepath.split('/')[-1]
        else:
            koa_filepath = None
            koa_id = None
        
        # Generate or use existing UUID using the dedicated method with consistent timestamp
        if self.meta.cal_id is None or regenerate_id:
            self.meta.cal_id = self.generate_calibration_id(last_updated=last_updated)
        
        # Create ORM instance with conditional KOA fields (inlined kwargs)
        orm = orm_class(
            id=self.meta.cal_id,
            filename=filename,
            master=self.meta.master_cal,
            cal_type=self.meta.cal_type.lower(),
            instera=self.meta.instrument.era,
            read_mode=self.meta.exposure.read_mode,
            target_name=self.meta.target.name,
            object_fiber1=self.meta.target.object_fiber1,
            object_fiber2=self.meta.target.object_fiber2,
            object_fiber3=self.meta.target.object_fiber3,
            spectrograph=self.meta.instrument.spectrograph,
            mjd_start=self.meta.exposure.mjd_start,
            datetime_obs=datetime_obs,
            drp_version=drp_version,
            note=note,
            last_updated=last_updated,
            koa_id=koa_id,
            koa_filepath=koa_filepath,
        )
        
        return orm
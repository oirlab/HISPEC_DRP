from __future__ import annotations
from ..base_model import DataModel, CalibrationModel
from ... import datamodels
from .hispec_meta import HISPECCoreMeta, HISPECCalibrationMeta
from .dqflags import DQ_FLAGS

from koa_middleware.file_utils import generate_koa_filepath

import uuid
from datetime import datetime
import astropy.time

__all__ = ['HISPECDataModel', 'HISPECCalibrationModel']

class HISPECDataModel(DataModel):

    META_MODEL = HISPECCoreMeta

    META_DEFAULTS = {
        'instrument_name' : 'HISPEC',
        'telescope' : 'Keck-II',
        'order_start' : 125,
        'order_end' : 85,
    }

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
        date_time = self.meta.datetime_obs
        spectrograph = self.meta.spectrograph
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
    

class HISPECCalibrationModel(CalibrationModel, HISPECDataModel):

    META_MODEL = HISPECCalibrationMeta

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
        spectrograph = self.meta.spectrograph
        date_time = self.meta.datetime_obs
        cal_type = self.CAL_TYPE
        master = self.meta.master_cal
        return self._generate_filename(
            spectrograph=spectrograph,
            date_time=date_time,
            cal_type=cal_type,
            master=master,
            suffix=suffix
        )
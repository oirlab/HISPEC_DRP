from .parvi_meta import PARVICoreMeta, PARVICalibrationMeta
from ..base_model import DataModel, CalibrationModel
from .dqflags import DQ_FLAGS

import astropy.time
from datetime import datetime

import uuid

__all__ = ['PARVIDataModel', 'PARVICalibrationModel']

class PARVIDataModel(DataModel):

    META_MODEL = PARVICoreMeta

    META_DEFAULTS = {
        'instrument_name' : 'PARVI',
        'telescope' : 'Palomar',
        'order_start' : 125,
        'order_end' : 85,
    }

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
        date_time = self.meta.datetime_obs
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
    

class PARVICalibrationModel(CalibrationModel, PARVIDataModel):

    META_MODEL = PARVICalibrationMeta

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
        date_time = self.meta.datetime_obs
        cal_type = self.CAL_TYPE
        master = self.meta.master_cal
        return self._generate_filename(
            date_time=date_time,
            cal_type=cal_type,
            master=master,
            suffix=suffix
        )
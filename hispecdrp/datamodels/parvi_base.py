from .parvi_meta import PARVIMeta
from .base_model import DataModel, CalibrationModel
from .dqflags import DQ_FLAGS

__all__ = ['PARVIDataModel', 'PARVICalibrationModel']

class PARVIDataModel(DataModel):

    INSTRUMENT_NAME = 'PARVI'
    INSTRUMENT_ALIAS = 'PR'

    META_MODEL = PARVIMeta
    ORM_TABLE_NAME = 'PARVI'

    DQ_FLAGS = DQ_FLAGS

    NUM_FIBERS = 2
    
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
            instrument_alias=self.INSTRUMENT_ALIAS,
            date_time=date_time,
            suffix=suffix
        )


class PARVICalibrationModel(CalibrationModel, PARVIDataModel):
    
    def generate_filename(self, suffix : str | None = None) -> str:
        """
        See _generate_filename for additional arguments.
        """
        master_cal = True
        return self._generate_filename(
            instrument_alias=self.INSTRUMENT_ALIAS,
            date_time=self.meta.datetime_obs,
            cal_type=self.CAL_TYPE,
            master=master_cal,
            suffix=suffix
        )
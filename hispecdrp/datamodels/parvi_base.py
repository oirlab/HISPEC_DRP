from .parvi_meta import PARVIMeta, PARVICalibrationMeta
from .base_model import DataModel, CalibrationModel
from .dqflags import DQ_FLAGS

import scipy.constants

__all__ = ['PARVIDataModel', 'PARVICalibrationModel']

class PARVIDataModel(DataModel):

    META_MODEL = PARVIMeta

    INSTRUMENT_ALIAS = 'PR'

    META_DEFAULTS = {
        'instrument_name' : 'PARVI',
        'telescope' : 'Palomar',
        'order_start' : 125,
        'order_end' : 85,
        'lfc_nu0' : scipy.constants.c / (1559.91370 * 1E-9),
        'lfc_delta_nu' : 10.000000174E9
    }

    DQ_FLAGS = DQ_FLAGS

    NUM_FIBERS = 4
    
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

    META_MODEL = PARVICalibrationMeta
    
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
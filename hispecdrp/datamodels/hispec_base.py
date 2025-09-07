from __future__ import annotations
from .base_model import DataModel, CalibrationModel
from .hispec_meta import HISPECCoreMeta, HISPECCalibrationMeta
from .dqflags import DQ_FLAGS

__all__ = ['HISPECDataModel', 'HISPECCalibrationModel']

class HISPECDataModel(DataModel):

    META_MODEL = HISPECCoreMeta

    META_DEFAULTS = {
        'instrument_name' : 'HISPEC',
        'telescope' : 'Keck-II',
        'order_start' : 125,
        'order_end' : 85,
    }

    INSTRUMENT_ALIASES = {'BSPEC': 'HB', 'RSPEC': 'HR'}

    DQ_FLAGS = DQ_FLAGS

    NUM_FIBERS = 3
    
    def generate_filename(self, suffix : str | None = None) -> str:
        """
        See _generate_filename for details.
        """
        date_time = self.meta.datetime_obs
        spectrograph = self.meta.spectrograph.upper()
        if spectrograph in self.INSTRUMENT_ALIASES:
            instrument_alias = self.INSTRUMENT_ALIASES[spectrograph]
        else:
            raise ValueError(f"Spectrograph must be BSPEC or RSPEC, got {spectrograph}")
        if suffix is None:
            if self.data_level_suffix is not None:
                suffix = self.data_level_suffix
        else:
            if self.data_level_suffix is not None:
                suffix = f"{self.data_level_suffix}-{suffix}"
        return self._generate_filename(
            instrument_alias=instrument_alias,
            date_time=date_time,
            suffix=suffix
        )
    

class HISPECCalibrationModel(CalibrationModel, HISPECDataModel):

    META_MODEL = HISPECCalibrationMeta
    
    def generate_filename(self, suffix : str | None = None) -> str:
        """
        See _generate_filename for additional arguments.
        """
        spectrograph = self.meta.spectrograph.upper()
        if spectrograph in self.INSTRUMENT_ALIASES:
            instrument_alias = self.INSTRUMENT_ALIASES[spectrograph]
        else:
            raise ValueError(f"Spectrograph must be BSPEC or RSPEC, got {spectrograph}")
        return self._generate_filename(
            instrument_alias=instrument_alias,
            date_time=self.meta.datetime_obs,
            cal_type=self.CAL_TYPE,
            master=self.meta.master_cal,
            suffix=suffix
        )
from .base_model import CalibrationModel
from .hispec_base import HISPECCalibrationModel
from .parvi_base import PARVICalibrationModel

from .extensions import create_L0_extensions

__all__ = ['HISPECDarkModel', 'PARVIDarkModel']


class DarkModel(CalibrationModel):
    EXTENSIONS = create_L0_extensions(name='DATA')
    CAL_TYPE = 'dark'


class HISPECDarkModel(HISPECCalibrationModel, DarkModel):
    pass


class PARVIDarkModel(PARVICalibrationModel, DarkModel):
    pass
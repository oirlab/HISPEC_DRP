from .base_model import CalibrationModel
from .hispec_base import HISPECCalibrationModel
from .parvi_base import PARVICalibrationModel

from .extensions import create_L0_extensions

__all__ = ['SaturationModel', 'HISPECSaturationModel', 'PARVISaturationModel']


class SaturationModel(CalibrationModel):
    EXTENSIONS = create_L0_extensions(name='DATA')
    CAL_TYPE = 'saturation'


class HISPECSaturationModel(HISPECCalibrationModel, SaturationModel):
    pass


class PARVISaturationModel(PARVICalibrationModel, SaturationModel):
    pass
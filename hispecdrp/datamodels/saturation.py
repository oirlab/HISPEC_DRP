from .base_model import CalibrationModel
from .hispec_base import HISPECCalibrationModel
from .parvi_base import PARVICalibrationModel

from .extensions import IMAGE_UINT16_HDU, DQ_L0

__all__ = ['SaturationModel', 'HISPECSaturationModel', 'PARVISaturationModel']


class SaturationModel(CalibrationModel):
    EXTENSIONS = {"SAT_THRESH": IMAGE_UINT16_HDU, **DQ_L0}
    CAL_TYPE = 'saturation'


class HISPECSaturationModel(HISPECCalibrationModel, SaturationModel):
    pass


class PARVISaturationModel(PARVICalibrationModel, SaturationModel):
    pass
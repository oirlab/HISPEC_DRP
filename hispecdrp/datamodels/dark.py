from .base_model import CalibrationModel
from .hispec_base import HISPECCalibrationModel
from .parvi_base import PARVICalibrationModel

from .extensions import IMAGE_FLOAT32_HDU,DQ_L0

__all__ = ['DarkModel', 'HISPECDarkModel', 'PARVIDarkModel']


class DarkModel(CalibrationModel):
    name = "RATE"
    EXTENSIONS = {
        f'{name}': IMAGE_FLOAT32_HDU,
        f'{name}_ERR': IMAGE_FLOAT32_HDU,
        f'{name}_VAR_RNOISE': IMAGE_FLOAT32_HDU,
        f'{name}_VAR_POISSON': IMAGE_FLOAT32_HDU,
        **DQ_L0
    }
    CAL_TYPE = 'dark'


class HISPECDarkModel(HISPECCalibrationModel, DarkModel):
    pass


class PARVIDarkModel(PARVICalibrationModel, DarkModel):
    pass
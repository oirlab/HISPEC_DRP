from .base_model import CalibrationModel
from .hispec_base import HISPECCalibrationModel
from .parvi_base import PARVICalibrationModel

from .extensions import IMAGE_FLOAT32_HDU, DQ_L0

__all__ = ['ReadNoiseModel', 'HISPECReadNoiseModel', 'PARVIReadNoiseModel']


class ReadNoiseModel(CalibrationModel):
    EXTENSIONS = {
        f'RN': IMAGE_FLOAT32_HDU,
        f'RN_ERR': IMAGE_FLOAT32_HDU,
        **DQ_L0
    }
    CAL_TYPE = 'rn'


class HISPECReadNoiseModel(HISPECCalibrationModel, ReadNoiseModel):
    pass


class PARVIReadNoiseModel(PARVICalibrationModel, ReadNoiseModel):
    pass
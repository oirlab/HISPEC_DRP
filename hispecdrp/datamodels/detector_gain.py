from .base_model import CalibrationModel
from .hispec_base import HISPECCalibrationModel
from .parvi_base import PARVICalibrationModel

from .extensions import IMAGE_FLOAT32_HDU, DQ_L0

__all__ = ['DetectorGainModel', 'HISPECDetectorGainModel', 'PARVIDetectorGainModel']


class DetectorGainModel(CalibrationModel):
    name = "GAIN"
    EXTENSIONS = {
        f'{name}': IMAGE_FLOAT32_HDU,
        f'{name}_ERR': IMAGE_FLOAT32_HDU,
        f'{name}_VAR_RNOISE': IMAGE_FLOAT32_HDU,
        f'{name}_VAR_POISSON': IMAGE_FLOAT32_HDU,
        **DQ_L0
    }
    CAL_TYPE = 'detgain'


class HISPECDetectorGainModel(HISPECCalibrationModel, DetectorGainModel):
    pass


class PARVIDetectorGainModel(PARVICalibrationModel, DetectorGainModel):
    pass
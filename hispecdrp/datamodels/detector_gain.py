from .base_model import CalibrationModel
from .hispec_base import HISPECCalibrationModel
from .parvi_base import PARVICalibrationModel

from .extensions import create_L0_extensions

__all__ = ['DetectorGainModel', 'HISPECDetectorGainModel', 'PARVIDetectorGainModel']


class DetectorGainModel(CalibrationModel):
    EXTENSIONS = create_L0_extensions(name='DATA')
    CAL_TYPE = 'detgain'


class HISPECDetectorGainModel(HISPECCalibrationModel, DetectorGainModel):
    pass


class PARVIDetectorGainModel(PARVICalibrationModel, DetectorGainModel):
    pass
from .base_model import CalibrationModel
from .hispec_base import HISPECCalibrationModel
from .parvi_base import PARVICalibrationModel

from .extensions import  IMAGE_FLOAT32_HDU,DQ_L0

__all__ = ['DetectorFlatModel', 'HISPECDetectorFlatModel', 'PARVIDetectorFlatModel']


class DetectorFlatModel(CalibrationModel):
    name = "FLAT"
    EXTENSIONS = {
        f'{name}': IMAGE_FLOAT32_HDU,
        f'{name}_ERR': IMAGE_FLOAT32_HDU,
        f'{name}_VAR_RNOISE': IMAGE_FLOAT32_HDU,
        f'{name}_VAR_POISSON': IMAGE_FLOAT32_HDU,
        **DQ_L0
    }
    CAL_TYPE = 'detflat'


class HISPECDetectorFlatModel(HISPECCalibrationModel, DetectorFlatModel):
    pass


class PARVIDetectorFlatModel(PARVICalibrationModel, DetectorFlatModel):
    pass
from .base_model import CalibrationModel
from .hispec_base import HISPECCalibrationModel
from .parvi_base import PARVICalibrationModel

from .extensions import IMAGE_FLOAT32_HDU, DQ_L0

__all__ = ['BiasModel', 'HISPECBiasModel', 'PARVIBiasModel']


class BiasModel(CalibrationModel):
    name = "BIAS"
    EXTENSIONS = {
        f'{name}': IMAGE_FLOAT32_HDU,
        f'{name}_ERR': IMAGE_FLOAT32_HDU,
        f'{name}_VAR_RNOISE': IMAGE_FLOAT32_HDU,
        f'{name}_VAR_POISSON': IMAGE_FLOAT32_HDU,
        **DQ_L0
    }
    CAL_TYPE = 'bias'


class HISPECBiasModel(HISPECCalibrationModel, BiasModel):
    pass


class PARVIBiasModel(PARVICalibrationModel, BiasModel):
    pass
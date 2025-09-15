from .base_model import CalibrationModel
from .hispec_base import HISPECCalibrationModel
from .parvi_base import PARVICalibrationModel

from .extensions import IMAGE_FLOAT32_HDU, DQ_L0

__all__ = ['BiasModel', 'HISPECBiasModel', 'PARVIBiasModel']


class BiasModel(CalibrationModel):
    EXTENSIONS = {
        f'BIAS': IMAGE_FLOAT32_HDU,
        f'BIAS_ERR': IMAGE_FLOAT32_HDU,
        f'kTC': IMAGE_FLOAT32_HDU,
        **DQ_L0
    }
    CAL_TYPE = 'bias'


class HISPECBiasModel(HISPECCalibrationModel, BiasModel):
    pass


class PARVIBiasModel(PARVICalibrationModel, BiasModel):
    pass
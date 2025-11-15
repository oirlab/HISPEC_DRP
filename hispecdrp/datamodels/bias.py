from .base_model import CalibrationModel
from .extensions import IMAGE_FLOAT32_HDU, DQ_L0

__all__ = ['BiasModel']


class BiasModel(CalibrationModel):
    EXTENSIONS = {
        f'BIAS': IMAGE_FLOAT32_HDU,
        f'BIAS_ERR': IMAGE_FLOAT32_HDU,
        f'KTC': IMAGE_FLOAT32_HDU,
        **DQ_L0
    }
    CAL_TYPE = 'bias'
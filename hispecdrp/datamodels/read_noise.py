from .base_model import CalibrationModel
from .extensions import IMAGE_FLOAT32_HDU, DQ_L0

__all__ = ['ReadNoiseModel']


class ReadNoiseModel(CalibrationModel):
    EXTENSIONS = {
        f'RN': IMAGE_FLOAT32_HDU,
        f'RN_ERR': IMAGE_FLOAT32_HDU,
        **DQ_L0
    }
    CAL_TYPE = 'rn'
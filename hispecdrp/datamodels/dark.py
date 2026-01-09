from .base_model import CalibrationModel
from .extensions import IMAGE_FLOAT32_HDU,DQ_L0

__all__ = ['DarkModel']


class DarkModel(CalibrationModel):
    EXTENSIONS = {
        f'RATE': IMAGE_FLOAT32_HDU,
        f'RATE_ERR': IMAGE_FLOAT32_HDU,
        f'RATE_VAR_RNOISE': IMAGE_FLOAT32_HDU,
        f'RATE_VAR_POISSON': IMAGE_FLOAT32_HDU,
        **DQ_L0
    }
    CAL_TYPE = 'dark'
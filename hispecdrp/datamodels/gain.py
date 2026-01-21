from .base_model import CalibrationModel

from .extensions import IMAGE_FLOAT32_HDU, DQ_L0

__all__ = ['GainModel']


class GainModel(CalibrationModel):
    EXTENSIONS = {
        f'GAIN': IMAGE_FLOAT32_HDU,
        f'RN': IMAGE_FLOAT32_HDU,
        f'COV': IMAGE_FLOAT32_HDU,
        **DQ_L0
    }
    CAL_TYPE = 'gain'
from .base_model import CalibrationModel

from .extensions import IMAGE_FLOAT32_HDU, DQ_L0

__all__ = ['DetectorGainModel']


class DetectorGainModel(CalibrationModel):
    EXTENSIONS = {
        f'GAIN': IMAGE_FLOAT32_HDU,
        f'GAIN_ERR': IMAGE_FLOAT32_HDU,
        f'GAIN_VAR_RNOISE': IMAGE_FLOAT32_HDU,
        f'GAIN_VAR_POISSON': IMAGE_FLOAT32_HDU,
        **DQ_L0
    }
    CAL_TYPE = 'detgain'
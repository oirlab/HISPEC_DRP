from .base_model import CalibrationModel

from .extensions import IMAGE_FLOAT32_HDU, IMAGE_UINT16_HDU, DQ_L0

__all__ = ['NonlinearCorrectionModel']

class NonlinearCorrectionModel(CalibrationModel):
    EXTENSIONS = {
        'COEFFS': IMAGE_FLOAT32_HDU,
        'NONLIN_THRESH' : IMAGE_UINT16_HDU,
        'NONLIN_MAX' : IMAGE_UINT16_HDU,
        **DQ_L0
    }
    CAL_TYPE = 'nonlin'
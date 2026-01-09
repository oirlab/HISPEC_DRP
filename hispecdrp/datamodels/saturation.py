from .base_model import CalibrationModel
from .extensions import IMAGE_UINT16_HDU, DQ_L0

__all__ = ['SaturationModel']


class SaturationModel(CalibrationModel):
    EXTENSIONS = {"SAT_THRESH": IMAGE_UINT16_HDU, **DQ_L0}
    CAL_TYPE = 'saturation'
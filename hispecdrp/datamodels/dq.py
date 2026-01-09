from .base_model import CalibrationModel
from .extensions import  DQ_L0

__all__ = ['DQModel']


class DQModel(CalibrationModel):
    EXTENSIONS = DQ_L0
    CAL_TYPE = 'dq'
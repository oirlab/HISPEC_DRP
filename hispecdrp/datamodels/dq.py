from .base_model import CalibrationModel
from .hispec_base import HISPECCalibrationModel
from .parvi_base import PARVICalibrationModel

from .extensions import  DQ_L0

from astropy.io import fits
import numpy as np

__all__ = ['DQModel', 'HISPECDQModel', 'PARVIDQModel']


class DQModel(CalibrationModel):
    EXTENSIONS = DQ_L0
    CAL_TYPE = 'dq'


class HISPECDQModel(HISPECCalibrationModel, DQModel):
    pass


class PARVIDQModel(PARVICalibrationModel, DQModel):
    pass
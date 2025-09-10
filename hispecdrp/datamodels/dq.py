from .base_model import CalibrationModel
from .hispec_base import HISPECCalibrationModel
from .parvi_base import PARVICalibrationModel

from .extensions import create_L0_extensions

from astropy.io import fits
import numpy as np

__all__ = ['DQModel', 'HISPECDQModel', 'PARVIDQModel']


class DQModel(CalibrationModel):
    EXTENSIONS = {'DQ': {'hdu_type' : fits.ImageHDU, 'dtype' : np.float32}}
    CAL_TYPE = 'dq'


class HISPECDQModel(HISPECCalibrationModel, DQModel):
    pass


class PARVIDQModel(PARVICalibrationModel, DQModel):
    pass
from astropy.io import fits
from .hispec_calibration_base import HISPECCalibrationModel
import numpy as np

__all__ = ['HISPECDarkModel']

class HISPECDarkModel(HISPECCalibrationModel):


    EXTENSIONS = {
        'DATA': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float32
        },
        'ERR_ALL': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float32
        },
        'ERR_FLOOR': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float32
        },
        'DQ': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.uint32
        }
    }

    CAL_TYPE = 'dark'

    @property
    def level(self):
        return '0'
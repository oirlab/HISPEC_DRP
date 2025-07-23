from astropy.io import fits
from .hispec_calibration_base import HISPECCalibrationModel
import numpy as np

__all__ = ['HISPECTracePosModel']

class HISPECTracePosModel(HISPECCalibrationModel):

    EXTENSIONS = {
        'PROF_FIB1': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        },
        'PROF_FIB2': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        },
        'PROF_FIB3': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        },
    }

    CAL_TYPE = 'tracepos'
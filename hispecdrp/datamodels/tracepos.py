from astropy.io import fits
from .hispec_calibration_base import HISPECCalibrationModel
import numpy as np

__all__ = ['HISPECTracePosModel']

class HISPECTracePosModel(HISPECCalibrationModel):

    EXTENSIONS = {
        'POS_FIB1': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        },
        'POS_FIB2': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        },
        'POS_FIB3': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        },
        'COEFF_FIB1': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        },
        'COEFF_FIB2': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        },
        'COEFF_FIB3': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        },
    }

    CAL_TYPE = 'tracepos'
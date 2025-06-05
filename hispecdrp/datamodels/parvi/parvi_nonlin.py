from astropy.io import fits
from .parvi_calibration_base import PARVICalibrationModel
import numpy as np

__all__ = ['PARVINonlinearCoeffModel']

class PARVINonlinearCoeffModel(PARVICalibrationModel):

    EXTENSIONS = {
        'COEFFS': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        },
        'THRESH' : {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.uint16
        },
        'MAX' : {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.uint16
        }
    }

    CAL_TYPE = 'nonlin'
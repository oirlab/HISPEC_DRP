from astropy.io import fits
from .parvi_base import PARVICalibrationModel
import numpy as np

__all__ = ['PARVINonlinearCorrectionModel']

class PARVINonlinearCorrectionModel(PARVICalibrationModel):

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
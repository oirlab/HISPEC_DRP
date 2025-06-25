from astropy.io import fits
from .hispec_base import HISPECCalibrationModel
import numpy as np

__all__ = ['HISPECNonlinearCorrectionModel']

class HISPECNonlinearCorrectionModel(HISPECCalibrationModel):

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
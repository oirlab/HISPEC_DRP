from astropy.io import fits
from .hispec_calibration_base import HISPECCalibrationModel
import numpy as np

__all__ = ['HISPECFlatModel']

class HISPECFlatModel(HISPECCalibrationModel):


    EXTENSIONS = {
        'DATA': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float32
        },
        'ERR': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float32
        },
        'VAR_RNOISE': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float32
        },
        'VAR_POISSON': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float32
        },
        'DQ': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.uint32
        }
    }

    CAL_TYPE = 'detflat'
from astropy.io import fits
from .hispec_base import HISPECDataModel
import numpy as np

__all__ = ['HISPECRampModel']

class HISPECRampModel(HISPECDataModel):

    EXTENSIONS = {
        'DATA': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.uint16,
        },
        'TIMES': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        },
        'DQ': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        },
        'ERR': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        },
        'VAR_RNOISE': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float32
        },
        'VAR_POISSON': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float32
        },
        'MID_TIME': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        },
    }
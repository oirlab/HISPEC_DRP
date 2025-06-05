from astropy.io import fits
from .parvi_base import PARVIDataModel
import numpy as np

__all__ = ['PARVIL0Model']

class PARVIL0Model(PARVIDataModel):

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
        },
        'MID_TIME': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        },
    }

    @property
    def data_level_suffix(self):
        return 'rate'
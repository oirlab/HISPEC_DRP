
from astropy.io import fits
from .hispec_base import HISPECDataModel
import numpy as np

class HISPECL0(HISPECDataModel):

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
        },
        'MID_TIME': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        },
    }

    @property
    def level(self):
        return 0
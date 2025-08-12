from astropy.io import fits
from .hispec_base import HISPECDataModel
import numpy as np

__all__ = ['HISPECRampModel']

_READTIMES_HDU = {
    'hdu_type' : fits.BinTableHDU,
    'columns' : {
        'READNUM' : np.uint16,
        'TIMES' : np.float64,
        'EXPTIME' : np.float64,
    }
}

class HISPECRampModel(HISPECDataModel):

    EXTENSIONS = {
        'DATA': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.uint16,
        },
        'TIMES': _READTIMES_HDU,
        'DQ': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.uint32
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
        'MID_TIME': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        },
    }

    @property
    def data_level_suffix(self) -> str:
        return "raw"
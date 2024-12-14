from astropy.io import fits
from .parvi_base import PARVIDataModel

class PARVIL0(PARVIDataModel):

    extensions = {
        'DATA': fits.ImageHDU,
        'ERR_ALL': fits.ImageHDU,
        'ERR_FLOOR': fits.ImageHDU,
        'DQ': fits.ImageHDU,
        'MID_TIME': fits.ImageHDU
    }

from astropy.io import fits
from .hispec_base import HISPECDataModel

class HISPECL0(HISPECDataModel):

    extensions = {
        'DATA': fits.ImageHDU,
        'ERR_ALL': fits.ImageHDU,
        'ERR_FLOOR': fits.ImageHDU,
        'DQ': fits.ImageHDU,
        'MID_TIME': fits.ImageHDU
    }
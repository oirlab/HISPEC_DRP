from astropy.io import fits
from .hispec_base import HISPECDataModel

class HISPECRawRead(HISPECDataModel):

    extensions = {
        'DATA': fits.ImageHDU,
        'DQ': fits.ImageHDU,
    }
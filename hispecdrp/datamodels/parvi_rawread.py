from astropy.io import fits
from .parvi_base import PARVIDataModel

class PARVIRawRead(PARVIDataModel):

    extensions = {
        'DATA': fits.ImageHDU,
        'DQ': fits.ImageHDU,
    }
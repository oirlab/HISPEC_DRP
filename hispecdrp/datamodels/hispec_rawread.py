from astropy.io import fits
from .hispec_base import HISPECDataModel
import numpy as np

class HISPECRawRead(HISPECDataModel):

    EXTENSIONS = {
        'DATA': {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.uint16
        }
    }
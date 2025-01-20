
from astropy.io import fits
from .hispec_base import HISPECDataModel
import numpy as np

class HISPECL1(HISPECDataModel):
    EXTENSIONS = {}
    for i in range(1, 4):
        # Uncalibrated spectrum, error, dq
        EXTENSIONS[f'SPEC_UNCAL_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
        EXTENSIONS[f'SPEC_UNCAL_ERR_ALL_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
        EXTENSIONS[f'SPEC_UNCAL_ERR_FLOOR_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
        EXTENSIONS[f'SPEC_DQ_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.uint32}

        # Mid time
        EXTENSIONS[f'MID_TIME_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float64}

        # Wavelength solution
        EXTENSIONS[f'WS_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float64}
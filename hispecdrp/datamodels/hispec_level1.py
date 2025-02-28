from astropy.io import fits
from .hispec_base import HISPECDataModel
import numpy as np

__all__ = ['HISPECL1Model']

_L1_EXTENSIONS = {}
for i in range(1, 4):
    # Uncalibrated spectrum, error, dq
    _L1_EXTENSIONS[f'SPEC_UNCAL_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _L1_EXTENSIONS[f'SPEC_UNCAL_ERR_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _L1_EXTENSIONS[f'SPEC_UNCAL_VAR_POISSON_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _L1_EXTENSIONS[f'SPEC_UNCAL_VAR_RNOISE_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _L1_EXTENSIONS[f'SPEC_DQ_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.uint32}

    # Mid time
    _L1_EXTENSIONS[f'MID_TIME_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float64}

    # Wavelength solution
    _L1_EXTENSIONS[f'WS_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float64}


class HISPECL1Model(HISPECDataModel):
    
    EXTENSIONS = _L1_EXTENSIONS

    @property
    def level(self):
        return '1'
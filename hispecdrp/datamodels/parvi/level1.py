from astropy.io import fits
from .parvi_base import PARVIDataModel
import numpy as np

__all__ = ['PARVIL1Model']

_L1_EXTENSIONS = {}
for i in range(1, 5):
    # Uncalibrated spectrum, error, dq
    _L1_EXTENSIONS[f'SPEC_UNCAL_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _L1_EXTENSIONS[f'SPEC_UNCAL_ERR_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _L1_EXTENSIONS[f'SPEC_UNCAL_VAR_POISSON_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _L1_EXTENSIONS[f'SPEC_UNCAL_VAR_RNOISE_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _L1_EXTENSIONS[f'SPEC_DQ_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.uint32}

    # Mid time
    _L1_EXTENSIONS[f'MID_TIME_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float64}

    # Wavelength solution
    _L1_EXTENSIONS[f'WAVE_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float64}


class PARVIL1Model(PARVIDataModel):
    
    EXTENSIONS = _L1_EXTENSIONS

    @property
    def data_level_suffix(self):
        return 'specuncal'
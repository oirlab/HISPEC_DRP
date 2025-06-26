
from astropy.io import fits
from .hispec_base import HISPECCalibrationModel
import numpy as np

__all__ = ['HISPECGaussianLSFModel']

_GAUSSLSF_EXTENSIONS = {}
for i in range(1, 5):
    _GAUSSLSF_EXTENSIONS[f'SPEC_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _GAUSSLSF_EXTENSIONS[f'SPEC_ERR_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _GAUSSLSF_EXTENSIONS[f'SPEC_VAR_POISSON_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _GAUSSLSF_EXTENSIONS[f'SPEC_VAR_RNOISE_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _GAUSSLSF_EXTENSIONS[f'SPEC_DQ_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _GAUSSLSF_EXTENSIONS[f'WAVE_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float64}
    _GAUSSLSF_EXTENSIONS[f'LSF_PARAMS_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _GAUSSLSF_EXTENSIONS[f'LFC_FITPARAMS_FIB{i}'] = {
        'hdu_type': fits.BinTableHDU,
        'columns': {
            'AMPLITUDE': np.float64,
            'AMPLITUDE_ERR': np.float64,
            'MU_PIX': np.float64,
            'MU_ERR_PIX': np.float64,
            'MU_ERR_WAVE': np.float64,
            'SIGMA_PIX': np.float64,
            'SIGMA_ERR_PIX': np.float64,
            'SIGMA_WAVE': np.float64,
            'SIGMA_ERR_WAVE': np.float64,
            'B': np.float64,
            'B_ERR': np.float64,
            'MODE_NUM': np.int16,
            'WAVE': np.float64,
            'ORDER': np.uint16,
        }
    }


class HISPECGaussianLSFModel(HISPECCalibrationModel):
    CAL_TYPE = 'lsf'
    EXTENSIONS = _GAUSSLSF_EXTENSIONS
    HEADER_SCHEMAS = [
        'hispec_gausslsf.yaml',
    ]
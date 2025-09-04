
from astropy.io import fits
from .parvi_base import PARVICalibrationModel
import numpy as np

__all__ = ['PARVILFCFitParamsModel']

_LFCFITPARAMS_EXTENSIONS = {}
for i in range(1, 5):
    _LFCFITPARAMS_EXTENSIONS[f'SPEC_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _LFCFITPARAMS_EXTENSIONS[f'SPEC_ERR_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _LFCFITPARAMS_EXTENSIONS[f'SPEC_VAR_POISSON_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _LFCFITPARAMS_EXTENSIONS[f'SPEC_VAR_RNOISE_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _LFCFITPARAMS_EXTENSIONS[f'SPEC_DQ_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.uint32}
    _LFCFITPARAMS_EXTENSIONS[f'WAVE_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float64}
    _LFCFITPARAMS_EXTENSIONS[f'LFC_FITPARAMS_FIB{i}'] = {
        'hdu_type': fits.BinTableHDU,
        'columns': {
            'AMPLITUDE': np.float64,
            'AMPLITUDE_ERR': np.float64,
            'MU_PIX': np.float64,
            'MU_ERR_PIX': np.float64,
            'MU_ERR_WAVE': np.float64,
            'FWHM_PIX': np.float64,
            'FWHM_ERR_PIX': np.float64,
            'FWHM_WAVE': np.float64,
            'FWHM_ERR_WAVE': np.float64,
            'B': np.float64,
            'B_ERR': np.float64,
            'MODE_NUM': np.int16,
            'WAVE': np.float64,
            'ORDER': np.uint16,
        }
    }


class PARVILFCFitParamsModel(PARVICalibrationModel):
    CAL_TYPE = 'lfc_fitparams'
    EXTENSIONS = _LFCFITPARAMS_EXTENSIONS
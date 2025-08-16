
from astropy.io import fits
from .parvi_base import PARVICalibrationModel
import numpy as np

__all__ = ['PARVIReferenceEtalonModel']


_ETALONFITPARAMS_HDU = {
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

_DRIFT_HDU = {
    'hdu_type' : fits.BinTableHDU,
    'columns' : {
        'PIXEL' : np.float64,
        'ORDER' : np.uint16,
        'WAVE' : np.float64,
        'DRIFT_PIX' : np.float64,
        'DRIFT_ERR_PIX' : np.float64,
        'DRIFT_WAVE' : np.float64,
        'DRIFT_ERR_WAVE' : np.float64,
        'DRIFT_VEL' : np.float64,
        'DRIFT_ERR_VEL' : np.float64,
    }
}

_EXTENSIONS = {}

for i in range(1, 5):

    # Calibrated spectrum, error, dq
    _EXTENSIONS[f'SPEC_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _EXTENSIONS[f'SPEC_ERR_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _EXTENSIONS[f'SPEC_VAR_POISSON_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _EXTENSIONS[f'SPEC_VAR_RNOISE_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _EXTENSIONS[f'SPEC_DQ_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.uint32}

    # Mid time
    _EXTENSIONS[f'MID_TIME_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float64}

    # Wavelength solution
    _EXTENSIONS[f'WAVE_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float64}

    # Etalon fit parameters
    _EXTENSIONS[f'ETALON_FITPARAMS_FIB{i}'] = _ETALONFITPARAMS_HDU

    # Drift correction
    _EXTENSIONS[f'DRIFT_FIB{i}'] = _DRIFT_HDU


class PARVIReferenceEtalonModel(PARVICalibrationModel):

    EXTENSIONS = _EXTENSIONS

    CAL_TYPE = 'ref_etalon'

from astropy.io import fits
from .hispec_base import HISPECDataModel
import numpy as np

__all__ = ['HISPECL2Model']

_BARYCORR_HDU = {
    'hdu_type' : fits.BinTableHDU,
    'columns' : {
        'MJDUTC' : np.float64,
        'MJDUTCW' : np.float64,
        'MBJD' : np.float64,
        'MBJDW' : np.float64,
        'BERV' : np.float64,
        'BERVW' : np.float64,
    }
}

_RVCCF_HDU = {
    'hdu_type' : fits.BinTableHDU,
    'columns' : {
        'MBJDW' : np.float64,
        'BERVW' : np.float64,
        'RV' : np.float64,
        'RV_ERR' : np.float64,
        'WAVELENGTH' : np.float64,
        'RESOLUTION' : np.float64,
        'TELL_ABS' : np.float64,
        'BIS' : np.float64,
        'RV_ABS' : np.float64,
        'RV_ABS_ERR' : np.float64,
    }
}

_RVCCFAVG_HDU = {
    'hdu_type' : fits.BinTableHDU,
    'columns' : {
        'MBJDW' : np.float64,
        'BERVW' : np.float64,
        'RV' : np.float64,
        'RV_ERR' : np.float64,
        'WAVELENGTH' : np.float64,
        'RESOLUTION' : np.float64,
        'TELL_ABS' : np.float64,
        'RV_ABS' : np.float64,
        'RV_ABS_ERR' : np.float64,
    }
}

_RVFWM_HDU = {
    'hdu_type' : fits.BinTableHDU,
    'columns' : {
        'MBJDW' : np.float64,
        'BERVW' : np.float64,
        'RV' : np.float64,
        'RV_ERR' : np.float64,
        'WAVELENGTH' : np.float64,
        'RESOLUTION' : np.float64,
        'TELL_ABS' : np.float64,
        'RV_ABS' : np.float64,
        'RV_ABS_ERR' : np.float64,
        'AIRMASS_FIT' : np.float64,
        'AIRMASS_ERR_FIT' : np.float64,
        'PWV_FIT' : np.float64,
        'PWV_ERR_FIT' : np.float64,
        'TELLURIC_VEL_FIT' : np.float64,
        'TELLURIC_VEL_ERR_FIT' : np.float64,
    }
}

_RVFWMAVG_HDU = {
    'hdu_type' : fits.BinTableHDU,
    'columns' : {
        'MBJDW' : np.float64,
        'BERVW' : np.float64,
        'RV' : np.float64,
        'RV_ERR' : np.float64,
        'WAVELENGTH' : np.float64,
        'RESOLUTION' : np.float64,
        'TELL_ABS' : np.float64,
        'BIS' : np.float64,
        'RV_ABS' : np.float64,
        'RV_ABS_ERR' : np.float64,
    }
}

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

_L2_EXTENSIONS = {}

for i in range(1, 4):
    # Uncalibrated spectrum, error
    _L2_EXTENSIONS[f'SPEC_UNCAL_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _L2_EXTENSIONS[f'SPEC_UNCAL_ERR_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _L2_EXTENSIONS[f'SPEC_UNCAL_VAR_POISSON_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _L2_EXTENSIONS[f'SPEC_UNCAL_VAR_RNOISE_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}

    # Calibrated spectrum, error, dq
    _L2_EXTENSIONS[f'SPEC_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _L2_EXTENSIONS[f'SPEC_ERR_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _L2_EXTENSIONS[f'SPEC_VAR_POISSON_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _L2_EXTENSIONS[f'SPEC_VAR_RNOISE_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _L2_EXTENSIONS[f'SPEC_DQ_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.uint32}

    # Mid time
    _L2_EXTENSIONS[f'MID_TIME_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float64}

    # Barycenter correction
    _L2_EXTENSIONS[f'BARYCORR_FIB{i}'] = _BARYCORR_HDU

    # Wavelength solution
    _L2_EXTENSIONS[f'WAVE_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float64}

    # Transmission correction
    _L2_EXTENSIONS[f'TRANS_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _L2_EXTENSIONS[f'TRANS_ERR_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}

    # Sky correction
    _L2_EXTENSIONS[f'SKY_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _L2_EXTENSIONS[f'SKY_ERR_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}

    # Star model
    _L2_EXTENSIONS[f'STAR_MOD_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}

    # E2E throughput
    _L2_EXTENSIONS[f'E2E_TPUT_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}

    # Telluric correction
    _L2_EXTENSIONS[f'TEL_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _L2_EXTENSIONS[f'TEL_ERR_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}

    # Etalon fit parameters
    _L2_EXTENSIONS[f'ETALON_FITPARAMS_FIB{i}'] = _ETALONFITPARAMS_HDU

    # Drift correction
    _L2_EXTENSIONS[f'DRIFT_FIB{i}'] = _DRIFT_HDU

    # RVs
    _L2_EXTENSIONS[f'RVCCF_FIB{i}'] = _RVCCF_HDU
    _L2_EXTENSIONS[f'RVCCFAVG_FIB{i}'] = _RVCCFAVG_HDU
    _L2_EXTENSIONS[f'RVFWM_FIB{i}'] = _RVFWM_HDU
    _L2_EXTENSIONS[f'RVFWMAVG_FIB{i}'] = _RVFWMAVG_HDU


class HISPECL2Model(HISPECDataModel):

    EXTENSIONS = _L2_EXTENSIONS

    @property
    def data_level_suffix(self):
        return 'spec'
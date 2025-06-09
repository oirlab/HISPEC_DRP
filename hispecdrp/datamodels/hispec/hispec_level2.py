
from astropy.io import fits
from .hispec_base import HISPECDataModel
import numpy as np

__all__ = ['HISPECL2Model']

_BARYCORR_HDU = {
    'hdu_type' : fits.BinTableHDU,
    'columns' : {
        'ORDER' : np.uint16,
        'WAVE' : np.float64,
        'JDUTC' : np.float64,
        'JDUTCW' : np.float64,
        'HJD' : np.float64,
        'HJDW' : np.float64,
        'BJD' : np.float64,
        'BJDW' : np.float64,
        'BERV' : np.float64,
        'BERVW' : np.float64,
    }
}

_RVCCF_HDU = {
    'hdu_type' : fits.BinTableHDU,
    'columns' : {
        'BJDUTCW' : np.float64,
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
        'BJDUTCW' : np.float64,
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

    # Flux calibration
    _L2_EXTENSIONS[f'FLUX_ABS_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _L2_EXTENSIONS[f'FLUX_ABS_ERR_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _L2_EXTENSIONS[f'FLUX_ABS_VAR_POISSON_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
    _L2_EXTENSIONS[f'FLUX_ABS_VAR_RNOISE_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}

    # Drift correction NOTE: FORMAT TBD, depends how drifts are calculated (per order, detector, pixel, etc.)
    _L2_EXTENSIONS[f'DRIFT_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}

    # RVs
    _L2_EXTENSIONS[f'RVCCF_FIB{i}'] = _RVCCF_HDU
    _L2_EXTENSIONS[f'RVCCFAVG_FIB{i}'] = _RVCCFAVG_HDU


class HISPECL2Model(HISPECDataModel):

    EXTENSIONS = _L2_EXTENSIONS

    @property
    def data_level_suffix(self):
        return 'spec'
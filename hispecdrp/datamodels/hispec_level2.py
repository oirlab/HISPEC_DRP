
from astropy.io import fits
from .hispec_base import HISPECDataModel
import numpy as np
import pandas as pd

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

class HISPECL2(HISPECDataModel):
    EXTENSIONS = {}
    for i in range(1, 4):
        # Uncalibrated spectrum, error
        EXTENSIONS[f'SPEC_UNCAL_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
        EXTENSIONS[f'SPEC_UNCAL_ERR_ALL_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
        EXTENSIONS[f'SPEC_UNCAL_ERR_FLOOR_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}

        # Calibrated spectrum, error, dq
        EXTENSIONS[f'SPEC_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
        EXTENSIONS[f'SPEC_ERR_ALL_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
        EXTENSIONS[f'SPEC_ERR_FLOOR_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
        EXTENSIONS[f'SPEC_DQ_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.uint32}

        # Mid time
        EXTENSIONS[f'MID_TIME_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float64}

        # Barycenter correction
        EXTENSIONS[f'BARYCORR_FIB{i}'] = _BARYCORR_HDU

        # Wavelength solution
        EXTENSIONS[f'WS_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float64}

        # Fiber flat
        EXTENSIONS[f'FIBERFLAT_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
        EXTENSIONS[f'FIBERFLAT_ERR_ALL_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
        EXTENSIONS[f'FIBERFLAT_ERR_FLOOR_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}

        # Sky fiber
        EXTENSIONS[f'SKY_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
        EXTENSIONS[f'SKY_ERR_ALL_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
        EXTENSIONS[f'SKY_ERR_FLOOR_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}

        # Star model
        EXTENSIONS[f'STAR_MOD_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}

        # E2E throughput
        EXTENSIONS[f'E2E_TPUT_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}

        # Telluric correction
        EXTENSIONS[f'TEL_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
        EXTENSIONS[f'TEL_ERR_ALL_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
        EXTENSIONS[f'TEL_ERR_FLOOR_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}

        # Flux calibration
        EXTENSIONS[f'FLUX_ABS_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
        EXTENSIONS[f'FLUX_ABS_ERR_ALL_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
        EXTENSIONS[f'FLUX_ABS_ERR_FLOOR_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}

        # Simult etalon
        EXTENSIONS[f'ET_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
        EXTENSIONS[f'ET_ERR_ALL_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
        EXTENSIONS[f'ET_ERR_FLOOR_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
        EXTENSIONS[f'ET_CEN_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float64}
        EXTENSIONS[f'ET_CEN_ERR_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float64}

        # RVs
        EXTENSIONS[f'RVCCF_FIB{i}'] = _RVCCF_HDU
        EXTENSIONS[f'RVCCFAVG_FIB{i}'] = _RVCCFAVG_HDU
        EXTENSIONS[f'CCF_FIB{i}'] = {'hdu_type': fits.ImageHDU, 'dtype': np.float32}
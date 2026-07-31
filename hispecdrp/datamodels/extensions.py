from astropy.io import fits
import numpy as np

# NOTE:
# _HDU = no key specified
# No _HDU = key already specified


# Common Image HDUs
IMAGE_UINT8_HDU = {'hdu_type' : fits.ImageHDU, 'dtype' : np.uint8}
IMAGE_UINT16_HDU = {'hdu_type' : fits.ImageHDU, 'dtype' : np.uint16}
IMAGE_UINT32_HDU = {'hdu_type' : fits.ImageHDU, 'dtype' : np.uint32}
IMAGE_FLOAT32_HDU = {'hdu_type' : fits.ImageHDU, 'dtype' : np.float32}
IMAGE_FLOAT64_HDU = {'hdu_type' : fits.ImageHDU, 'dtype' : np.float64}

# Only specified in DataModel
RECEIPT = {
    'RECEIPT': {
        'hdu_type' : fits.BinTableHDU,
        'columns' : {
            'CLASS' : str, # Primitive class
            'PROC_START_TIME' : str, # Processing time YYYY-MM-DDTHH:MM:SS.sss
            'PROC_END_TIME' : str, # Processing time YYYY-MM-DDTHH:MM:SS.sss
            'FILENAME' : str, # Base filename (is this useful?)
            'ARGUMENTS' : str, # JSON rep of args. This may become problematic with arrays.
            'DRP_VERSION' : str, # DRP semver
            'DRP_GIT_COMMIT_ID' : str, # full SHA1
            'STATUS' : str, # COMPLETED, FAILED, SKIPPED
        }
    }
}

CALIBRATIONS = {
    'CALIBRATIONS': {
        'hdu_type': fits.BinTableHDU,
        'columns': {
            'CAL_ROLE': str, # Role of calibration
            'CAL_TYPE': str, # Type of calibration
            'FILENAME': str, # Base filename of the calibration
            'ID': str, # UUID
            'DATETIME_OBS': str, # Observation date and time YYYY-MM-DDTHH:MM:SS.sss
            'LAST_PROCESSED': str, # When calibration was last processed. This is NOT equal to the last_updated timestamp in the calibration DB.
            'TIME_APPLIED' : str, # Time calibration was applied YYYY-MM-DDTHH:MM:SS.sss
            'CLASS' : str, # Primitive class that used the calibration.
            'DRP_VERSION': str, # DRP semver calibration was processed with (not necessarily current version)
        }
    }
}

DATA_ANCESTORS = {
    'DATA_ANCESTORS': {
        'hdu_type': fits.BinTableHDU,
        'columns': {
            'FILENAME' : str,
            'CLASS' : str,
            'TIME_APPLIED' : str,
            # ''... controlled by include_ancestors
        }
    }
}


#### Raw

DQ_RAW = {'DQ_RAW': IMAGE_UINT8_HDU}

#### L0 (rate maps)

DQ_L0 = {'DQ': IMAGE_UINT32_HDU}

def create_L0_extensions(name : str = 'RATE') -> dict[str, dict]:
    ext = {
        f'{name}': IMAGE_FLOAT32_HDU,
        f'{name}_ERR': IMAGE_FLOAT32_HDU,
        f'{name}_VAR_RNOISE': IMAGE_FLOAT32_HDU,
        f'{name}_VAR_POISSON': IMAGE_FLOAT32_HDU,
    }
    ext.update(DQ_L0)
    ext['MID_TIME'] = IMAGE_FLOAT64_HDU
    return ext


#### Spectral level extensions (per-fiber)

def create_dataspec_extensions(fiber : str, prefix : str = 'SPEC') -> dict[str, dict]:
    return {
        f'{prefix}_FIB{fiber}': IMAGE_FLOAT32_HDU,
        f'{prefix}_ERR_FIB{fiber}': IMAGE_FLOAT32_HDU,
        f'{prefix}_VAR_POISSON_FIB{fiber}': IMAGE_FLOAT32_HDU,
        f'{prefix}_VAR_RNOISE_FIB{fiber}': IMAGE_FLOAT32_HDU,
    }

def create_specdq_extensions(fiber : str) -> dict[str, dict]:
    return {f'SPEC_DQ_FIB{fiber}' : IMAGE_UINT32_HDU}

def create_sky_extensions(fiber : str) -> dict[str, dict]:
    return {
        f'SKY_FIB{fiber}': IMAGE_FLOAT32_HDU,
        f'SKY_ERR_FIB{fiber}': IMAGE_FLOAT32_HDU,
    }

def create_trans_extensions(fiber : str) -> dict[str, dict]:
    return {
        f'TRANS_FIB{fiber}': IMAGE_FLOAT32_HDU,
        f'TRANS_ERR_FIB{fiber}': IMAGE_FLOAT32_HDU,
    }

def create_throughput_extensions(fiber : str) -> dict[str, dict]:
    return {f'THROUGHPUT_FIB{fiber}': IMAGE_FLOAT32_HDU}

def create_wave_extensions(fiber : str) -> dict[str, dict]:
    return {f'WAVE_FIB{fiber}': IMAGE_FLOAT64_HDU}

CHEBYSHEV_NORM_HDU = {
    'hdu_type': fits.BinTableHDU,
    'columns': {
        'FIBER': str,
        'PIXEL_NORM_MIN': np.float64,
        'PIXEL_NORM_MAX': np.float64,
        'ORDER_NORM_MIN': np.int16,
        'ORDER_NORM_MAX': np.int16,
    }
}

def create_chebyshev_extensions(fiber : str) -> dict[str, dict]:
    return {
        f'CHEB_COEFFS_FIB{fiber}': IMAGE_FLOAT64_HDU,
        f'CHEB_NORM_FIB{fiber}': CHEBYSHEV_NORM_HDU
    }

#### Wavelength solution drifts

DRIFT_HDU = {
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

def create_drift_extensions(fiber : str) -> dict[str, dict]:
    return {f'DRIFT_FIB{fiber}': DRIFT_HDU}


#### Etalon fit parameters

ETALONFITPARAMS_HDU = {
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

def create_etalonfitparam_extensions(fiber : str) -> dict[str, dict]:
    return {f'ETALON_FITPARAMS_FIB{fiber}': ETALONFITPARAMS_HDU}


#### LFC Fit parameters

LFCFITPARAMS_HDU = {
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

def create_lfcfitparam_extensions(fiber : str) -> dict[str, dict]:
    return {f'LFC_FITPARAMS_FIB{fiber}': LFCFITPARAMS_HDU}


#### RVs

# Element dtype for each 1D CCF vector stored in the ``CCF`` column (the table column itself is ``object``).
RVCCF_CCF_ELEM_DTYPE = np.float64

RVCCF_HDU = {
    'hdu_type' : fits.BinTableHDU,
    'columns' : {
        'MBJDW' : np.float64,  # flux-weighted modified BJD
        'BERVW' : np.float64,  # flux-weighted barycentric correction (km/s)
        'RV' : np.float64,  # chunk RV (m/s)
        'RV_ERR' : np.float64,  # chunk RV uncertainty (m/s)
        'ORDER' : np.int16,  # echelle order of the chunk
        'WAVELENGTH' : np.float64,  # chunk center wavelength (µm)
        'RESOLUTION' : np.float64,  # resolving power R = λ/Δλ
        'TELL_ABS' : np.float64,  # telluric absorption metric
        'BIS' : np.float64,  # Queloz bisector inverse slope (m/s)
        'BIS_ERR' : np.float64,  # uncertainty on BIS (m/s)
        'RV_ABS' : np.float64,  # absolute-frame RV (m/s)
        'RV_ABS_ERR' : np.float64,  # uncertainty on RV_ABS (m/s)
        'INLIER' : np.uint8,  # 1 = used in aggregate, 0 = rejected by median/MAD outlier
        'CHUNK_ID' : np.int32,  # serial chunk index in the exposure
        'N_GOOD' : np.int32,  # good pixels in chunk (-1 if unknown)
        'V_MIN' : np.float64,  # CCF velocity grid min (m/s)
        'V_MAX' : np.float64,  # CCF velocity grid max (m/s)
        'V_STEP' : np.float64,  # CCF velocity sampling step (m/s)
        'N_VEL' : np.int32,  # length of CCF(v)
        'CCF' : object,  # object column: each row is a 1D np.ndarray, dtype ``RVCCF_CCF_ELEM_DTYPE``; empty if absent
    }
}

RVCCFAVG_HDU = {
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

RVFWM_HDU = {
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

RVFWMAVG_HDU = {
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

def create_rv_extensions(fiber : str) -> dict[str, dict]:
    return {
        f'RVCCF_FIB{fiber}': RVCCF_HDU,
        f'RVCCFAVG_FIB{fiber}': RVCCFAVG_HDU,
        f'RVFWM_FIB{fiber}': RVFWM_HDU,
        f'RVFWMAVG_FIB{fiber}': RVFWMAVG_HDU
    }


#### Telluric models

def create_telluric_extensions(fiber : str) -> dict[str, dict]:
    return {
        f'TEL_FIB{fiber}': IMAGE_FLOAT32_HDU,
        f'TEL_ERR_FIB{fiber}': IMAGE_FLOAT32_HDU
    }


#### Stellar models

def create_starmodel_extensions(fiber : str) -> dict[str, dict]:
    return {f'STAR_MOD_FIB{fiber}' : IMAGE_FLOAT32_HDU}


#### Throughput

def create_throughput_extensions(fiber : str) -> dict[str, dict]:
    return {f'THROUGHPUT_FIB{fiber}' : IMAGE_FLOAT32_HDU}


#### Barycentric correction

BARYCORR_HDU = {
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

def create_midtime_extensions(fiber : str) -> dict[str, dict]:
    return {f'MID_TIME_FIB{fiber}' : IMAGE_FLOAT64_HDU}

def create_barycorr_extensions(fiber : str) -> dict[str, dict]:
    return {f'BARYCORR_FIB{fiber}' : BARYCORR_HDU}




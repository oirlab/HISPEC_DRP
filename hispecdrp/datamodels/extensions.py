from __future__ import annotations

from astropy.io import fits
from astropy.table import Column, Table
import numpy as np

# NOTE:
# _HDU = no key specified
# No _HDU = key already specified


# Common Image HDUs
def IMAGE_HDU(
    dtype : type[np.generic],
    ndim: int,
    default: float | int | None = None,
    per_fiber: bool = False
) -> dict:
    """
    Build an image extension descriptor.

    Parameters
    ----------
    dtype
        Numpy dtype of the array (np.float64, np.int32, etc.).
    ndim : int
        Number of dimensions the array is expected to have.
    default : float | int | None, optional
        Fill value used to lazily auto-initialize the array to
        ``np.full(model.shape, default, dtype=dtype)`` the first time the
        attribute is accessed, but only when ``ndim == len(model.shape)``.
        Defaults to ``None``, which opts out of auto-initialization; the
        attribute then stays ``None`` until explicitly assigned.
    per_fiber : bool, optional
        If True, declare this extension once (without a ``_FIB{n}`` suffix)
        and let ``DataModel._get_extensions`` expand it into one
        ``<NAME>_FIB{n}`` entry per fiber, for ``n in 1..model.num_fibers``.
    """
    info = {'hdu_type': fits.ImageHDU, 'dtype': dtype, 'ndim': ndim}
    if default is not None:
        info['default'] = default
    if per_fiber:
        info['per_fiber'] = True
    return info


def TABLE_HDU(columns: dict, per_fiber: bool = False) -> dict:
    """
    Build a table extension descriptor.

    Parameters
    ----------
    columns : dict
        Mapping of column name -> dtype (e.g. ``str``, ``np.float64``, ``object``
        for a variable-length array per row, as used by the ``CCF`` column).
    per_fiber : bool, optional
        If True, declare this extension once (without a ``_FIB{n}`` suffix)
        and let ``DataModel._get_extensions`` expand it into one
        ``<NAME>_FIB{n}`` entry per fiber, for ``n in 1..model.config.num_fibers``. Table extensions are always eagerly initialized as an empty (0-row) Table at datamodel construction, regardless of this flag.
    """
    info = {'hdu_type': fits.BinTableHDU, 'columns': columns}
    if per_fiber:
        info['per_fiber'] = True
    return info


# Only specified in DataModel
RECEIPT = {
    'RECEIPT': TABLE_HDU({
        'CLASS' : str, # Primitive class
        'PROC_START_TIME' : str, # Processing time YYYY-MM-DDTHH:MM:SS.sss
        'PROC_END_TIME' : str, # Processing time YYYY-MM-DDTHH:MM:SS.sss
        'FILENAME' : str, # Base filename (is this useful?)
        'PARAMETERS' : str, # JSON rep of args. This may become problematic with arrays.
        'DRP_VERSION' : str, # DRP semver
        'DRP_GIT_COMMIT_ID' : str, # full SHA1
        'STATUS' : str, # COMPLETED, FAILED, SKIPPED
        'CONFIG_FILE' : str, # Path to the configuration file
    })
}


def migrate_receipt_table(table: Table | None) -> Table | None:
    """
    Upgrade a RECEIPT table to the current schema by adding any missing columns.
    """
    if table is None:
        return None
    if not isinstance(table, Table):
        return table

    missing = [c for c in RECEIPT['RECEIPT']['columns'] if c not in table.colnames]
    if not missing:
        return table

    out = table.copy()
    for name in missing:
        dtype = RECEIPT['RECEIPT']['columns'][name]
        out.add_column(Column(name=name, data=np.full(len(out), '', dtype=dtype)))
    return out


CALIBRATIONS = {
    'CALIBRATIONS': TABLE_HDU({
        'CAL_ROLE': str, # Role of calibration
        'CAL_TYPE': str, # Type of calibration
        'FILENAME': str, # Base filename of the calibration
        'ID': str, # UUID
        'DATETIME_OBS': str, # Observation date and time YYYY-MM-DDTHH:MM:SS.sss
        'LAST_PROCESSED': str, # When calibration was last processed. This is NOT equal to the last_updated timestamp in the calibration DB.
        'TIME_APPLIED' : str, # Time calibration was applied YYYY-MM-DDTHH:MM:SS.sss
        'CLASS' : str, # Primitive class that used the calibration.
        'DRP_VERSION': str, # DRP semver calibration was processed with (not necessarily current version)
    })
}

DATA_ANCESTORS = {
    'DATA_ANCESTORS': TABLE_HDU({
        'FILENAME' : str,
        'CLASS' : str,
        'TIME_APPLIED' : str,
        # ''... controlled by include_ancestors
    })
}


#### Raw

DQ_RAW = {'DQ_RAW': IMAGE_HDU(np.uint8, ndim=3, default=0)}

#### L0 (rate maps)

DQ_L0 = {'DQ': IMAGE_HDU(np.uint32, ndim=2, default=0)}

def create_L0_extensions(name : str = 'RATE') -> dict[str, dict]:
    ext = {
        f'{name}': IMAGE_HDU(np.float32, ndim=2, default=0),
        f'{name}_ERR': IMAGE_HDU(np.float32, ndim=2, default=0),
        f'{name}_VAR_RNOISE': IMAGE_HDU(np.float32, ndim=2, default=0),
        f'{name}_VAR_POISSON': IMAGE_HDU(np.float32, ndim=2, default=0),
    }
    ext.update(DQ_L0)
    ext['MID_TIME'] = IMAGE_HDU(np.float64, ndim=2, default=0)
    return ext


#### Spectral level extensions (per-fiber)
#
# Each of these is declared ONCE, without a fiber suffix, using `per_fiber=True`.
# `DataModel._get_extensions` expands each into `<NAME>_FIB{n}` for `n in 1..model.config.num_fibers` when a concrete model's EXTENSIONS includes it.

def create_dataspec_extensions(prefix : str = 'SPEC') -> dict[str, dict]:
    return {
        prefix: IMAGE_HDU(np.float32, ndim=2, default=0, per_fiber=True),
        f'{prefix}_ERR': IMAGE_HDU(np.float32, ndim=2, default=0, per_fiber=True),
        f'{prefix}_VAR_POISSON': IMAGE_HDU(np.float32, ndim=2, default=0, per_fiber=True),
        f'{prefix}_VAR_RNOISE': IMAGE_HDU(np.float32, ndim=2, default=0, per_fiber=True),
    }

def create_specdq_extensions() -> dict[str, dict]:
    return {'SPEC_DQ': IMAGE_HDU(np.uint32, ndim=2, default=0, per_fiber=True)}

def create_sky_extensions() -> dict[str, dict]:
    return {
        'SKY': IMAGE_HDU(np.float32, ndim=2, default=0, per_fiber=True),
        'SKY_ERR': IMAGE_HDU(np.float32, ndim=2, default=0, per_fiber=True),
    }

def create_trans_extensions() -> dict[str, dict]:
    return {
        'TRANS': IMAGE_HDU(np.float32, ndim=2, default=1, per_fiber=True),
        'TRANS_ERR': IMAGE_HDU(np.float32, ndim=2, default=1, per_fiber=True),
    }

def create_throughput_extensions() -> dict[str, dict]:
    return {'THROUGHPUT': IMAGE_HDU(np.float32, ndim=2, default=1, per_fiber=True)}

def create_wave_extensions() -> dict[str, dict]:
    return {'WAVE': IMAGE_HDU(np.float64, ndim=2, default=0, per_fiber=True)}

CHEBYSHEV_NORM_HDU = TABLE_HDU({
    'FIBER': str,
    'PIXEL_NORM_MIN': np.float64,
    'PIXEL_NORM_MAX': np.float64,
    'ORDER_NORM_MIN': np.int16,
    'ORDER_NORM_MAX': np.int16,
}, per_fiber=True)

def create_chebyshev_extensions() -> dict[str, dict]:
    return {
        # NOTE: shape is (n_coeffs_intra_order, n_coeffs_inter_order), unrelated to
        # model.shape (Nord x 4096), so no default is given (not auto-initialized).
        'CHEB_COEFFS': IMAGE_HDU(np.float64, ndim=2, per_fiber=True),
        'CHEB_NORM': CHEBYSHEV_NORM_HDU,
    }

#### Wavelength solution drifts

DRIFT_HDU = TABLE_HDU({
    'PIXEL' : np.float64,
    'ORDER' : np.uint16,
    'WAVE' : np.float64,
    'DRIFT_PIX' : np.float64,
    'DRIFT_ERR_PIX' : np.float64,
    'DRIFT_WAVE' : np.float64,
    'DRIFT_ERR_WAVE' : np.float64,
    'DRIFT_VEL' : np.float64,
    'DRIFT_ERR_VEL' : np.float64,
}, per_fiber=True)

def create_drift_extensions() -> dict[str, dict]:
    return {'DRIFT': DRIFT_HDU}


#### Etalon fit parameters

ETALONFITPARAMS_HDU = TABLE_HDU({
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
}, per_fiber=True)

def create_etalonfitparam_extensions() -> dict[str, dict]:
    return {'ETALON_FITPARAMS': ETALONFITPARAMS_HDU}


#### LFC Fit parameters

LFCFITPARAMS_HDU = TABLE_HDU({
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
}, per_fiber=True)

def create_lfcfitparam_extensions() -> dict[str, dict]:
    return {'LFC_FITPARAMS': LFCFITPARAMS_HDU}


#### RVs

# Element dtype for each 1D CCF vector stored in the ``CCF`` column (the table column itself is ``object``).
RVCCF_CCF_ELEM_DTYPE = np.float64

RVCCF_HDU = TABLE_HDU({
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
}, per_fiber=True)

RVCCFAVG_HDU = TABLE_HDU({
    'MBJDW' : np.float64,
    'BERVW' : np.float64,
    'RV' : np.float64,
    'RV_ERR' : np.float64,
    'WAVELENGTH' : np.float64,
    'RESOLUTION' : np.float64,
    'TELL_ABS' : np.float64,
    'RV_ABS' : np.float64,
    'RV_ABS_ERR' : np.float64,
}, per_fiber=True)

RVFWM_HDU = TABLE_HDU({
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
}, per_fiber=True)

RVFWMAVG_HDU = TABLE_HDU({
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
}, per_fiber=True)

def create_rv_extensions() -> dict[str, dict]:
    return {
        'RVCCF': RVCCF_HDU,
        'RVCCFAVG': RVCCFAVG_HDU,
        'RVFWM': RVFWM_HDU,
        'RVFWMAVG': RVFWMAVG_HDU,
    }


#### Telluric models

def create_telluric_extensions() -> dict[str, dict]:
    return {
        'TEL': IMAGE_HDU(np.float32, ndim=2, default=0, per_fiber=True),
        'TEL_ERR': IMAGE_HDU(np.float32, ndim=2, default=0, per_fiber=True),
    }


#### Stellar models

def create_starmodel_extensions() -> dict[str, dict]:
    return {'STAR_MOD' : IMAGE_HDU(np.float32, ndim=2, default=0, per_fiber=True)}


#### Barycentric correction

BARYCORR_HDU = TABLE_HDU({
    'MJDUTC' : np.float64,
    'MJDUTCW' : np.float64,
    'MBJD' : np.float64,
    'MBJDW' : np.float64,
    'BERV' : np.float64,
    'BERVW' : np.float64,
}, per_fiber=True)

def create_midtime_extensions() -> dict[str, dict]:
    return {'MID_TIME' : IMAGE_HDU(np.float64, ndim=2, default=0, per_fiber=True)}

def create_barycorr_extensions() -> dict[str, dict]:
    return {'BARYCORR' : BARYCORR_HDU}

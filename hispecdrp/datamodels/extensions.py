from __future__ import annotations

import types

from astropy.io import fits
import numpy as np

# NOTE:
# _HDU = no key specified
# No _HDU = key already specified


# Symbolic axis-name strings recognized in an IMAGE_HDU `shape=` spec, mixed
# in with literal ints as needed, e.g. ('N_ORDERS', 'DETECTOR_SIZE_Y') or
# ('N_READS', 'DETECTOR_SIZE_Y', 'DETECTOR_SIZE_X'). No constants for these
# -- just write the bare (case-insensitive, but conventionally uppercase)
# string:
#
#   DETECTOR_SIZE_Y / DETECTOR_SIZE_X -- the detector plane's (ny, nx).
#   N_ORDERS   -- number of echelle orders.
#   N_FIBERS   -- number of fibers.
#   N_READS    -- number of UTR reads.
#   ANY        -- never resolvable, by design; marks an axis generic
#                 auto-init can't build (e.g. a per-instance fit result with
#                 no symbol), forcing a fall-through to
#                 DataModel.make_default_array.
#
# Resolved per-model-instance by DataModel._resolve_shape_axis, in order:
# (1) inferred from any *other* extension on the same model that's already
# been assigned real data and whose own declared `shape=` spec uses this
# same symbol (see DataModel._infer_shape_symbols) -- e.g. explicitly
# assigning `model.bias = np.ones((10, 10))` lets DETECTOR_SIZE_Y/X resolve
# to 10 for every other extension on that model, with nothing hardcoded
# about "bias" being special; (2) each symbol's own independent source, if
# it has one (DETECTOR_SIZE_Y/X fall back to the constructor's `shape=`
# override, then config.DETECTOR_SIZE; N_ORDERS to
# meta.order_start/order_end; N_FIBERS to config.num_fibers). N_READS has
# no independent source -- it's only ever resolvable via inference.


# Common Image HDUs
def IMAGE_HDU(
    dtype : type[np.generic] | tuple[type[np.generic], ...],
    shape : tuple | None = None,
    default: float | int | None = None,
    enumerate: str | None = None,
    title: str | None = None,
    units: str | None = None,
) -> dict:
    """
    Build an image extension descriptor.

    Parameters
    ----------
    dtype
        Numpy dtype of the array (np.float64, np.int32, etc.). For an
        extension with more than one legitimate dtype (e.g. RampModel.DATA,
        uint16 for real detector data vs. float32 for simulations run with
        ``convert_to_uint16=False``), pass a precedence-ordered tuple or a
        ``X | Y`` union (e.g. ``np.uint16 | np.float32``); the first entry is
        canonical -- used to build auto-initialized default arrays, and as the
        dtype-correction target when the actual array matches none of them.
        A single dtype is stored as-is (not wrapped in a tuple).
    shape : tuple | None, optional
        The array's shape, as a tuple of literal ints and/or the symbolic
        axis-name strings ``'DETECTOR_SIZE_Y'``, ``'DETECTOR_SIZE_X'``,
        ``'N_ORDERS'``, ``'N_FIBERS'``, ``'N_READS'``, ``'ANY'`` (matched
        case-insensitively, but write them uppercase by convention), e.g.
        ``('N_ORDERS', 'DETECTOR_SIZE_Y')`` or ``('N_READS', 'DETECTOR_SIZE_Y', 'DETECTOR_SIZE_X')``.
        Axis order is always Y before X. Resolved per-instance by
        :meth:`DataModel._resolve_shape`. May be omitted (``None``) when this
        extension's real shape is always identical to the model's primary
        extension's (see :attr:`DataModel.primary_extension_name`) -- e.g.
        ``KTC`` next to ``BIAS``, or ``RATE_ERR`` next to ``RATE`` -- in which
        case it's filled in from the primary's own ``shape=`` at construction
        time instead of being repeated here (see
        :meth:`DataModel._populate_shape_info`). Only useful on a model
        whose ``primary_extension_name`` is knowable without first resolving
        any extension's shape (the base first-declared-key default); a model
        with a dynamic per-fiber override (returning ``None`` until some
        fiber's data is set) can't use this and must always give its own
        ``shape=``.
    default : float | int | None, optional
        Fill value used to lazily auto-initialize the array to
        ``np.full(resolved_shape, default, dtype=dtype)`` the first time the
        attribute is accessed, once every axis of ``shape`` resolves to a
        concrete size (an unresolvable axis --
        e.g. containing ``ANY``, or a symbol with no source available on this
        model -- means this automatic path doesn't apply). Defaults to
        ``None``. Regardless of whether ``default`` is given, building the
        array falls to :meth:`DataModel.make_default_array` whenever the
        automatic path doesn't apply -- override it in a subclass for
        extensions needing custom construction (e.g. polynomial coefficients,
        whose axis sizes are a per-instance fit result with no symbol at all).
        With no ``default`` and no such override, the attribute stays ``None``
        until explicitly assigned.
    enumerate : str | None, optional
        If given, declare this extension once (without a suffix) and let
        ``DataModel._get_extensions`` expand it into one
        ``<NAME>_<enumerate>{n}`` entry per fiber (the format is always
        ``prefix_{enumerate}#``), for ``n in 1..model.num_fibers``.
    title : str | None, optional
        Short human-readable description, purely for documentation -- shown
        as the "Description" column of the ``Extensions`` table that
        :meth:`DataModel.extensions_table` generates from ``EXTENSIONS`` and
        splices into the model class's docstring. Never read by any
        auto-init or I/O logic.
    units : str | None, optional
        Physical units, purely for documentation -- shown as the "Units"
        column of the same generated table. Never read by any auto-init or
        I/O logic.
    """
    if isinstance(dtype, types.UnionType):
        dtype = dtype.__args__
    info = {'hdu_type': fits.ImageHDU, 'dtype': dtype, 'shape': shape}
    if default is not None:
        info['default'] = default
    if enumerate is not None:
        info['enumerate'] = enumerate
    if title is not None:
        info['title'] = title
    if units is not None:
        info['units'] = units
    return info


def TABLE_HDU(
    columns: dict,
    enumerate: str | None = None,
    title: str | None = None,
    units: str | None = None,
) -> dict:
    """
    Build a table extension descriptor.

    Parameters
    ----------
    columns : dict
        Mapping of column name -> dtype (e.g. ``str``, ``np.float64``, ``object``
        for a variable-length array per row, as used by the ``CCF`` column).
    enumerate : str | None, optional
        If given, declare this extension once (without a suffix) and let
        ``DataModel._get_extensions`` expand it into one
        ``<NAME>_<enumerate>{n}`` entry per fiber (the format is always
        ``prefix_{enumerate}#``), for ``n in 1..model.config.num_fibers``.
        Table extensions are always eagerly initialized as an empty (0-row)
        Table at datamodel construction, regardless of this flag.
    title : str | None, optional
        Short human-readable description, purely for documentation -- see
        :func:`IMAGE_HDU`'s ``title``.
    units : str | None, optional
        Physical units, purely for documentation -- see :func:`IMAGE_HDU`'s
        ``units``.
    """
    info = {'hdu_type': fits.BinTableHDU, 'columns': columns}
    if enumerate is not None:
        info['enumerate'] = enumerate
    if title is not None:
        info['title'] = title
    if units is not None:
        info['units'] = units
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

# DQ_RAW is only ever combined with 'DATA' (shape=(N_READS, DETECTOR_SIZE_Y,
# DETECTOR_SIZE_X)) as the primary extension (see RampModel), so its own
# identical shape is inferred rather than repeated (see
# DataModel._populate_shape_info).
DQ_RAW = {
    'DQ_RAW': IMAGE_HDU(
        dtype=np.uint8,
        default=0,
        title='Data quality flags unique to each pixel and read.',
        shape=('N_READS', 'DETECTOR_SIZE_Y', 'DETECTOR_SIZE_X'),
    )
}

#### L0 (rate maps)

DQ_L0 = {
    'DQ': IMAGE_HDU(
        dtype=np.uint32,
        shape=('DETECTOR_SIZE_Y', 'DETECTOR_SIZE_X'),
        default=0,
        title='Data quality.'
    )
}

def create_L0_extensions(name : str = 'RATE', title : str = 'Rate map.', units : str | None = 'e-/s') -> dict[str, dict]:
    # name (the primary) declares shape= explicitly; its siblings below infer
    # it (see DataModel._populate_shape_info) since it's always identical.
    var_units = _units_squared(units)
    base = _base_title(title)
    ext = {
        f'{name}': IMAGE_HDU(
            dtype=np.float32,
            shape=('DETECTOR_SIZE_Y', 'DETECTOR_SIZE_X'),
            default=0,
            title=title,
            units=units
        ),
        f'{name}_ERR': IMAGE_HDU(
            dtype=np.float32,
            default=0,
            title=f'{base} error (all sources).',
            units=units
        ),
        f'{name}_VAR_RNOISE': IMAGE_HDU(
            dtype=np.float32,
            default=0,
            title=f'{base} variance (read noise).',
            units=var_units
        ),
        f'{name}_VAR_POISSON': IMAGE_HDU(
            dtype=np.float32,
            default=0,
            title=f'{base} variance (Poisson).',
            units=var_units
        ),
    }
    ext.update(DQ_L0)
    ext['MID_TIME'] = IMAGE_HDU(
        dtype=np.float64,
        default=0,
        title='Exposure mid-point time for each pixel.',
        units='MJD'
    )
    return ext


#### Spectral level extensions (per-fiber)
#
# Each of these is declared ONCE, without a fiber suffix, using `enumerate='FIB'`.
# `DataModel._get_extensions` expands each into `<NAME>_FIB{n}` for `n in 1..model.config.num_fibers` when a concrete model's EXTENSIONS includes it.

def _units_squared(units : str | None) -> str | None:
    return f'({units})^2' if units else None


def _base_title(title : str | None) -> str | None:
    """Strip a trailing '.' so a derived title (e.g. '{base} error.') doesn't get a stray one mid-sentence."""
    return title[:-1] if title and title.endswith('.') else title


def create_dataspec_extensions(
    prefix : str = 'SPEC',
    title : str | None = None,
    units : str | None = None,
    **kwargs
) -> dict[str, dict]:
    var_units = _units_squared(units)
    base = _base_title(title)
    return {
        prefix: IMAGE_HDU(
            dtype=np.float32,
            shape=('N_ORDERS', 'DETECTOR_SIZE_Y'),
            default=0,
            enumerate='FIB',
            title=title,
            units=units,
            **kwargs
        ),
        f'{prefix}_ERR': IMAGE_HDU(
            dtype=np.float32,
            shape=('N_ORDERS', 'DETECTOR_SIZE_Y'),
            default=0,
            enumerate='FIB',
            title=base and f'{base} error.',
            units=units,
            **kwargs
        ),
        f'{prefix}_VAR_POISSON': IMAGE_HDU(
            dtype=np.float32,
            shape=('N_ORDERS', 'DETECTOR_SIZE_Y'),
            default=0,
            enumerate='FIB',
            title=base and f'{base} variance (Poisson).',
            units=var_units,
            **kwargs
        ),
        f'{prefix}_VAR_RNOISE': IMAGE_HDU(
            dtype=np.float32,
            shape=('N_ORDERS', 'DETECTOR_SIZE_Y'),
            default=0,
            enumerate='FIB',
            title=base and f'{base} variance (read noise).',
            units=var_units,
            **kwargs
        ),
    }

def create_specdq_extensions(title : str = 'Data quality.', prefix : str = 'SPEC', **kwargs) -> dict[str, dict]:
    return {
        f'{prefix}_DQ': IMAGE_HDU(
            dtype=np.uint32,
            shape=('N_ORDERS', 'DETECTOR_SIZE_Y'),
            default=0,
            enumerate='FIB',
            title=title,
            **kwargs
        )
    }

def create_sky_extensions(title : str | None = None, units : str | None = None, **kwargs) -> dict[str, dict]:
    base = _base_title(title)
    return {
        'SKY': IMAGE_HDU(
            dtype=np.float32,
            shape=('N_ORDERS', 'DETECTOR_SIZE_Y'),
            default=0,
            enumerate='FIB',
            title=title,
            units=units,
            **kwargs
        ),
        'SKY_ERR': IMAGE_HDU(
            dtype=np.float32,
            shape=('N_ORDERS', 'DETECTOR_SIZE_Y'),
            default=0,
            enumerate='FIB',
            title=base and f'{base} error.',
            units=units,
            **kwargs
        ),
    }

def create_trans_extensions(title : str = 'Transmission correction.', **kwargs) -> dict[str, dict]:
    base = _base_title(title)
    return {
        'TRANS': IMAGE_HDU(
            dtype=np.float32,
            shape=('N_ORDERS', 'DETECTOR_SIZE_Y'),
            default=1,
            enumerate='FIB',
            title=title,
            **kwargs
        ),
        'TRANS_ERR': IMAGE_HDU(
            dtype=np.float32,
            shape=('N_ORDERS', 'DETECTOR_SIZE_Y'),
            default=1,
            enumerate='FIB',
            title=f'{base} error.',
            **kwargs
        ),
    }

def create_throughput_extensions(**kwargs) -> dict[str, dict]:
    return {
        'THROUGHPUT': IMAGE_HDU(
            dtype=np.float32,
            shape=('N_ORDERS', 'DETECTOR_SIZE_Y'),
            default=1,
            enumerate='FIB',
            **kwargs
        )
    }

def create_wave_extensions(**kwargs) -> dict[str, dict]:
    return {
        'WAVE': IMAGE_HDU(
            dtype=np.float64,
            shape=('N_ORDERS', 'DETECTOR_SIZE_Y'),
            default=0,
            enumerate='FIB',
            units='micron',
            **kwargs,
        )
    }

CHEBYSHEV_NORM_HDU = TABLE_HDU({
    'PIXEL_NORM_MIN': np.float64,
    'PIXEL_NORM_MAX': np.float64,
    'ORDER_NORM_MIN': np.int16,
    'ORDER_NORM_MAX': np.int16,
}, enumerate='FIB', title='Chebyshev pixel/order normalization bounds.')

def create_chebyshev_extensions() -> dict[str, dict]:
    return {
        # Real shape is (n_coeffs_intra_order, n_coeffs_inter_order) -- a
        # per-instance fit result with no symbol, hence ANY for both axes and
        # no default (not auto-initialized without a make_default_array override).
        'CHEB_COEFFS': IMAGE_HDU(
            dtype=np.float64,
            shape=('ANY', 'ANY'),
            enumerate='FIB',
            title='Chebyshev coefficients for the wavelength solution.'
        ),
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
}, enumerate='FIB', title='Drift correction table for each fiber.')

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
}, enumerate='FIB', title='Etalon fit parameters for each mode, order, and fiber.')

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
}, enumerate='FIB', title='LFC fit parameters for each LFC mode, order, and fiber.')

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
}, enumerate='FIB', title='Per-chunk RVs and raw CCF(v) (stellar RV).')

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
}, enumerate='FIB', title='RVs aggregated across chunks (CCF-based).')

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
}, enumerate='FIB', title='Per-chunk RVs (first-moment method).')

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
}, enumerate='FIB', title='RVs aggregated across chunks (first-moment method).')

def create_rv_extensions() -> dict[str, dict]:
    return {
        'RVCCF': RVCCF_HDU,
        'RVCCFAVG': RVCCFAVG_HDU,
        'RVFWM': RVFWM_HDU,
        'RVFWMAVG': RVFWMAVG_HDU,
    }


#### Telluric models

def create_telluric_extensions(title : str = 'Telluric model spectrum.') -> dict[str, dict]:
    return {
        'TEL': IMAGE_HDU(
            dtype=np.float32,
            shape=('N_ORDERS', 'DETECTOR_SIZE_Y'),
            default=1,
            enumerate='FIB',
            title=title
        ),
        'TEL_ERR': IMAGE_HDU(
            dtype=np.float32,
            shape=('N_ORDERS', 'DETECTOR_SIZE_Y'),
            default=0,
            enumerate='FIB',
            title=f'{_base_title(title)} error (all sources).'
        ),
    }


#### Stellar models

def create_starmodel_extensions(title : str = 'Stellar model spectrum.', units : str | None = None) -> dict[str, dict]:
    return {
        'STAR_MOD' : IMAGE_HDU(
            dtype=np.float32,
            shape=('N_ORDERS', 'DETECTOR_SIZE_Y'),
            default=0,
            enumerate='FIB',
            title=title,
            units=units
        )
    }


#### Absolute flux calibration

def create_fluxabs_extensions(units : str | None = 'W/m^2/micron') -> dict[str, dict]:
    var_units = _units_squared(units)
    return {
        'FLUX_ABS': IMAGE_HDU(
            dtype=np.float32,
            shape=('N_ORDERS', 'DETECTOR_SIZE_Y'),
            default=0,
            enumerate='FIB',
            title='Spectrum in absolute flux units.',
            units=units
        ),
        'FLUX_ABS_ERR': IMAGE_HDU(
            dtype=np.float32,
            shape=('N_ORDERS', 'DETECTOR_SIZE_Y'),
            default=0,
            enumerate='FIB',
            title='Absolute spectrum error (all sources).',
            units=units
        ),
        'FLUX_ABS_VAR_POISSON': IMAGE_HDU(
            dtype=np.float32,
            shape=('N_ORDERS', 'DETECTOR_SIZE_Y'),
            default=0,
            enumerate='FIB',
            title='Absolute spectrum variance (Poisson).',
            units=var_units
        ),
        'FLUX_ABS_VAR_RNOISE': IMAGE_HDU(
            dtype=np.float32,
            shape=('N_ORDERS', 'DETECTOR_SIZE_Y'),
            default=0,
            enumerate='FIB',
            title='Absolute spectrum variance (read noise).',
            units=var_units
        ),
        'FLUX_ABS_DQ': IMAGE_HDU(
            dtype=np.uint32,
            shape=('N_ORDERS', 'DETECTOR_SIZE_Y'),
            default=0,
            enumerate='FIB',
            title='Absolute spectrum data quality.'
        ),
    }


#### Barycentric correction

BARYCORR_HDU = TABLE_HDU({
    'MJDUTC' : np.float64,
    'MJDUTCW' : np.float64,
    'MBJD' : np.float64,
    'MBJDW' : np.float64,
    'BERV' : np.float64,
    'BERVW' : np.float64,
}, enumerate='FIB', title='Barycentric correction information.')

def create_midtime_extensions(title : str = 'Exposure mid-point times.', units : str = 'MJD') -> dict[str, dict]:
    return {'MID_TIME' : IMAGE_HDU(
        dtype=np.float64,
        shape=('N_ORDERS', 'DETECTOR_SIZE_Y'),
        default=0,
        enumerate='FIB',
        title=title,
        units=units
    )}

def create_barycorr_extensions() -> dict[str, dict]:
    return {'BARYCORR' : BARYCORR_HDU}

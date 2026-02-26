from __future__ import annotations
from astropy.io import fits
import os
from typing import TYPE_CHECKING
import astropy.time
from datetime import datetime
if TYPE_CHECKING:
    from ..datamodels import DataModel

from koa_middleware.utils import generate_koa_filehandle

import logging
logger = logging.getLogger(__name__)


def open(
    init,
    copy : bool = False,
    config : type | None = None,
    **kwargs
) -> 'DataModel':
    """
    Creates a DataModel from a FITS file.
    Automatically select the right DataModel class based on the 'DATAMODL' header keyword.

    Parameters
    ----------
    init
        filepath (str): Full path to FITS file.
        DataModel: Return the given DataModel instance or make a copy if ``copy=True``.
    copy : bool
        If True, return a copy of the DataModel instance.
        If ``init`` is a filepath, this is ignored.
    kwargs:
        Additional keyword arguments for the DataModel constructor.
    
    Returns
    -------
    DataModel
        The specific DataModel instance.
    """

    # If init is already a datamodel, copy and return
    from .base_model import DataModel
    if isinstance(init, DataModel):
        if copy:
            return init.copy()
        else:
            return init
    
    # Convert path to string
    if isinstance(init, str):
        with fits.open(init, memmap=True) as hdulist:
            model_class = get_model_type(hdulist)
            if config is None:
                config = get_instrument_config(hdulist)
        model = model_class(filepath=init, config=config, **kwargs)
    elif isinstance(init, fits.HDUList):
        model_class = get_model_type(init)
        if config is None:
            config = get_instrument_config(init)
        model = model_class(filepath=init, config=config, **kwargs)
    else:
        raise TypeError(f"init must be a file path string or a FITS HDUList, got {type(init).__name__}")

    # Return the model
    return model

def get_model_type(input: str | fits.HDUList):
    """
    Get the data model class from a FITS file path, HDUList, or model name string.

    Parameters
    ----------
    input : str or fits.HDUList
        - FITS HDUList: extracts DATAMODL from primary header
        - str ending in '.fits' or '.fit': interpreted as FITS file path
        - Other str: interpreted as model name

    Returns
    -------
    model_class : type
        The DataModel class.

    Raises
    ------
    ValueError
        If DATAMODL key is missing in the FITS header, or model name not found.
    TypeError
        If source is not HDUList or str.
    """
    from . import MODEL_REGISTER

    if isinstance(input, fits.HDUList):
        model_name = input[0].header.get("DATAMODL")
        if model_name is None:
            raise ValueError("No DATAMODL key found in primary FITS header.")

    elif isinstance(input, str):
        if os.path.isfile(input):
            with fits.open(input) as hdulist:
                model_name = hdulist[0].header.get("DATAMODL")
                if model_name is None:
                    raise ValueError(f"No DATAMODL key found in primary FITS header: {input}")
        else:
            # Treat as model name directly
            model_name = input

    else:
        raise TypeError("input must be a FITS file path, HDUList, or model name string.")

    model_class = MODEL_REGISTER.get(model_name)
    if model_class is None:
        raise ValueError(f"Model type '{model_name}' not found in MODEL_REGISTER.")

    return model_class

def get_instrument_config(
    input : str | fits.HDUList | None = None,
    instrument_name : str | None = None
) -> type:
    """
    Get the instrument configuration class from either a FITS filename or an HDUList.
    
    Parameters
    ----------
    input : str or fits.HDUList, optional
        Either a FITS filename path or an already-opened HDUList object.
        The INSTRUME keyword will be read from the primary header.
    instrument_name : str, optional
        The instrument name. If provided, this will be used instead of reading the instrument name from the FITS header.
    
    Returns
    -------
    config_class : type
        The instrument config class if found.
    
    Raises
    ------
    TypeError
        If input is neither a string nor an HDUList.
    ValueError
        If no INSTRUME keyword is found in the header, or if the instrument
        config is not found in INSTRUMENT_CONFIGS.
    """
    from . import INSTRUMENT_CONFIGS

    if instrument_name is None:
        assert input is not None, "Either input or instrument_name must be provided."
        instrument_name = get_instrument_name(input)
    
    if instrument_name is not None:
        config_class = INSTRUMENT_CONFIGS.get(instrument_name.lower(), None)
    else:
        config_class = None
    
    return config_class

def get_instrument_name(input : str | fits.HDUList):
    """
    Get the instrument name from either a FITS filename or an HDUList.
    
    Parameters
    ----------
    input : str or fits.HDUList
        Either a FITS filename path or an already-opened HDUList object.
        The INSTRUME keyword will be read from the primary header.
    
    Returns
    -------
    instrument_name : str
        The instrument name if found.
    
    Raises
    ------
    TypeError
        If input is neither a string nor an HDUList.
    ValueError
        If no INSTRUME keyword is found in the header.
    """
    if isinstance(input, fits.HDUList):
        instrument_name = input[0].header.get("INSTRUME")

    elif isinstance(input, str):
        if not os.path.isfile(input):
            raise ValueError(f"File does not exist: {input}")
        with fits.open(input) as hdulist:
            instrument_name = hdulist[0].header.get("INSTRUME")

    else:
        raise TypeError("input must be a FITS file path or HDUList.")

    return instrument_name


    
    # # Open the file or use the provided HDUList
    # if isinstance(input, str):
    #     with fits.open(input, memmap=True) as hdulist:
    #         instrument_name = hdulist[0].header.get('INSTRUME')
    # elif isinstance(input, fits.HDUList):
    #     instrument_name = input[0].header.get('INSTRUME')
    # else:
    #     raise TypeError(
    #         f"input must be a string (filename) or fits.HDUList, got {type(input).__name__}"
    #     )
    
    # # Check if INSTRUME keyword was found
    # if instrument_name is None:
    #     raise ValueError("No INSTRUME key found in FITS header")
    
    # return instrument_name

def group_by_illuminated_fiber(input: list[str | 'DataModel']) -> dict[str, object]:
    """
    Associate input models by fiber.

    Parameters
    ----------
    input : list[str | DataModel]
        List of file paths or DataModel instances.

    Returns
    -------
    fiber_map : dict[str, object]
        Dictionary mapping fiber identifiers to DataModel instances.
    """
    fiber_map = {}
    input_models = [open(f, meta_only=True) for f in input]

    for i, model in enumerate(input_models):
        for fiber in range(1, model.num_fibers + 1):
            val = getattr(model.meta, f'object_fiber{fiber}', None)
            if val is None or val == 'BLOCKED':
                continue  # skip blocked fibers
            key = str(fiber)
            if key in fiber_map:
                raise ValueError(f"Multiple models for fiber {key}")
            fiber_map[key] = input[i]

    return fiber_map

def group_by_fiber_combination(input: list[str | 'DataModel']) -> dict[tuple, list]:
    """
    Group input models by like-objects in each fiber.

    Parameters
    ----------
    input : list[str | DataModel]
        List of file paths or DataModel instances.

    Returns
    -------
    inputs_grouped : dict[tuple[str, ...], list[data]]
        Dictionary with keys as tuples of object names per fiber and values as lists of corresponding input models.
        Example:

            >>> inputs_grouped = {
                ('SCIENCE', 'SKY', 'BLOCKED'): [model1, model2],
                ('BLOCKED', 'SCIENCE', 'SKY') : [model3, model4]
            }

    """
    inputs_grouped = {}
    input_models = [open(f, meta_only=True) for f in input]

    ref_set = None

    for i, model in enumerate(input_models):
        objs = []
        for fiber in range(1, model.num_fibers + 1):
            val = getattr(model.meta, f'object_fiber{fiber}', None)
            objs.append('BLOCKED' if val is None or val == 'BLOCKED' else val)

        objs = tuple(objs)
        objs_set = frozenset(o for o in objs if o != 'BLOCKED')

        if ref_set is None:
            ref_set = objs_set
        elif objs_set != ref_set:
            raise ValueError(f"Inconsistent illuminated fiber types: {ref_set} vs {objs_set}")

        inputs_grouped.setdefault(objs, []).append(input[i])

    return inputs_grouped

def _set_datetime_meta(model : 'DataModel'):
    if model.meta.datetime_obs is None and model.meta.mjd_start is None:
        logger.warning("No time info is present, adopting current time for datetime_obs and mjd_start.")
        model.meta.datetime_obs = datetime.now().isoformat(timespec="milliseconds")
        model.meta.mjd_start = astropy.time.Time(model.meta.datetime_obs, format="isot", scale="utc").mjd
    elif model.meta.datetime_obs is not None and model.meta.mjd_start is None:
        dt = astropy.time.Time(model.meta.datetime_obs, format="isot", scale="utc")
        model.meta.mjd_start = dt.mjd
    elif model.meta.datetime_obs is None and model.meta.mjd_start is not None:
        model.meta.datetime_obs = astropy.time.Time(model.meta.mjd_start, format="mjd", scale="utc").isot

    dt = astropy.time.Time(model.meta.datetime_obs, format="isot", scale="utc")
    if model.meta.koa_date_obs is None:
        model.meta.koa_date_obs = dt.strftime("%Y-%m-%d")
    if model.meta.koa_utc_obs is None:
        model.meta.koa_utc_obs = dt.strftime("%H:%M:%S.%f").rstrip("0").rstrip(".")

def _set_koa_meta(model: 'DataModel') -> 'DataModel':
    model.meta.koa_id = model.meta.filename
    model.meta.koa_filepath = generate_koa_filehandle(
        model.meta.instrument_name,
        model.meta.datetime_obs,
        model.meta.koa_id
    )
    return model

def get_instrument_alias(instrument_name : str, spectrograph : str | None = None) -> str:
    """
    Get the instrument alias for a given instrument name.

    Parameters
    ----------
    instrument_name : str
        The instrument name.

    Returns
    -------
    instrument_alias : str
        The instrument alias.
    """
    config = get_instrument_config(instrument_name=instrument_name)
    return config.get_instrument_alias(spectrograph=spectrograph)


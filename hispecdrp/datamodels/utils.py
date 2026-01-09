from __future__ import annotations
from astropy.io import fits
from .meta_data import MetaModel
import numpy as np
import warnings


def open(init, copy : bool = False, config : type | None = None, **kwargs):
    """
    Creates a DataModel from a FITS file. Automatically select the right DataModel class based on the `DATAMODL` header keyword.
    More types might be added later.

    Args:
        init:
            filepath: Full path to FITS file.
            DataModel: Return the given DataModel instance or make a copy if `copy=True`.
        copy (bool): If True, return a copy of the DataModel instance. If `init` is a `str`, this is ignored.
        kwargs: Additional keyword arguments for the DataModel constructor.
    
    Returns:
        DataModel : The specific DataModel instance.
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
            model_class = get_model_type_from_hdulist(hdulist)
            if config is None:
                config = get_config_from_hdulist(hdulist)
        model = model_class(filepath=init, config=config, **kwargs)
    elif isinstance(init, fits.HDUList):
        model_class = get_model_type_from_hdulist(init)
        if config is None:
            config = get_config_from_hdulist(init)
        model = model_class(filepath=init, config=config, **kwargs)

    # Return the model
    return model


    
def get_config_from_hdulist(hdu_list : fits.HDUList):
    """
    Get the instrument config class associated with a filepath.

    Parameters
    ----------
    hdu_list : fits.HDUList
        Input HDU list.

    Returns
    -------
    config_class : type
        The config class if found.
    """
    primary = hdu_list[0]
    instrument_name = primary.header.get('INSTRUME')
    if instrument_name is None:
        raise ValueError(f"No INSTRUME key found in FITS header: {instrument_name}")
    else:
        from . import INSTRUMENT_CONFIGS
        config_class = INSTRUMENT_CONFIGS.get(instrument_name, None)
        if config_class is None:
            raise ValueError(f"Instrument config for '{instrument_name}' not found.")
        return config_class


def get_model_type_from_hdulist(hdu_list : fits.HDUList):
    """
    Get the model type from the primary header, lookup to get class

    Parameters
    ----------
    hdu_list : fits.HDUList
        The initializer.

    Returns
    -------
    model_class : type
        The name of the model class if found.
    """
    primary = hdu_list[0]
    model_type = primary.header.get('DATAMODL')
    if model_type is None:
        raise ValueError(f"No DATAMODL key found in FITS header: {model_type}")
    else:
        return get_model_type_from_name(model_type)


def get_model_type_from_name(model_name : str):
    """
    Get the model type from the primary header, lookup to get class

    Parameters
    ----------
    model_name : str
        Name of the data model class (e.g. 'L0Model').

    Returns
    -------
    model_class : type
        The DataModel class.
    """
    from . import MODEL_REGISTER
    model_class = MODEL_REGISTER.get(model_name, None)
    if model_class is None:
        raise ValueError(f"Model type '{model_name}' not found in MODEL_REGISTER.")
    else:
        return model_class


def associate_fibers(input: list[str | 'DataModel']) -> dict[str, object]:
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


def group_input_by_fiber_object(input: list[str | 'DataModel']) -> dict[tuple, list]:
    """
    Group input models by objects in each fiber.

    Parameters
    ----------
    input : list[str | DataModel]
        List of file paths or DataModel instances.

    Returns
    -------
    inputs_grouped : dict[str, object]
        Dictionary with keys as tuples of object names per fiber and values as lists of corresponding input models.
        Example: {('SCIENCE', 'SKY', 'BLOCKED'): [model1, model2], ('BLOCKED', 'SCIENCE', 'SKY') : [model3, model4]}
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


def merge_meta(models: list['DataModel']) -> MetaModel:
    """
    Merge the meta information from multiple DataModel instances.

    Parameters
    ----------
    models : list[DataModel]
        List of DataModel instances to merge meta information from.

    Returns
    -------
    meta : MetaModel
        Merged metadata model.
    """
    meta_class = models[0].meta.__class__
    merged_data = {}

    for key, field in meta_class.model_fields.items():
        values = [getattr(model.meta, key) for model in models]
        if not values:
            continue
        merge_rule = field.json_schema_extra.get('merge_rule', 'first') or 'first'
        if merge_rule == 'drop':
            merged_value = None
        elif merge_rule == 'first':
            merged_value = values[0]
        elif merge_rule == 'last':
            merged_value = values[-1]
        elif merge_rule == 'sum':
            merged_value = np.sum(np.array(values).astype(float))
        elif merge_rule == 'max':
            merged_value = np.max(np.array(values).astype(float))
        elif merge_rule == 'min':
            merged_value = np.min(np.array(values).astype(float))
        elif merge_rule == 'mean':
            merged_value = np.mean(np.array(values).astype(float))
        elif merge_rule == 'coalesce':
            first_non_none = next((v for v in values if v is not None), None)
            other_values = [v for v in values if v is not None and v != first_non_none]
            if other_values:
                warnings.warn("Multiple differing non-None values detected, using the first one")
            merged_value = first_non_none
        elif merge_rule == 'min_datetime':
            datetime_values = [np.datetime64(v) for v in values if v is not None]
            merged_value = str(np.min(np.array(datetime_values))) if datetime_values else None
        elif merge_rule == 'max_datetime':
            datetime_values = [np.datetime64(v) for v in values if v is not None]
            merged_value = str(np.max(np.array(datetime_values))) if datetime_values else None
        elif merge_rule == 'mean_datetime':
            datetime_values = [np.datetime64(v) for v in values if v is not None]
            merged_value = str(np.mean(np.array(datetime_values))) if datetime_values else None
        elif callable(merge_rule):
            merged_value = merge_rule(values)
        else:
            raise ValueError(f"Unknown merge rule '{merge_rule}' for meta field '{key}'")
        merged_data[key] = merged_value
    return meta_class(**merged_data)
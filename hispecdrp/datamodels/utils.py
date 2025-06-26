from astropy.io import fits


def open(init, copy : bool = False, **kwargs):
    """
    Creates a DataModel from a FITS file. More types might be added later.

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
            model_class = model_type_from_hdu(hdulist)
        model = model_class(filepath=init, **kwargs)
    elif isinstance(init, fits.HDUList):
        model_class = model_type_from_hdu(init)
        model = model_class(hdulist=init, **kwargs)

    # Return the model
    return model


def model_type_from_hdu(hdu_list : fits.HDUList):
    """
    Get the model type from the primary header, lookup to get class

    Args:
        init (HDUList): The initializer.

    Returns:
        type: The DataModel class if found.
    """
    primary = hdu_list[0]
    model_type = primary.header.get('DATAMODL')
    if model_type is None:
        raise ValueError(f"No DATAMODL key found in FITS header: {model_type}")
    else:
        return model_type_from_name(model_type)


def model_type_from_name(model_type : str):
    """
    Get the model type from the primary header, lookup to get class

    Args:
        model_type (str): Name of the data model class.

    Returns:
        type: The DataModel class.
    """
    from . import MODEL_REGISTER
    model_class = MODEL_REGISTER.get(model_type, None)
    if model_class is None:
        raise ValueError(f"Model type '{model_type}' not found in MODEL_REGISTER.")
    else:
        return model_class

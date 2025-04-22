from astropy.io import fits


def open(init, copy : bool = False, **kwargs):
    """
    Creates a DataModel from a FITS file. More types might be added later.

    Parameters:
        init:
            - filepath: Full path to FITS file.
            - DataModel: Return the given DataModel instance or make a copy if `copy=True`.
        copy (bool): If True, return a copy of the DataModel instance. If `init` is a `str`, this is ignored.
        **kwargs: Additional keyword arguments for the DataModel constructor.
    
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
            model_class = class_from_model_type(hdulist)
        model = model_class(filepath=init, **kwargs)
    elif isinstance(init, fits.HDUList):
        model_class = class_from_model_type(init)
        model = model_class(hdulist=init, **kwargs)

    # Return the model
    return model


def class_from_model_type(hdu_list : fits.HDUList):
    """
    Get the model type from the primary header, lookup to get class

    Parameters:
        init (HDUList): The initializer.

    Returns:
        type | None: The DataModel class if found.
    """

    from . import MODEL_REGISTER
    primary = hdu_list[0]
    model_type = primary.header.get('DATAMODL')
    if model_type is None:
        return None
    else:
        try:
            model_class = MODEL_REGISTER[model_type]
        except KeyError:
            raise ValueError(f"Unknown model type: {model_type}")

    # Return the class
    return model_class


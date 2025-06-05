import astropy.time

from .parvi_base import PARVIDataModel

__all__ = ['get_parvi_meta']


def get_parvi_meta(model : PARVIDataModel, meta : dict | None = None):
    """
    Populate certain model metadata defaults for development and testing data for now.

    Args:
        model (datamodels.DataModel): The data model to update.
    """

    if meta is not None:
        update_model_meta(model, meta)

    # Add exposure metadata
    get_parvi_exposure_meta(model)

    # Date time
    if model.meta.date_obs is None:
        dt = astropy.time.Time(model.meta.exposure.datetime_start, format='isot')
        model.meta.date_obs = dt.strftime('%Y%m%d')
        model.meta.utc = dt.strftime('%H:%M:%S.%f')
    
    # Reference models
    if hasattr(model, 'CAL_TYPE'):
        model.meta.cal_type = model.CAL_TYPE
        if model.meta.cal_type is None:
            model.meta.master_cal = True
    return model


def get_parvi_exposure_meta(model : PARVIDataModel):
    if model.meta.exposure.datetime_start is None and model.meta.exposure.mjd_start is None:
        model.meta.exposure.datetime_start = '2024-09-24T00:00:00.000'  # Default value
    if model.meta.exposure.datetime_start is None and model.meta.exposure.mjd_start is not None:
        dt = astropy.time.Time(model.meta.exposure.mjd_start, format='mjd')
        model.meta.exposure.datetime_start = dt.isot
    elif model.meta.exposure.datetime_start is not None and model.meta.exposure.mjd_start is None:
        dt = astropy.time.Time(model.meta.exposure.datetime_start, format='isot')
        model.meta.exposure.mjd_start = dt.mjd
    if model.meta.exposure.exposure_type is None:
        model.meta.exposure.exposure_type = 'SCI'
    if model.meta.exposure.read_mode is None:
        model.meta.exposure.read_mode = 'DEFAULT'
    if model.meta.exposure.data_quality is None:
        model.meta.exposure.data_quality = 'GOOD'
    return model


def update_model_meta(model : PARVIDataModel, meta : dict):
    for key, value in meta.items():
        attrs = key.split('.')
        target = model.meta
        for attr in attrs[:-1]:
            if hasattr(target, attr):
                target = getattr(target, attr)
            else:
                target is None
        if target is not None:
            setattr(target, attrs[-1], value)
import numpy as np
from hispecdrp.datamodels import RampModel, L0Model, DQ_FLAGS

def add_refchan_ramp(
    model : RampModel,
    channel_size : int = 64,
    value: float = 0,
):
    """
    Extend ramp data arrays to include ref channel on right side.
    
    Parameters
    ----------
    model : RampModel
        Input ramp data model.
    channel_size : int, optional
        Number of columns in each readout channel, by default 64.
    value : float, optional
        Constant value used to fill the reference channel, by default 0.
    """

    # DATA ext
    pad_width = ((0, 0), (0, 0), (0, channel_size))
    model.data = np.pad(model.data, pad_width, mode="constant", constant_values=value)

    # DQ RAW ext
    if model.dq_raw is not None:
        model.dq_raw = np.pad(model.dq_raw, pad_width, mode="constant", constant_values=0)


def set_refpix_L0(
    model : L0Model,
    channel_size : int = 64,
    n_refpixs : int | None = None
):
    """
    Extend L0 data arrays to include ref channel on right side.
    
    Parameters
    ----------
    model : L0Model
        Input L0 data model.
    channel_size : int, optional
        Number of columns in each readout channel, by default 64.
    value : float, optional
        Constant value used to fill the reference channel, by default 0.
    """

    def pad_array(arr, value):
        if arr is None:
            return arr
        if arr.shape[0] == arr.shape[1]:
            return np.pad(arr, pad_width=((0, 0), (0, channel_size)), mode="constant", constant_values=value)
        else:
            return arr

    # Add reference channel to rate maps
    if model.has_extension_value('rate'):
        model.rate = pad_array(model.rate, 0)
    if model.has_extension_value('rate_err'):
        model.rate_err = pad_array(model.rate_err, 0)
    if model.has_extension_value('rate_var_poisson'):
        model.rate_var_poisson = pad_array(model.rate_var_poisson, 0)
    if model.has_extension_value('rate_var_rnoise'):
        model.rate_var_rnoise = pad_array(model.rate_var_rnoise, 0)
    if model.has_extension_value('dq'):
        model.dq = pad_array(model.dq, DQ_FLAGS['REFERENCE_PIXEL'])
    if model.has_extension_value('mid_time'):
        model.mid_time = pad_array(model.mid_time, np.nan)

    # Set reference pixels on boundary
    if n_refpixs is None:
        n_refpixs = model.meta.n_refpixs or 4

    # Pad edge 4 pixels on all sides
    model.dq[:, :n_refpixs] |= DQ_FLAGS['REFERENCE_PIXEL']
    model.dq[:, -n_refpixs:] |= DQ_FLAGS['REFERENCE_PIXEL']
    model.dq[:n_refpixs, :] |= DQ_FLAGS['REFERENCE_PIXEL']
    model.dq[-n_refpixs:, :] |= DQ_FLAGS['REFERENCE_PIXEL']

    return model
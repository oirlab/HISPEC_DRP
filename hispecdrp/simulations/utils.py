import numpy as np
from .. import datamodels

def add_refchan_ramp(
    arr: np.ndarray,
    position: str,
    num_refpix: int = 64,
    value: float = 0,
):
    """
    Add a reference channel to each read of a ramp cube.
    Parameters
    ----------
    arr : np.ndarray
        Input array with shape (n_reads, ny, nx).
    position : str
        Side on which to add the reference channel, either 'left' or 'right'.
    num_refpix : int, optional
        Number of reference pixels to add along the x-axis, by default 64.
    value : float, optional
        Constant value used to fill the reference channel, by default 0.
    Returns
    -------
    np.ndarray
        Array with reference channel added along the last axis.
    Raises
    ------
    ValueError
        If the input array is not 3D or position is invalid.
    """
    pos = position.lower()
    if arr.ndim != 3:
        raise ValueError("Expected array with shape (n_reads, ny, nx)")
    if pos == "right":
        pad_width = ((0, 0), (0, 0), (0, num_refpix))
    elif pos == "left":
        pad_width = ((0, 0), (0, 0), (num_refpix, 0))
    else:
        raise ValueError("position must be 'left' or 'right'")
    return np.pad(arr, pad_width, mode="constant", constant_values=value)

def add_refchan_L0(
    arr: np.ndarray,
    position: str,
    num_refpix: int = 64,
    value: float = 0,
):
    """
    Add a reference channel to a 2D L0 frame.
    Parameters
    ----------
    arr : np.ndarray
        Input array with shape (ny, nx).
    position : str
        Side on which to add the reference channel, either 'left' or 'right'.
    num_refpix : int, optional
        Number of reference pixels to add along the x-axis, by default 64.
    value : float, optional
        Constant value used to fill the reference channel, by default 0.
    Returns
    -------
    np.ndarray
        Array with reference channel added along the last axis.
    Raises
    ------
    ValueError
        If the input array is not 2D or position is invalid.
    """
    pos = position.lower()
    if arr.ndim != 2:
        raise ValueError("Expected array with shape (ny, nx)")
    if pos == "right":
        pad_width = ((0, 0), (0, num_refpix))
    elif pos == "left":
        pad_width = ((0, 0), (num_refpix, 0))
    else:
        raise ValueError("position must be 'left' or 'right'")
    return np.pad(arr, pad_width, mode="constant", constant_values=value)


def set_init_dq_L0(model : datamodels.DataModel):
    
    shape = (4096, 4096 + 64)
    dq = np.zeros(shape, dtype=np.uint32)
    
    # Set all ref pixels to DQ=REF CHAN BIT
    ref_flag = datamodels.DQ_FLAGS['REFERENCE_PIXEL']

    n = model.meta.n_refpixs
    channel_size = 64
    
    # 4-pixel border on all sides
    dq[:n, :] |= ref_flag           # top
    dq[-n:, :] |= ref_flag          # bottom
    dq[:, :n] |= ref_flag           # left
    dq[:, -n:] |= ref_flag          # right

    pos = model.meta.ref_channel_position.lower()

    # 64-pixel wide reference channel on the right side
    if pos == 'right':
        dq[:, -channel_size:] |= ref_flag
    elif pos == 'left':
        dq[:, :channel_size] |= ref_flag

    model.dq = dq

import numpy as np
from ...datamodels.dqflags import DQ_FLAGS

def get_mid_delta_times(times, dq_raw):
    """
    Compute the mid time of the exposure accounting for the DQ flags: DO_NOT_USE, SATURATED, or JUMP_DET.

    Parameters
    ----------
    times : np.ndarray
        1D array of read times (length n_reads).
    dq_raw : np.ndarray uint32
        3D array of data quality flags (shape n_reads, ny, nx).

    Returns
    -------
    mid_delta_time : np.ndarray
        2D array (ny, nx) of the mid times.
    """
    n_reads,ny,nx=dq_raw.shape
    avg_bin_times = (times[0:n_reads - 1] + times[1::]) / 2
    avg_bin_times_3d = np.tile(avg_bin_times[:, None, None], (1, ny, nx))

    mask = (dq_raw & DQ_FLAGS['DO_NOT_USE']) | (dq_raw & DQ_FLAGS['SATURATED'])
    mask[1:n_reads] += mask[0:n_reads - 1]
    avg_bin_times_3d[np.where(mask[1::] > 0)] = np.nan

    mask2 = (dq_raw & DQ_FLAGS['JUMP_DET'])
    avg_bin_times_3d[np.where(mask2[1::] > 0)] = np.nan

    mid_delta_time = np.nanmean(avg_bin_times_3d, axis=0)
    return mid_delta_time
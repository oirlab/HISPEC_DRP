import numpy as np
from ...datamodels.dqflags import DQ_FLAGS
from numba import njit, prange

saturation_flag = DQ_FLAGS["SATURATED"]
jump_det_flag = DQ_FLAGS["JUMP_DET"]
do_not_use_flag = DQ_FLAGS["DO_NOT_USE"]

@njit(inline="always")
def mean_numba_if_not_zero(arr):
    total = 0.0
    n = 0
    for i in range(arr.size):
        if arr[i] != 0:
            total += arr[i]
            n=n+1
    if n != 0:
        return total / n
    else:
        return 0

@njit()
def get_mid_delta_times_ramp(times, dq_raw):
    """
    Compute the mid time of the exposure accounting for the DQ flags: DO_NOT_USE, SATURATED, or JUMP_DET.

    Parameters
    ----------
    times : np.ndarray
        1D array of read times (length n_reads).
    dq_raw : np.ndarray uint32
        1D array of data quality flags (length n_reads).

    Returns
    -------
    mid_delta_time : float
        The average mid time of the valid exposures, or NaN if no valid exposures exist.
    """
    n_reads = len(times)
    avg_bin_times = (times[0:n_reads - 1] + times[1::]) / 2

    mask = ((dq_raw & do_not_use_flag) | (dq_raw & saturation_flag)).astype(np.bool)
    mask2 = (dq_raw & jump_det_flag).astype(np.bool)
    avg_bin_times = avg_bin_times * ~(mask[1:n_reads] | mask[0:n_reads - 1] | mask2[1:n_reads])

    mid_delta_time = mean_numba_if_not_zero(avg_bin_times)
    return mid_delta_time


@njit()#inline="always",nogil=True,
def get_ramp_chunks(dq: np.ndarray):
    """
    Numba-compatible core. Returns (starts, ends) arrays of indices of dtype int64.
    Drops everything at and after the first SATURATED index.
    Keeps the JUMP_DET read but splits before it (i.e., previous chunk ends at i-1,
    next chunk starts at i).
    Excludes DO_NOT_USE reads entirely (may split).
    NOTE: does NOT filter out single-read chunks; wrapper does that.
    """
    n = dq.size

    # Find first saturated index; exclude it and everything after.
    n_use = 1
    for i in range(n):
        # checking backward for saturation to speed up processing since saturation will be at the end of the ramp
        if (dq[n-i-1] & (saturation_flag | do_not_use_flag)) == 0:
            n_use = n-i
            break
    #If all bad, n_use will be 1, and we return empty arrays below
    #If no trailing bad reads, then n_use == n
    if n_use == 1:
        return np.empty(0, dtype=np.int64),np.empty(0, dtype=np.int64)

    # Preallocate worst-case space (each valid run -> separate chunk)
    starts = np.empty(n_use//2+1, dtype=np.int64)
    ends   = np.empty(n_use//2+1, dtype=np.int64)
    count = 0

    in_chunk = False
    start = 0

    for i in range(n_use):
        dqv = dq[i]

        # If bad, close chunk and move to next read
        if (dqv & do_not_use_flag) != 0:
            if in_chunk: # then close chunk
                end = i
                if end - start >= 2:
                    starts[count] = start
                    ends[count]   = end
                    count += 1
                in_chunk = False
            continue

        # If jump, close chunk and start new one
        if (dqv & jump_det_flag) != 0:
            if in_chunk: # then close chunk
                end = i
                if end - start >= 2:
                    starts[count] = start
                    ends[count]   = end
                    count += 1
                in_chunk = False

        # Open chunk if needed
        if not in_chunk:
            start = i
            in_chunk = True


    # Flush trailing chunk
    if in_chunk:
        end = n_use
        if end - start >= 2:
            starts[count] = start
            ends[count]   = end
            count += 1


    return  starts[:count], ends[:count]

@njit(nogil=True,parallel=True,cache=True)
def get_mid_delta_times(times, dq_cube):
    """
    Parallelized wrapper for get_mid_delta_times to apply on a 3D DQ array.
    Loops over the 2nd and 3rd dimensions (y, x) and applies get_mid_delta_times to each pixel.

    Parameters
    ----------
    times : np.ndarray
        1D array of read times (length n_reads).
    dq_cube : np.ndarray
        3D array of DQ flags with shape (n_reads, ny, nx).

    Returns
    -------
    mid_time_map : np.ndarray
        2D array (ny, nx) of mid times for each pixel.
    """
    n_reads, ny, nx = dq_cube.shape
    mid_time_map = np.empty((ny, nx), dtype=np.float64)
    for j in prange(ny):
        for k in range(nx):
            mid_time_map[j, k] = get_mid_delta_times_ramp(times, dq_cube[:, j, k])
    return mid_time_map

@njit(inline="always")
def bitwise_or_reduce(arr):
    result = np.uint32(0)
    for k in range(arr.shape[0]):
        result |= arr[k]
    return result
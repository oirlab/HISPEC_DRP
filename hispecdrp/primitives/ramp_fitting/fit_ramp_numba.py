from numba import njit, prange
import numpy as np
from ...datamodels.dqflags import DQ_FLAGS
from .utils import get_mid_delta_times
import multiprocessing as mp

saturation_flag = DQ_FLAGS["SATURATED"]
jump_det_flag = DQ_FLAGS["JUMP_DET"]
do_not_use_flag = DQ_FLAGS["DO_NOT_USE"]

# TODO: Implement jump detection, look into more sophisticated methods
# https://iopscience.iop.org/article/10.1088/1538-3873/ad38d9/pdf
# https://www.stsci.edu/files/live/sites/www/files/home/roman/_documents/Roman-STScI-000394_DeterminingTheBestFittingSlope.pdf
# NOTE: If we want to use numba, we may want to cache some arrays
# NOTE: like np.sum(times**2) since times are same for all pixels, etc.


from numba import njit
import numpy as np

from numba import njit
import numpy as np
from ...datamodels.dqflags import DQ_FLAGS

__all__ = ['fit_ramps_ols', 'fit_ramps_mcds']


@njit(nogil=True)
def get_ramp_chunks(dq: np.ndarray):
    """
    Numba-compatible core. Returns (starts, ends) arrays of dtype int64.
    Drops everything at and after the first SATURATED index.
    Keeps the JUMP_DET read but splits before it (i.e., previous chunk ends at i-1,
    next chunk starts at i).
    Excludes DO_NOT_USE reads entirely (may split).
    NOTE: does NOT filter out single-read chunks; wrapper does that.
    """
    n = dq.size

    # Find first saturated index; exclude it and everything after.
    n_use = n
    for i in range(n):
        if (dq[i] & saturation_flag) != 0:
            n_use = i
            break

    # Preallocate worst-case space (each valid run -> separate chunk)
    starts = np.empty(n, dtype=np.int64)
    ends   = np.empty(n, dtype=np.int64)
    count = 0

    if n_use == 0:
        return starts[:0], ends[:0]

    in_chunk = False
    start = 0

    for i in range(n_use):
        dqv = dq[i]

        # DO_NOT_USE ? close before this read and skip it
        if (dqv & do_not_use_flag) != 0:
            if in_chunk:
                end = i - 1
                if end >= start:
                    starts[count] = start
                    ends[count]   = end
                    count += 1
                in_chunk = False
            continue

        # Open chunk if needed
        if not in_chunk:
            start = i
            in_chunk = True

        # JUMP_DET ? close chunk up to i-1, start new at i (keeping i)
        if (dqv & jump_det_flag) != 0:
            end = i - 1
            if end >= start:
                starts[count] = start
                ends[count]   = end
                count += 1
            start = i  # keep the jump read in the next chunk
            in_chunk = True

    # Flush trailing chunk
    if in_chunk:
        end = n_use - 1
        if end >= start:
            starts[count] = start
            ends[count]   = end
            count += 1

    return starts[:count], ends[:count]

@njit(nogil=True)
def _fit_ramp_slope_pedestal(times, ramp, dq, rdnoise, gain, dark_current):
    """
    Fits a line to the given data (times, ramp) with a free y-intercept.
    """
    first_indices,last_indices = get_ramp_chunks(dq)

    slope_num_sum = 0
    slope_denom_sum = 0
    rnoise_denom_sum = 0
    poisson_denom_sum = 0
    b_num_sum = 0
    b_denom_sum = 0

    n_chunk = 0
    for k in range(len(first_indices)):
        if first_indices[k] == last_indices[k]:
            continue
        _times = times[first_indices[k]: last_indices[k]+1]
        _ramp = ramp[first_indices[k]: last_indices[k]+1]

        n = len(_times)/rdnoise**2
        times_tot = np.sum(_times)/rdnoise**2
        times2_tot = np.sum(_times**2)/rdnoise**2
        ramp_tot = np.sum(_ramp)/rdnoise**2
        timesramp_tot = np.sum(_times * _ramp)/rdnoise**2
        deno = (n * times2_tot - times_tot**2)

        slope = (n * timesramp_tot - times_tot * ramp_tot) / deno
        b = (times2_tot * ramp_tot - times_tot * timesramp_tot)/ deno

        var_rnoise = n / deno
        var_poisson = np.abs((slope + dark_current) * gain) / (_times[-1] - _times[0])
        b_var = times2_tot/deno

        if var_rnoise > 0 and var_poisson > 0 and b_var > 0:
            n_chunk+=1
            slope_num_sum += slope/(var_rnoise+var_poisson)
            slope_denom_sum += 1/(var_rnoise+var_poisson)
            rnoise_denom_sum += 1/var_rnoise
            poisson_denom_sum += 1/var_poisson
            b_num_sum += b/b_var
            b_denom_sum += 1/b_var
        # slope_num_sum = slope/(var_rnoise+var_poisson)
        # slope_denom_sum = 1/(var_rnoise+var_poisson)
        # rnoise_denom_sum = 1/var_rnoise
        # poisson_denom_sum = 1/var_poisson
        # b_num_sum = b/b_var
        # b_denom_sum = 1/b_var

    if n_chunk == 0:
        slope, slope_err,var_rnoise, var_poisson, b, b_err = 0.0, -1.0, -1.0, -1.0, 0.0, -1.0
    else:
        slope = slope_num_sum/slope_denom_sum
        slope_err = np.sqrt(1/slope_denom_sum)
        var_rnoise = 1/rnoise_denom_sum
        var_poisson = 1/poisson_denom_sum
        b= b_num_sum/b_denom_sum
        b_err = np.sqrt(1/b_denom_sum)
    return slope, slope_err, var_rnoise, var_poisson, b, b_err

@njit(nogil=True)
def _fit_ramps_slope_pedestal(times, ramps, dq, rdnoise, gain, dark_current):
    _, ny, nx = ramps.shape
    slope_image = np.zeros(shape=(ny, nx), dtype=np.float32)
    slope_err = np.zeros(shape=(ny, nx), dtype=np.float32)
    var_rnoise = np.zeros(shape=(ny, nx), dtype=np.float32)
    var_poisson = np.zeros(shape=(ny, nx), dtype=np.float32)
    b = np.zeros(shape=(ny, nx), dtype=np.float32)
    b_err = np.zeros(shape=(ny, nx), dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            slope_image[i, j], slope_err[i,j], var_rnoise[i, j], var_poisson[i, j],b[i,j],b_err[i,j] = _fit_ramp_slope_pedestal(
                                                times, ramps[:, i, j], dq[:, i, j],
                                                rdnoise[i, j], gain[i, j], dark_current[i, j])
    return slope_image, slope_err, var_rnoise, var_poisson,b,b_err


def fit_ramps_ols(times, ramps, dq=None, rdnoise=None, gain=None, dark_current=None, start_times=None,max_cores=1):
    """
    Top-level function for fitting ramps with or without a fixed y-intercept with no correlated noise.
    Dispatches to the appropriate specialized function based on whether b is provided.

    Args:
        times (np.ndarray): Read times.
        ramps (np.ndarray): 3D array of ramp data with shape (n_reads, ny, nx).
        dq (np.ndarray, optional): 2D array of data quality flags across detector (ny, nx).
        rdnoise (np.ndarray, optional): 2D array of read noise across detector (ny, nx).
        gain (np.ndarray, optional): 2D array of gain across detector (ny, nx).
        dark_current (np.ndarray, optional): 2D array of dark current (DN/s) across detector (ny, nx).
            Caution: Only use when the ramps have been already dark subtracted.
        start_times (np.ndarray, optional): 2D array of MJD times of the first read across detector (ny, nx).
        max_cores (int): Number of cores for parallelization.

    Returns:
        dict[str, np.ndarray]: Dictionary with keys matching L0 rate objects.
    """
    n_reads,ny, nx = ramps.shape

    if dq is None:
        dq = np.zeros((ny, nx)).astype(np.uint32)
    if rdnoise is None:
        rdnoise = np.zeros((ny, nx)).astype(np.float32)
    if gain is None:
        gain = np.ones((ny, nx)).astype(np.float32)
    if dark_current is None:
        dark_current = np.zeros((ny, nx)).astype(np.float32)
    if start_times is None:
        start_times = np.zeros((ny, nx)).astype(np.float32)

    result = _fit_ramps_slope_pedestal(times, ramps,dq, rdnoise, gain, dark_current)

    output = {}
    output['rate'] = result[0]
    output['rate_err'] = result[1]
    output['rate_var_rnoise'] = result[2]
    output['rate_var_poisson'] = result[3]
    output['mid_time'] = start_times + get_mid_delta_times(times, dq)
    if len(result) >4:
        output['bias'] = result[4]
        output['bias_err'] = result[5]
    # if all reads are flagged as bad, set output DQ of rate to DO_NOT_USE
    _if_all_reads_bad = np.bitwise_and.reduce(dq & DQ_FLAGS["DO_NOT_USE"], axis=0)
    _dq = dq-dq & DQ_FLAGS["DO_NOT_USE"]
    output['DQ'] = np.bitwise_or.reduce(_dq, axis=0)+_if_all_reads_bad

    where_bad = np.where(result[1] == -1)
    output['DQ'][where_bad] |= DQ_FLAGS["DO_NOT_USE"]
    output['rate'][where_bad] = np.nan
    output['rate_err'][where_bad] = np.nan
    output['rate_var_rnoise'][where_bad] = np.nan
    output['rate_var_poisson'][where_bad] = np.nan

    return output


##############################
############ MCDS ############
##############################



@njit(nogil=True)
def _fit_ramp_mcds(times, ramp, dq, rdnoise, gain, dark_current, num_coadd=1):
    first_indices,last_indices = get_ramp_chunks(dq)

    slope_num_sum = 0
    slope_denom_sum = 0
    rnoise_denom_sum = 0
    poisson_denom_sum = 0

    n_chunk = 0
    for k in range(len(first_indices)):
        if first_indices[k] == last_indices[k]:
            continue
        _times = times[first_indices[k]: last_indices[k]+1]
        _ramp = ramp[first_indices[k]: last_indices[k]+1]

        n1 = np.sum(_ramp[:num_coadd])
        n2 = np.sum(_ramp[-num_coadd:])
        d1 = np.sum(_times[:num_coadd])
        d2 = np.sum(_times[-num_coadd:])
        if num_coadd > 1:
            n1 = n1 / num_coadd
            n2 = n2 / num_coadd
            d1 = d1 / num_coadd
            d2 = d2 / num_coadd
        slope = (n2 - n1) / (d2 - d1)

        var_rnoise = (2*num_coadd)*rdnoise**2/ (d2 - d1)**2
        var_poisson = np.abs((slope+dark_current)*gain)/(d2 - d1)

        if var_rnoise > 0 and var_poisson > 0:
            n_chunk+=1
            slope_num_sum += slope/(var_rnoise+var_poisson)
            slope_denom_sum += 1/(var_rnoise+var_poisson)
            rnoise_denom_sum += 1/var_rnoise
            poisson_denom_sum += 1/var_poisson
        # slope_num_sum = slope/(var_rnoise+var_poisson)
        # slope_denom_sum = 1/(var_rnoise+var_poisson)
        # rnoise_denom_sum = 1/var_rnoise
        # poisson_denom_sum = 1/var_poisson

    if n_chunk == 0:
        slope, slope_err,var_rnoise, var_poisson = 0.0, -1.0, -1.0, -1.0
    else:
        slope = slope_num_sum/slope_denom_sum
        slope_err = np.sqrt(1/slope_denom_sum)
        var_rnoise = 1/rnoise_denom_sum
        var_poisson = 1/poisson_denom_sum
    return slope, slope_err, var_rnoise, var_poisson


@njit(nogil=True)
def _fit_ramps_mcds(times, ramps, dq, rdnoise, gain, dark_current,num_coadd=1):
    _, ny, nx = ramps.shape
    slope_image = np.zeros(shape=(ny, nx), dtype=np.float32)
    slope_err = np.zeros(shape=(ny, nx), dtype=np.float32)
    var_rnoise = np.zeros(shape=(ny, nx), dtype=np.float32)
    var_poisson = np.zeros(shape=(ny, nx), dtype=np.float32)
    for i in range(ny):
        for j in range(nx):
            out  = _fit_ramp_mcds(times,
                                  ramps[:, i, j], dq[:,i,j], rdnoise[i,j], gain[i,j], dark_current[i,j],
                                  num_coadd=num_coadd)
            slope_image[i, j],slope_err[i,j], var_rnoise[i, j], var_poisson[i, j] = out
    return slope_image, slope_err, var_rnoise, var_poisson

def _fit_ramps_mcds_worker(args):
    """
    Worker function for multiprocessing. Wraps _fit_ramps_mcds to accept a single argument (tuple/list).
    Args:
        args (tuple): Contains (times, ramps, dq, rdnoise, gain, dark_current, num_coadd)
    Returns:
        tuple: Output from _fit_ramps_mcds
    """
    # G = _G
    # use views to avoid per-task copies
    # return _fit_ramps_mcds(
    #     G['times'],
    #     G['ramps'][:, i:i+1, :],
    #     G['dq'][:, i:i+1, :],
    #     G['rdnoise'][i:i+1, :],
    #     G['gain'][i:i+1, :],
    #     G['dark_current'][i:i+1, :],
    #     G['num_coadd'])
    times, ramps, dq, rdnoise, gain, dark_current, num_coadd = args
    return _fit_ramps_mcds(times, ramps, dq, rdnoise, gain, dark_current, num_coadd)

# _G = {}
# def _init_globals(times, ramps, dq, rdnoise, gain, dark_current, num_coadd):
#     _G['times'] = times
#     _G['ramps'] = ramps
#     _G['dq'] = dq
#     _G['rdnoise'] = rdnoise
#     _G['gain'] = gain
#     _G['dark_current'] = dark_current
#     _G['num_coadd'] = num_coadd

def fit_ramps_mcds(times, ramps, dq=None, rdnoise=None, gain=None, dark_current=None, start_times=None,num_coadd=1, max_cores=1):
    """
    Fits ramps using the MCDS method.

    Args:
        times (np.ndarray): Read times.
        ramps (np.ndarray): 3D array of ramp data with shape (n_reads, ny, nx).
        dq (np.ndarray): 2D array of data quality flags across detector (ny, nx)
        rdnoise (np.ndarray): 2D array of read noise across detector (ny, nx)
        gain (np.ndarray): 2D array of gain across detector (ny, nx)
        dark_current (np.ndarray): 2D array of dark current (DN/s) across detector (ny, nx).
            Caution: Only use when the ramps have been already dark subtracted.
        start_times (np.ndarray): 2D array of MJD times of the first read across detector (ny, nx).
        num_coadd (int): Number of coadds.
        max_cores (int): Number of cores for parallelization.

    Returns:
        dict[str,
         np.ndarray]: Dictionary with keys matching L0 rate objects.
    """
    n_reads,ny, nx = ramps.shape

    # Create remaining dummy variables if input arrays are not provided
    if dq is None:
        dq = np.zeros((ny, nx)).astype(np.uint32)
    if rdnoise is None:
        rdnoise = np.zeros((ny, nx)).astype(np.float32)
    if gain is None:
        gain = np.ones((ny, nx)).astype(np.float32)
    if dark_current is None:
        dark_current = np.zeros((ny, nx)).astype(np.float32)
    if start_times is None:
        start_times = np.zeros((ny, nx)).astype(np.float32)

    slope_image = np.zeros(shape=(ny, nx), dtype=np.float32)
    slope_err = np.zeros(shape=(ny, nx), dtype=np.float32)
    var_rnoise = np.zeros(shape=(ny, nx), dtype=np.float32)
    var_poisson = np.zeros(shape=(ny, nx), dtype=np.float32)

    print("max_cores",max_cores)
    if max_cores > 1:
        print("coucou parallel")
        if 0:
            # Prepare input for each row
            indata = []
            for i in range(ny):
                indata.append((
                    times,
                    ramps[:, i, :][:, np.newaxis, :],  # shape (n_reads,1, nx)
                    dq[:, i, :][:, np.newaxis, :],
                    rdnoise[i, :][np.newaxis, :],
                    gain[i, :][np.newaxis, :],
                    dark_current[i, :][np.newaxis, :],
                    num_coadd
                ))
            with mp.Pool(max_cores) as pool:
                results = pool.map(_fit_ramps_mcds_worker, indata)

            # # Build arrays first, then create pool:
            # # print("start method:", mp.get_start_method())
            # with mp.Pool(processes=max_cores, initializer=_init_globals,
            #              initargs=(times, ramps, dq, rdnoise, gain, dark_current, num_coadd)) as pool:
            #     # print("workers:", pool._processes)
            #     results = list(pool.imap(_fit_ramps_mcds_worker, range(ny), chunksize=8))

            # Unpack results into arrays
            for i, (slope_row, slope_err_row, var_r_row, var_p_row) in enumerate(results):
                slope_image[i, :] = slope_row[0, :]
                slope_err[i, :] = slope_err_row[0, :]
                var_rnoise[i, :] = var_r_row[0, :]
                var_poisson[i, :] = var_p_row[0, :]


        # Choose a block size so we submit ~max_cores tasks (or a few more),
        # not ny tasks. This heavily reduces IPC/pickling overhead.
        block_rows = int(np.ceil(ny / max_cores))
        block_rows = max(1, block_rows)

        tasks = []
        row_starts = []
        for row_start in range(0, ny, block_rows):
            row_end = min(ny, row_start + block_rows)
            nblock = row_end - row_start

            # Slice a contiguous block of rows
            ramps_block        = ramps[:, row_start:row_end, :]               # (n_reads, nblock, nx)
            dq_block           = dq[:, row_start:row_end, :]                  # (n_reads, nblock, nx)
            rdnoise_block      = rdnoise[row_start:row_end, :]                # (nblock, nx)
            gain_block         = gain[row_start:row_end, :]                   # (nblock, nx)
            dark_current_block = dark_current[row_start:row_end, :]           # (nblock, nx)

            tasks.append((
                times,
                ramps_block,
                dq_block,
                rdnoise_block,
                gain_block,
                dark_current_block,
                num_coadd
            ))
            row_starts.append(row_start)

        with mp.Pool(processes=max_cores) as pool:
            # Map over blocks (each task handles many rows)
            for row_start, (s, se, vr, vp) in zip(row_starts,pool.map(_fit_ramps_mcds_worker, tasks)):
                row_end = row_start + s.shape[0]  # s is (nblock, nx)
                slope_image[row_start:row_end, :] = s
                slope_err[row_start:row_end, :]   = se
                var_rnoise[row_start:row_end, :]  = vr
                var_poisson[row_start:row_end, :] = vp
    else:
        slope_image, slope_err, var_rnoise, var_poisson = _fit_ramps_mcds(
            times, ramps, dq, rdnoise, gain, dark_current,num_coadd=num_coadd)

    slope_err[np.where(slope_err < 0)] = np.nan
    var_rnoise[np.where(var_rnoise < 0)] = np.nan
    var_poisson[np.where(var_poisson < 0)] = np.nan

    output = {}
    output['rate'] = slope_image
    output['rate_err'] = slope_err
    output['rate_var_rnoise'] = var_rnoise
    output['rate_var_poisson'] = var_poisson
    output['mid_time'] = start_times + get_mid_delta_times(times, dq)
    # if all reads are flagged as bad, set output DQ of rate to DO_NOT_USE
    _if_all_reads_bad = np.bitwise_and.reduce(dq & DQ_FLAGS["DO_NOT_USE"], axis=0)
    _dq = dq-dq & DQ_FLAGS["DO_NOT_USE"]
    output['DQ'] = np.bitwise_or.reduce(_dq, axis=0)+_if_all_reads_bad

    where_bad = np.where(np.isnan(slope_err))
    output['DQ'][where_bad] |= DQ_FLAGS["DO_NOT_USE"]
    return output

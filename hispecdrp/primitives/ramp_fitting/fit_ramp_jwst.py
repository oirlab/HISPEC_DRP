from ... import datamodels

import numpy as np

from stcal.ramp_fitting import ramp_fit
from stcal.ramp_fitting.ramp_fit_class import RampData

import multiprocessing as mp
import ctypes
from ...utils.parallelization_utils import _arraytonumpy
from multiprocessing.sharedctypes import RawArray

from ...datamodels.dqflags import DQ_FLAGS
from .utils import get_mid_delta_times
import psutil, os
from tqdm import tqdm

__all__ = ['fit_ramps_jwst']

def _fit_ramps_jwst_row(indata):
    """
    Passes data for each row of a ramp to stcal.ramp_fitting.likely_fit methods.

    Args:
        data Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]: 
            Tuple containing the row index, and the following data arrays for a single row:
                ramp row index, read times, ramp data, readnoise, gain, data quality, dark current, algorithm
    Returns:
        i (int): index of input row from the ramps, same as input.
        slope (np.ndarray): slope image (1, nx).
        dq_out (np.ndarray): output data quality image with flags from algorithm (1, nx).
        var_poisson (np.ndarray): output variance from poisson noise (1, nx).
        var_rnoise (np.ndarray): output variance from read noise (1, nx).
        err (np.ndarray): output error in slope image (1, nx).
    """

    i, times, data_3d, readnoise_2d, gain_2d, dq_3d, dark_current_2d, algorithm = indata

    groupdq = dq_3d.astype(np.uint8)
    mask = dq_3d & DQ_FLAGS["DO_NOT_USE"]
    pixeldq_2d = np.bitwise_or.reduce(dq_3d-mask, axis=(0,1))

    # We are implicitly fitting by row, so columns is only other axis and nrows=1
    nints, nframes, ny, nx = data_3d.shape

    # Set static parameters for stcal.ramp_fitting methods
    save_opt=False # Save optional output
    weighting="optimal"
    suppress_one_group=False # Suppress saturated ramps with good 0th group

    # Initialize ramp data model
    ramp_data = RampData()

    flags = datamodels.DQ_FLAGS
    # Explicitly set flags required by JWST pipeline
    ramp_data.flags_do_not_use = flags['DO_NOT_USE']
    ramp_data.flags_jump_det = flags['JUMP_DET']
    ramp_data.flags_saturated = flags['SATURATED']
    ramp_data.flags_no_gain_val = flags['UNRELIABLE_FLAT'] # Check if this is best/optimal flag
    ramp_data.flags_unreliable_slope = flags['UNRELIABLE_SLOPE']
    ramp_data.flags_chargeloss = flags['OUTLIER'] # Check if this is best/optimal flag - only required by OLS_C

    # Explicitly set data arrays
    ramp_data.data = data_3d
    ramp_data.groupdq = groupdq
    ramp_data.pixeldq = pixeldq_2d
    ramp_data.average_dark_current = dark_current_2d

    # Get average exposure time
    diffs = [times[i+1] - times[i] for i in range(len(times)-1)]
    group_time = np.average(diffs)

    # Set other required parameters
    ramp_data.algorithm=algorithm
    ramp_data.start_row=0
    ramp_data.num_rows=ny
    ramp_data.nframes=1 # this corresponds to nints instead of nframes as above

    # Required by OLS_C for frame time
    ramp_data.frame_time=0.0
    ramp_data.group_time=group_time
    ramp_data.groupgap=0
    ramp_data.drop_frames1=0

    ramp_data.suppress_one_group_ramps=suppress_one_group

    # Times information is set as read_pattern parameter of RampData
    ramp_data.read_pattern = list(times)

    #todo: we are disabling the jump detection for now, because we cannot get the group DQ flags to return from the function.
    # and we might need it to estimate the mid_time correctly.
    ramp_data.rejection_threshold = np.inf

    # Run ramp fitting
    if algorithm=='LIKELY':
        if nframes < ramp_fit.likely_fit.LIKELY_MIN_NGROUPS:
            raise Exception(f'Input ramps have {nframes} frames which is less than LIKELY method requirement of {ramp_fit.likely_fit.LIKELY_MIN_NGROUPS} frames')
        # Call directly to likely ramp fit code
        image_info, integ_info, opt_info = ramp_fit.likely_fit.likely_ramp_fit(
            ramp_data, readnoise_2d, gain_2d
        )
    elif algorithm=='OLS_C':
        # Call to C extension for OLS slope fitting
        image_info, _, _ = ramp_fit.ols_fit.ols_slope_fitter(
            ramp_data, gain_2d, readnoise_2d, weighting, save_opt)

    # Parse output image info that we need to pass to output data model
    outdata, dq, var_poisson, var_rnoise, err = image_info

    return(i, outdata, dq, var_poisson, var_rnoise, err)

def _fit_ramps_worker(args):
    """
    Worker function for multiprocessing. Runs _fit_ramps_jwst_row on a block of rows using shared arrays.
    Args:
        args (tuple): (block_start, block_end, times, algorithm, types_tuple)
    Returns:
        tuple: (block_start, block_end, results)
    """
    block_start, block_end, times, algorithm, types_tuple = args
    mp_data_type, mp_float_type, mp_dq_type = types_tuple

    ramps_np = _arraytonumpy(shared_data, shared_data_shape, dtype=mp_data_type)
    dq_raw_np = _arraytonumpy(shared_dq_raw, shared_dq_raw_shape, dtype=mp_dq_type)
    gain_np = _arraytonumpy(shared_gain, shared_gain_shape, dtype=mp_float_type)
    dark_np = _arraytonumpy(shared_dark, shared_dark_shape, dtype=mp_float_type)
    rdnoise_np = _arraytonumpy(shared_read_noise, shared_read_noise_shape, dtype=mp_float_type)
    start_times_np = _arraytonumpy(shared_start_times, shared_start_times_shape, dtype=mp_float_type)

    slope_image_np = _arraytonumpy(shared_slope_image, shared_slope_image_shape, dtype=mp_float_type)
    slope_errors_np = _arraytonumpy(shared_slope_errors, shared_slope_errors_shape, dtype=mp_float_type)
    var_poisson_image_np = _arraytonumpy(shared_var_poisson_image, shared_var_poisson_image_shape, dtype=mp_float_type)
    var_rnoise_image_np = _arraytonumpy(shared_var_rnoise_image, shared_var_rnoise_image_shape, dtype=mp_float_type)
    dq_image_np = _arraytonumpy(shared_dq_image, shared_dq_image_shape, dtype=mp_dq_type)
    mid_times_np = _arraytonumpy(shared_mid_times, shared_mid_times_shape, dtype=mp_float_type)

    results = []
    for row in range(block_start, block_end):
        ramps_3d = ramps_np[:, [row], :]
        dq_3d = dq_raw_np[:, [row], :]
        ramps_3d = ramps_3d[np.newaxis, :]
        dq_3d = dq_3d[np.newaxis, :]

        args_row = (
            row,
            times,
            ramps_3d,
            rdnoise_np[[row], :],
            gain_np[[row], :],
            dq_3d,
            dark_np[[row], :],
            algorithm
        )
        results.append(_fit_ramps_jwst_row(args_row))

    for row_element in results:
        row, outdata, dq, var_poisson, var_rnoise, err = row_element

        slope_image_np[row, :] = outdata
        slope_errors_np[row, :] = err
        dq_image_np[row, :] = dq
        var_poisson_image_np[row, :] = var_poisson
        var_rnoise_image_np[row, :] = var_rnoise

    mid_times_np[block_start:block_end, :] = start_times_np[block_start:block_end, :] +\
                                             get_mid_delta_times(times, dq_raw_np[:, block_start:block_end, :])

    return (block_start, block_end)

def fit_ramps_jwst(times, ramps, max_cores=1, algorithm='JWST_LIKELY', dq_raw=None, rdnoise=None, gain=None,
                   dark_current=None,start_times=None):
    """
    Top-level function for fitting ramps with methods in stcal.ramp_fitting.

    Args:
        times (np.ndarray): Read times.
        ramps (np.ndarray): 3D array of ramp data with shape (n_reads,ny, nx).
        max_cores (int): Number of cores for parallelization.
        algorithm (str): Ramp fitting algorithm to use.  Options are 'JWST_LIKELY' or 'JWST_OLS'.
        dq_raw (np.ndarray): 3D array of data quality flags across detector (n_reads, ny, nx)
        rdnoise (np.ndarray): 2D array of read noise across detector (ny, nx)
        gain (np.ndarray): 2D array of gain across detector (ny, nx)
        dark_current (np.ndarray): 2D array of dark current across detector (ny, nx)
        start_times (float): MJD time of the start of the first read.
    Returns:
        dict[str, np.ndarray]:
    """

    n_reads, ny, nx = ramps.shape

    if algorithm.lower() in ('jwst_likely', 'jwst_like', 'jwst_likelihood'):
        algorithm = 'LIKELY'
    elif algorithm.lower() in ('jwst_ols', 'jwst_ols_c'):
        algorithm = 'OLS_C'
    else:
        raise ValueError(f'Unrecognized JWST ramp fitting algorithm {algorithm}')



    if ramps.dtype is np.uint16:
        mp_data_type = ctypes.c_ushort
    else:
        mp_data_type = ctypes.c_float
    mp_float_type = ctypes.c_float
    mp_dq_type = ctypes.c_uint
    types_tuple = (mp_data_type, mp_float_type, mp_dq_type)

    slope_image_mp = RawArray(mp_float_type, nx * ny)
    slope_image_shape = (ny, nx)
    slope_image_np = _arraytonumpy(slope_image_mp, slope_image_shape, dtype=mp_float_type)
    slope_image_np[:] = np.zeros((ny, nx), dtype=np.float32)

    slope_errors_mp = RawArray(mp_float_type, nx * ny)
    slope_errors_shape = (ny, nx)
    slope_errors_np = _arraytonumpy(slope_errors_mp, slope_errors_shape, dtype=mp_float_type)
    slope_errors_np[:] = np.zeros((ny, nx), dtype=np.float32)

    var_poisson_image_mp = RawArray(mp_float_type, nx * ny)
    var_poisson_image_shape = (ny, nx)
    var_poisson_image_np = _arraytonumpy(var_poisson_image_mp, var_poisson_image_shape, dtype=mp_float_type)
    var_poisson_image_np[:] = np.zeros((ny, nx), dtype=np.float32)

    var_rnoise_image_mp = RawArray(mp_float_type, nx * ny)
    var_rnoise_image_shape = (ny, nx)
    var_rnoise_image_np = _arraytonumpy(var_rnoise_image_mp, var_rnoise_image_shape, dtype=mp_float_type)
    var_rnoise_image_np[:] = np.zeros((ny, nx), dtype=np.float32)

    dq_image_mp = RawArray(mp_dq_type, nx * ny)
    dq_image_shape = (ny, nx)
    dq_image_np = _arraytonumpy(dq_image_mp, dq_image_shape, dtype=mp_dq_type)
    dq_image_np[:] = np.zeros((ny, nx), dtype=np.uint32)

    start_times_mp = RawArray(mp_float_type, nx * ny)
    start_times_shape = (ny, nx)
    start_times_np = _arraytonumpy(start_times_mp, start_times_shape, dtype=mp_float_type)
    if start_times is None:
        start_times_np[:] = np.zeros((ny, nx)).astype(np.float32)
    else:
        start_times_np[:] = start_times

    mid_times_mp = RawArray(mp_float_type, nx * ny)
    mid_times_shape = (ny, nx)
    mid_times_np = _arraytonumpy(mid_times_mp, mid_times_shape, dtype=mp_float_type)
    mid_times_np[:] = np.zeros((ny, nx), dtype=np.float32)

    gain_mp = RawArray(mp_float_type, nx * ny)
    gain_shape = (ny, nx)
    gain_np = _arraytonumpy(gain_mp, gain_shape, dtype=mp_float_type)
    if gain is None:
        # TODO? - optimize gain when array is not passed
        gain_np[:] = np.ones((ny, nx)).astype(np.float32)
    else:
        gain_np[:] = gain

    dark_mp = RawArray(mp_float_type, nx * ny)
    dark_shape = (ny, nx)
    dark_np = _arraytonumpy(dark_mp, dark_shape, dtype=mp_float_type)
    if dark_current is None:
        dark_np[:] = np.zeros((ny, nx)).astype(np.float32)
    else:
        dark_np[:] = dark_current

    rdnoise_mp = RawArray(mp_float_type, nx * ny)
    rdnoise_shape = (ny, nx)
    rdnoise_np = _arraytonumpy(rdnoise_mp, dark_shape, dtype=mp_float_type)
    if rdnoise is None:
        # TODO? - optimize rdnoise when array is not passed
        rdnoise_np[:] = np.ones((ny, nx)).astype(np.float32)
    else:
        rdnoise_np[:] = rdnoise


    if max_cores > 1:

        ramps_mp = RawArray(mp_data_type, nx * ny * n_reads)
        ramps_shape = (n_reads, ny, nx)
        ramps_np = _arraytonumpy(ramps_mp, ramps_shape, dtype=mp_data_type)
        ramps_np[:] = ramps  # copy original array into shared array
        ramps = ramps_np  # free the original array from memory to avoid duplicates

        dq_raw_mp = RawArray(mp_dq_type, nx * ny * n_reads)
        dq_raw_shape = (n_reads, ny, nx)
        dq_raw_np = _arraytonumpy(dq_raw_mp, dq_raw_shape, dtype=mp_dq_type)
        if dq_raw is None:
            dq_raw_np[:] = np.zeros((n_reads, ny, nx), dtype=np.uint32)
        else:
            dq_raw_np[:] = dq_raw  # copy original array into shared array
            dq_raw = dq_raw_np  # free the original array from memory to avoid duplicates


        # Prepare blocks
        block_size = max(1, ny // (4*max_cores)) if max_cores > 1 else ny
        args_list = []
        for block_start in range(0, ny, block_size):
            block_end = min(block_start + block_size, ny)

            # block_start, block_end, times, algorithm, types_tuple = args
            args = (block_start, block_end, times, algorithm, types_tuple)
            args_list.append(args)

        _init_args = (
            ramps_mp, ramps_shape,
            dq_raw_mp, dq_raw_shape,
            gain_mp, gain_shape,
            dark_mp, dark_shape,
            rdnoise_mp, rdnoise_shape,
            slope_image_mp, slope_image_shape,
            slope_errors_mp, slope_errors_shape,
            var_poisson_image_mp, var_poisson_image_shape,
            var_rnoise_image_mp, var_rnoise_image_shape,
            dq_image_mp, dq_image_shape,
            start_times_mp, start_times_shape,
            mid_times_mp, mid_times_shape
        )

        ctx = mp.get_context("spawn")  # avoids unsafe fork after OpenMP init
        tpool = ctx.Pool(
            processes=max_cores,
            initializer=_tpool_init_jwst_ramp,
            initargs=_init_args,
            maxtasksperchild=50
        )
        # tpool = mp.Pool(processes=max_cores,
        #                 initializer=_tpool_init_jwst_ramp,
        #                 initargs=_init_args,
        #                 maxtasksperchild=50)
        try:
            tasks = [tpool.apply_async(_fit_ramps_worker, args=(args_tuple,))
                     for args_tuple in args_list]

            for t in tqdm(tasks, desc="Processing blocks"):
                t.wait()
        finally:
            tpool.close()
            tpool.join()
    else:
        results = []
        for row in np.arange(ny):
            print(f"Processing row {row+1} of {ny}.")
            ramps_3d = ramps[:,[row],:]
            dq_3d = dq_raw[:,[row],:]
            # Reshape to add the nints axis that JWST pipeline requires
            ramps_3d = ramps_3d[np.newaxis,:]
            dq_3d = dq_3d[np.newaxis,:]

            args = (row,
                    times,
                    ramps_3d,
                    # Make sure to add a new axis to each array to maintain number of dimensions
                    rdnoise_np[[row], :],
                    gain_np[[row], :],
                    dq_3d,
                    dark_np[[row], :],
                    algorithm)

            results.append(_fit_ramps_jwst_row(args))

        for row_element in results:
            row, outdata, dq, var_poisson, var_rnoise, err = row_element

            slope_image_np[row,:] = outdata
            slope_errors_np[row,:] = err
            dq_image_np[row, :] = dq
            var_poisson_image_np[row, :] = var_poisson
            var_rnoise_image_np[row, :] = var_rnoise

        mid_times_np = start_times + get_mid_delta_times(times, dq_raw)

    output = {}
    output['rate'] = slope_image_np
    output['rate_err'] = slope_errors_np
    output['rate_var_rnoise'] = var_rnoise_image_np
    output['rate_var_poisson'] = var_poisson_image_np
    output['mid_time'] = mid_times_np
    output['DQ'] = dq_image_np

    return output


def _tpool_init_jwst_ramp(data, data_shape,
                dq_raw, dq_raw_shape,
                gain, gain_shape,
                dark, dark_shape,
                read_noise, read_noise_shape,
                slope_image, slope_image_shape,
                slope_errors, slope_errors_shape,
                var_poisson_image, var_poisson_image_shape,
                var_rnoise_image, var_rnoise_image_shape,
                dq_image, dq_image_shape,
                start_times, start_times_shape,
                mid_times, mid_times_shape
                ):
    """
    Initializer function for the thread pool that initializes various shared variables. Main things to note that all
    except the shapes are shared arrays (mp.Array).
    """

    global shared_data, shared_data_shape, \
        shared_dq_raw, shared_dq_raw_shape, \
        shared_gain, shared_gain_shape, \
        shared_dark, shared_dark_shape, \
        shared_read_noise, shared_read_noise_shape, \
        shared_slope_image, shared_slope_image_shape, \
        shared_slope_errors, shared_slope_errors_shape, \
        shared_var_poisson_image, shared_var_poisson_image_shape, \
        shared_var_rnoise_image, shared_var_rnoise_image_shape, \
        shared_dq_image, shared_dq_image_shape, \
        shared_start_times, shared_start_times_shape, \
        shared_mid_times, shared_mid_times_shape

    shared_data = data
    shared_data_shape = data_shape

    shared_dq_raw = dq_raw
    shared_dq_raw_shape = dq_raw_shape

    shared_gain = gain
    shared_gain_shape = gain_shape

    shared_dark = dark
    shared_dark_shape = dark_shape

    shared_read_noise = read_noise
    shared_read_noise_shape = read_noise_shape

    shared_slope_image = slope_image
    shared_slope_image_shape = slope_image_shape

    shared_slope_errors = slope_errors
    shared_slope_errors_shape = slope_errors_shape

    shared_var_poisson_image = var_poisson_image
    shared_var_poisson_image_shape = var_poisson_image_shape

    shared_var_rnoise_image = var_rnoise_image
    shared_var_rnoise_image_shape = var_rnoise_image_shape

    shared_dq_image = dq_image
    shared_dq_image_shape = dq_image_shape

    shared_start_times = start_times
    shared_start_times_shape = start_times_shape

    shared_mid_times = mid_times
    shared_mid_times_shape = mid_times_shape
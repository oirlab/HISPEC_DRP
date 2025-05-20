from ... import datamodels

import numpy as np

from stcal.ramp_fitting import ramp_fit
from stcal.ramp_fitting.ramp_fit_class import RampData

import multiprocessing as mp

__all__ = ['fit_ramps_jwst']

def _fit_ramps_likely_row(indata):
    """
    Passes data for each row of a ramp to stcal.ramp_fitting.likely_fit methods.

    Args:
        data Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]: 
            Tuple containing the row index, and the following data arrays for a single row:
                ramps, read times, ramp data, readnoise, gain, data quality, dark current, bias, algorithm
    Returns:
        Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            Row index integer and arrays containing the following output data arrays for a single row:
                slope, data quality, variance of Poisson noise, variance of read noise, and error in slope
    """

    i, times, data_3d, readnoise_2d, gain_2d, pixeldq_2d, dark_current_2d, bias_2d, algorithm = indata

    # We are implicitly fitting by row, so columns is only other axis and nrows=1
    nints, nframes, ny, nx = data_3d.shape

    # Set static parameters for stcal.ramp_fitting methods
    max_cores="1" # Run with 1 core since we are parallelizing outside this method
    save_opt=False # Save optional output
    weighting="optimal"
    suppress_one_group=False # Suppress saturated ramps with good 0th group

    # Spoof unused data structure for the groupdq - can be refactored if want to use data quality flags at the group/frame level
    groupdq = np.zeros(data_3d.shape).astype(np.uint32)

    flags = datamodels.hispec_dqflags.DQ_FLAGS
    # Need to designate one of the flags to be NO_GAIN_VALUE
    flags['NO_GAIN_VALUE'] = flags['UNRELIABLE_FLAT']

    # Initialize ramp data model
    ramp_data = RampData()

    ramp_data.set_arrays(data_3d,
        groupdq, 
        pixeldq_2d, 
        dark_current_2d,
        None
    )

    # Set other required parameters
    ramp_data.algorithm=algorithm
    ramp_data.set_dqflags(flags)
    ramp_data.zeroframe=bias_2d
    ramp_data.start_row=0
    ramp_data.num_rows=ny
    ramp_data.nframes=1 # this corresponds to nints instead of nframes as above
    ramp_data.suppress_one_group_ramps=suppress_one_group

    # Times information is set as read_pattern parameter of RampData
    ramp_data.read_pattern = list(times)

    # Run ramp fitting
    image_info, _, _ = ramp_fit.ramp_fit_data(
        ramp_data, save_opt, readnoise_2d, gain_2d, algorithm, 
        weighting, max_cores, flags)

    return(i, *image_info)

def fit_ramps_jwst(times, ramps, max_cores=1, algorithm='JWST_LIKELY', dq=None, rdnoise=None, gain=None, dark_current=None, bias=None):
    """
    Top-level function for fitting ramps with stcal.ramp_fitting.likely_fit.

    Args:
        times (np.ndarray): Read times.
        ramps (np.ndarray): 3D array of ramp data with shape (ny, nx, n_reads).
        max_cores (int): Number of cores for parallelization.
        dq (np.ndarray): 2D array of data quality flags across detector (ny, nx)
        rdnoise (np.ndarray): 2D array of read noise across detector (ny, nx)
        gain (np.ndarray): 2D array of gain across detector (ny, nx)
        dark_current (np.ndarray): 2D array of dark current across detector (ny, nx)
        bias (np.ndarray): 2D array of bias across detector (ny, nx)
    Returns:
        dict[str, np.ndarray]: Dictionary with keys 'slope' and 'slope_err' containing the fitted slopes and errors.  The output also contains 'dq' for output data quality, 'var_poisson' for variance in Poisson noise, and 'var_rnoise' for variance in read noise.
    """

    ny, nx, n_reads = ramps.shape

    if algorithm.lower() in ('jwst_likely', 'jwst_like', 'jwst_likelihood'):
        algorithm = 'LIKELY'
    elif algorithm.lower() in ('jwst_ols', 'jwst_ols_c'):
        algorithm = 'OLS_C'
    else:
        raise ValueError(f'Unrecognized JWST ramp fitting algorithm {algorithm}')

    # JWST assumes that ramps are in first index, so transpose to n_reads x ny x nx
    ramps = ramps.transpose(2, 0, 1)

    # Create remaining dummy variables if input arrays are not provided
    if dq is None:
        dq = np.zeros((ny, nx)).astype(np.uint32)
    if rdnoise is None:
        # TODO - optimize read noise when array is not passed
        rdnoise = np.zeros((ny, nx)) + np.std(ramps[0, :, :])
    if gain is None:
        # TODO - optimize gain when array is not passed
        gain = np.zeros((ny, nx)) + 2.28
    if dark_current is None:
        dark_current = np.zeros((ny, nx))
    if bias is None:
        bias = np.zeros((ny, nx))

    with mp.Pool(max_cores) as pool:
        # TODO - parallelize with memory sharing instead of this way
        data = []
        for row in np.arange(ny):
            ramps_3d = ramps[:,[row],:]
            # Reshape to add the nints axis that JWST pipeline requires
            ramps_3d = ramps_3d[np.newaxis,:]

            data.append((row,
                         times,
                         ramps_3d,
                         # Make sure to add a new axis to each array to maintain number of dimensions
                         rdnoise[[row], :],
                         gain[[row], :],
                         dq[[row], :],
                         dark_current[[row], :],
                         # JWST bias/zero frame has shape nints x ny x nx
                         bias[[row], :][np.newaxis, :],
                         algorithm))

        results = pool.map(_fit_ramps_likely_row, data)

    # Parse process results into output images
    slope_image = np.zeros(shape=(ny, nx), dtype=np.float32)
    slope_errors = np.zeros(shape=(ny, nx), dtype=np.float32)
    var_poisson_image = np.zeros(shape=(ny, nx), dtype=np.float32)
    var_rnoise_image = np.zeros(shape=(ny, nx), dtype=np.float32)
    dq_image = np.zeros(shape=(ny, nx), dtype=np.uint32)
    for row_element in results:
        row, outdata, dq, var_poisson, var_rnoise, err = row_element

        slope_image[row,:] = outdata
        slope_errors[row,:] = err
        dq_image[row, :] = dq
        var_poisson_image[row, :] = var_poisson 
        var_rnoise_image[row, :] = var_rnoise

    return(dict(slope=slope_image, 
                slope_err=slope_errors, 
                dq=dq_image, 
                var_poisson=var_poisson_image, 
                var_rnoise=var_rnoise_image))

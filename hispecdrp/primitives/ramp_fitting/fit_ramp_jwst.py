from ... import datamodels

import numpy as np

from stcal.ramp_fitting import ramp_fit
from stcal.ramp_fitting.ramp_fit_class import RampData

import multiprocessing as mp

__all__ = ['fit_ramps_jwst']

def _fit_ramps_jwst_row(indata):
    """
    Passes data for each row of a ramp to stcal.ramp_fitting.likely_fit methods.

    Args:
        data Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]: 
            Tuple containing the row index, and the following data arrays for a single row:
                ramps, read times, ramp data, readnoise, gain, data quality, dark current, algorithm
    Returns:
        i (int): index of input row from the ramps, same as input.
        slope (np.ndarray): slope image (1, nx).
        dq_out (np.ndarray): output data quality image with flags from algorithm (1, nx).
        var_poisson (np.ndarray): output variance from poisson noise (1, nx).
        var_rnoise (np.ndarray): output variance from read noise (1, nx).
        err (np.ndarray): output error in slope image (1, nx).
    """

    i, times, data_3d, readnoise_2d, gain_2d, pixeldq_2d, dark_current_2d, algorithm = indata

    # We are implicitly fitting by row, so columns is only other axis and nrows=1
    nints, nframes, ny, nx = data_3d.shape

    # Set static parameters for stcal.ramp_fitting methods
    save_opt=False # Save optional output
    weighting="optimal"
    suppress_one_group=False # Suppress saturated ramps with good 0th group

    # Spoof unused data structure for the groupdq - can be refactored if want to use data quality flags at the group/frame level
    groupdq = np.zeros(data_3d.shape).astype(np.uint8)

    # Initialize ramp data model
    ramp_data = RampData()

    flags = datamodels.hispec.dqflags.DQ_FLAGS
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

    # Run ramp fitting
    if algorithm=='LIKELY':
        if nframes < ramp_fit.likely_fit.LIKELY_MIN_NGROUPS:
            raise Exception(f'Input ramps have {nframes} frames which is less than LIKELY method requirement of {ramp_fit.likely_fit.LIKELY_MIN_NGROUPS} frames')
        # Call directly to likely ramp fit code
        image_info, _, _ = ramp_fit.likely_fit.likely_ramp_fit(
            ramp_data, readnoise_2d, gain_2d
        )
    elif algorithm=='OLS_C':
        # Call to C extension for OLS slope fitting
        image_info, _, _ = ramp_fit.ols_fit.ols_slope_fitter(
            ramp_data, gain_2d, readnoise_2d, weighting, save_opt)

    # Parse output image info that we need to pass to output data model
    outdata, dq, var_poisson, var_rnoise, err = image_info

    return(i, outdata, dq, var_poisson, var_rnoise, err)

def fit_ramps_jwst(times, ramps, max_cores=1, algorithm='JWST_LIKELY', dq=None, rdnoise=None, gain=None, dark_current=None):
    """
    Top-level function for fitting ramps with methods in stcal.ramp_fitting.

    Args:
        times (np.ndarray): Read times.
        ramps (np.ndarray): 3D array of ramp data with shape (ny, nx, n_reads).
        max_cores (int): Number of cores for parallelization.
        dq (np.ndarray): 2D array of data quality flags across detector (ny, nx)
        rdnoise (np.ndarray): 2D array of read noise across detector (ny, nx)
        gain (np.ndarray): 2D array of gain across detector (ny, nx)
        dark_current (np.ndarray): 2D array of dark current across detector (ny, nx)
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
        rdnoise = np.ones((ny, nx)).astype(np.float32)
    if gain is None:
        # TODO - optimize gain when array is not passed
        gain = np.ones((ny, nx)).astype(np.float32)
    if dark_current is None:
        dark_current = np.zeros((ny, nx)).astype(np.float32)

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
                        algorithm))

        results = pool.map(_fit_ramps_jwst_row, data)

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


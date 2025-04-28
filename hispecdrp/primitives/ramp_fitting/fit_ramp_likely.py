from stcal.ramp_fitting import likely_fit, likely_algo_classes

import multiprocessing as mp
import numpy as np

# TODO: benchmark likely fit method using jump detection for run time
# add memory sharing in parallelization
# pass back var_rdnoise and var_poisson calculated directly by this method

__all__ = ['fit_ramps_likely']

def _likely_ramp_fit_row(data):
    """
    Passes data for each row of a ramp to stcal.ramp_fitting.likely_fit methods.

    Args:
        data Tuple[int, stcal.ramp_fitting.likely_algo_classes.Covar, np.ndarray, np.ndarray, np.ndarray]: 
            Tuple containing the row index, covariance object of read times, 
            difference of input image array, read noise array, and gain array.
    Returns:
        Tuple[int, stcal.ramp_fitting.likely_algo_classes.RampResult]: 
            Row index integer and RampResult output class containing slope and slope error.
    """

    row, covar_times, diff, rdnoise, gain = data

    if jump_detect:
        d2use, countrates = likely_fit.mask_jumps(diff, covar_times, rdnoise, gain)
        rateguess = countrates * (countrates > 0)
    else:
        rateguess = np.median(diff, axis=0)[np.newaxis, :]
        rateguess *= rateguess > 0
        d2use = None
    
    result = likely_fit.fit_ramps(
        diff,
        covar_times,
        gain,
        rdnoise,
        diffs2use=d2use,
        count_rate_guess=rateguess,
    )

    return(row, result)

def fit_ramps_likely(times, ramps, max_cores=1, dq=None, rdnoise=None, gain=None):
    """
    Fits ramps using the likely method.

    See references from Brandt (2024) - 
    https://iopscience.iop.org/article/10.1088/1538-3873/ad38da

    Args:
        times (np.ndarray): Read times.
        ramps (np.ndarray): 3D array of ramp data with shape (ny, nx, n_reads).
        max_cores (int): Number of cores for parallelization.
        dq (np.ndarray): 2D array of data quality flags across detector (ny, nx)
        rdnoise (np.ndarray): 2D array of read noise across detector (ny, nx)
        gain (np.ndarray): 2D array of gain across detector (ny, nx)
    Returns:
        dict[str, np.ndarray]: Dictionary with keys 'slope' and 'slope_err' containing the fitted slopes and errors.
    """

    covar_times = likely_algo_classes.Covar(times)

    ny, nx, nreads = ramps.shape

    # likely_fit methods assume reads are on first axis
    ramps = ramps.transpose(2, 0, 1)

    if nreads < 4:
        raise Exception('Likelihood fit should be performed with nreads >= 4')

    # Create dummy variables if input arrays are not provided
    if dq is None:
        dq = np.zeros((ny, nx)).astype(np.uint8)
    if rdnoise is None:
        rdnoise = np.zeros((ny, nx)) + np.std(ramps[0])
    if gain is None:
        gain = np.zeros((ny, nx))+1.0

    # See references from Brandt (2024) - 
    # https://iopscience.iop.org/article/10.1088/1538-3873/ad38da
    # Eqn (5)
    diff = (ramps[1:] - ramps[:-1]) / covar_times.delta_t[:,np.newaxis, np.newaxis]

    with mp.Pool(max_cores) as pool:
        # TODO - pass arrays via memory sharing instead of splitting this way
        data = [(row, covar_times, diff[:, row], rdnoise[row], gain[row]) 
            for row in np.arange(nrows)]

        results = pool.map(likely_ramp_fit_row, data)

    # Parse process results into output images
    slope_image = np.zeros(shape=(ny, nx), dtype=np.float32)
    slope_errors = np.zeros(shape=(ny, nx), dtype=np.float32)
    for row_element in results:
        row_idx = row_element[0]
        result = row_element[1]

        slope_image[row_idx,:] = result.countrate
        slope_errors[row_idx,:] = result.uncert

    return dict(slope=slope_image, slope_err=slope_errors)

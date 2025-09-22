from ... import datamodels

import numpy as np

import multiprocessing as mp

__all__ = ['jump_detection_difference']

def _jump_detection_difference_row(indata):

    """
    Finds jumps for row of a ramp using simple two-point difference algorithm

    Args:
        data Tuple[int, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            Tuple containing the following information for a single row:
                ramp row index, rejection threshold, ramp data, readnoise, gain, data quality
    Returns:
        i (int): index of input row from the ramps, same as input.
        dq_out (np.ndarray): output data quality image with flags from algorithm (1, nx).
    """

    i, rejection_threshold, ramp_data, rdnoise, gain, dq_in = indata

    n_reads, ny, nx = ramp_data.shape
    dq_out = np.zeros((n_reads, ny, nx), dtype=np.uint32)

    # Calculate differences between successive reads
    diffs = np.diff(ramp_data, axis=0)

    # Estimate the typical value of differences and the noise in the differences
    median_diff = np.median(diffs, axis=0) * gain
    photon_noise_squared = np.abs(median_diff) # Assuming Poisson
    total_noise_squared = (rdnoise**2) * 2 + photon_noise_squared # Factor of 2 for diffs
    sigma_diff = np.sqrt(total_noise_squared) / gain # Convert back to ADU

    # Identify jumps
    jumps = np.abs(diffs) > (rejection_threshold * sigma_diff)

    # Pad the first axis by 1 so jumps has same shape as ramp_data
    padding = np.zeros((1, ny, nx), dtype=bool)
    jumps = np.concatenate((padding, jumps), axis=0)

    # Identify the correct flag for jumps from the dqflags datamodel
    jump_flag = datamodels.DQ_FLAGS['JUMP_DET']

    # Modify dq_out to flag where jumps occurred
    dq_out[jumps] = jump_flag

    return(i, dq_out)

def jump_detection_difference(ramp_data: np.ndarray, dq_in: np.ndarray, gain=None, rdnoise=None, max_cores=1, rejection_threshold=4.0) -> np.ndarray:
    """
    Simple jump detection for a 3D array of up the ramp data.

    Args:
        ramp_data (np.ndarray): 3D array of the UTR data with shape n_ramps x ny x nx.
        dq_in (np.ndarray): Input data quality flags for the UTR data with shape n_ramps x ny x nx.
        gain (np.ndarray): Input gain array with shape ny x nx
        rdnoise (np.ndarray): Input read noise array with shape ny x nx
        rejection_threshold (float): Sigma threshold for jump detection.
    Returns:
        np.ndarray: New data quality flags indicating where jumps are detected.
    """

    n_reads, ny, nx = ramp_data.shape

    # TODO - determine how to use the input DQ flags to ignore bad data

    if rdnoise is None:
        # TODO - optimize read noise when array is not passed
        rdnoise = np.zeros((ny, nx)).astype(np.float32)
    if gain is None:
        # TODO - optimize gain when array is not passed
        gain = np.ones((ny, nx)).astype(np.float32)

    if n_reads <= 2:
        return dq_out

    with mp.Pool(max_cores) as pool:
        # TODO - parallelize with memory sharing instead of this way
        data = []
        for row in np.arange(ny):
            ramps_3d = ramp_data[:,[row],:]

            data.append((row,
                         rejection_threshold,
                         ramps_3d,
                         # Make sure to add a new axis to each array to maintain number of dimensions
                         rdnoise[[row], :],
                         gain[[row], :],
                         dq_in[:,[row], :]))

        results = pool.map(_jump_detection_difference_row, data)

    # Reassemble output 
    dq_out = np.zeros(shape=(n_reads, ny, nx), dtype=np.uint32)

    for row_element in results:
        row, dq = row_element

        dq_out[:, [row], :] = dq

    return dq_out

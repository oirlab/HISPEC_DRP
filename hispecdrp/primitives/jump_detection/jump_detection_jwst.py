from ... import datamodels

import numpy as np

from stcal.jump.jump import detect_jumps_data
from stcal.jump.jump_class import JumpData

import multiprocessing as mp

__all__ = ['jump_detection_jwst']

def _jump_detection_jwst_row(indata):
    """
    Finds jumps for row of a ramp using stcal.jump.jump.detect_jumps_data

    Args:
        data Tuple[int, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            Tuple containing the following information for a single row:
                ramp row index, rejection threshold, ramp data, readnoise, gain, data quality
    Returns:
        i (int): index of input row from the ramps, same as input.
        dq_out (np.ndarray): output data quality image with flags from algorithm (1, nx).
    """

    i, rejection_threshold, ramp_data, rdnoise, gain, dq_in = indata

    # We are implicitly detecting by row, so columns is only other axis and nrows=1
    nints, nframes, ny, nx = ramp_data.shape

    # Initialize ramp data model
    jump_data = JumpData()

    # Set data arrays and parameters
    jump_data.nframes = nframes
    jump_data.data = ramp_data
    jump_data.gain_2d = gain
    jump_data.rnoise_2d = rdnoise

    # Spoof unused data structure for the groupdq - can be refactored if want to use data quality flags at the group/frame level
    # TODO - update these from input dq
    jump_data.gdq = np.zeros((nints, nframes, ny, nx), dtype=np.uint8)
    jump_data.pdq = np.zeros((ny, nx), dtype=np.uint32)

    # Tunable parameters - update to be able to tweak these:
    # https://github.com/spacetelescope/stcal/blob/main/src/stcal/jump/jump_class.py
    jump_data.rejection_threshold = rejection_threshold

    # If set to True (default is True), it will cause the four perpendicular
    # neighbors of all detected jumps to also be flagged as a jump.
    jump_data.flag_4_neighbors = False

    # Value in units of sigma that sets the upper limit for flagging of neighbors.
    # Any jump above this cutoff will not have its neighbors flagged.
    jump_data.max_jump_to_flag_neighbors = np.inf

    # Value in units of sigma that sets the lower limit for flagging of neighbors (marginal
    # detections). Any primary jump below this value will not have its neighbors flagged.
    jump_data.min_jump_to_flag_neighbors = np.inf

    # The minimum contour area to trigger the creation of enclosing ellipses or circles.
    jump_data.min_jump_area = (1,)

    # The minimum area of saturated pixels at the center of a snowball. Only
    # contours with area above the minimum will create snowballs.
    jump_data.min_sat_area = 0

    # The factor that is used to increase the size of the enclosing
    # circle/ellipse jump flagged pixels.
    jump_data.expand_factor = 0.0

    # If true there must be a saturation circle within the radius of the jump
    # circle to trigger the creation of a snowball. All true snowballs appear
    # to have at least one saturated pixel.
    jump_data.sat_required_snowball = False

    # The minimum radius of the saturated core of a snowball for the core to be extended
    jump_data.min_sat_radius_extend = 0.0

    # The number of pixels to expand the saturated core of detected snowballs
    jump_data.sat_expand = 0

    # The distance from the edge of the detector where saturated cores are not
    # required for snowball detection
    jump_data.edge_size = 0

    # We are parallelizing at row level, so set max_cores=1
    jump_data.max_cores = "1"

    # Explicitly set flags required by JWST pipeline
    jump_data.fl_good = datamodels.DQ_FLAGS['GOOD']
    jump_data.fl_sat = datamodels.DQ_FLAGS['SATURATED']
    jump_data.fl_jump = datamodels.DQ_FLAGS['JUMP_DET']
    jump_data.fl_ngv = datamodels.DQ_FLAGS['UNRELIABLE_FLAT'] # Check if this is best/optimal flag
    jump_data.fl_dnu = datamodels.DQ_FLAGS['DO_NOT_USE']
    jump_data.fl_ref = datamodels.DQ_FLAGS['REFERENCE_PIXEL']

    gdq, pdq, total_primary_crs, number_extended_events, stddev = detect_jumps_data(jump_data)

    # Ignore first axis since we have nints=1
    gdq = gdq[0, :].astype(np.uint32)

    return(i, gdq)

def jump_detection_jwst(ramp_data: np.ndarray, dq_in: np.ndarray, gain=None, rdnoise=None, max_cores=1, rejection_threshold=4.0) -> np.ndarray:
    """
    Top-level function for running jump detection with stcal.jump.jump.detect_jumps_data

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

    dq_out = np.zeros((n_reads, ny, nx), dtype=np.uint32)

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
            # Reshape to add the nints axis that JWST pipeline requires
            ramps_3d = ramps_3d[np.newaxis,:]

            data.append((row,
                         rejection_threshold,
                         ramps_3d,
                         # Make sure to add a new axis to each array to maintain number of dimensions
                         rdnoise[[row], :],
                         gain[[row], :],
                         dq_in[:, [row], :]))

        results = pool.map(_jump_detection_jwst_row, data)

    # Reassemble output 
    dq_out = np.zeros(shape=(n_reads, ny, nx), dtype=np.uint32)

    for row_element in results:
        row, dq = row_element

        dq_out[:,[row], :] = dq

    return dq_out


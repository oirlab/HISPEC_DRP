from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np
from ...datamodels.dqflags import DQ_FLAGS
from typing import Sequence
from copy import copy

from numba import njit, prange
from ...utils.parallelization_utils import _arraytonumpy, numba_thread_scope


__all__ = ['MakeDetectorFlatPrimitive']

COLD_FLAG =  DQ_FLAGS['COLD']
DO_NOT_USE_FLAG = DQ_FLAGS["DO_NOT_USE"]

class MakeDetectorFlatPrimitive(BasePrimitive):
    """
    Create a detector flat calibration model from a combined rate map.
    This primitive expects the coadd primitive to have already been run on a list of flat field exposures.

    todo: add median white lamp intensity to metadata for instrument monitoring

    Parameters
    ----------
    input : L0Model
        Input flat field rate map to converted to detector flat calibration. Do not input a list.
    max_cores : int, optional
        Maximum number of CPU cores to use for parallel processing. Default is 1.
    cold_pixel_threshold : float, optional
        Threshold normalized flat value below which pixels are flagged as cold in the DQ array. Default is 0.9.

    Returns
    -------
    output : DetectorFlatModel
        The derived detector flat calibration model containing the normalized flat field map and updated DQ array.
    """

    ARGUMENTS = {
        'output_model' : 'DetectorFlatModel',
        'max_cores' : 1,
        'cold_pixel_threshold' : 0.9,  # Threshold below which pixels are
    }

    def _perform(self):
        if  isinstance(self.input, Sequence):
            raise ValueError("Rate maps must be combined before computing the detector flat. Input should not be a list.")

        input_model = datamodels.open(self.input)
        # Define output model from inputs
        output_detflat = self.get_output_model(input_model)

        median_flat_whitelamp_intensity = np.nanmedian(input_model.rate)

        # Median white lamp intensity for instrument monitoring
        output_detflat.meta.median_flat_whitelamp_intensity = median_flat_whitelamp_intensity

        output_detflat.flat = input_model.rate/median_flat_whitelamp_intensity
        output_detflat.flat_err = input_model.rate_err/median_flat_whitelamp_intensity
        output_detflat.flat_var_rnoise = input_model.rate_var_rnoise/median_flat_whitelamp_intensity**2
        output_detflat.flat_var_poisson = input_model.rate_var_poisson/median_flat_whitelamp_intensity**2
        output_detflat.dq = copy(input_model.dq)

        with numba_thread_scope(self.max_cores):
            _detect_cold_pixels(output_detflat.flat, output_detflat.dq,self.cold_pixel_threshold)

        return dict(output=output_detflat)


@njit(nogil=True, parallel=True, cache=True)
def _detect_cold_pixels(flat_rate: np.ndarray, dq: np.ndarray, cold_pixel_threshold: float = 0.9):
    ny, nx = flat_rate.shape
    for j in prange(ny):
        for i in range(nx):
            if flat_rate[j, i] < cold_pixel_threshold:
                dq[j, i] |= COLD_FLAG


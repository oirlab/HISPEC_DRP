from ..base_primitive import BasePrimitive
from ...datamodels.dqflags import DQ_FLAGS
import numpy as np
from copy import copy
from typing import Sequence
from ... import datamodels
from ..jump_detection.jump_detection_difference import median_axis0_3d

from numba import njit, prange
from ...utils.parallelization_utils import numba_thread_scope

__all__ = ['MakeSaturationPrimitive', 'check_saturation']


class MakeSaturationPrimitive(BasePrimitive):
    """
    Create a saturation calibration model from a list of raw ramp exposures.
    This primitive derives the per-pixel saturation threshold by taking the median of the last reads
    of a list of raw ramp exposures.

    Parameters
    ----------
    input : list of str or list of RampModel
        List of input file paths or ramp models to process.
    offset : float, optional
        Value to subtract from the last read median to define the saturation threshold. This can help account for read noise or other systematics.
        Default is 100.
    max_cores : int, optional
        Maximum number of CPU cores to use for parallel processing. Default is 1.

    Returns
    -------
    output : SaturationModel
        The derived saturation calibration model containing the per-pixel saturation thresholds.

    """
    
    ARGUMENTS = {
        'output_model' : 'SaturationModel',
        'offset' : 100, # subtract from last read to define saturation threshold in case read noise or other systematics are an issue.
        'max_cores' : 1,
    }

    def _perform(self):
        if isinstance(self.input, (datamodels.DataModel, str)):
            _input_list_tmp = [self.input]
        if  isinstance(self.input, Sequence):
            _input_list_tmp = self.input
        nfiles = len(_input_list_tmp)
        self.log.info(f"Deriving the detector saturation threshold by taking the median of the last reads of a list of {nfiles} inputs.")

        input_models = [datamodels.open(f) for f in _input_list_tmp]

        # Define output model from inputs
        output_saturation = self.get_output_model(input_models)

        last_read_arr = []
        for _input in input_models:
            last_read_arr.append(_input.data[-1,:,:])
        last_read_arr = np.array(last_read_arr)

        with numba_thread_scope(self.max_cores):
            output_saturation.SAT_THRESH = median_axis0_3d(last_read_arr)-self.offset

        # Output
        return dict(output=output_saturation)
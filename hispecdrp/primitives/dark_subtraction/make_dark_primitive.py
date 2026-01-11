from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np
from ...datamodels.dqflags import DQ_FLAGS
from typing import Sequence
from copy import copy

from numba import njit, prange
from ...utils.parallelization_utils import numba_thread_scope


__all__ = ['MakeDarkPrimitive']

HOT_FLAG =  DQ_FLAGS['HOT']
DO_NOT_USE_FLAG = DQ_FLAGS["DO_NOT_USE"]

class MakeDarkPrimitive(BasePrimitive):

    ARGUMENTS = {
        'output_model' : 'DarkModel',
        'max_cores' : 1,
        'hot_pixel_threshold' : 5,  # e-/s,  Threshold above which pixels are bad
    }

    # CALIBRATIONS = {}

    def _perform(self):
        if  isinstance(self.input, Sequence):
            raise ValueError("Rate maps must be combined before computing the detector dark. Input should not be a list.")

        input_model = datamodels.open(self.input)
        # Define output model from inputs
        output_detdark = self.get_output_model(input_model)

        # median_dark_level = np.nanmedian(input_model.rate)
        # Median dark rate for instrument monitoring
        # output_detdark.meta["MEDDARK"] = median_dark_level

        output_detdark.rate = input_model.rate
        output_detdark.rate_err = input_model.rate_err
        output_detdark.rate_var_rnoise = input_model.rate_var_rnoise
        output_detdark.rate_var_poisson = input_model.rate_var_poisson
        output_detdark.dq = copy(input_model.dq)

        with numba_thread_scope(self.max_cores):
            _detect_hot_pixels(output_detdark.rate, output_detdark.dq,self.hot_pixel_threshold)

        return dict(output=output_detdark)


@njit(nogil=True, parallel=True, cache=True)
def _detect_hot_pixels(dark_rate: np.ndarray, dq: np.ndarray, hot_pixel_threshold: float = 0.9):
    ny, nx = dark_rate.shape
    for j in prange(ny):
        for i in range(nx):
            if dark_rate[j, i] > hot_pixel_threshold:
                dq[j, i] |= HOT_FLAG


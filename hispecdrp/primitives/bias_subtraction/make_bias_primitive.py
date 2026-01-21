from ..base_primitive import BasePrimitive
from ...datamodels.dqflags import DQ_FLAGS
import numpy as np
from copy import copy
from typing import Sequence
from ... import datamodels
from ..jump_detection.jump_detection_difference import median_axis0_3d

from numba import njit, prange
from ...utils.parallelization_utils import numba_thread_scope

__all__ = ['MakeBiasPrimitive']

UNRELIABLE_BIAS_FLAG =  DQ_FLAGS['UNRELIABLE_BIAS']
DO_NOT_USE_FLAG = DQ_FLAGS["DO_NOT_USE"]

class MakeBiasPrimitive(BasePrimitive):
    """
    Derive the detector bias and kTC noise from a set of dark exposures.
    1) Median combine the first read to create the bias frame.
    2) Compute the kTC noise as the standard deviation across the first reads,
       using sigma-clipping to reject outliers.
    3) Flag pixels with high bias level or high kTC noise in the DQ array.

    todo: add median bias level and median kTC level to metadata for instrument monitoring

    Parameters
    ----------
    input : list of str or list of RampModel
        List of input file paths or ramp models to process.
    max_cores : int, optional
        Maximum number of CPU cores to use for parallel processing. Default is 1.
    kTC_threshold : float, optional
        Threshold kTC above which pixels are flagged as bad in the DQ array. Default is 100 DN.
    bias_threshold : float, optional
        Threshold bias above which pixels are flagged as bad in the DQ array. Default is 1000 DN.

    Returns
    -------
    output : BiasModel
        The derived bias calibration model containing the bias frame, kTC noise map, and updated DQ array.

    """

    ARGUMENTS = {
        'output_model' : 'BiasModel',
        'max_cores' : 1,
        'kTC_threshold' : 100,  # DN,  Threshold kTC above which pixels are bad
        'bias_threshold' : 1000,  # DN,  Threshold bias above which pixels are bad
    }

    # CALIBRATIONS = {}

    def _perform(self):
        if isinstance(self.input, (datamodels.DataModel, str)):
            _input_list_tmp = [self.input]
        if  isinstance(self.input, Sequence):
            _input_list_tmp = self.input
        nfiles = len(_input_list_tmp)
        self.log.info(f"Deriving the detector bias and kTC noise from a list of {nfiles} inputs.")

        input_models = [datamodels.open(f) for f in _input_list_tmp]

        # Define output model from inputs
        output_bias = self.get_output_model(input_models)

        bias_arr = []
        for _input in input_models:
            bias_arr.append(_input.data[0,:,:])
        bias_arr = np.array(bias_arr)

        with numba_thread_scope(self.max_cores):
            output_bias.bias = median_axis0_3d(bias_arr)
            output_bias.ktc = std_axis0_3d_sigclip(bias_arr,nsig=5,min_keep=10)
            output_bias.dq = copy(input_models[0].dq)
            _find_bad_pixels_bias(output_bias.bias, output_bias.ktc , output_bias.dq,
                                    self.kTC_threshold, self.bias_threshold)

        # median_bias_level = np.nanmedian(output_bias.bias)
        # Median bias for instrument monitoring
        # output_detdark.meta["MEDBIAS"] = median_bias_level
        # median_kTC_level = np.nanmedian(output_bias.ktc)
        # Median kTC noise level for instrument monitoring
        # output_detdark.meta["MEDKTC"] = median_kTC_level

        return dict(output=output_bias)


@njit(nogil=True, parallel=True, cache=True)
def _find_bad_pixels_bias(bias: np.ndarray, kTC: np.ndarray, dq: np.ndarray,
                       kTC_threshold: float = 100, bias_threshold: float = 1000):
    ny, nx = bias.shape
    for j in prange(ny):
        for i in range(nx):
            if kTC[j, i] > kTC_threshold:
                dq[j, i] |= UNRELIABLE_BIAS_FLAG
            if bias[j, i] > bias_threshold:
                dq[j, i] |= UNRELIABLE_BIAS_FLAG

@njit(nogil=True, parallel=True, cache=True)
def std_axis0_3d_sigclip(bias_arr, nsig=5.0, min_keep=10):
    """
    Sigma-clipped std over axis=0 for bias_arr with shape (n_files, ny, nx)

    Parameters
    ----------
    bias_arr : float array (n_files, ny, nx)
    nsig     : sigma threshold
    min_keep : minimum number of samples required after clipping

    Returns
    -------
    std : (ny, nx) array
    """
    n_files, ny, nx = bias_arr.shape
    std = np.zeros((ny, nx), dtype=bias_arr.dtype)

    for y in prange(ny):
        for x in range(nx):

            # ---------
            # Pass 1: initial mean/std (Welford)
            mean = 0.0
            M2 = 0.0
            count = 0

            for f in range(n_files):
                v = bias_arr[f, y, x]
                count += 1
                delta = v - mean
                mean += delta / count
                M2 += delta * (v - mean)

            if count < 2:
                std[y, x] = 0.0
                continue

            sigma = np.sqrt(M2 / count)

            # ---------
            # Pass 2: sigma clipping + recompute stats
            mean2 = 0.0
            M2_2 = 0.0
            count2 = 0
            thresh = nsig * sigma

            for f in range(n_files):
                v = bias_arr[f, y, x]
                if abs(v - mean) <= thresh:
                    count2 += 1
                    delta = v - mean2
                    mean2 += delta / count2
                    M2_2 += delta * (v - mean2)

            if count2 < min_keep:
                # fallback: unclipped std
                std[y, x] = sigma
            else:
                std[y, x] = np.sqrt(M2_2 / count2)

    return std
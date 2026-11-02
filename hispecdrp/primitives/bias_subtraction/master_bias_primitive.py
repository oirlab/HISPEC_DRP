from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np
from ...datamodels.dqflags import DQ_FLAGS
from typing import Sequence
from copy import copy
import logging

from ...utils.parallelization_utils import numba_thread_scope

logger = logging.getLogger(__name__)
from .make_bias_primitive import _find_bad_pixels_bias

__all__ = ['MakeMasterBiasPrimitive']

UNRELIABLE_BIAS_FLAG =  DQ_FLAGS['UNRELIABLE_BIAS']
DO_NOT_USE_FLAG = DQ_FLAGS["DO_NOT_USE"]

class MakeMasterBiasPrimitive(BasePrimitive):
    """
    Create a master bias calibration model from a list of bias calibrations (BiasModel).

    A master calibration is identified by setting the `meta.master_cal` attribute to True.

    TODO: implement automatic selection of biases to be combined for making the master bias

    Parameters
    ----------
    input : list of str or list of BiasModel
        List of input file paths or bias models to process.
    max_cores : int, optional
        Maximum number of CPU cores to use for parallel processing. Default is 1.
    kTC_threshold : float, optional
        Threshold kTC above which pixels are flagged as bad in the DQ array. Default is 100 DN.
    bias_threshold : float, optional
        Threshold bias above which pixels are flagged as bad in the DQ array. Default is 1000 DN.

    Returns
    -------
    output : BiasModel
        The derived master bias calibration model containing the averaged bias frame,
        averaged kTC noise map, and updated DQ array.

    """

    PARAMETERS = {
        'output_model' : 'BiasModel',
        'kTC_threshold' : 100,  # DN,  Threshold kTC above which pixels are bad
        'bias_threshold' : 1000,  # DN,  Threshold bias above which pixels are bad
    }

    def _perform(self):

        if self.dataset is not None: # How do you call a primitive without input?
            _inputs = self.dataset
        else:
            # todo implement automatic query of caldb to get all the darks between some dates with xyz filters
            raise NotImplementedError("Automatic calibration retrieval not yet implemented. List of darks must be provided as input.")


        input_type = type(_inputs)

        # Make sure input is a list of models
        if input_type in (str, datamodels.DataModel):
            logger.warning("Input is a single model or path, returning without coadding.")
            output_bias = _inputs.copy()
        if input_type in (list, tuple):
            input_models = [datamodels.open(f) for f in _inputs]
            ny,nx = input_models[0].bias.shape
            _bias = np.zeros((ny,nx), dtype=input_models[0].bias.dtype)
            _dq = np.zeros((ny,nx), dtype=input_models[0].dq.dtype)
            _kTC2 = np.zeros((ny,nx), dtype=input_models[0].ktc.dtype)
            for _input_model in input_models:
                _bias += _input_model.bias
                _kTC2 += _input_model.ktc**2
                _dq = np.bitwise_or(_dq, _input_model.dq)

            output_bias = self.get_output_model(input_models)

            output_bias.bias = _bias / len(input_models)
            output_bias.ktc = np.sqrt(_kTC2 / len(input_models))
            output_bias.dq = _dq

        max_cores = self.max_cores or 1
        with numba_thread_scope(max_cores):
            _find_bad_pixels_bias(
                output_bias.bias, output_bias.ktc , output_bias.dq,
                self.kTC_threshold, self.bias_threshold
            )

        output_bias.meta.master_cal = True
        median_bias_level = np.nanmedian(output_bias.bias)
        output_bias.meta.median_bias_level = median_bias_level
        median_kTC_level = np.nanmedian(output_bias.ktc)
        output_bias.meta.median_kTC_level = median_kTC_level

        return dict(output=output_bias)


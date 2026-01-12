from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np
from ...datamodels.dqflags import DQ_FLAGS
from typing import Sequence
from copy import copy

from numba import njit, prange
from ...utils.parallelization_utils import numba_thread_scope
from ..coadd.coadd_frames_primitive import coadd_group
from .make_bias_primitive import _find_bad_pixels_bias

__all__ = ['MakeMasterBiasPrimitive']

UNRELIABLE_BIAS_FLAG =  DQ_FLAGS['UNRELIABLE_BIAS']
DO_NOT_USE_FLAG = DQ_FLAGS["DO_NOT_USE"]

class MakeMasterBiasPrimitive(BasePrimitive):
    """

    """

    ARGUMENTS = {
        'output_model' : 'BiasModel',
        'kTC_threshold' : 100,  # DN,  Threshold kTC above which pixels are bad
        'bias_threshold' : 1000,  # DN,  Threshold bias above which pixels are bad
        'max_cores' : 1,
    }

    CALIBRATIONS = {
        # Todo to implement for automatic selection of darks to be combined for making the master dark
        # 'bias': {
        #     'selector': BiasSelector,
        #     'selector_kwargs': {}
        # }
    }


    def _perform(self):

        if self.input is not None: # How do you call a primitive without input?
            _inputs = self.input
        else:
            # todo implement automatic query of caldb to get all the darks between some dates with xyz filters
            raise NotImplementedError("Automatic calibration retrieval not yet implemented. List of darks must be provided as input.")


        input_type = type(_inputs)

        # Make sure input is a list of models
        if input_type in (str, datamodels.DataModel):
            self.log.warning("Input is a single model or path, returning without coadding.")
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

            output_bias.bias =_bias / len(input_models)
            output_bias.ktc = np.sqrt(_kTC2 / len(input_models))
            output_bias.dq = _dq

        # Flag this calibration as a master calibration
        output_bias.meta.master_cal = True

        with numba_thread_scope(self.max_cores):
            _find_bad_pixels_bias(output_bias.bias, output_bias.ktc , output_bias.dq,
                                    self.kTC_threshold, self.bias_threshold)

        # median_bias_level = np.nanmedian(output_bias.bias)
        # Median bias for instrument monitoring
        # output_detdark.meta["MEDBIAS"] = median_bias_level
        # median_kTC_level = np.nanmedian(output_bias.ktc)
        # Median kTC noise level for instrument monitoring
        # output_detdark.meta["MEDKTC"] = median_kTC_level

        return dict(output=output_bias)


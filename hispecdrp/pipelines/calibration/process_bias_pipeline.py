from hispecdrp import datamodels
from ..calibration_pipeline_base import CalibrationPipeline
from ...primitives import (DQInitPrimitive, MakeBiasPrimitive, RegisterCalibrationPrimitive)
from typing import Sequence

__all__ = ['ProcessBiasPipeline']


class ProcessBiasPipeline(CalibrationPipeline):

    PRIMITIVES = {
        'dq_init': DQInitPrimitive,
        'make_bias': MakeBiasPrimitive,
        'register_bias_cal' : RegisterCalibrationPrimitive,
    }

    def _perform(self):

        # Making sure that the input is a list of models for processing
        if isinstance(self.input, (datamodels.DataModel, str)):
            _input_list = [self.input]
        if  isinstance(self.input, Sequence):
            _input_list = self.input

        # First, loop over all inputs to apply DQ init
        results = []
        self.log.info(f"Looping over a list of {len(_input_list)} inputs to apply dq_init.")
        for k, _input in enumerate(_input_list):
            self.log.info(f"Processing input {k+1}/{len(_input_list)}: {_input}")
            model_to_process = datamodels.open(_input)
            _result = self.dq_init.apply(model_to_process)
            results.append(_result)

        # Fit gain using all processed inputs
        bias_result = self.make_bias.apply(results)

        # Register the detector gain calibration to the calibration database
        # self.register_bias_cal.apply(bias_result)

        return dict(output=dict(results=results, bias=bias_result))

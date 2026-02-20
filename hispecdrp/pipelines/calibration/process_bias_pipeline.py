import logging

from hispecdrp import datamodels
from ..calibration_pipeline_base import CalibrationPipeline
from ...primitives import (DQInitPrimitive, MakeBiasPrimitive, RegisterCalibrationPrimitive)
from typing import Sequence

logger = logging.getLogger(__name__)

__all__ = ['ProcessBiasPipeline']


class ProcessBiasPipeline(CalibrationPipeline):
    """
    Pipeline to process bias frames and create a master bias calibration.

    This pipeline applies the following primitives in order:

    - :class:`~hispecdrp.primitives.dq_init.dqinit_primitive.DQInitPrimitive`
    - :class:`~hispecdrp.primitives.bias_subtraction.make_bias_primitive.MakeBiasPrimitive`
    - :class:`~hispecdrp.primitives.register_calibration.register_cal_primitive.RegisterCalibrationPrimitive`

    Any primitive can be skipped by setting its ``run`` argument to ``False`` when initializing the pipeline.
    """

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
        logger.info(f"Looping over a list of {len(_input_list)} inputs to apply dq_init.")
        for k, _input in enumerate(_input_list):
            logger.info(f"Processing input {k+1}/{len(_input_list)}: {_input}")
            model_to_process = datamodels.open(_input)
            _result = self.dq_init.apply(model_to_process)
            results.append(_result)

        # Fit gain using all processed inputs
        bias_result = self.make_bias.apply(results)

        # Register the detector gain calibration to the calibration database
        # self.register_bias_cal.apply(bias_result)

        return dict(output=dict(results=results, bias=bias_result))

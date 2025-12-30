from hispecdrp import datamodels
from ..calibration_pipeline_base import CalibrationPipeline
from ...primitives import (DQInitPrimitive, CheckSaturationPrimitive,BiasSubtractionPrimitive,JumpDetectionPrimitive,
                           FitDetectorGainPrimitive,RegisterCalibrationPrimitive)
from typing import Sequence

__all__ = ['ProcessDetflatsPipeline']


class ProcessDetflatsPipeline(CalibrationPipeline):

    PRIMITIVES = {
        'dq_init': DQInitPrimitive,
        'saturation': CheckSaturationPrimitive,
        'bias': BiasSubtractionPrimitive,
        'jump_detec': JumpDetectionPrimitive,
        'fit_det_gain': FitDetectorGainPrimitive,
        'register_detgain_cal' : RegisterCalibrationPrimitive,
        # register read noise cal
        # repeat same for detector flats
    }

    def _perform(self):

        # Making sure that the input is a list of models for processing
        if isinstance(self.input, (datamodels.DataModel, str)):
            _input_list = [self.input]
        if  isinstance(self.input, Sequence):
            _input_list = self.input

        # First, loop over all inputs to apply DQ init, saturation check, and bias subtraction
        results = []
        self.log.info(f"Looping over a list of {len(_input_list)} inputs to apply dq_init, saturation, and bias subtraction.")
        for k, _input in enumerate(_input_list):
            self.log.info(f"Processing input {k+1}/{len(_input_list)}: {_input}")
            model_to_process = datamodels.open(_input)
            _result = self.dq_init.apply(model_to_process)
            _result = self.saturation.apply(_result)
            _result = self.bias.apply(_result)
            _result = self.jump_detec.apply(_result)
            results.append(_result)

        # Fit gain using all processed inputs
        detgain_result = self.fit_det_gain.apply(results)

        # Register the detector gain calibration to the calibration database
        # self.register_detgain_cal.apply(detgain_result)

        return dict(output=dict(detgain=detgain_result, detflat=None))

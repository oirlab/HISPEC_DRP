from hispecdrp import datamodels
from ..calibration_pipeline_base import CalibrationPipeline
from ...primitives import (DQInitPrimitive, CheckSaturationPrimitive, NonlinearCorrectionPrimitive,
                           JumpDetectionPrimitive,FitRampPrimitive, DarkSubtractionPrimitive,
                           BiasSubtractionPrimitive,DetectorFlatPrimitive, DetectorGainPrimitive,
                           FitSaturationPrimitive,FitNonLinPrimitive,
                           RegisterCalibrationPrimitive,
                           CoaddFramesPrimitive)
from typing import Sequence

__all__ = ['ProcessDetSatFlatsPipeline']


class ProcessDetSatFlatsPipeline(CalibrationPipeline):

    PRIMITIVES = {
        'dq_init': DQInitPrimitive,
        'fit_saturation': FitSaturationPrimitive,
        'register_saturation_cal' : RegisterCalibrationPrimitive,
        'saturation': CheckSaturationPrimitive,
        'fit_nonlin': FitNonLinPrimitive,
        'register_nonlin_cal' : RegisterCalibrationPrimitive,
        'bias': BiasSubtractionPrimitive,
        'nonlin_corr': NonlinearCorrectionPrimitive,
        'jump_detec': JumpDetectionPrimitive,
        'ramp_fit': FitRampPrimitive,
        'dark': DarkSubtractionPrimitive,
        'det_gain': DetectorGainPrimitive,
        'det_flat': DetectorFlatPrimitive,
    }

    def _perform(self):

        # Making sure that the input is a list of models for processing
        if isinstance(self.input, (datamodels.DataModel, str)):
            _input_list = [self.input]
        if  isinstance(self.input, Sequence):
            _input_list = self.input

        # First, loop over all inputs to apply DQ init, saturation check, and bias subtraction
        results = []
        self.log.info(f"Looping over a list of {len(_input_list)} inputs to apply dq_init.")
        for k, _input in enumerate(_input_list):
            self.log.info(f"Processing input {k+1}/{len(_input_list)}: {_input}")
            model_to_process = datamodels.open(_input)
            _result = self.dq_init.apply(model_to_process)
            results.append(_result)

        # Fit for nonlinearity using all processed inputs
        saturation_result = self.fit_saturation.apply(results)

        # Register the nonlinear correction calibration to the calibration database
        # self.register_nonlin_cal.apply(nonlin_result)

        self.log.info(f"Continue loop over inputs to apply saturation check.")
        for k, _result in enumerate(results):
            self.log.info(f"Processing input {k+1}/{len(results)}: {_result}")
            _result = self.saturation.apply(_result)
            results[k] = _result

        # Fit for nonlinearity using all processed inputs
        nonlin_result = self.fit_nonlin.apply(results)

        # Register the nonlinear correction calibration to the calibration database
        # self.register_nonlin_cal.apply(nonlin_result)

        self.log.info(f"Continue loop over inputs to apply bias, nonlin, jump_detec, ramp_fit, dark, det_gain, and det_flat.")
        for k, _result in enumerate(results):
            self.log.info(f"Processing input {k+1}/{len(results)}: {_result}")
            _result = self.bias.apply(_result)
            _result = self.nonlin_corr.apply(_result)
            _result = self.jump_detec.apply(_result)
            _result = self.ramp_fit.apply(_result)
            _result = self.dark.apply(_result)
            _result = self.det_gain.apply(_result)
            _result = self.det_flat.apply(_result)
            results[k] = _result

        return dict(output=dict(results=results, nonlin=nonlin_result, saturation=saturation_result))

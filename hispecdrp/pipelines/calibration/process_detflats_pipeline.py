from hispecdrp import datamodels
from ..calibration_pipeline_base import CalibrationPipeline
from ...primitives import (DQInitPrimitive, CheckSaturationPrimitive, NonlinearCorrectionPrimitive,
                           JumpDetectionPrimitive, FitRampPrimitive, DarkSubtractionPrimitive,
                           BiasSubtractionPrimitive, DetectorFlatCorrectionPrimitive, GainCorrectionPrimitive,
                           MakeGainPrimitive, MakeDetectorFlatPrimitive, RegisterCalibrationPrimitive,
                           CoaddFramesPrimitive)
from typing import Sequence

__all__ = ['ProcessDetFlatsPipeline']


class ProcessDetFlatsPipeline(CalibrationPipeline):

    PRIMITIVES = {
        'dq_init': DQInitPrimitive,
        'saturation_check': CheckSaturationPrimitive,
        'bias_sub': BiasSubtractionPrimitive,
        'nonlin_corr': NonlinearCorrectionPrimitive,
        'jump_detec': JumpDetectionPrimitive,
        'make_gain': MakeGainPrimitive,
        'register_gain_cal' : RegisterCalibrationPrimitive,
        'ramp_fit': FitRampPrimitive,
        'dark_sub': DarkSubtractionPrimitive,
        'gain_corr': GainCorrectionPrimitive,
        'coadd': CoaddFramesPrimitive,
        'make_detflat': MakeDetectorFlatPrimitive,
        'register_det_flat_cal' : RegisterCalibrationPrimitive,
        'detflat_corr': DetectorFlatCorrectionPrimitive,
    }

    def _perform(self):

        # Making sure that the input is a list of models for processing
        if isinstance(self.input, (datamodels.DataModel, str)):
            _input_list = [self.input]
        if  isinstance(self.input, Sequence):
            _input_list = self.input

        # First, loop over all inputs to apply DQ init, saturation check, and bias subtraction
        results = []
        self.log.info(f"Looping over a list of {len(_input_list)} inputs to apply dq_init, saturation, bias subtraction, and nonlin_corr.")
        for k, _input in enumerate(_input_list):
            self.log.info(f"Processing input {k+1}/{len(_input_list)}: {_input}")
            model_to_process = datamodels.open(_input)
            _result = self.dq_init.apply(model_to_process)
            _result = self.saturation_check.apply(_result)
            _result = self.bias_sub.apply(_result)
            _result = self.nonlin_corr.apply(_result)
            _result = self.jump_detec.apply(_result)
            results.append(_result)

        # Fit gain using all processed inputs
        gain_result = self.make_gain.apply(results)

        # Register the detector gain calibration to the calibration database
        # self.register_gain_cal.apply(gain_result)

        self.log.info(f"Continue loop over inputs to apply ramp_fit, dark, and det_gain.")
        for k, _result in enumerate(results):
            self.log.info(f"Processing input {k+1}/{len(results)}: {_result}")
            _result = self.ramp_fit.apply(_result)
            _result = self.dark_sub.apply(_result)
            _result = self.gain_corr.apply(_result)
            results[k] = _result

        # Get the flat field using all processed inputs. Coadding first.
        coadded_rate = self.coadd.apply(results)[0]
        detflat_result = self.make_detflat.apply(coadded_rate)

        # Register the detector gain calibration to the calibration database
        # self.register_det_flat_cal.apply(detflat_result)

        self.log.info(f"Continue loop over inputs to apply flat fielding.")
        for k, _result in enumerate(results):
            self.log.info(f"Processing input {k+1}/{len(results)}: {_result}")
            _result = self.detflat_corr.apply(_result)
            results[k] = _result

        return dict(output=dict(results=results, gain=gain_result, detflat=detflat_result))

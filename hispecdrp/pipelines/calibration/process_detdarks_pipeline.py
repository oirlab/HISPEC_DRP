from hispecdrp import datamodels
from ..calibration_pipeline_base import CalibrationPipeline
from ...primitives import (DQInitPrimitive, FitDetectorBiasPrimitive, CheckSaturationPrimitive, NonlinearCorrectionPrimitive,
                           JumpDetectionPrimitive,FitRampPrimitive, DarkSubtractionPrimitive,
                           BiasSubtractionPrimitive,DetectorFlatPrimitive, DetectorGainPrimitive,
                           FitDetectorReadnoisePrimitive,FitDetectorDarkPrimitive, RegisterCalibrationPrimitive,
                           CoaddFramesPrimitive)
from typing import Sequence

__all__ = ['ProcessDetDarksPipeline']


class ProcessDetDarksPipeline(CalibrationPipeline):

    PRIMITIVES = {
        'dq_init': DQInitPrimitive,
        'fit_bias': FitDetectorBiasPrimitive,
        'register_bias_cal' : RegisterCalibrationPrimitive,
        'saturation': CheckSaturationPrimitive,
        'bias': BiasSubtractionPrimitive,
        'nonlin_corr': NonlinearCorrectionPrimitive,
        'jump_detec': JumpDetectionPrimitive,
        'fit_det_rn': FitDetectorReadnoisePrimitive,
        'register_detrn_cal' : RegisterCalibrationPrimitive,
        'ramp_fit': FitRampPrimitive,
        'coadd': CoaddFramesPrimitive,
        'fit_det_dark': FitDetectorDarkPrimitive,
        'register_detdark_cal' : RegisterCalibrationPrimitive,
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

        # First, loop over all inputs to apply DQ init
        results = []
        self.log.info(f"Looping over a list of {len(_input_list)} inputs to apply dq_init, saturation, bias subtraction, nonlin_corr, and jump detection.")
        for k, _input in enumerate(_input_list):
            self.log.info(f"Processing input {k+1}/{len(_input_list)}: {_input}")
            model_to_process = datamodels.open(_input)
            _result = self.dq_init.apply(model_to_process)
            _result = self.saturation.apply(_result)
            _result = self.bias.apply(_result)
            _result = self.nonlin_corr.apply(_result)
            _result = self.jump_detec.apply(_result)
            results.append(_result)

        # Fit gain using all processed inputs
        detrn_result = self.fit_det_rn.apply(results)

        # Register the detector gain calibration to the calibration database
        # self.register_detrn_cal.apply(detrn_result)

        self.log.info(f"Continue loop over inputs to apply ramp_fit.")
        for k, _result in enumerate(results):
            self.log.info(f"Processing input {k+1}/{len(results)}: {_result}")
            _result = self.ramp_fit.apply(_result)
            results[k] = _result

        # Get the flat field using all processed inputs. Coadding first.
        coadded_rate = self.coadd.apply(results)[0]
        detdark_result = self.fit_det_dark.apply(coadded_rate)

        # Register the detector gain calibration to the calibration database
        # self.register_detdark_cal.apply(detflat_result)

        self.log.info(f"Continue loop over inputs to apply dark, gain, and flat fielding.")
        for k, _result in enumerate(results):
            self.log.info(f"Processing input {k+1}/{len(results)}: {_result}")
            _result = self.dark.apply(_result)
            _result = self.det_gain.apply(_result)
            _result = self.det_flat.apply(_result)
            results[k] = _result

        return dict(output=dict(results=results, detrn=detrn_result, detdark=detdark_result))

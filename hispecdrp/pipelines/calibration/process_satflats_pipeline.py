from hispecdrp import datamodels
from ..calibration_pipeline_base import CalibrationPipeline
from ...primitives import (DQInitPrimitive, CheckSaturationPrimitive, NonlinearCorrectionPrimitive,
                           JumpDetectionPrimitive, FitRampPrimitive, DarkSubtractionPrimitive,
                           BiasSubtractionPrimitive, DetectorFlatCorrectionPrimitive, GainCorrectionPrimitive,
                           MakeSaturationPrimitive, MakeNonLinPrimitive,
                           RegisterCalibrationPrimitive,
                           CoaddFramesPrimitive)
from typing import Sequence
import logging

logger = logging.getLogger(__name__)

__all__ = ['ProcessSatFlatsPipeline']


class ProcessSatFlatsPipeline(CalibrationPipeline):
    """
    Pipeline to process saturation flat frames and create master calibrations for nonlinearity and saturation levels.

    This pipeline applies the following primitives in order:

    - :class:`~hispecdrp.primitives.dq_init.dqinit_primitive.DQInitPrimitive`
    - :class:`~hispecdrp.primitives.saturation.make_saturation_primitive.MakeSaturationPrimitive`
    - :class:`~hispecdrp.primitives.register_calibration.register_cal_primitive.RegisterCalibrationPrimitive`
    - :class:`~hispecdrp.primitives.saturation.saturation_primitive.CheckSaturationPrimitive`
    - :class:`~hispecdrp.primitives.bias_subtraction.bias_primitive.BiasSubtractionPrimitive`
    - :class:`~hispecdrp.primitives.nonlinear_correction.make_nonlin_primitive.MakeNonLinPrimitive`
    - :class:`~hispecdrp.primitives.register_calibration.register_cal_primitive.RegisterCalibrationPrimitive`
    - :class:`~hispecdrp.primitives.nonlinear_correction.nonlin_primitive.NonlinearCorrectionPrimitive`
    - :class:`~hispecdrp.primitives.jump_detection.jump_detection_primitive.JumpDetectionPrimitive`
    - :class:`~hispecdrp.primitives.ramp_fitting.ramp_fit_primitive.FitRampPrimitive`
    - :class:`~hispecdrp.primitives.dark_subtraction.dark_primitive.DarkSubtractionPrimitive`
    - :class:`~hispecdrp.primitives.gain.gain_primitive.GainCorrectionPrimitive`
    - :class:`~hispecdrp.primitives.detector_flat.detflat_primitive.DetectorFlatCorrectionPrimitive`

    Any primitive can be skipped by setting its ``run`` argument to ``False`` when initializing the pipeline.
    """

    PRIMITIVES = {
        'dq_init': DQInitPrimitive,
        'make_saturation': MakeSaturationPrimitive,
        'register_saturation_cal' : RegisterCalibrationPrimitive,
        'saturation_check': CheckSaturationPrimitive,
        'bias_sub': BiasSubtractionPrimitive,
        'make_nonlin': MakeNonLinPrimitive,
        'register_nonlin_cal' : RegisterCalibrationPrimitive,
        'nonlin_corr': NonlinearCorrectionPrimitive,
        'jump_detec': JumpDetectionPrimitive,
        'ramp_fit': FitRampPrimitive,
        'dark_sub': DarkSubtractionPrimitive,
        'gain_corr': GainCorrectionPrimitive,
        'detflat_corr': DetectorFlatCorrectionPrimitive,
    }

    def _perform(self):

        # Making sure that the input is a list of models for processing
        if isinstance(self.dataset, (datamodels.DataModel, str)):
            _input_list = [self.dataset]
        if  isinstance(self.dataset, Sequence):
            _input_list = self.dataset

        # First, loop over all inputs to apply DQ init, saturation check, and bias subtraction
        results = []
        logger.info(f"Looping over a list of {len(_input_list)} inputs to apply dq_init.")
        for k, _input in enumerate(_input_list):
            logger.info(f"Processing input {k+1}/{len(_input_list)}: {_input}")
            model_to_process = datamodels.open(_input)
            _result = self.dq_init.apply(model_to_process)
            results.append(_result)

        # Fit for nonlinearity using all processed inputs
        saturation_result = self.make_saturation.apply(results)

        # Register the nonlinear correction calibration to the calibration database
        # self.register_nonlin_cal.apply(nonlin_result)

        logger.info(f"Continue loop over inputs to apply saturation check.")
        for k, _result in enumerate(results):
            logger.info(f"Processing input {k+1}/{len(results)}: {_result}")
            _result = self.saturation_check.apply(_result)
            _result = self.bias_sub.apply(_result)
            results[k] = _result

        # Fit for nonlinearity using all processed inputs
        nonlin_result = self.make_nonlin.apply(results)

        # Register the nonlinear correction calibration to the calibration database
        # self.register_nonlin_cal.apply(nonlin_result)

        logger.info(f"Continue loop over inputs to apply bias, nonlin_corr, jump_detec, ramp_fit, dark_sub, gain_corr, and detflat_corr.")
        for k, _result in enumerate(results):
            logger.info(f"Processing input {k+1}/{len(results)}: {_result}")
            _result = self.nonlin_corr.apply(_result)
            _result = self.jump_detec.apply(_result)
            _result = self.ramp_fit.apply(_result)
            _result = self.dark_sub.apply(_result)
            _result = self.gain_corr.apply(_result)
            _result = self.detflat_corr.apply(_result)
            results[k] = _result

        return dict(output=dict(results=results, nonlin=nonlin_result, saturation=saturation_result))

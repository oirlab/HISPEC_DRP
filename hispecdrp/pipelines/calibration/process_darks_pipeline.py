from hispecdrp import datamodels
from ..calibration_pipeline_base import CalibrationPipeline
from ...primitives import (DQInitPrimitive, MakeBiasPrimitive, CheckSaturationPrimitive, NonlinearCorrectionPrimitive,
                           JumpDetectionPrimitive, FitRampPrimitive, DarkSubtractionPrimitive,
                           BiasSubtractionPrimitive, DetectorFlatCorrectionPrimitive, GainCorrectionPrimitive,
                           MakeReadnoisePrimitive, MakeDarkPrimitive, RegisterCalibrationPrimitive,
                           CoaddFramesPrimitive)
from typing import Sequence

__all__ = ['ProcessDarksPipeline']


class ProcessDarksPipeline(CalibrationPipeline):
    """
    Pipeline to process dark frames and create master calibrations for read noise and dark current.

    This pipeline applies the following primitives in order:

    - :class:`~hispecdrp.primitives.dq_init.dqinit_primitive.DQInitPrimitive`
    - :class:`~hispecdrp.primitives.bias_subtraction.make_bias_primitive.MakeBiasPrimitive`
    - :class:`~hispecdrp.primitives.register_calibration.register_cal_primitive.RegisterCalibrationPrimitive`
    - :class:`~hispecdrp.primitives.saturation.saturation_primitive.CheckSaturationPrimitive`
    - :class:`~hispecdrp.primitives.bias_subtraction.bias_primitive.BiasSubtractionPrimitive`
    - :class:`~hispecdrp.primitives.nonlinear_correction.nonlin_primitive.NonlinearCorrectionPrimitive`
    - :class:`~hispecdrp.primitives.jump_detection.jump_detection_primitive.JumpDetectionPrimitive`
    - :class:`~hispecdrp.primitives.readnoise.make_rn_primitive.MakeReadnoisePrimitive`
    - :class:`~hispecdrp.primitives.register_calibration.register_cal_primitive.RegisterCalibrationPrimitive`
    - :class:`~hispecdrp.primitives.ramp_fitting.ramp_fit_primitive.FitRampPrimitive`
    - :class:`~hispecdrp.primitives.coadd.coadd_frames_primitive.CoaddFramesPrimitive`
    - :class:`~hispecdrp.primitives.dark_subtraction.make_dark_primitive.MakeDarkPrimitive`
    - :class:`~hispecdrp.primitives.register_calibration.register_cal_primitive.RegisterCalibrationPrimitive`
    - :class:`~hispecdrp.primitives.dark_subtraction.dark_primitive.DarkSubtractionPrimitive`
    - :class:`~hispecdrp.primitives.gain.gain_primitive.GainCorrectionPrimitive`
    - :class:`~hispecdrp.primitives.detector_flat.detflat_primitive.DetectorFlatCorrectionPrimitive`

    Any primitive can be skipped by setting its ``run`` argument to ``False`` when initializing the pipeline.
    """
    PRIMITIVES = {
        'dq_init': DQInitPrimitive,
        'make_bias': MakeBiasPrimitive,
        'register_bias_cal' : RegisterCalibrationPrimitive,
        'saturation_check': CheckSaturationPrimitive,
        'bias_sub': BiasSubtractionPrimitive,
        'nonlin_corr': NonlinearCorrectionPrimitive,
        'jump_detec': JumpDetectionPrimitive,
        'make_rn': MakeReadnoisePrimitive,
        'register_rn_cal' : RegisterCalibrationPrimitive,
        'ramp_fit': FitRampPrimitive,
        'coadd': CoaddFramesPrimitive,
        'make_dark': MakeDarkPrimitive,
        'register_dark_cal' : RegisterCalibrationPrimitive,
        'dark_sub': DarkSubtractionPrimitive,
        'gain_corr': GainCorrectionPrimitive,
        'detflat_corr': DetectorFlatCorrectionPrimitive,
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
            _result = self.saturation_check.apply(_result)
            _result = self.bias_sub.apply(_result)
            _result = self.nonlin_corr.apply(_result)
            _result = self.jump_detec.apply(_result)
            results.append(_result)

        # Fit gain using all processed inputs
        rn_result = self.make_rn.apply(results)

        # Register the detector gain calibration to the calibration database
        # self.register_rn_cal.apply(rn_result)

        self.log.info(f"Continue loop over inputs to apply ramp_fit.")
        for k, _result in enumerate(results):
            self.log.info(f"Processing input {k+1}/{len(results)}: {_result}")
            _result = self.ramp_fit.apply(_result)
            results[k] = _result

        # Get the flat field using all processed inputs. Coadding first.
        coadded_rate = self.coadd.apply(results)[0]
        dark_result = self.make_dark.apply(coadded_rate)

        # Register the detector gain calibration to the calibration database
        # self.register_dark_cal.apply(dark_result)

        self.log.info(f"Continue loop over inputs to apply dark, gain, and flat fielding.")
        for k, _result in enumerate(results):
            self.log.info(f"Processing input {k+1}/{len(results)}: {_result}")
            _result = self.dark_sub.apply(_result)
            _result = self.gain_corr.apply(_result)
            _result = self.detflat_corr.apply(_result)
            results[k] = _result

        return dict(output=dict(results=results, rn=rn_result, dark=dark_result))

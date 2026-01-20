from hispecdrp import datamodels
from ..base_pipeline import BasePipeline
from ...primitives import (DQInitPrimitive, CheckSaturationPrimitive, NonlinearCorrectionPrimitive,
                           JumpDetectionPrimitive, FitRampPrimitive, DarkSubtractionPrimitive,
                           BiasSubtractionPrimitive, DetectorFlatCorrectionPrimitive, GainCorrectionPrimitive,
                           RefPixelCorrectionPrimitive)

__all__ = ['Stage0Pipeline']


class Stage0Pipeline(BasePipeline):
    """
    Stage 0 detector-level reduction pipeline.
    Process Raw UTR data (RampModel class) to produce calibrated slope images (L0Model class).

    This pipeline applies the following primitives in order:
    - :class:`hispecdrp.primitives.dq_init.dqinit_primitive.DQInitPrimitive`
    - :class:`hispecdrp.primitives.saturation.saturation_primitive.CheckSaturationPrimitive`
    - :class:`hispecdrp.primitives.bias_primitive.BiasSubtractionPrimitive`
    - :class:`hispecdrp.primitives.refpix.refpix_primitive.RefPixelCorrectionPrimitive`
    - :class:`hispecdrp.primitives.nonlinear_correction.nonlin_primitive.NonlinearCorrectionPrimitive`
    - :class:`hispecdrp.primitives.jump_detection.jump_detection_primitive.JumpDetectionPrimitive`
    - :class:`hispecdrp.primitives.ramp_fitting.ramp_fit_primitive.FitRampPrimitive`
    - :class:`hispecdrp.primitives.dark_subtraction.dark_primitive.DarkSubtractionPrimitive`
    - :class:`hispecdrp.primitives.gain.gain_primitive.GainCorrectionPrimitive`
    - :class:`hispecdrp.primitives.detector_flat.detflat_primitive.DetectorFlatCorrectionPrimitive`

    """

    PRIMITIVES = {
        'dq_init': DQInitPrimitive,
        'saturation_check': CheckSaturationPrimitive,
        'bias_sub': BiasSubtractionPrimitive,
        'refpix_corr': RefPixelCorrectionPrimitive,
        'nonlin_corr': NonlinearCorrectionPrimitive,
        'jump_detec': JumpDetectionPrimitive,
        'ramp_fit': FitRampPrimitive,
        'dark_sub': DarkSubtractionPrimitive,
        'gain_corr': GainCorrectionPrimitive,
        'detflat_corr': DetectorFlatCorrectionPrimitive,
    }

    def _perform(self):
        model_to_process = datamodels.open(self.input)
        result = self.dq_init.apply(model_to_process)
        result = self.saturation_check.apply(result)
        result = self.bias_sub.apply(result)
        result = self.refpix_corr.apply(result)
        result = self.nonlin_corr.apply(result)
        result = self.jump_detec.apply(result)
        result = self.ramp_fit.apply(result)
        result = self.dark_sub.apply(result)
        result = self.gain_corr.apply(result)
        result = self.detflat_corr.apply(result)
        return dict(output=result)
from hispecdrp import datamodels
from ..base_pipeline import BasePipeline
from ...primitives import (DQInitPrimitive, CheckSaturationPrimitive, NonlinearCorrectionPrimitive,
                           JumpDetectionPrimitive,FitRampPrimitive, DarkSubtractionPrimitive,
                           BiasSubtractionPrimitive,DetectorFlatPrimitive, DetectorGainPrimitive)

__all__ = ['Stage0Pipeline']


class Stage0Pipeline(BasePipeline):

    PRIMITIVES = {
        'dq_init': DQInitPrimitive,
        'saturation': CheckSaturationPrimitive,
        'bias': BiasSubtractionPrimitive,
        'nonlin_corr': NonlinearCorrectionPrimitive,
        'jump_detec': JumpDetectionPrimitive,
        'ramp_fit': FitRampPrimitive,
        'dark': DarkSubtractionPrimitive,
        'det_gain': DetectorGainPrimitive,
        'det_flat': DetectorFlatPrimitive,
    }

    def _perform(self):
        model_to_process = datamodels.open(self.input)
        result = self.dq_init.apply(model_to_process)
        result = self.saturation.apply(result)
        result = self.bias.apply(result)
        result = self.nonlin_corr.apply(result)
        result = self.jump_detec.apply(result)
        result = self.ramp_fit.apply(result)
        result = self.dark.apply(result)
        result = self.det_gain.apply(result)
        result = self.det_flat.apply(result)
        return dict(output=result)
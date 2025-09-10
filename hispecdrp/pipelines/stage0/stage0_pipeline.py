from hispecdrp import datamodels
from ..base_pipeline import BasePipeline
from ...primitives import (CheckSaturationPrimitive,NonlinearCorrectionPrimitive, FitRampPrimitive,
                           DarkSubtractionPrimitive, BiasSubtractionPrimitive,
                           DetectorFlatPrimitive, DetectorGainPrimitive)

__all__ = ['Stage0Pipeline']


class Stage0Pipeline(BasePipeline):

    PRIMITIVES = {
        'saturation': CheckSaturationPrimitive,
        'bias': BiasSubtractionPrimitive,
        'nonlin_corr': NonlinearCorrectionPrimitive,
        'ramp_fit': FitRampPrimitive,
        'dark': DarkSubtractionPrimitive,
        'det_gain': DetectorGainPrimitive,
        'det_flat': DetectorFlatPrimitive,
    }

    def _perform(self):
        model_to_process = datamodels.open(self.input)
        result = self.saturation.apply(model_to_process)
        result = self.bias.apply(result)
        result = self.nonlin_corr.apply(result)
        result = self.ramp_fit.apply(result)
        result = self.dark.apply(result)
        result = self.det_gain.apply(result)
        result = self.det_flat.apply(result)
        return dict(output=result)
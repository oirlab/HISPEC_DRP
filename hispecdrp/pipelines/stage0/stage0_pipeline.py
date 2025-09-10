from hispecdrp import datamodels
from ..base_pipeline import BasePipeline
from ...primitives import NonlinearCorrectionPrimitive, FitRampPrimitive, DarkSubtractionPrimitive, DetectorFlatPrimitive

__all__ = ['Stage0Pipeline']


class Stage0Pipeline(BasePipeline):

    PRIMITIVES = {
        'nonlin_corr': NonlinearCorrectionPrimitive,
        'ramp_fit': FitRampPrimitive,
        'dark': DarkSubtractionPrimitive,
        'det_flat': DetectorFlatPrimitive,
    }

    def _perform(self):
        model_to_process = datamodels.open(self.input)
        result = self.nonlin_corr.apply(model_to_process)
        result = self.ramp_fit.apply(result)
        result = self.dark.apply(result)
        result = self.det_flat.apply(result)
        return dict(output=result)
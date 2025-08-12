from hispecdrp import datamodels
from ..base_pipeline import BasePipeline
from ...primitives import NonlinearCorrectionPrimitive, FitRampPrimitive

__all__ = ['Stage0Pipeline']


class Stage0Pipeline(BasePipeline):

    PRIMITIVES = {
        'nonlin_corr': NonlinearCorrectionPrimitive,
        'ramp_fit': FitRampPrimitive,
    }

    def _perform(self):
        model_to_process = datamodels.open(self.input)
        result = self.nonlin_corr.apply(model_to_process)
        result = self.ramp_fit.apply(result)
        return dict(output=result)
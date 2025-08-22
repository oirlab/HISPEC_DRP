from hispecdrp import datamodels
from ..base_pipeline import BasePipeline
from ...primitives import DarkSubtractionPrimitive, DetectorFlatPrimitive

__all__ = ['Stage1Pipeline']

class Stage1Pipeline(BasePipeline):

    PRIMITIVES = {
        'dark_sub': DarkSubtractionPrimitive,
        'det_flat': DetectorFlatPrimitive,
    }

    def _perform(self):
        model_to_process = datamodels.open(self.input)
        result = self.dark_sub.apply(model_to_process)
        result = self.det_flat.apply(result)
        return dict(output=result)
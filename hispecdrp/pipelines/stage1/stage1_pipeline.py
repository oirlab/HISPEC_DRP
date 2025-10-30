from hispecdrp import datamodels
from ..base_pipeline import BasePipeline
from ...primitives import SpectralExtractionPrimitive

__all__ = ['Stage1Pipeline']

class Stage1Pipeline(BasePipeline):

    PRIMITIVES = {
        'spec_extract' : SpectralExtractionPrimitive
    }

    def _perform(self):
        input_model = datamodels.open(self.input)
        result = self.spec_extract.apply(input_model)
        return dict(output=result)
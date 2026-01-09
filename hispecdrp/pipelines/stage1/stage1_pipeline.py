from hispecdrp import datamodels
from ..base_pipeline import BasePipeline
from ...primitives import SpectralExtractionPrimitive, SetInitialWLSPrimitive

__all__ = ['Stage1Pipeline']

class Stage1Pipeline(BasePipeline):

    PRIMITIVES = {
        'spec_extract' : SpectralExtractionPrimitive,
        'set_wavesol' : SetInitialWLSPrimitive,
    }

    def _perform(self):
        input_model = datamodels.open(self.input)
        result = self.spec_extract.apply(input_model)
        result = self.set_wavesol.apply(result)
        return dict(output=result)
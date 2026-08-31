from hispecdrp import datamodels
from ..base_pipeline import BasePipeline
from ...primitives import SpectralExtractionPrimitive, SetInitialWLSPrimitive

__all__ = ['Stage1Pipeline']

class Stage1Pipeline(BasePipeline):

    """
    Pipeline to process level 0 (rate map) data to generate level 1 (uncalibrated spectra) data.

    This pipeline performs the following steps:

    - :class:`~hispecdrp.primitives.spectral_extraction.spectral_extraction_primitive.SpectralExtractionPrimitive`
    - :class:`~hispecdrp.primitives.wavesol.set_wavesol_primitive.SetInitialWLSPrimitive`
    
    """

    PRIMITIVES = {
        'spec_extract' : SpectralExtractionPrimitive,
        'set_wavesol' : SetInitialWLSPrimitive,
    }

    def _perform(self):
        input_model = datamodels.open(self.dataset)
        result = self.spec_extract.apply(input_model)
        result = self.set_wavesol.apply(result)
        return dict(output=result)
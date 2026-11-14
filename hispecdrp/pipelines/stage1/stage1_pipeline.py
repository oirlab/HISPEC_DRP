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
        results = []
        for i, _input in self._wrap_dataset():
            model = self.spec_extract.apply(_input)
            model = self.set_wavesol.apply(model)
            results.append(model)
        return dict(output=self._unwrap_dataset(results))
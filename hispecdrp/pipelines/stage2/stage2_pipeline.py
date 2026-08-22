from hispecdrp import datamodels
from ..base_pipeline import BasePipeline
from ...primitives import (
    BarycenterCorrectionPrimitive,
    BlazeCorrectionPrimitive,
    SkySubtractionPrimitive,
    StellarRVPrimitive,
)

__all__ = ["Stage2Pipeline"]


class Stage2Pipeline(BasePipeline):
    """
    Stage 2 spectral calibration pipeline.

    Consumes Level 1 spectra with wavelength grids (e.g. output of
    :class:`~hispecdrp.pipelines.stage1.stage1_pipeline.Stage1Pipeline`), applies
    blaze correction to produce calibrated L2 spectra, subtracts sky background using sky fiber, computes barycentric
    corrections, then runs stellar RV (default line-mask CCF) on that L2 product.

    Primitives (in order):

    - :class:`~hispecdrp.primitives.blaze_correction.blaze_correction_primitive.BlazeCorrectionPrimitive`
    - :class:`~hispecdrp.primitives.sky_subtraction.sky_primitive.SkySubtractionPrimitive`
    - :class:`~hispecdrp.primitives.barycenter_correction.barycorr_primitive.BarycenterCorrectionPrimitive`
    - :class:`~hispecdrp.primitives.stellar_rv.stellar_rv_primitive.StellarRVPrimitive`
    """

    PRIMITIVES = {
        "blaze_corr": BlazeCorrectionPrimitive,
        "sky_sub": SkySubtractionPrimitive,
        "barycorr": BarycenterCorrectionPrimitive,
        "stellar_rv": StellarRVPrimitive,
    }

    def _perform(self):
        model_to_process = datamodels.open(self.dataset)
        result = self.blaze_corr.apply(model_to_process)
        result = self.sky_sub.apply(result)
        result = self.barycorr.apply(result)
        result = self.stellar_rv.apply(result)
        return dict(output=result)

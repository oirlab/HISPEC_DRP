from hispecdrp import datamodels
from ..base_pipeline import BasePipeline
from ...primitives import BlazeCorrectionPrimitive, CrossCorrelationPrimitive

__all__ = ["Stage2Pipeline"]


class Stage2Pipeline(BasePipeline):
    """
    Pipeline to process Level 1 (uncalibrated spectra) to Level 2 (calibrated spectra with RV).

    This pipeline performs the following steps:

    - :class:`~hispecdrp.primitives.blaze_correction.blaze_correction_primitive.BlazeCorrectionPrimitive`
      Blaze correction using the fiber flat; produces calibrated spectra (L2).
    - :class:`~hispecdrp.primitives.cross_correlation.cross_correlation_primitive.CrossCorrelationPrimitive`
      Cross-correlation with a spectral template; computes per-chunk and aggregated radial velocities (m/s)
      and writes RVCCF_* and RVCCFAVG_* extensions to the L2 model.

    The cross-correlation step requires a template. Pass it via ``prim_args`` when running the pipeline,
    e.g. ``prim_args={'cross_corr': {'template_path': 'path/to/template.h5'}}``.
    """

    PRIMITIVES = {
        "blaze_corr": BlazeCorrectionPrimitive,
        "cross_corr": CrossCorrelationPrimitive,
    }

    def _perform(self):
        input_model = datamodels.open(self.input)
        result = self.blaze_corr.apply(input_model)
        result = self.cross_corr.apply(result)
        return dict(output=result)

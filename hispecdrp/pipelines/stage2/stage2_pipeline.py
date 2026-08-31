import logging

from hispecdrp import datamodels
from ..base_pipeline import BasePipeline
from ...primitives import (
    BarycenterCorrectionPrimitive,
    BlazeCorrectionPrimitive,
    SkySubtractionPrimitive,
    StellarRVPrimitive,
)
from ...primitives.barycenter_correction.barycenter_corr import (
    berv_m_s_to_ccf_zb,
    read_berv_m_s_uniform_from_model,
)

logger = logging.getLogger(__name__)

__all__ = ["Stage2Pipeline"]


def _primitive_config_has_explicit_ccf_zb(config: dict) -> bool:
    if not config:
        return False
    if "ccf_zb" in config:
        return True
    params = config.get("parameters")
    return isinstance(params, dict) and "ccf_zb" in params


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

    def init_primitives(
        self,
        context=None,
        primitives=None,
    ):
        primitives = primitives or {}
        pipe_stellar = self.config.primitives.get("stellar_rv", {})
        self._stellar_rv_ccf_zb_explicit = (
            _primitive_config_has_explicit_ccf_zb(primitives.get("stellar_rv", {}))
            or _primitive_config_has_explicit_ccf_zb(pipe_stellar)
        )
        super().init_primitives(context, primitives)

    def _populate_stellar_rv_ccf_zb_from_barycorr(self, model: datamodels.DataModel) -> None:
        """Set uniform ``ccf_zb`` on the stellar-RV primitive from ``BARYCORR``."""
        if self._stellar_rv_ccf_zb_explicit:
            return
        berv_m_s = read_berv_m_s_uniform_from_model(model)
        if berv_m_s is None:
            return
        zb = berv_m_s_to_ccf_zb(berv_m_s)
        self.stellar_rv.ccf_zb = zb
        logger.info(
            "Stage2: stellar RV ccf_zb=%.9e from BARYCORR BERV=%.3f m/s",
            zb,
            float(berv_m_s),
        )

    def _perform(self):
        model_to_process = datamodels.open(self.dataset)
        result = self.blaze_corr.apply(model_to_process)
        result = self.sky_sub.apply(result)
        result = self.barycorr.apply(result)
        self._populate_stellar_rv_ccf_zb_from_barycorr(result)
        result = self.stellar_rv.apply(result)
        return dict(output=result)

from hispecdrp import datamodels
from ..calibration_pipeline_base import CalibrationPipeline
from ...primitives import FitLFCModesPrimitive, WLSChebyshev2DPrimitive, GaussianLSFPrimitive, RegisterCalibrationPrimitive

from keckdrpframework.models.arguments import Arguments

__all__ = ['ProcessLFCPipeline']


class ProcessLFCPipeline(CalibrationPipeline):

    PRIMITIVES = {
        'fit_lfc_modes': FitLFCModesPrimitive,
        'lfc_wavesol': WLSChebyshev2DPrimitive,
        'lsf': GaussianLSFPrimitive,
        'register_wave_cal' : RegisterCalibrationPrimitive,
        'register_lsf_cal' : RegisterCalibrationPrimitive,
    }

    def _perform(self):
        model_to_process = datamodels.open(self.args.name)
        result = self.fit_lfc_modes.apply(model_to_process)
        result = self.lfc_wavesol.apply(result)
        self.register_wave_cal.apply(result)
        result = self.lsf.apply(result)
        self.register_lsf_cal.apply(result)
        return Arguments(name=result)
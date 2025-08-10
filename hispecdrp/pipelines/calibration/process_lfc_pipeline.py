from hispecdrp import datamodels
from ..calibration_pipeline_base import CalibrationPipeline
from ...primitives import FitLFCLinesPrimitive, CalculateChebyshev2DWLSPrimitive, CalculateGaussianLSFPrimitive, RegisterCalibrationPrimitive

from keckdrpframework.models.arguments import Arguments

__all__ = ['ProcessLFCPipeline']


class ProcessLFCPipeline(CalibrationPipeline):

    PRIMITIVES = {
        'fit_lfc_lines': FitLFCLinesPrimitive,
        'lfc_wavesol': CalculateChebyshev2DWLSPrimitive,
        'register_wave_cal' : RegisterCalibrationPrimitive,
        'lsf': CalculateGaussianLSFPrimitive,
        'register_lsf_cal' : RegisterCalibrationPrimitive,
    }

    def _perform(self):
        input_model = datamodels.open(self.name)
        line_fit_result = self.fit_lfc_lines.apply(input_model)
        wavesol_result = self.lfc_wavesol.apply(line_fit_result)
        self.register_wave_cal.apply(wavesol_result)
        lsf_result = self.lsf.apply(wavesol_result)
        self.register_lsf_cal.apply(lsf_result)
        return Arguments(name=dict(wavesol=wavesol_result, lsf=lsf_result))
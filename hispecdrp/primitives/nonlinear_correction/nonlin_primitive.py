from ..base_primitive import BasePrimitive
from ...calibrations import HISPECNonlinearCorrectionSelector, PARVINonlinearCorrectionSelector
from ... import datamodels
import numpy as np

from keckdrpframework.models.arguments import Arguments

from .nonlincorr_numba import correct_nonlinearity

__all__ = ['NonlinearCorrectionPrimitive']


class NonlinearCorrectionPrimitive(BasePrimitive):
    """
    ARGUMENTS:
        nonlin: Nonlinear model to apply to the input data.
    """

    ARGUMENTS = {
        'suffix' : 'nonlin',
        'nonlin' : None
    }

    def _perform(self):
        input_model = datamodels.open(self.args.name)
        if self.args.nonlin is None:
            nonlin_path = self.get_calibration(input_model)
        else:
            nonlin_path = self.args.nonlin
        nonlin_model = datamodels.open(nonlin_path)
        result = input_model.copy()
        self.log.info(f"Applying linearity correction with {nonlin_model} to {input_model}")
        data = result.data.astype(np.float64)
        correct_nonlinearity(data, nonlin_model.coeffs)
        result.data = data.astype(np.float32)
        return Arguments(name=result)

    def get_calibration_selector(self, name):
        input_model = datamodels.open(name, meta_only=True)
        if isinstance(input_model, datamodels.HISPECDataModel):
            return HISPECNonlinearCorrectionSelector()
        elif isinstance(input_model, datamodels.PARVIDataModel):
            return PARVINonlinearCorrectionSelector()
        else:
            raise ValueError(f"Unsupported input model type: {type(input_model)}")
from ..base_primitive import BasePrimitive
from ...calibrations import NonlinearCorrectionSelector
from ... import datamodels
import numpy as np

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
        input_model = datamodels.open(self.name)
        if self.nonlin is None:
            nonlin_path = self.get_calibration(input_model, selector=NonlinearCorrectionSelector())
        else:
            nonlin_path = self.nonlin
        nonlin_model = datamodels.open(nonlin_path)
        result = input_model.copy()
        self.log.info(f"Applying linearity correction with {nonlin_model} to {input_model}")
        data = result.data.astype(np.float64)
        correct_nonlinearity(data, nonlin_model.coeffs)
        result.data = data.astype(np.float32)
        return dict(name=result, nonlin=nonlin_path)
from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np

from .nonlincorr_numba import correct_nonlinearity

__all__ = ['NonlinearCorrectionPrimitive']


class NonlinearCorrectionPrimitive(BasePrimitive):
    """
    ARGUMENTS:
        coeffs (NonlinearCoeffModel | str | None): Coefficients model to apply to the input data.
    """

    class_alias = 'nonlin_corr'

    ARGUMENTS = {
        'suffix' : 'nonlincorr',
        'coeffs' : None
    }


    def _perform(self):
        input_model = self.open_model(self.args.name, copy=False)
        if self.args.coeffs is None:
            coeffs_path = self.get_calibration_file(input_model, cal_type='nonlin')
        else:
            coeffs_path = self.args.coeffs
        nonlin_model = self.open_model(coeffs_path, copy=False)
        result = input_model.copy()
        self.log.info(f"Applying linearity correction with {nonlin_model} to {input_model}")
        data = result.data.astype(np.float64)
        correct_nonlinearity(data, nonlin_model.coeffs)
        result.data = data.astype(np.float32)
        return result
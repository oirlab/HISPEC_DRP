from ..base_primitive import BasePrimitive
from ...calibrations import NonlinearCorrectionSelector
from ... import datamodels
import numpy as np

from .nonlincorr_numba import correct_nonlinearity
from numba import get_num_threads, set_num_threads


__all__ = ['NonlinearCorrectionPrimitive']


class NonlinearCorrectionPrimitive(BasePrimitive):

    ARGUMENTS = {
        'suffix' : 'nonlincorr',
        'ouput_model' : 'input',
        'max_cores' : 1,
    }

    CALIBRATIONS = {
        'nonlin': {
            'selector': NonlinearCorrectionSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):
        set_num_threads(self.max_cores)

        input_model = datamodels.open(self.input)

        nonlin, _ = self.get_calibration(input_model, 'nonlin')
        nonlin_model = datamodels.open(nonlin)

        result = self.get_output_model(input_model)
        self.log.info(f"Applying linearity correction with {nonlin_model} to {input_model}")

        data = result.data.astype(np.float64)
        correct_nonlinearity(data, nonlin_model.coeffs)
        result.data = data.astype(np.float32)

        return dict(output=result, nonlin=nonlin)
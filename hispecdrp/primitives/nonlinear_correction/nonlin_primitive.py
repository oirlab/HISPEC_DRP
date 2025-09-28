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

        if result.data.dtype != np.float32:
            self.log.info(f"Converting UTR data from {result.data.dtype} to float32")
            result.data = result.data.astype(np.float32)
        self.log.info(f"Polynomial coefficients are {nonlin_model.coeffs.dtype}.")

        correct_nonlinearity(result.data,
                             result.dq_raw,
                             result.dq,
                             nonlin_model.coeffs,
                             nonlin_model.nonlin_max,
                             nonlin_model.dq)

        return dict(output=result, nonlin=nonlin)
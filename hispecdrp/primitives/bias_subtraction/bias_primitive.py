from ..base_primitive import BasePrimitive
from ...calibrations import BiasSelector
from ... import datamodels
from numba import njit, prange
from numba import get_num_threads, set_num_threads
import numpy as np

__all__ = ['BiasSubtractionPrimitive']


class BiasSubtractionPrimitive(BasePrimitive):
    
    ARGUMENTS = {
        'suffix' : 'biassub',
        'output_model' : 'input',
        'max_cores' : 1
    }

    CALIBRATIONS = {
        'bias': {
            'selector': BiasSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):
        set_num_threads(self.max_cores)

        # Load the input model
        input_model = datamodels.open(self.input)

        # Get the bias model
        bias, _ = self.get_calibration(input_model, 'bias')
        bias_model = datamodels.open(bias)

        # Output
        result = self.get_output_model(input_model)

        # Log
        self.log.info(f"Applying bias correction with {bias} to {input_model}")

        if result.data.dtype != np.float32:
            self.log.info(f"Converting UTR data from {result.data.dtype} to float32")
            result.data = result.data.astype(np.float32)

        # Perform bias subtraction
        bias_subtraction(result, bias_model)

        # Output
        return dict(output=result, bias=bias)


def bias_subtraction(input_model: datamodels.RampModel, bias_model: datamodels.BiasModel) -> datamodels.RampModel:
    """
    Subtract the bias from the ramps.

    Args:
        input_model: Input RampModel.
        bias_model: Bias calibration model.
    
    Returns:
        RampModel: The bias-subtracted data.
    """
    if not hasattr(input_model, 'DATA'):
        raise Exception("DATA extension not found in input model for bias subtraction.")
    if not hasattr(input_model, 'DQ_RAW'):
        raise Exception("DQ_RAW extension not found in input model for bias subtraction.")

    _bias_subtraction_numba(
        input_model.data,
        input_model.dq_raw,
        bias_model.bias,
        bias_model.dq,
    )
    return


@njit(nogil=True, parallel=True, cache=True)
def _bias_subtraction_numba(data, dq_raw, bias, bias_dq):
    """
    Subtract bias and propagate DQ flags.

    Args:
        data:    (n_read, ny, nx) float32/float64 ramp data
        dq_raw:  (n_read, ny, nx) uint32 DQ array (modified in place)
        bias:    (ny, nx) float32 bias calibration image
        bias_dq: (ny, nx) uint32 DQ map for bias model

    Returns:
        data (in place), dq_raw (in place)
    """
    n_read, ny, nx = data.shape
    for j in prange(ny):  # parallelize across spatial rows
        for i in range(nx):
            b = bias[j, i]
            dq_val = bias_dq[j, i]
            data[:, j, i] = data[:, j, i] - b
            dq_raw[:, j, i] |= dq_val
    return
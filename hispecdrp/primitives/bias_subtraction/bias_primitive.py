from ..base_primitive import BasePrimitive
from ...calibrations import BiasSelector
from ... import datamodels
from ...utils.parallelization_utils import numba_thread_scope
from numba import njit, prange
import numpy as np
import logging

logger = logging.getLogger(__name__)

__all__ = ['BiasSubtractionPrimitive']


class BiasSubtractionPrimitive(BasePrimitive):
    """
    Subtract the bias level from the input data.

    This is the base signal from which accumulated charges are measured.
    The bias voltage is to prevent negative values to be read given the
    unsigned nature of the values produced by the ADC. Bias subtraction
    is useful to set the y-intercept term of the UTR data close to 0.

    This primitive applies the appropriate bias calibration to the
    input data product. The bias frame is subtracted from the science data,
    the associated uncertainties are propagated by adding errors in
    quadrature, and data quality (DQ) flags are combined using a bitwise OR.

    Parameters
    ----------
    input : str or RampModel
        Input file path or data model to process. Currently, this is expected
        to be a raw (ramp) data product.
    suffix : str or None, optional
        Suffix to append to the output filename. Default is ``'biassub'``.
    max_cores : int, optional
        Maximum number of CPU cores to use for parallel processing. Default is 1 (no parallelization).

    Calibrations
    ------------
    bias : BiasModel
        - selector: :class:`~hispecdrp.calibrations.selectors.BiasSelector`

    Returns
    -------
    output : DataModel
        Bias-subtracted data model with updated error and data quality
        arrays.
    """

    ARGUMENTS = {
        'suffix' : 'biassub',
        'max_cores' : 1
    }

    CALIBRATIONS = {
        'bias': {
            'selector': BiasSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):

        # Load the input model
        input_model = datamodels.open(self.input)

        # Get the bias model
        bias, _ = self.get_calibration(input_model, 'bias')
        bias_model = datamodels.open(bias)

        # Output
        result = self.get_output_model(input_model)

        if not hasattr(result, 'DATA'):
            raise Exception("DATA extension not found in input model for bias subtraction.")
        if not hasattr(result, 'DQ'):
            raise Exception("DQ extension not found in input model for bias subtraction.")

        # Log
        logger.info(f"Applying bias correction with {bias} to {input_model}")

        if result.data.dtype != np.float32:
            logger.info(f"Converting UTR data from {result.data.dtype} to float32")
            result.data = result.data.astype(np.float32)

        # Perform bias subtraction
        _bias_subtraction(
            result.data, result.dq,
            bias_model.bias, bias_model.dq,
            max_cores=self.max_cores
        )

        # Output
        return dict(output=result, bias=bias)


def _bias_subtraction(
    input_data : np.ndarray,
    input_dq : np.ndarray,
    bias_data : np.ndarray,
    bias_dq : np.ndarray,
    max_cores : int = 1
):
    """
    Subtract the bias from the ramps.

    Parameters
    ----------
    input_data
        Input ramp data with shape (Nreads, Ny, Nx).
    input_dq
        Input DQ array with shape (Ny, Nx).
    bias_data
        Bias data with shape (Ny, Nx).
    bias_dq
        Bias DQ array with shape (Ny, Nx).
    max_cores
        Maximum number of CPU cores to use.
    """
    with numba_thread_scope(max_cores):
        _bias_subtraction_numba(input_data, input_dq, bias_data, bias_dq)
    return input_data, input_dq


@njit(nogil=True, parallel=True, cache=True)
def _bias_subtraction_numba(data : np.ndarray, dq : np.ndarray, bias : np.ndarray, bias_dq : np.ndarray):
    n_read, ny, nx = data.shape
    for j in prange(ny):
        for i in range(nx):
            data[:, j, i] = data[:, j, i] - bias[j, i]
            dq[j, i] |= bias_dq[j, i]
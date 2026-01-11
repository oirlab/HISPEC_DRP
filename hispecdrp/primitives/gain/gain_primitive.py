
from ..base_primitive import BasePrimitive
from ... import datamodels
from ...calibrations import GainSelector
from ...utils.parallelization_utils import numba_thread_scope
import numpy as np
from numba import njit, prange


__all__ = ['GainCorrectionPrimitive', 'gain_correction']


class GainCorrectionPrimitive(BasePrimitive):

    ARGUMENTS = {
        'suffix' : 'gain',
        'output_model' : 'input',
        'max_cores' : 1,
    }

    CALIBRATIONS = {
        'gain': {
            'selector': GainSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):

        # Load the input model
        input_model = datamodels.open(self.input)

        # Get the gain model
        gain, _ = self.get_calibration(input_model, 'gain')
        gain_model = datamodels.open(gain)

        # Output
        result = self.get_output_model(input_model)

        if not hasattr(input_model, 'RATE'):
            raise Exception("RATE extension not found in input model for detector gain correction.")
        if not hasattr(input_model, 'RATE_ERR'):
            raise Exception("RATE_ERR extension not found in input model for detector gain correction.")
        if not hasattr(input_model, 'DQ'):
            raise Exception("DQ extension not found in input model for detector gain correction.")
        if not hasattr(gain_model, 'GAIN'):
            raise Exception("GAIN extension not found in gain model for dark subtraction.")
        if not hasattr(gain_model, 'DQ'):
            raise Exception("DQ extension not found in gain model for dark subtraction.")

        # Log
        self.log.info(f"Applying detector gain correction with {gain} to {input_model}")

        # Perform detector flat correction
        gain_correction(
            result.rate, result.rate_err, result.dq,
            gain_model.gain, np.sqrt(gain_model.cov[0,0]), gain_model.dq,
            max_cores=self.max_cores
        )

        # Output
        return dict(output=result, gain=gain)


def gain_correction(
    input_rate : np.ndarray, input_err : np.ndarray, input_dq : np.ndarray,
    gain : np.ndarray, gain_err : np.ndarray, gain_dq : np.ndarray,
    max_cores : int = 1
):
    """
    Apply detector gain correction to 2D rate image.

    Parameters
    ----------
    input_rate : np.ndarray
        2D array of input rates.
    input_err : np.ndarray
        2D array of input rate errors.
    input_dq : np.ndarray
        2D array of input data quality flags.
    gain : np.ndarray
        2D array of gain values.
    gain_err : np.ndarray
        2D array of gain errors.
    gain_dq : np.ndarray
        2D array of gain data quality flags.
    max_cores : int, optional
        Maximum number of CPU cores to use for parallel processing. Default is 1.

    Returns
    -------
    input_rate : np.ndarray
        Gain-corrected rate array.
    input_err : np.ndarray
        Updated rate error array.
    input_dq : np.ndarray
        Updated data quality flags array.
    """
    with numba_thread_scope(max_cores):
        _gain_correction_numba(
            input_rate, input_err, input_dq,
            gain, gain_err, gain_dq
        )
    return input_rate, input_err, input_dq

@njit(nogil=True, parallel=True, cache=True)
def _gain_correction_numba(
    input_rate : np.ndarray, input_err : np.ndarray, input_dq : np.ndarray,
    gain : np.ndarray, gain_err : np.ndarray, gain_dq : np.ndarray
):
    ny, nx = input_rate.shape
    for j in prange(ny):
        for i in range(nx):
            g = gain[j, i]
            r = input_rate[j, i]
            input_rate[j, i] = r * g
            gr = np.abs(input_rate[j, i])
            e_in = input_err[j, i]
            ge_in = gain_err[j, i]

            # Approximate error propagation for division of random variables
            # https://en.wikipedia.org/wiki/Propagation_of_uncertainty
            input_err[j, i] = gr * np.sqrt((e_in / r) ** 2 + (ge_in / g) ** 2)

            input_dq[j, i] |= gain_dq[j, i]
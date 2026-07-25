from ..base_primitive import BasePrimitive
from ...calibrations import DarkSelector
from ... import datamodels
import numpy as np
from numba import njit, prange
from ...utils.parallelization_utils import numba_thread_scope
import logging

logger = logging.getLogger(__name__)

__all__ = ['DarkSubtractionPrimitive', 'dark_subtraction_rate']


class DarkSubtractionPrimitive(BasePrimitive):
    """
    Subtract the dark current from the input data.

    This primitive applies the appropriate dark calibration to the
    input data product. The dark frame is subtracted from the science data,
    the associated uncertainties are propagated by adding errors in
    quadrature, and data quality (DQ) flags are combined using a bitwise OR.

    Parameters
    ----------
    input : str or L0Model
        Input file path or data model to process. Currently, this is expected
        to be a level 0 (rate map) data product.
    suffix : str or None, optional
        Suffix to append to the output filename. Default is ``'darksub'``.
    max_cores : int, optional
        Maximum number of CPU cores to use for Numba parallel processing. Default is 1.

    Calibrations
    ------------
    dark : DarkModel
        - selector: :class:`~hispecdrp.calibrations.selectors.DarkSelector`

    Returns
    -------
    output
        The dark-subtracted data model with updated error and data quality arrays.
    """
    ARGUMENTS = {
        'suffix' : 'darksub',
        'max_cores' : 1,
    }

    CALIBRATIONS = {
        'dark': {
            'selector': DarkSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):

        # Load the input model
        input_model = datamodels.open(self.input)

        # Get the dark model
        dark, _ = self.get_calibration(input_model, 'dark')
        dark_model = datamodels.open(dark)

        # Output
        result = self.get_output_model(input_model)

        # Log
        logger.info(f"Applying dark correction with {dark_model} to {input_model}")

        # Perform dark subtraction
        dark_subtraction_rate(
            result.rate, result.rate_err, result.dq,
            dark_model.rate, dark_model.rate_err, dark_model.dq,
            max_cores=self.max_cores
        )

        # Output
        return dict(output=result, dark=dark)


def dark_subtraction_rate(
    input_rate: np.ndarray, input_err: np.ndarray, input_dq: np.ndarray,
    dark_rate: np.ndarray, dark_err: np.ndarray, dark_dq: np.ndarray,
    max_cores : int = 1
):
    """
    Subtract the dark current rate from the input rate.
    Input and dark rates are DN/s.
    The error arrays are added in quadrature.
    The data quality arrays are combined with a bitwise OR.
    The corrections are done in-place.
    Dark should be corrected before gain correction.

    Parameters
    ----------
    input_rate : np.ndarray
        2D array of rate values (DN/s)
    input_err : np.ndarray
        2D array of error values (DN/s)
    input_dq : np.ndarray
        2D array of data quality flags
    dark_rate : np.ndarray
        2D array of dark rate values (DN/s)
    dark_err : np.ndarray
        2D array of dark error values (DN/s)
    dark_dq : np.ndarray
        2D array of dark data quality flags
    max_cores : int
        Maximum number of CPU cores to use for parallel processing.

    Returns
    -------
    input_rate : np.ndarray
        Dark-subtracted rate array (modified in place)
    input_err : np.ndarray
        Updated error array (modified in place)
    input_dq : np.ndarray
        Updated data quality array (modified in place)
    """
    with numba_thread_scope(max_cores):
        _dark_subtraction_rate_numba(
            input_rate, input_err, input_dq,
            dark_rate, dark_err, dark_dq
        )
    return input_rate, input_err, input_dq


@njit(nogil=True, parallel=True, cache=True)
def _dark_subtraction_rate_numba(
    input_rate : np.ndarray, input_err : np.ndarray, input_dq : np.ndarray,
    dark_rate : np.ndarray, dark_err : np.ndarray, dark_dq : np.ndarray
):
    ny, nx = input_rate.shape
    for j in prange(ny):
        for i in range(nx):
            # Subtract dark rate
            input_rate[j, i] = input_rate[j, i] - dark_rate[j, i]

            # Propagate error
            input_err[j, i] = np.sqrt(input_err[j, i] * input_err[j, i] + dark_err[j, i] * dark_err[j, i])

            # Propagate DQ
            input_dq[j, i] |= dark_dq[j, i]
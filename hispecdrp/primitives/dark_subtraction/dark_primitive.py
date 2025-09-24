from ..base_primitive import BasePrimitive
from ...calibrations import DarkSelector
from ... import datamodels
import numpy as np
from numba import njit, prange
from numba import get_num_threads, set_num_threads

__all__ = ['DarkSubtractionPrimitive']


class DarkSubtractionPrimitive(BasePrimitive):
    
    ARGUMENTS = {
        'suffix' : 'darksub',
        'output_model' : 'input',
        'max_cores' : 1,
    }

    CALIBRATIONS = {
        'dark': {
            'selector': DarkSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):
        set_num_threads(self.max_cores)

        # Load the input model
        input_model = datamodels.open(self.input)

        # Get the dark model
        dark, _ = self.get_calibration(input_model, 'dark')
        dark_model = datamodels.open(dark)

        # Output
        result = self.get_output_model(input_model)

        # Log
        self.log.info(f"Applying dark correction with {dark} to {input_model}")

        # Perform dark subtraction
        output_model = dark_subtraction_rate(result, dark_model)

        # Output
        return dict(output=output_model, dark=dark)


def dark_subtraction_rate(input_model: datamodels.L0Model, dark_model: datamodels.DarkModel) -> datamodels.L0Model:
    """
    Subtract the dark model from an input rate (level 0) model.
    Dark rates are DN/s not e-/s. Dark should be corrected before gain correction.

    Args:
        input_model: Input level 0 model.
        dark_model: Dark calibration model.
    
    Returns:
        HISPECL0Model: The dark-subtracted data.
    """
    if not hasattr(input_model, 'RATE'):
        raise Exception("RATE extension not found in input model for dark subtraction.")
    if not hasattr(input_model, 'RATE_ERR'):
        raise Exception("RATE_ERR extension not found in input model for dark subtraction.")
    if not hasattr(input_model, 'DQ'):
        raise Exception("DQ extension not found in input model for dark subtraction.")
    if not hasattr(dark_model, 'RATE'):
        raise Exception("RATE extension not found in dark model for dark subtraction.")
    if not hasattr(dark_model, 'RATE_ERR'):
        raise Exception("RATE_ERR extension not found in dark model for dark subtraction.")
    if not hasattr(dark_model, 'DQ'):
        raise Exception("DQ extension not found in dark model for dark subtraction.")

    out=_dark_subtraction_rate_numba(input_model.rate, input_model.rate_err, input_model.dq, dark_model.rate, dark_model.rate_err, dark_model.dq)

    input_model.rate, input_model.rate_err, input_model.dq = out

    return input_model


@njit(nogil=True, parallel=True, cache=True)
def _dark_subtraction_rate_numba(rate, err, dq, dark_rate, dark_err, dark_dq):
    """
    Subtract dark current from 2D rate image, propagate errors and DQ.

    Args:
        rate:       (ny, nx) float32/float64 input rate data
        err:        (ny, nx) float32/float64 input error map
        dq:     (ny, nx) uint32 input DQ map (modified in place)
        dark_rate:  (ny, nx) float32 dark calibration
        dark_err:   (ny, nx) float32 dark error map
        dark_dq:    (ny, nx) uint32 dark DQ map

    Returns:
        rate (in place), err (in place or None), dq_raw (in place or None)
    """
    ny, nx = rate.shape
    for j in prange(ny):
        for i in range(nx):
            # Subtract dark rate
            rate[j, i] = rate[j, i] - dark_rate[j, i]

            # Propagate error
            err[j, i] = np.sqrt(err[j, i] * err[j, i] + dark_err[j, i] * dark_err[j, i])

            # Propagate DQ
            dq[j, i] |= dark_dq[j, i]

    return rate, err, dq
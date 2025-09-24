
from ..base_primitive import BasePrimitive
from ... import datamodels
from ...calibrations import DetectorGainSelector
import numpy as np
from numba import njit, prange
from numba import get_num_threads, set_num_threads

__all__ = ['DetectorGainPrimitive']


class DetectorGainPrimitive(BasePrimitive):

    ARGUMENTS = {
        'suffix' : 'detgain',
        'output_model' : 'input',
        'max_cores' : 1,
    }

    CALIBRATIONS = {
        'detgain': {
            'selector': DetectorGainSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):
        set_num_threads(self.max_cores)

        # Load the input model
        input_model = datamodels.open(self.input)

        # Get the detgain model
        detgain, _ = self.get_calibration(input_model, 'detgain')
        detgain_model = datamodels.open(detgain)

        # Output
        result = self.get_output_model(input_model)

        # Log
        self.log.info(f"Applying detector gain correction with {detgain} to {input_model}")

        # Perform detector flat correction
        output_model = detector_gain_correction(result, detgain_model)

        # Output
        return dict(output=output_model, detgain=detgain)


def detector_gain_correction(input_model : datamodels.L0Model, detgain_model : datamodels.DetectorGainModel) -> datamodels.DataModel:
    """
    Correct the detector level flat field.

    Args:
        input_model (DataModel): Input model.
        detflat_model (HISPECFlatModel): Input detector flat model.
    
    Returns:
        DataModel: The model corrected in place.
    """
    if not hasattr(input_model, 'RATE'):
        raise Exception("RATE extension not found in input model for detector gain correction.")
    if not hasattr(input_model, 'RATE_ERR'):
        raise Exception("RATE_ERR extension not found in input model for detector gain correction.")
    if not hasattr(input_model, 'DQ'):
        raise Exception("DQ extension not found in input model for detector gain correction.")
    if not hasattr(detgain_model, 'GAIN'):
        raise Exception("GAIN extension not found in gain model for dark subtraction.")
    if not hasattr(detgain_model, 'GAIN_ERR'):
        raise Exception("GAIN_ERR extension not found in gain model for dark subtraction.")
    if not hasattr(detgain_model, 'DQ'):
        raise Exception("DQ extension not found in gain model for dark subtraction.")

    # Run Numba kernel
    input_model.rate, input_model.rate_err, input_model.dq = _detector_gain_correction_numba(
        input_model.rate,
        input_model.rate_err,
        input_model.dq,
        detgain_model.gain,
        detgain_model.gain_err,
        detgain_model.dq,
    )

    return input_model



@njit(nogil=True, parallel=True, cache=True)
def _detector_gain_correction_numba(rate, err, dq, gain, gain_err, gain_dq):
    """
    Apply detector gain correction to 2D rate image.

    Args:
        rate:    (ny, nx) float32 input rate data
        err:     (ny, nx) float32 input error map
        dq:      (ny, nx) uint32 input DQ map
        gain:    (ny, nx) float32 gain map
        gain_err:    (ny, nx) float32 gain error map
        gain_dq: (ny, nx) uint32 gain DQ map

    Returns:
        rate (in place), err (in place or None), dq (in place or None)
    """
    ny, nx = rate.shape
    for j in prange(ny):
        for i in range(nx):
            g = gain[j, i]
            r = rate[j, i]
            rate[j, i] = r * g
            if rate[j, i] > 0:
                gr = rate[j, i]
            else:
                gr = -rate[j, i]

            e_in = err[j, i]
            ge_in = gain_err[j, i]

            # approximate error propagation for multiplication of random variables
            # https://en.wikipedia.org/wiki/Propagation_of_uncertainty
            err[j, i] = gr * np.sqrt((e_in / r) ** 2 + (ge_in / g) ** 2)

            dq[j, i] |= gain_dq[j, i]

    return rate, err, dq
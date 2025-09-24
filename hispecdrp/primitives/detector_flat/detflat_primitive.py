
from ..base_primitive import BasePrimitive
from ... import datamodels
from ...calibrations import DetectorFlatSelector
import numpy as np
from numba import njit, prange
from numba import get_num_threads, set_num_threads

__all__ = ['DetectorFlatPrimitive']


class DetectorFlatPrimitive(BasePrimitive):

    ARGUMENTS = {
        'suffix' : 'detflat',
        'output_model' : 'input',
        'max_cores' : 1,
    }

    CALIBRATIONS = {
        'detflat': {
            'selector': DetectorFlatSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):
        set_num_threads(self.max_cores)

        # Load the input model
        input_model = datamodels.open(self.input)

        # Get the detflat model
        detflat, _ = self.get_calibration(input_model, 'detflat')
        detflat_model = datamodels.open(detflat)

        # Output
        result = self.get_output_model(input_model)

        # Log
        self.log.info(f"Applying detector flat correction with {detflat} to {input_model}")

        # Perform detector flat correction
        output_model = detector_flat_correction(result, detflat_model)

        # Output
        return dict(output=output_model, detflat=detflat)


def detector_flat_correction(input_model : datamodels.L0Model, detflat_model : datamodels.DetectorFlatModel) -> datamodels.DataModel:
    """
    Correct the detector level flat field.

    Args:
        input_model (DataModel): Input model.
        detflat_model (HISPECFlatModel): Input detector flat model.
    
    Returns:
        DataModel: The model corrected in place.
    """
    if not hasattr(input_model, 'RATE'):
        raise Exception("RATE extension not found in input model for detector flat correction.")
    if not hasattr(input_model, 'RATE_ERR'):
        raise Exception("RATE_ERR extension not found in input model for detector flat correction.")
    if not hasattr(input_model, 'DQ'):
        raise Exception("DQ extension not found in input model for detector flat correction.")
    if not hasattr(detflat_model, 'FLAT'):
        raise Exception("FLAT extension not found in flat model for dark subtraction.")
    if not hasattr(detflat_model, 'FLAT_ERR'):
        raise Exception("FLAT_ERR extension not found in flat model for dark subtraction.")
    if not hasattr(detflat_model, 'DQ'):
        raise Exception("DQ extension not found in flat model for dark subtraction.")

    # Run Numba kernel
    input_model.rate, input_model.rate_err, input_model.dq = _detector_flat_correction_numba(
        input_model.rate,
        input_model.rate_err,
        input_model.dq,
        detflat_model.flat,
        detflat_model.flat_err,
        detflat_model.dq,
    )

    return input_model


@njit(nogil=True, parallel=True, cache=True)
def _detector_flat_correction_numba(rate, err, dq, flat, flat_err, flat_dq):
    """
    Apply detector flat correction to 2D rate image.

    Args:
        rate:     (ny, nx) float32/float64 input rate data
        err:      (ny, nx) float32/float64 input error map
        dq:       (ny, nx) uint32 input DQ map (modified in place)
        flat:     (ny, nx) float32 detector flat map
        flat_err: (ny, nx) float32 flat error map
        flat_dq:  (ny, nx) uint32 flat DQ map

    Returns:
        rate (in place), err (in place or None), dq (in place or None)
    """
    ny, nx = rate.shape
    for j in prange(ny):
        for i in range(nx):
            f = flat[j, i]
            r = rate[j, i]
            rate[j, i] = r / f
            if rate[j, i] > 0:
                z = rate[j, i]
            else:
                z = -rate[j, i]

            e_in   = err[j, i]
            fe_in  = flat_err[j, i]

            # approximate error propagation for division of random variables
            # https://en.wikipedia.org/wiki/Propagation_of_uncertainty
            err[j, i] = z * np.sqrt((e_in / r) ** 2 + (fe_in / f) ** 2)

            dq[j, i] |= flat_dq[j, i]

    return rate, err, dq
from ..base_primitive import BasePrimitive
from ...calibrations import SaturationSelector
from ... import datamodels
from ...datamodels.dqflags import DQ_FLAGS
from numba import njit, prange
from numba import get_num_threads, set_num_threads
import numpy as np

__all__ = ['CheckSaturationPrimitive']

SATURATED_FLAG =  DQ_FLAGS["SATURATED"]
DO_NOT_USE_FLAG = DQ_FLAGS["DO_NOT_USE"]

class CheckSaturationPrimitive(BasePrimitive):
    
    ARGUMENTS = {
        'suffix' : 'checksat',
        'output_model' : 'input',
        'max_cores' : 1,
    }

    CALIBRATIONS = {
        'saturation': {
            'selector': SaturationSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):
        set_num_threads(self.max_cores)

        # Load the input model
        input_model = datamodels.open(self.input)

        # Get the saturation threshold model
        sat, _ = self.get_calibration(input_model, 'saturation')
        saturation_model = datamodels.open(sat)

        # Output
        result = self.get_output_model(input_model)

        # Log
        self.log.info(f"Applying saturation check with {saturation_model} to {input_model}")

        # Perform saturation check
        output_model = check_saturation(result, saturation_model)

        # Output
        return dict(output=output_model, saturation=sat)


def check_saturation(input_model: datamodels.RampModel, saturation_model: datamodels.SaturationModel) -> datamodels.RampModel:
    """
    Flag saturated pixels in ramps.

    Args:
        input_model: Input level 0 model.
        sat_model: Calibration model of saturation threshold map.
    
    Returns:
        RampModel: A new ramp model with saturated pixels flagged in the DQ array.
    """
    if not hasattr(input_model, "DQ_RAW"):
        raise AttributeError("Input model has no DQ_RAW array to flag saturation.")

    input_model.dq_raw = _check_saturation_numba(
        input_model.data,
        input_model.dq_raw,
        saturation_model.sat_thresh,
        saturation_model.dq,
    )

    return input_model


@njit(nogil=True, parallel=True, cache=True)
def _check_saturation_numba(data, dq_raw, sat_thresh, sat_dq):
    """
    Core saturation flagging routine (Numba-friendly).

    Args:
        data:        (n_read, ny, nx) float32/float64 ramp data
        dq_raw:      (n_read, ny, nx) uint32 DQ array (modified in place)
        sat_thresh:  (ny, nx) float32 saturation threshold map
        sat_dq:      (ny, nx) uint32 DQ map from saturation model
    """
    n_read, ny, nx = data.shape
    for j in prange(ny):  # parallelize over spatial rows
        for i in range(nx):
            thresh = sat_thresh[j, i]
            dq_val = sat_dq[j, i]
            for r in range(n_read):
                if data[r, j, i] >= thresh:
                    dq_raw[r, j, i] |= (SATURATED_FLAG | DO_NOT_USE_FLAG)
                dq_raw[r, j, i] |= dq_val
    return dq_raw
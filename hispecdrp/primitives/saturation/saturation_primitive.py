from ..base_primitive import BasePrimitive
from ...calibrations import SaturationSelector
from ... import datamodels
import numpy as np
from ...datamodels.dqflags import DQ_FLAGS

__all__ = ['CheckSaturationPrimitive']


class CheckSaturationPrimitive(BasePrimitive):
    
    ARGUMENTS = {
        'suffix' : 'checksat',
        'output_model' : 'input'
    }

    CALIBRATIONS = {
        'saturation': {
            'selector': SaturationSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):

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
    where_saturated = np.where(input_model.data >= saturation_model.sat_thresh[None,:,:])
    if hasattr(input_model, 'DQ_RAW'):
        input_model.dq_raw[where_saturated] |= DQ_FLAGS["SATURATED"] | DQ_FLAGS["DO_NOT_USE"]
        input_model.dq_raw |= saturation_model.dq[None,:,:]
    else:
        raise AttributeError("Input model has no DQ_RAW array to flag saturation.")
    return input_model

from ..base_primitive import BasePrimitive
from ... import datamodels
from ...calibrations import DetectorGainSelector

import numpy as np

__all__ = ['DetectorGainPrimitive']


class DetectorGainPrimitive(BasePrimitive):

    ARGUMENTS = {
        'suffix' : 'detgain',
        'output_model' : 'input'
    }

    CALIBRATIONS = {
        'detgain': {
            'selector': DetectorGainSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):

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
    input_model.data = input_model.data * detgain_model.data
    if hasattr(input_model, 'err'):
        input_model.err = np.sqrt(detgain_model.data) * input_model.err
    if hasattr(input_model, 'dq'):
        input_model.dq |= detgain_model.dq
    return input_model

from hispecdrp.datamodels.hispec import dark
from ..base_primitive import BasePrimitive
from ... import datamodels
from ...calibrations import DetectorFlatSelector

import numpy as np

__all__ = ['DetectorFlatPrimitive']


class DetectorFlatPrimitive(BasePrimitive):

    ARGUMENTS = {
        'suffix' : 'detflat',
    }

    CALIBRATIONS = {
        'detflat': {
            'selector': DetectorFlatSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):

        # Load the input model
        input_model = datamodels.open(self.input)

        # Get the detflat model
        detflat, _ = self.get_calibration(input_model, 'detflat')
        detflat_model = datamodels.open(detflat)

        # Output
        result = input_model.copy()

        # Log
        self.log.info(f"Applying detector flat correction with {detflat} to {input_model}")

        # Perform detector flat correction
        output_model = detector_flat_correction(result, detflat_model)

        # Output
        return dict(output=output_model, detflat=detflat)


def detector_flat_correction(input_model : datamodels.DataModel, detflat_model : datamodels.DataModel) -> datamodels.DataModel:
    """
    Correct the detector level flat field.

    Args:
        input_model (DataModel): Input model.
        detflat_model (HISPECFlatModel): Input detector flat model.
    
    Returns:
        DataModel: The model corrected in place.
    """
    z = input_model.data / detflat_model.data
    input_model.err = z * np.sqrt((input_model.err / input_model.data)**2 + (detflat_model.err / detflat_model.data)**2)
    input_model.dq |= detflat_model.dq
    input_model.data = z
    return input_model
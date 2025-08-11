
from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np

__all__ = ['DetectorFlatPrimitive']


class DetectorFlatPrimitive(BasePrimitive):
    """
    Corrects the detector level pixel sensitivity.

    ARGUMENTS:
        detflat (DetectorFlatModel | str | None): Detector flat model to apply to the input data.
            If None, the detector flat model will be queried.
    """

    ARGUMENTS = {
        'detflat' : None
    }

    def _perform(self):
        input_model = datamodels.open(self.name)
        if self.detflat is not None:
            flat_model = datamodels.open(self.detflat)
        else:
            detflat_path = self.get_calibration(input_model, use_cached=True)
            flat_model = datamodels.open(detflat_path)
        result = input_model.copy()
        self.log.info(f"Applying detector flat correction with {flat_model} to {input_model}")
        result = detector_flat_correction(result, flat_model)
        return dict(name=result, detflat=self.detflat)


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
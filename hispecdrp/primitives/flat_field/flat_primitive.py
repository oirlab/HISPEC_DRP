from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np

__all__ = ['FlatFieldPrimitive']


class FlatFieldPrimitive(BasePrimitive):
    """
    ARGUMENTS:
        flat (FlatModel | str | None): Flat model to apply to the input data.
            If None, the flat model will be queried.
    """

    class_alias = 'flat_field'

    ARGUMENTS = {
        'suffix' : 'flatfield',
        'flat' : None
    }


    def _perform(self):
        input_model = datamodels.open(self.args.name, copy=False)
        if self.args.flat is None:
            flat_path = self.get_calibration_file(input_model, cal_type='flat')
        else:
            flat_path = self.args.flat
        flat_model = datamodels.open(flat_path, copy=False)
        result = input_model.copy()
        self.log.info(f"Applying flat field correction with {flat_model} to {input_model}")
        result = flatfield_correction(result, flat_model)
        return result
    

def flatfield_correction(
        input_model : datamodels.HISPECL0Model,
        flat_model : datamodels.HISPECFlatModel,
    ) -> datamodels.HISPECRampModel:
    """
    Flat field correction.

    Args:
        input_model (HISPECL0Model): Input level 0 model.
        flat_model (HISPECFlatModel): Flat model.
    
    Returns:
        HISPECL0Model: The flatfield corrected data.
    """
    z = input_model.data / flat_model.data
    if hasattr(input_model, 'err_all'):
        input_model.err_all = z * np.sqrt((input_model.err_all / input_model.data)**2 + (flat_model.err_all / flat_model.data)**2)
    if hasattr(input_model, 'err_floor'):
        input_model.err_floor = z * np.sqrt((input_model.err_floor / input_model.data)**2 + (flat_model.err_floor / flat_model.data)**2)
    if hasattr(input_model, 'dq'):
        input_model.dq |= flat_model.dq
    input_model.data = z
    return input_model
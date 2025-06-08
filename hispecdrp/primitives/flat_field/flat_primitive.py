from ..base_primitive import BasePrimitive
from ...calibrations import get_calibration_store, CalibrationSelector, FlatFieldSelector
from ... import datamodels
from keckdrpframework.models.arguments import Arguments
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
            flat_path = self.get_calibration_file(input_model, selector=FlatFieldSelector(), use_cached=True)
        else:
            flat_path = self.args.flat
        flat_model = datamodels.open(flat_path, copy=False)
        result = input_model.copy()
        self.log.info(f"Applying flat field correction with {flat_model} to {input_model}")
        result = flatfield_correction(result, flat_model)
        return Arguments(name=result)
    

    def get_calibration(self, name, selector : CalibrationSelector | None = None, use_cached : bool = True, **kwargs):
        cal_store = get_calibration_store()
        if selector is None:
            selector = FlatFieldSelector()
        dark_file = cal_store.get_calibration(name, selector, use_cached=use_cached, **kwargs)
        return dark_file
    

def flatfield_correction(
        input_model : datamodels.HISPECL0Model,
        flat_model : datamodels.HISPECFlatModel,
    ) -> datamodels.HISPECL0Model:
    """
    Flat field correction.

    Args:
        input_model (HISPECL0Model): Input level 0 model.
        flat_model (HISPECFlatModel): Flat model.
    
    Returns:
        HISPECL0Model: The flatfield corrected data.
    """
    z = input_model.data / flat_model.data
    if hasattr(input_model, 'err'):
        input_model.err = z * np.sqrt((input_model.err / input_model.data)**2 + (flat_model.err / flat_model.data)**2)
    if hasattr(input_model, 'dq'):
        input_model.dq |= flat_model.dq
    input_model.data = z
    return input_model
from ..base_primitive import BasePrimitive
from ...calibrations import get_calibration_store, CalibrationSelector, DarkSelector
from ... import datamodels
import numpy as np

__all__ = ['DarkSubtractionPrimitive']


class DarkSubtractionPrimitive(BasePrimitive):
    """
    ARGUMENTS:
        dark (DataModel | str | None): Dark model to apply to the input data.
            If None, the dark model will be queried.
    """

    class_alias = 'dark_sub'

    ARGUMENTS = {
        'suffix' : 'darksub',
        'dark' : None
    }


    def _perform(self):
        input_model = datamodels.open(self.args.name, copy=False)
        if self.args.dark is None:
            dark_path = self.get_calibration(input_model, selector=DarkSelector(), use_cached=True)
        else:
            dark_path = self.args.dark
        dark_model = datamodels.open(dark_path, copy=False)
        result = input_model.copy()
        self.log.info(f"Applying dark correction with {dark_model} to {input_model}")
        if isinstance(result, datamodels.HISPECRampModel):
            result = dark_subtraction_ramp(result, dark_model)
        elif isinstance(result, datamodels.HISPECL0Model):
            result = dark_subtraction_rate(result, dark_model)
        return result
    
    def get_calibration(self, name, selector : CalibrationSelector | None = None, use_cached : bool = True, **kwargs):
        if selector is None:
            selector = DarkSelector()
        return super().get_calibration(name, selector, use_cached=use_cached, **kwargs)
        

def dark_subtraction_ramp(
        input_model : datamodels.HISPECRampModel,
        dark_model : datamodels.HISPECDarkModel,
    ) -> datamodels.HISPECRampModel:
    """
    Subtract the dark model from an input ramp model.
    
    Args:
        input_model (RampModel): Input ramp model.
        dark_model (DarkModel): Dark model.
    
    Returns:
        RampModel: The dark-subtracted data.
    """
    input_model.data = input_model.data - dark_model.data[:, :, None] # NOTE: THIS WILL NEED TO BE UPDATED TO INCLUDE ITIME
    # TODO: Sort out error handling for ramps.
    return input_model


def dark_subtraction_rate(
        input_model : datamodels.HISPECL0Model,
        dark_model : datamodels.HISPECDarkModel,
    ) -> datamodels.HISPECL0Model:
    """
    Subtract the dark model from an input rate (level 0) model. The correction is done in-place.

    Args:
        input_model (HISPECL0Model): Input level 0 model.
        dark_model (HISPECDarkModel): Dark model.
    
    Returns:
        HISPECL0Model: The dark-subtracted data.
    """
    input_model.data = input_model.data - dark_model.data
    if hasattr(input_model, 'err'):
        input_model.err = np.sqrt(input_model.err**2 + dark_model.err**2)
    if hasattr(input_model, 'dq'):
        input_model.dq |= dark_model.dq
    return input_model
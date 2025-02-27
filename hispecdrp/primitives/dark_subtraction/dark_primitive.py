from ..base_primitive import BasePrimitive
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
            dark_path = self.get_calibration_file(input_model, cal_type='dark')
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
    input_model.data = input_model.data - dark_model.data[:, :, None]
    # TODO: Sort this out. Currently ramp data has no error or DQ arrays.
    # if hasattr(result, 'err_all'):
    #     result.err_all = np.sqrt(result.err_all**2 + dark_model.err_all**2)
    # if hasattr(result, 'err_floor'):
    #     result.err_floor = np.sqrt(result.err_floor**2 + dark_model.err_floor**2)
    # if hasattr(result, 'dq'):
    #     result.dq |= dark_model.dq
    return input_model


def dark_subtraction_rate(
        input_model : datamodels.HISPECL0Model,
        dark_model : datamodels.HISPECDarkModel,
    ) -> datamodels.HISPECRampModel:
    """
    Subtract the dark model from an input rate (level 0) model.

    Args:
        input_model (HISPECL0Model): Input level 0 model.
        dark_model (HISPECDarkModel): Dark model.
    
    Returns:
        HISPECL0Model: The dark-subtracted data.
    """
    input_model.data = input_model.data - dark_model.data
    if hasattr(input_model, 'err_all'):
        input_model.err_all = np.sqrt(input_model.err_all**2 + dark_model.err_all**2)
    if hasattr(input_model, 'err_floor'):
        input_model.err_floor = np.sqrt(input_model.err_floor**2 + dark_model.err_floor**2)
    if hasattr(input_model, 'dq'):
        input_model.dq |= dark_model.dq
    return input_model
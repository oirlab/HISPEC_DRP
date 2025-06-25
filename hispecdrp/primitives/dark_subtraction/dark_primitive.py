from ..base_primitive import BasePrimitive
from ...calibrations import HISPECDarkSelector, PARVIDarkSelector
from ... import datamodels
from keckdrpframework.models.arguments import Arguments
import numpy as np

__all__ = ['DarkSubtractionPrimitive']


class DarkSubtractionPrimitive(BasePrimitive):
    """
    ARGUMENTS:
        dark (DataModel | str | None): Dark model to apply to the input data.
            If None, the dark model will be queried.
    """

    ARGUMENTS = {
        'suffix' : 'darksub',
        'dark' : None,
    }

    def _perform(self):
        input_model = datamodels.open(self.args.name)
        if self.args.dark is None:
            dark_path = self.get_calibration(input_model, use_cached=True)
        else:
            dark_path = self.args.dark
        dark_model = datamodels.open(dark_path)
        result = input_model.copy()
        self.log.info(f"Applying dark correction with {dark_model} to {input_model}")
        result = dark_subtraction_rate(result, dark_model)
        return Arguments(name=result)

    def get_calibration_selector(self, name):
        input_model = datamodels.open(name, meta_only=True)
        if isinstance(input_model, datamodels.HISPECDataModel):
            return HISPECDarkSelector()
        elif isinstance(input_model, datamodels.PARVIDataModel):
            return PARVIDarkSelector()
        else:
            raise ValueError(f"Unsupported input model type: {type(input_model)}")

        
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
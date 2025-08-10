from ..base_primitive import BasePrimitive
from ...calibrations import DarkSelector
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
    }

    CALIBRATIONS = {
        'dark': {
            'selector': DarkSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):

        # Load the input model
        input_model = datamodels.open(self.name)

        # Get the dark model
        dark_path = self.get_calibration(input_model, 'dark')
        dark_model = datamodels.open(dark_path)
        
        # Output
        result = input_model.copy()

        # Log
        self.log.info(f"Applying dark correction with {dark_model} to {input_model}")

        # Perform dark subtraction
        output_model = dark_subtraction_rate(result, dark_model)

        # Output
        return Arguments(name=output_model, dark=dark_model)


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
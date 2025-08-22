from ..base_primitive import BasePrimitive
from ...calibrations import DarkSelector
from ... import datamodels
import numpy as np

__all__ = ['DarkSubtractionPrimitive']


class DarkSubtractionPrimitive(BasePrimitive):
    
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
        input_model = datamodels.open(self.input)

        # Get the dark model
        dark, _ = self.get_calibration(input_model, 'dark')
        dark_model = datamodels.open(dark)

        # Output
        result = input_model.copy()

        # Log
        self.log.info(f"Applying dark correction with {dark} to {input_model}")

        # Perform dark subtraction
        output_model = dark_subtraction_rate(result, dark_model)

        # Output
        return dict(output=output_model, dark=dark)


def dark_subtraction_rate(input_model, dark_model):
    """
    Subtract the dark model from an input rate (level 0) model. The correction is done in-place.

    Args:
        input_model: Input level 0 model.
        dark_model: Dark calibration model.
    
    Returns:
        HISPECL0Model: The dark-subtracted data.
    """
    input_model.data = input_model.data - dark_model.data
    if hasattr(input_model, 'err'):
        input_model.err = np.sqrt(input_model.err**2 + dark_model.err**2)
    if hasattr(input_model, 'dq'):
        input_model.dq |= dark_model.dq
    return input_model
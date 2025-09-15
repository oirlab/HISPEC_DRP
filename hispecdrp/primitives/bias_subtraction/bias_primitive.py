from ..base_primitive import BasePrimitive
from ...calibrations import BiasSelector
from ... import datamodels
import numpy as np

__all__ = ['BiasSubtractionPrimitive']


class BiasSubtractionPrimitive(BasePrimitive):
    
    ARGUMENTS = {
        'suffix' : 'biassub',
        'output_model' : 'input'
    }

    CALIBRATIONS = {
        'bias': {
            'selector': BiasSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):

        # Load the input model
        input_model = datamodels.open(self.input)

        # Get the bias model
        bias, _ = self.get_calibration(input_model, 'bias')
        bias_model = datamodels.open(bias)

        # Output
        result = self.get_output_model(input_model)

        # Log
        self.log.info(f"Applying bias correction with {bias} to {input_model}")

        # Perform bias subtraction
        output_model = bias_subtraction(result, bias_model)

        # Output
        return dict(output=output_model, bias=bias)


def bias_subtraction(input_model: datamodels.RampModel, bias_model: datamodels.BiasModel) -> datamodels.RampModel:
    """
    Subtract the bias from the ramps.

    Args:
        input_model: Input RampModel.
        bias_model: Bias calibration model.
    
    Returns:
        RampModel: The bias-subtracted data.
    """
    input_model.data = input_model.data - bias_model.bias[None, :, :]
    if hasattr(input_model, 'DQ_RAW'):
        input_model.dq_raw |= bias_model.dq[None,:,:]
    return input_model
from ..base_primitive import BasePrimitive
from ...calibrations import DQSelector
from ... import datamodels
import numpy as np

__all__ = ['DQInitPrimitive']


class DQInitPrimitive(BasePrimitive):
    
    ARGUMENTS = {
        'suffix' : 'dqinit',
        'output_model' : 'input'
    }

    CALIBRATIONS = {
        'dq': {
            'selector': DQSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):

        # Load the input model
        input_model = datamodels.open(self.input)

        # Get the dark model
        dq, _ = self.get_calibration(input_model, 'dq')
        dq_model = datamodels.open(dq)

        # Output
        result = self.get_output_model(input_model)

        # Log
        self.log.info(f"Applying DQ initialization with {dq} to {input_model}")

        if hasattr(input_model, 'DQ_RAW'):
            result.dq_raw = input_model.dq_raw | dq_model.dq[None,:,:]

        # Output
        return dict(output=result, bias=dq)

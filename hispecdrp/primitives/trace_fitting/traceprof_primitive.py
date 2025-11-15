from ..base_primitive import BasePrimitive
from ... import datamodels
from typing import Sequence

__all__ = ['TraceProfilePrimitive']


class TraceProfilePrimitive(BasePrimitive):

    ARGUMENTS = {
        'suffix' : 'traceprof',
        'output_model' : 'TraceProfModel',
    }

    def _perform(self):
        
        # Associate inputs by fiber number
        if isinstance(self.input, (datamodels.DataModel, str)):
            input = datamodels.associate_fibers([self.input])
        if  isinstance(self.input, Sequence):
            input = datamodels.associate_fibers(self.input)
        elif isinstance(self.input, dict):
            input = self.input
        else:
            raise ValueError(f"Unsupported input type: {type(self.input)}")
        
        # Prepare output model
        output_model = self.get_output_model(input, include_data=False)

        # Loop over fibers and copy profile data
        for fiber in input.keys():
            input_model = datamodels.open(input[fiber])
            setattr(output_model, f'PROF_FIB{fiber}', input_model.rate)
            setattr(output_model, f'PROF_ERR_FIB{fiber}', input_model.rate_err)
            self.add_data_ancestors(output_model, input_model)

        return dict(output=output_model)
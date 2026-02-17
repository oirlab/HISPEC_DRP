from ..base_primitive import BasePrimitive
from ... import datamodels
from typing import Sequence

__all__ = ['TraceProfilePrimitive']


class TraceProfilePrimitive(BasePrimitive):

    """
    Calculate the trace positions for each fiber/order.

    Parameters
    ----------
    input : DataModel or list of DataModels or dict of DataModels
        Input data. Can be a single DataModel or filename, a sequence of DataModels or filenames, or a dict mapping fiber numbers to DataModels or filenames.
    """

    ARGUMENTS = {
        'output_model' : 'TraceProfModel',
    }

    def _perform(self):
        
        # Associate inputs by fiber number
        if isinstance(self.input, (datamodels.DataModel, str)):
            input = datamodels.group_by_illuminated_fiber([self.input])
        if  isinstance(self.input, Sequence):
            input = datamodels.group_by_illuminated_fiber(self.input)
        elif isinstance(self.input, dict):
            input = self.input
        else:
            raise ValueError(f"Unsupported input type: {type(self.input)}")
        
        # Prepare output model
        output_model = datamodels.TraceProfModel.from_datamodels(input)

        # Loop over fibers and copy profile data
        for fiber in input.keys():
            input_model = datamodels.open(input[fiber])
            setattr(output_model, f'PROF_FIB{fiber}', input_model.rate)
            setattr(output_model, f'PROF_ERR_FIB{fiber}', input_model.rate_err)

        return dict(output=output_model)
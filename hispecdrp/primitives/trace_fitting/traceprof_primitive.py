from ..base_primitive import BasePrimitive
from ... import datamodels

__all__ = ['TraceProfilePrimitive']


class TraceProfilePrimitive(BasePrimitive):

    ARGUMENTS = {
        'suffix' : 'traceprof',
        'output_model' : 'TraceProfModel',
    }

    def _perform(self):
        input_type = type(self.input)
        if input_type in (datamodels.DataModel, str):
            input = self.associate_fibers([self.input])
        elif input_type in (list, tuple):
            input = self.associate_fibers(self.input)
        elif input_type is dict:
            input = self.input
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
        first_model = datamodels.open(input[list(input.keys())[0]])
        output_model = self.get_output_model(first_model)
        for fiber in input.keys():
            input_model = datamodels.open(input[fiber])
            setattr(output_model, f'PROF_FIB{fiber}', input_model.rate)
            setattr(output_model, f'PROF_ERR_FIB{fiber}', input_model.rate_err)
            self.add_data_ancestors(output_model, input_model)

        return dict(output=output_model)
from ..base_primitive import BasePrimitive
from ... import datamodels

__all__ = ['FlatFieldPrimitive']


class FlatFieldPrimitive(BasePrimitive):
    """
    ARGUMENTS:
        flat (datamodels.DataModel | str | None): Flat field model to apply to the input data.
            If None, the flat field model will be queried.
    """

    class_alias = 'flat_field'

    ARGUMENTS = {
        'suffix' : 'flatfield',
        'flat' : None
    }

    def _perform(
            self,
            name : datamodels.HISPECL0 | str,
        ):
        model_to_process = self.open_model(name)
        return model_to_process
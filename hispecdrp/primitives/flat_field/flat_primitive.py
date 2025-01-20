from ..base_primitive import BasePrimitive
from ... import datamodels

__all__ = ['FlatFieldPrimitive']


class FlatFieldPrimitive(BasePrimitive):
    """
    ARGUMENTS:
        flat (datamodels.DataModel | str | None): Flat field model to apply to the input data.
            If None, the flat field model will be queried.
    """

    ARGUMENTS = {
        'suffix' : 'flatfield',
        'flat' : None
    }

    def _perform(
            self,
            name : datamodels.DataModel | str,
        ):
        return datamodels.open(name)
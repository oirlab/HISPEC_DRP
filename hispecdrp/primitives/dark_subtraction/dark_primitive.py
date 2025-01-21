from ..base_primitive import BasePrimitive
from ... import datamodels

__all__ = ['DarkSubtractionPrimitive']


class DarkSubtractionPrimitive(BasePrimitive):
    """
    ARGUMENTS:
        dark (datamodels.DataModel | str | None): Dark model to apply to the input data.
            If None, the dark model will be queried.
    """

    class_alias = 'darksub'

    ARGUMENTS = {
        'suffix' : 'darksub',
        'dark' : None
    }

    def _perform(
            self,
            name : datamodels.HISPECL0 | str,
        ):
        model_to_process = self.open_model(name)
        return model_to_process
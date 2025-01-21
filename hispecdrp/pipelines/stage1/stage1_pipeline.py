
from ..base_pipeline import BasePipeline
from ... import datamodels
from ...primitives import DarkSubtractionPrimitive, FlatFieldPrimitive

__all__ = ['Stage1Pipeline']

class Stage1Pipeline(BasePipeline):

    class_alias = 'stage1'

    PRIMITIVES = {
        'dark_sub': DarkSubtractionPrimitive,
        'flat_field': FlatFieldPrimitive,
    }


    def _perform(self, name : datamodels.HISPECL0 | str):
        result = self.dark_sub.apply(name)
        result = self.flat_field.apply(result)
        return result
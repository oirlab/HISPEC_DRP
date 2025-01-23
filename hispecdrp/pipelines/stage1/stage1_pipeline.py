
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


    def _perform(self):
        model_to_process = self.open_model(self.args.name)
        result = self.dark_sub.apply(model_to_process)
        result = self.flat_field.apply(result)
        return result
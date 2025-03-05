from hispecdrp import datamodels
from ..base_pipeline import BasePipeline
from ...primitives import DarkSubtractionPrimitive, FlatFieldPrimitive

__all__ = ['Stage1Pipeline']

class Stage1Pipeline(BasePipeline):

    class_alias = 'stage1'

    PRIMITIVES = {
        'dark_sub': DarkSubtractionPrimitive,
        'flat_field': FlatFieldPrimitive,
        'flux_extract':FluxExtractionPrimitive
    }


    def _perform(self):
        model_to_process = datamodels.open(self.args.name)
        result = self.dark_sub.apply(model_to_process)
        result = self.flat_field.apply(result)
        result = self.flux_extract.apply(result)
        return result
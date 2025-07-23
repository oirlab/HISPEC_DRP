from hispecdrp import datamodels
from ..base_pipeline import BasePipeline
from ...primitives import DarkSubtractionPrimitive, FlatFieldPrimitive, TracePositionPrimitive, TraceProfilePrimitive

__all__ = ['TraceFindingPipeline']

class TraceFindingPipeline(BasePipeline):

    class_alias = 'tracefinding'

    PRIMITIVES = {
        'dark_sub': DarkSubtractionPrimitive,
        'flat_field': FlatFieldPrimitive,
        'trace_pos':TracePositionPrimitive,
    }


    def _perform(self):
        ### Still not sure how this is going to work with multiple files
        result = self.dark_sub.apply(model)
        result = self.flat_field.apply(result)
        ### Coadd for each fiber??? Should do that hereish
        result = self.trace_pos.apply(result)
        return Arguments(name=result)
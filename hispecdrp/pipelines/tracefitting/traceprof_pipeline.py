from hispecdrp import datamodels
from ..base_pipeline import BasePipeline
from ...primitives import DarkSubtractionPrimitive, FlatFieldPrimitive, TracePositionPrimitive, TraceProfilePrimitive

__all__ = ['TraceFittingPipeline']

class TraceFittingPipeline(BasePipeline):

    class_alias = 'tracefitting'

    PRIMITIVES = {
        'dark_sub': DarkSubtractionPrimitive,
        'flat_field': FlatFieldPrimitive,
        'trace_pos':TracePositionPrimitive,
        'trace_prof':TraceProfilePrimitive
    }


    def _perform(self):
        ### Still not sure how this is going to work with multiple files
        result = self.dark_sub.apply(model)
        result = self.flat_field.apply(result)
        ### Coadd for each fiber??? Should do that hereish
        result = self.trace_pos.apply(result)
        ### Everything above here should have been run previously in the tracepos_pipeline???
        result = self.trace_prof.appy(result)
        return Arguments(name=result)
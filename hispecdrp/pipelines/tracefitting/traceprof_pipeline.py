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
        model_to_process = datamodels.open(self.args.name)
        result = self.dark_sub.apply(model_to_process)
        result = self.flat_field.apply(result)
        result = self.trace_pos.apply(result)
        result = self.trace_prof.appy(result)
        return result
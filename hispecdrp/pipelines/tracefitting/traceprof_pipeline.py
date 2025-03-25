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
        ### FIXME if we're doing this on a stack that stack needs
        ### to be processed and coadded before we hit the trace
        ### finding primitives
        model_to_process = datamodels.open(self.args.name)
        result = self.dark_sub.apply(model_to_process)
        result = self.flat_field.apply(result)
        ### Coadd for each fiber

        ### Feed in images for each fiber
        ### trace_pos is supposed to takea  list of 3 images,
        ### 1 per fiber - how do we do this?
        result = self.trace_pos.apply(result)
        result = self.trace_prof.appy(result)
        return result
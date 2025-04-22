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
        ### Self.args.name is a list of filenames, or 1 file name
        models_to_process = datamodels.open(self.args.name)
        for model in model_to_process:
            ### Loop over files for flat/dark subtraction, and get fiber info from metadata
            result = self.dark_sub.apply(model_to_process)
            result = self.flat_field.apply(result)
        

        ### Group fibers


            ### Coadd for each fiber

            ### Feed in images for each fiber
            ### trace_pos is supposed to takea  list of 3 images,
            ### 1 per fiber - how do we do this?
            result = self.trace_pos.apply(result)
            result = self.trace_prof.appy(result)

        ### Stitch fibers together into 1 cal file

        ### If fiber not present, create extension but don't populate

        return result
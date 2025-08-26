from hispecdrp import datamodels
from ..base_pipeline import BasePipeline
from ...primitives import DarkSubtractionPrimitive, FlatFieldPrimitive, TracePositionPrimitive, TraceProfilePrimitive

__all__ = ['TraceFittingPipeline']

class TraceFittingPipeline(BasePipeline):

    class_alias = 'tracfitting'

    PRIMITIVES = {
        'dark_sub': DarkSubtractionPrimitive,
        'flat_field': FlatFieldPrimitive,
        'trace_pos':TracePositionPrimitive,
        'coadd_frames':CoaddPrimitive
    }


    def _perform(self):
        ### Take in rate maps for the day's fiber flats
        ### 1 rate map or multiple? Probably a group
        ### Still not sure how this is going to work with multiple files
        result = self.dark_sub.apply(model)
        result = self.flat_field.apply(result)

        coadded = []
        for fiber in self.args.do_fibers:
            ### Get the datamodels that correspond to a given fiber
            names = self.args.names[self.args.names==fiber]
            coadded.append(self.coadd.apply(names))
        ### This should be a list with one entry per fiber
        ### I'm not sure this is how to make that list in a way 
        ### that do_fibers will parse correctly later
        result = self.trace_pos.apply(coadded)
        ### In theory there are cases (quicklook) where we don't want to do this 
        ### part
        result = self.trace_prof.apply(result)
        return Arguments(name=result)
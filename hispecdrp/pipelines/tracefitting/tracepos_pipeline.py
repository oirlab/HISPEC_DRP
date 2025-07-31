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
        ### Take in rate maps for the day's fiber flats
        ### 1 rate map or multiple? Probably a group
        ### Still not sure how this is going to work with multiple files
        result = self.dark_sub.apply(model)
        result = self.flat_field.apply(result)
        ### Coadd for each fiber??? Should do that hereish
        ### TODO write coadd primitive
        result = self.trace_pos.apply(result)
        
        ### Load existing trace_pos
        result = self.trac_prof.apply(result)
        ### Arguments vs datamodel? 
        ### Arguments lets us return ancillary data not packed into a 
        ### datamodel
        return Arguments(name=result)
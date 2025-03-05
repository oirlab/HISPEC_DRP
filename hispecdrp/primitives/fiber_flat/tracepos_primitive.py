from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np

__all__ = ['TracePositionPrimitive']


class TracePositionPrimitive(BasePrimitive):
    """
    ARGUMENTS:
        Deg (int | 4): Polynomial order to use for computing trace positions
    """

    class_alias = 'trace_position'

    ARGUMENTS = {
        'suffix' : 'tracepos',
        'deg':4
    }

    def _perform(self):
        ### FIXME Input model should be a flat/dark corrected rate map, how do we enforce that?
        input_model = datamodels.open(self.args.name, copy=False)

        result = input_model.copy()
        self.log.info(f"Performing trace finding on {input_model}")

       
        result = find_trace_pos(result, self.args.deg)
        return result
    



def find_trace_pos(
        input_model : datamodels.HISPECL0Model,
        deg : int
    ) -> datamodels.HISPECTracePosModel:

    ### Mask bad/low pixels

    ### Figure out number of orders in the frame

    ### Find peaks in each column

    ### Write trace position data
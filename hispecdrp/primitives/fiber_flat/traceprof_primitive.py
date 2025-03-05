from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np

__all__ = ['TraceProfilePrimitive']


class TraceProfilePrimitive(BasePrimitive):
    """
    ARGUMENTS:
        tracepos (datamodels.HISPECTraceProsModel | None): Trace centers to do fitting on
        Deg (int | 4): Polynomial order to use for smoothing trace profile parameters

    """

    class_alias = 'trace_profile'

    ARGUMENTS = {
        'suffix' : 'traceprof',
        'tracepos' : None,
        'deg':4
    }

    def _perform(self):
        ### FIXME Input model should be a flat/dark corrected rate map, how do we enforce that?
        ### By using a pipeline file
        input_model = datamodels.open(self.args.name, copy=False)

        ### Load the trace positions if not passed
        if self.args.tracepos is None:
            tracepos_path = self.get_calibration_file(input_model, cal_type='tracepos')
        else:
            tracepos_path = self.args.tracepos
        tracepos_model = datamodels.open(tracepos_path, copy=False)

        result = input_model.copy()
        self.log.info(f"Performing trace profile fitting on {input_model} with positions {tracepos_model}")

       
        result = fit_trace_prof(result, self.args.tracepos_model, self.args.deg)
        return result
    



def fit_trace_prof(
        input_model : datamodels.HISPECL0Model,
        trace_pos : datamodels.HISPECTracePosModel
        deg : int
    ) -> datamodels.HISPECTracePosModel:

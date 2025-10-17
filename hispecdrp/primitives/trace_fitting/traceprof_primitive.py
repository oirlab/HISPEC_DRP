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
        'width':7
    }

    def _perform(self):
        ### FIXME Input model should be a flat/dark corrected rate map, how do we enforce that?
        ### By using a pipeline file
        input_model = datamodels.open(self.args.name, copy=False)
        output_model = self.get_output_model(input_mdoel, model_suffix='HISPECTraceProfModel')
        out_dict = {}

        ### Load the trace positions if not passed
        if self.args.tracepos is None:
            tracepos_path = self.get_calibration_file(input_model, cal_type='tracepos')
        else:
            tracepos_path = self.args.tracepos
        tracepos_model = datamodels.open(tracepos_path, copy=False)


         for fiber in self.args.do_fibers:
            result = input_model.copy()
            ### I'm assuming fiber is an int for now, or has a corresponding int
            self.log.info(f"Performing trace finding on {input_model} fiber {fiber}")
            self.log.info(f"Performing trace profile fitting on {input_model} with positions {tracepos_model}")
            ### Get the right fiber locations from the trace position file
            fib_locs = self.args.tracepos_model[f'POS_FIB{fiber}']
            ### Do the calculation
            result = fit_trace_prof(result, fib_locs, self.args.width)
            ### Save to the extension for the corresponding fiber
            out_dict[f'PROF_FIB{fiber}'] = result
            out_dict[f'POS_FIB{fiber}'] fib_locs
        return Arguments(name=output_model,fit_results=out_dict)
    

def fit_trace_prof(
        input_model : datamodels.HISPECL0Model,
        trace_pos : np.ndarray,
        width: int
    ) -> datamodels.HISPECTracePosModel:
    
    ### Get the data
    data = input_model.data

    ### Instantiate output array - norder x npix x width
    out = np.zeros((trace_pos.shape[0],trace_pos.shape[1], 2*width+1))

    ### Loop over orders
    for i in range(trace_pos.shape[0]):
        ### Loop over pixels
        for j in range(trace_pos.shape[1]):
            ### Slice out the data within +/- width of the center
            center = int(trace_pos[i,j])
            dslice = data[center-width:center+width+1, j]
            ### Normalize so that it sums to 1 and save
            out[i,j,:] = dslice/np.nansum(dslice)
    return out

    
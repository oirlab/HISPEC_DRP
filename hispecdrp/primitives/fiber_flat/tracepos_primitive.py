from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np

__all__ = ['TracePositionPrimitive']


class TracePositionPrimitive(BasePrimitive):
    """
    ARGUMENTS:
        deg (int | 4): Polynomial order to use for computing trace positions
        threshold_peak (float | 500): Minimum peak counts to identify a trace
        cutsize (int | 10): Half width of cut used to compute flux-weighted trace center
    """

    class_alias = 'trace_position'

    ARGUMENTS = {
        'suffix' : 'tracepos',
        'deg':4
        'threshold_peak':500
        'cutsize':10
    }

    def _perform(self):
        input_model = datamodels.open(self.args.name)
        output_model = self.get_output_model(input_model, model_suffix='HISPECTracePosModel')
        out_dict = {}

        for fiber in self.args.do_fibers:
            ### I'm assuming fiber is an int for now, or has a corresponding int
            self.log.info(f"Performing trace finding on {input_model} fiber {fiber}")
            ### FIXME how do we index the fibers?
            fib_locs, fib_coeffs = find_trace_pos(input_model, fiber, self.args.deg, self.args.threshold_peak, self.args.cutsize) 
            out_dict[f'POS_FIB{fiber}'] = fib_locs
            out_dict[f'COEFF_FIB{fiber}'] = fib_coeffs
        
        return Arguments(name=output_model,fit_results=out_dict)
    
def find_trace_pos(
        input_model : datamodels.HISPECL0Model,
        fnum: int,
        deg : int,
        threshold_peak: float,
        cutsize: int
    ) -> datamodels.HISPECTracePosModel:

    
    out_dict = {}
    data = input_model.data

    ### Initial peak finding
    peaks,_ = signal.find_peaks(data[:,0],height=threshold_peak)
    
    ### Declare output arrays
    trace_locs = np.zeros((len(peaks),data.shape[1]))
    smooth_trace_locs = np.zeros(trace_locs.shape)
    coeffs = np.zeros((trace_locs.shape[0],deg+1))

    ### Loop through columns
    for i in range(data.shape[1]):
        ### Get index of peaks from scipy
        ### We could write our own jit-compatible version, but it's so
        ### fast that it doesn't seem worth it. The scipy version also
        ### has prominence/width criteria which might be helpful for
        ### avoiding bad pixels, once we have the actual assembled 
        ### detectors to check
        peaks, _ = signal.find_peaks(data[:,i],height=threshold_peak)
        ### Get flux-weighted center for each peak on its own
        for j, peak in enumerate(peaks):
            inds = np.arange(peak-cutsize,peak+cutsize+1)
            dslice = data[inds,i]
            trace_locs[j,i] = np.sum(inds*dslice)/np.sum(dslice)

    ### Do polynomial fit to traces
    xpts = np.arange(trace_locs.shape[1])
    for i in range(trace_locs.shape[0]):
        ypts = trace_locs[i]
        coeffs[i] = np.polyfit(xpts, ypts, deg)
        fit = np.poly1d(coeffs[i])
        smooth_trace_locs[i,:] = fit(xpts)

    return smooth_trace_locs, coeffs
   

    
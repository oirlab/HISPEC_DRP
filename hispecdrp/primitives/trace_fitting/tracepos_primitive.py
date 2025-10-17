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

        ### Make output data model with empty data 

        for fiber in self.args.do_fibers:
            ### I'm assuming fiber is an int for now, or has a corresponding int
            self.log.info(f"Performing trace finding on {input_model} fiber {fiber}")
            ### FIXME how do we index the fibers?
            fib_locs, fib_coeffs, fib_widths, fib_width_coeffs = find_trace_pos(input_model, fiber, self.args.deg, self.args.threshold_peak, self.args.cutsize) 
            out_dict[f'POS_FIB{fiber}'] = fib_locs
            out_dict[f'COEFF_FIB{fiber}'] = fib_coeffs
            out_dict[f'FWHM_FIB{fiber}'] = fib_widths
            out_dict[f'FWHM_COEFF_FIB{fiber}'] = fib_width_coeffs

        ### I think this is how you assign a dict to a datamodel where the keys are the extensions but I'm not sure
        output_model.data = out_dict
        
        ### arguments should also return image data for trace profile fitting
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
    trace_widths = np.zeros(trace_locs.shape)
    smooth_trace_locs = np.zeros(trace_locs.shape)
    smooth_trace_widths = np.zeros(trace_locs.shape)
    
    ### Loop through columns
    for i in range(data.shape[1]):
        ### Get index of peaks from scipy
        ### We could write our own jit-compatible version, but it's so
        ### fast that it doesn't seem worth it. The scipy version also
        ### has prominence/width criteria which might be helpful for
        ### avoiding bad pixels, once we have the actual assembled 
        ### detectors to check
        ### There are a number of additional checks we may need to add if 
        ### real fiber flats are noisy - the simulations are so photon-
        ### dominated that it's hard to be sure waht's needed
        peaks, _ = signal.find_peaks(data[:,i],height=threshold_peak)
        ### Get flux-weighted center for each peak on its own
        for j, peak in enumerate(peaks):
            inds = np.arange(peak-cutsize,peak+cutsize+1)
            dslice = data[inds,i]
            trace_locs[j,i] = np.sum(inds*dslice)/np.sum(dslice)
            ### This calculates the approx standard deviation of a gaussian fit
            ### to the trace profile
            trace_widths[j,i] = np.sqrt(np.sum((inds-peak)**2 * dslice)/np.sum(dslice))
            if trace_widths[j,i] > 1.9:
                trace_widths[j,i] = np.nan

    ### Do polynomial fit/smoothing to traces
    smooth_trace_locs, loc_coeffs = _polynomial_smooth(trace_locs,deg=deg)
    ### Convert trace widths to FWHM and smooth
    smooth_trace_widths, width_coeffs = _polynomial_smooth(2.355*trace_widths, deg=deg)

    return smooth_trace_locs, loc_coeffs, smooth_trace_widths, width_coeffs
   

   def _polynomial_smooth(data, deg=4):
    coeffs = np.zeros((data.shape[0],deg+1))
    smooth_data = np.zeros(data.shape)
    xpts = np.arange(data.shape[1])
    for i in range(data.shape[0]):
        ypts = data[i]
        goodinds = np.isfinite(ypts)
        coeffs[i] = np.polyfit(xpts[goodinds], ypts[goodinds], deg)
        fit = np.poly1d(coeffs[i])
        smooth_data[i,:] = fit(xpts)
    return smooth_data, coeffs
    

    
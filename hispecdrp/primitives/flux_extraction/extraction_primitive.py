from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np

__all__ = ['FluxExtractionPrimitive']


class FluxExtractionPrimitive(BasePrimitive):
    """
    ARGUMENTS:
        tracepos (datamodels.HISPECTracePosModel | str | None): Trace traceposs for flux extraction
            if None, the trace traceposs will be queried
        traceprof (datamodels.HISPECTraceProfModel | str | None): Profile traceprof to use for flux extraction
            If None, the fiber flat will be queried.
        method (str | 'box'): method to use for flux extraction.
            At present, only box is implemented
        boxwidth (int | 5): width to use for box extraction
    """

    class_alias = 'flux_extract'

    ARGUMENTS = {
        'suffix' : 'fluxextracted',
        'tracepos' : None
        'traceprof' : None
        method : 'box' 
        boxwidth : 5
    }

    def _perform(self):
        input_model = datamodels.open(self.args.name, copy=False)

        ### Load the trace positions if not passed
        if self.args.tracepos is None:
            tracepos_path = self.get_calibration_file(input_model, cal_type='tracepos')
        else:
            tracepos_path = self.args.tracepos
        tracepos_model = datamodels.open(tracepos_path, copy=False)

        ### If using box extraction, the traceprof should be empty
        if self.args.method.lower() == 'optimal':
            if self.args.traceprof is None:
                traceprof_path = self.get_calibration_file(input_model, cal_type='traceprof')
            else:
                traceprof_path = self.args.traceprof
            traceprof_model = datamodels.open(tracepos_path, copy=False)
        elif self.args.metho.lower() == 'box':
            traceprof_model = None
        else:
            self.log.error(f"Invalid flux extraction method {self.args.method.lower()}")
            raise ValueError(f"Invalid flux extraction method {self.args.method.lower()}")


        result = input_model.copy()
        if self.args.method.lower() == 'box'
            self.log.info(f"Performing {method} extraction on {input_model} with tracepos {tracepos_model} and width {boxwidth}")
        else:
            self.log.info(f"Performing {method} extraction on {input_model} with tracepos {tracepos_model} and traceprof {traceprof_mdoel}")
        
        result = flux_extraction(result, tracepos_model, traceprof_model, method, boxwidth)
        return result
    



def flux_extraction(
        input_model : datamodels.HISPECL0Model,
        tracepos_model : datamodels.HISPECTracePosModel,
        traceprof_model : datamodels.HISPECTraceProfModel ,
        method : str,
        boxwidth : int
    ) -> datamodels.HISPECL1Model:
    """
    Flux extraction.

    Args:
        input_model (HISPECL0Model): Input level 0 model.
        tracepos_model (HISPECTracePosModel): Trace center tracepos
        traceprof_model (HISPECTraceProfModel): traceprof for optimal extraction
        method (str): Type of extraction to perform.
        boxwidth (int): size of box to use for box extraction
    
    Returns:
        HISPECL1Model: The extracted 1D sepectra.
    """

    ### How do we compute average reference time for each wavelength

    ### Dictionary that will be filled with output data
    out_dict = {}

    ### Basic box extraction for now
    if method == 'box':
        dy = (boxwidth - 1) // 2
        ### Loop over fibers
        for i in range(1,4):
            ### Get the trace positions for the fiber
            pos = tracepos_model[f'POS_FIB{i}']
            ### Set up the output arrays
            flux = np.empty(pos.shape)
            err, rnoise, poisson = np.empty(pos.shape), np.empty(pos.shape), np.empty(pos.shape)
            dq = np.empty(pos.shape)
            ### Loop over orders and wavlength channels for the specified fiber
            ### We can definitely find ways to speed up the nested loop
            for order in range(pos.shape[0]):
                for channel in range(pos.shape[1]):
                    y = int(np.round(pos[order, channel]),0)
                    ### Check axes with some simulated data
                    ### Simple sum of the flux
                    flux[order, channel] = np.sum(input_model.data[y-dy:y+dy+1,channel])
                    ### Assuming the errors add in quadrature but can revisit
                    err[order, channel] = np.sqrt(np.sum(input_model.err[y-dy:y+dy+1,channel]**2))
                    ### Assuming there are no covariances so that this is a straight sum
                    rnoise[order, channel] = np.sum(input_model.var_rnoise[y-dy:y+dy+1,channel])
                    poisson[order, channel] = np.sum(input_model.var_poisson[y-dy:y+dy+1,channel])
                    ### Just check if any pixels in the slice were bad or do something more complicated?
                    dq = int(~np.any(input_model.dq[y-dy:y+dy+1,channel] != 0))
            ### Save the results for each fiber to a separate extension
            out_dict[f'SPEC_UNCAL_FIB{i}'] = flux
            out_dict[f'SPEC_UNCAL_ERR_FIB{i}'] = err
            out_dict[f'SPEC_UNCAL_VAR_POISSON_FIB{i}'] = poisson
            out_dict[f'SPEC_UNCAL_VAR_RNOISE{i}'] = rnoise
            out_dict[f'SPEC_DQ_FIB{i}'] = dq
            ### How do we calculate mid time
            # output[f'MID_TIME_FIB{i}']
            ### Are we attaching wavelength solution now?
        return output

    elif method == 'optimal':
        for i in range(1,4):
            ### Get the trace positions for the fiber
            pos = tracepos_model[f'POS_FIB{i}']
            profiles = traceprof_model[f'POF_FIB{i}']
            ### Set up the output arrays
            flux = np.empty(pos.shape)
            err, rnoise, poisson = np.empty(pos.shape), np.empty(pos.shape), np.empty(pos.shape)
            dq = np.empty(pos.shape)
            ### Loop over orders and wavlength channels for the specified fiber
            ### We can definitely find ways to speed up the nested loop
            for channel in range(pos.shape[0]):
                ys = pos[channel,:]
                profs = profiles[channel,:]
                
                ### Now we have a 3x norders free parameter fit

                ### Sort that out into the right orders/fibers


            ### Save the results for each fiber to a separate extension
            out_dict[f'SPEC_UNCAL_FIB{i}'] = flux
            out_dict[f'SPEC_UNCAL_ERR_FIB{i}'] = err
            out_dict[f'SPEC_UNCAL_VAR_POISSON_FIB{i}'] = poisson
            out_dict[f'SPEC_UNCAL_VAR_RNOISE{i}'] = rnoise
            out_dict[f'SPEC_DQ_FIB{i}'] = dq


    ### Write the output model
    model_result = HISPECL1Model(data=out_dict,meta=input_model.meta)

    return model_result

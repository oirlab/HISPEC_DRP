from ..base_primitive import BasePrimitive
from ... import datamodels

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
        ### FIXME if we're going to process a large number of files we don't want to 
        ### load the calibrations from disk every time, it's much more efficient to 
        ### keep them in memory and loop over the science frames
        if self.args.tracepos is None:
            tracepos_path = self.get_calibration_file(input_model, cal_type=)
        else:
            tracepos_path = self.args.flat
        tracepos_model = datamodels.open(tracepos_path, copy=False)

        ### If using box extraction, the traceprof should be empty
        ### FIXME how much error checking are we doing on inputs
        if self.args.method != 'box':
            if self.args.traceprof is None:
                traceprof_path = self.get_calibration_file(input_model, cal_type=)
            else:
                traceprof_path = self.args.traceprof
            traceprof_model = datamodels.open(tracepos_path, copy=False)
        else:
            traceprof_model = None



        result = input_model.copy()
        if method == 'box'
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
        tracepos_model (): Trace center traceposs
        traceprof_model (): traceprof for optimal extraction
        method (str): Type of extraction to perform.
        boxwidth (int): size of box to use for box extraction
    
    Returns:
        HISPECL1Model: The extracted 1D sepectra.
    """

    ### How do we compute average referebce time for each wavelength

    ### Set up the L1 model we write at the end

    ### Basic box extraction for now
    if method == 'box':
        for i in range(1,len)

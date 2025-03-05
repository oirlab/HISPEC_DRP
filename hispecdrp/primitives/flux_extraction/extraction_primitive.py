from ..base_primitive import BasePrimitive
from ... import datamodels

__all__ = ['FluxExtractionPrimitive']


class FluxExtractionPrimitive(BasePrimitive):
    """
    ARGUMENTS:
        locations (datamodels.DataModel | str | None): Trace locations for flux extraction
            if None, the trace locations will be queried
        weights (datamodels.DataModel | str | None): Profile weights to use for flux extraction
            If None, the fiber flat will be queried.
        method (str | 'box'): method to use for flux extraction.
            At present, only box is implemented
        boxwidth (int | 5): width to use for box extraction
    """

    class_alias = 'flat_field'

    ARGUMENTS = {
        'suffix' : 'fluxextracted',
        'locations' : None
        'weights' : None
        method : 'box' 
        boxwidth : 5
    }

    def _perform(self):
        input_model = datamodels.open(self.args.name, copy=False)
        ### FIXME if we're going to process a large number of files we don't want to 
        ### load the calibrations from disk every time, it's much more efficient to 
        ### keep them in memory and loop over the science frames
        if self.args.locations is None:
            location_path = self.get_calibration_file(input_model, cal_type=)
        else:
            location_path = self.args.flat
        location_model = datamodels.open(location_path, copy=False)

        ### If using box extraction, the weights should be empty
        ### FIXME how much error checking are we doing on inputs
        if self.args.method != 'box':
            if self.args.weights is None:
                weights_path = self.get_calibration_file(input_model, cal_type=)
            else:
                weights_path = self.args.weights
            weights_model = datamodels.open(location_path, copy=False)
        else:
            weights_model = None



        result = input_model.copy()
        if method == 'box'
            self.log.info(f"Performing {method} extraction on {input_model} with locations {location_model} and width {boxwidth}")
        else:
            self.log.info(f"Performing {method} extraction on {input_model} with locations {location_model} and weights {weights_mdoel}")
        
        result = flux_extraction(result, location_model, weights_model, method, boxwidth)
        return result
    



def flux_extraction(
        input_model : datamodels.HISPECL0Model,
        location_model : datamodels.HISPEC,
        weights_model : ,
        method : str,
        boxwidth : int
    ) -> datamodels.HISPECL1Model:
    """
    Flux extraction.

    Args:
        input_model (HISPECL0Model): Input level 0 model.
        location_model (): Trace center locations
        weights_model (): Weights for optimal extraction
        method (str): Type of extraction to perform.
        boxwidth (int): size of box to use for box extraction
    
    Returns:
        HISPECL1Model: The extracted 1D sepectra.
    """

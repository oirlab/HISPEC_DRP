from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np

from .fit_ramp_numba import fit_ramps_utr, fit_ramps_mcds

__all__ = ['FitRampPrimitive']


class FitRampPrimitive(BasePrimitive):
    """
    ARGUMENTS:
        method (str): Ramp fitting method. Options are 'cds', 'mcds', 'utr'. Default is UTR.
        num_coadd (int): Number of coadds for MCDS method. Default is 1, equivalent to method=CDS.
    """

    class_alias = 'ramp_fit'

    ARGUMENTS = {
        'suffix' : 'rampfit',
        'method' : 'utr',
        'num_coadd' : 1
    }


    def _perform(self):
        input_model = self.open_model(self.args.name, copy=False)
        self.log.info(f"Fitting ramp to {input_model} with method={self.args.method.lower()}")
        if self.args.method.lower() == 'utr':
            result = fit_ramps_utr(input_model.times.astype(np.float32), input_model.data.astype(np.float32), b=0)
            slopes, slopes_err = result['slope'], result['slope_err']
        elif self.args.method.lower() == 'mcds':
            result = fit_ramps_mcds(input_model.times.astype(np.float32), input_model.data.astype(np.float32), num_coadd=self.num_coadd)
            slopes, slopes_err = result['slope'], result['slope_err']

        # For now, create a simple DQ model for the L0 model
        dq_out = np.zeros(slopes.shape, dtype=np.uint16)

        # Create output model
        model_result = datamodels.HISPECL0Model(data=dict(data=slopes, err_floor=slopes_err, err_all=slopes_err, dq=dq_out), meta=input_model.meta)

        return model_result
    
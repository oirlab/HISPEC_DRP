from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np

from .fit_ramp_numba import fit_ramps_ols, fit_ramps_mcds
from .fit_ramp_jwst import fit_ramps_jwst

__all__ = ['FitRampPrimitive']


class FitRampPrimitive(BasePrimitive):
    """
    ARGUMENTS:
        method (str): Ramp fitting method. Options are 'cds', 'mcds', 'ols', 'likely'. Default is ols.
        num_coadd (int): Number of coadds for MCDS method. Default is 1, equivalent to method='cds'.
        max_cores (int): maximum number of cores to use during ramp fitting.  Default is 1.
    """

    class_alias = 'ramp_fit'

    ARGUMENTS = {
        'suffix' : 'rampfit',
        'method' : 'ols',
        'num_coadd' : 1,
        'max_cores' : 1,
    }


    def _perform(self):
        input_model = datamodels.open(self.args.name, copy=False)
        self.log.info(f"Fitting ramp to {input_model} with method={self.args.method.lower()}")

        var_poisson = None
        var_rnoise = None
        dq_out = None

        if self.args.method.lower() == 'ols':
            result = fit_ramps_ols(input_model.times.astype(np.float32), input_model.data.astype(np.float32), b=0)
            slopes, slopes_err = result['slope'], result['slope_err']
        elif self.args.method.lower() in ('mcds', 'cds'):
            result = fit_ramps_mcds(input_model.times.astype(np.float32), input_model.data.astype(np.float32), num_coadd=self.args.num_coadd)
            slopes, slopes_err = result['slope'], result['slope_err']
        elif self.args.method.lower() in ('jwst_likely', 'jwst_ols'):
            result = fit_ramps_jwst(input_model.times.astype(np.float32), input_model.data.astype(np.float32), max_cores=self.args.max_cores, algorithm=self.args.method.lower())
            slopes, slopes_err = result['slope'], result['slope_err']
            # Pass Poisson, read noise, dq arrays out
            var_poisson = result['var_poisson']
            var_rnoise = result['var_rnoise']
            dq_out = result['dq']
        else:
            self.log.error(f"Invalid ramp fit method {self.args.method.lower()}")
            raise ValueError(f"Invalid ramp fit method {self.args.method.lower()}")

        # Calculate variances
        if var_poisson is None:
            var_poisson = np.sqrt(np.abs(slopes))
        if var_rnoise is None:
            var_rnoise = np.zeros(slopes.shape, dtype=np.float32) # TODO: Calculate effective read noise from fit (?)

        # For now, create a simple DQ model for the L0 model
        # Updated to 32-bit to follow data model - CDK
        if dq_out is None:
            dq_out = np.zeros(slopes.shape, dtype=np.uint32)

        # Create output model
        model_result = datamodels.HISPECL0Model(
            data=dict(
                data=slopes,
                err=slopes_err,
                var_poisson=var_poisson,
                var_rnoise=var_rnoise,
                dq=dq_out
            ),
            meta=input_model.meta
        )

        return model_result
    

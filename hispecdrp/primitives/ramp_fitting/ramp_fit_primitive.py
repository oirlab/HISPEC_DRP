from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np
from ...calibrations import DetectorGainSelector,ReadNoiseSelector #,DarkSelector

from .fit_ramp_numba import fit_ramps_ols, fit_ramps_mcds
from .fit_ramp_jwst import fit_ramps_jwst
from ...datamodels.dqflags import DQ_FLAGS

from numba import get_num_threads, set_num_threads

__all__ = ['FitRampPrimitive']


class FitRampPrimitive(BasePrimitive):

    ARGUMENTS = {
        'suffix' : 'rampfit',
        'method' : 'jwst_ols',
        'mcds_num_coadd' : None,
        'max_cores' : 1,
        'output_model' : 'L0Model',
    }

    CALIBRATIONS = {
        'detgain': {
            'selector': DetectorGainSelector,
            'selector_kwargs': {}
        },
        # 'dark': { # would need to be defined if we implement dark subtraction before ramp fitting.
        #     'selector': DarkSelector,
        #     'selector_kwargs': {}
        # },
        'rn': {
            'selector': ReadNoiseSelector,
            'selector_kwargs': {}
        }
    }
    def _perform(self):
        set_num_threads(self.max_cores)

        input_model = datamodels.open(self.input, copy=False)
        self.log.info(f"Fitting ramp to {input_model} with method={self.method.lower()}")

        detgain_cal, _ = self.get_calibration(input_model, 'detgain')
        detgain_model = datamodels.open(detgain_cal)
        # Dark would need to be defined if we implement dark subtraction before ramp fitting.
        # dark_cal, _ = self.get_calibration(input_model, 'dark')
        # dark_model = datamodels.open(dark_cal)
        rn_cal, _ = self.get_calibration(input_model, 'rn')
        rn_model = datamodels.open(rn_cal)

        if self.mcds_num_coadd is None:
            n_read = input_model.data.shape[0]
            _mcds_num_coadd = n_read // 2
        else:
            _mcds_num_coadd = self.mcds_num_coadd

        delta_readtimes = input_model.get_delta_readtimes()
        first_read_mjd = input_model.get_first_read_mjd()

        if self.method.lower() == 'ols':
            result = fit_ramps_ols(delta_readtimes,
                                    input_model.data,
                                    dq_raw=input_model.dq_raw,
                                    dq=input_model.dq,
                                    rdnoise=rn_model.rn,
                                    gain=detgain_model.gain,
                                    start_times=first_read_mjd,
                                    dark_current=None,
                                    max_cores=self.max_cores) # Not used because we assume dark subtraction is done in rate maps.
        elif self.method.lower() in ('cds'):
            result = fit_ramps_mcds(delta_readtimes,
                                    input_model.data,
                                    dq_raw=input_model.dq_raw,
                                    dq=input_model.dq,
                                    rdnoise=rn_model.rn,
                                    gain=detgain_model.gain,
                                    start_times=first_read_mjd,
                                    dark_current=None, # Not used because we assume dark subtraction is done in rate maps.
                                    num_coadd=1,
                                    max_cores=self.max_cores)
        elif self.method.lower() in ('mcds'):
            result = fit_ramps_mcds(delta_readtimes,
                                    input_model.data,
                                    dq_raw=input_model.dq_raw,
                                    dq=input_model.dq,
                                    rdnoise=rn_model.rn,
                                    gain=detgain_model.gain,
                                    start_times=first_read_mjd,
                                    dark_current=None, # Not used because we assume dark subtraction is done in rate maps.
                                    num_coadd=_mcds_num_coadd,
                                    max_cores=self.max_cores)
        elif self.method.lower() in ('jwst_likely', 'jwst_ols'):
            result = fit_ramps_jwst(delta_readtimes, input_model.data.astype(np.float32),
                                    dq_raw=input_model.dq_raw,
                                    dq=input_model.dq,
                                    rdnoise=rn_model.rn,
                                    gain=detgain_model.gain,
                                    start_times=first_read_mjd,
                                    dark_current=None,
                                    max_cores=self.max_cores, algorithm=self.method.lower())
        else:
            self.log.error(f"Invalid ramp fit method {self.method.lower()}")
            raise ValueError(f"Invalid ramp fit method {self.method.lower()}")

        output_model = self.get_output_model(input_model)
        output_model.rate = result['rate']
        output_model.rate_err = result['rate_err']
        output_model.rate_var_poisson = result['rate_var_poisson']
        output_model.rate_var_rnoise = result['rate_var_rnoise']
        output_model.dq = result['DQ']
        output_model.mid_time = result['mid_time']

        return dict(output=output_model, method=self.method)
    

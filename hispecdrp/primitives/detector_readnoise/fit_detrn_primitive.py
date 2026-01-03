from ..base_primitive import BasePrimitive
from ... import datamodels
from ...calibrations import DetectorGainSelector,ReadNoiseSelector
import numpy as np
from typing import Sequence
from ...datamodels.dqflags import DQ_FLAGS
from ..detector_gain.fit_detgain_primitive import fit_gain_and_readnoise


__all__ = ['FitDetectorReadnoisePrimitive']

SATURATED_FLAG_uint8 =  np.uint8(DQ_FLAGS["SATURATED"])
DO_NOT_USE_FLAG_uint8 = np.uint8(DQ_FLAGS["DO_NOT_USE"])
JUMP_DET_FLAG_uint8 = np.uint8(DQ_FLAGS["JUMP_DET"])

class FitDetectorReadnoisePrimitive(BasePrimitive):

    ARGUMENTS = {
        'suffix' : 'detrn',
        'max_cores' : 1,
        'init_RN' : None,  # None | in,  Define the initial readnoise value used for the maximum likelihood grid search.
        # If None, use calibration database, otherwise specify a np.array (eg, np.linspace(1, 20, 101)).
        'mode': 'per_pixel', # str, 'per_pixel' | 'channel' | 'full_detector', mode to fit the parameters.
        # 'per_pixel' compute readnoise for each pixel independently,
        # 'channel' (Not implemented yet) compute readnoise for each readout channel independently,
        # 'full_detector' compute readnoise for the full detector at once. This can be used when a single exposure is available.
        'method': 'quad',# str, 'quad' | 'grid' | 'both', method to fit the parameters.
        # 'quad' uses quadratic approximation,
        # 'grid' uses maximum likelihood grid search.
        # 'both' start with grid search and then apply quadratic approximation.
        'init_delta_RNs' : None,  # np.array, the grid sampling is defined as init_RN + init_delta_RNs for the maximum likelihood grid search.
        # specify a np.array (eg, np.linspace(1, 20, 101)).
        'quad_delta_RN' : 0.5, # float, used only in 'quad' mode, defines the initial delta read noise around init_RN for the quadratic approximation.
        'countratesidentical': False, # bool, if True, assume all ramps have identical count rates.
    }

    CALIBRATIONS = {
        'detgain': {
            'selector': DetectorGainSelector,
            'selector_kwargs': {}
        },
        'rn_init': {
            'selector': ReadNoiseSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):
        if self.mode.lower() != 'full_detector' and self.mode.lower() != 'per_pixel':
            raise NotImplementedError(
                f"Only 'full_detector' and 'per_pixel' modes are implemented currently. Not {self.mode}")

        if isinstance(self.input, (datamodels.DataModel, str)):
            _input_list_tmp = [self.input]
        if isinstance(self.input, Sequence):
            _input_list_tmp = self.input
        nfiles = len(_input_list_tmp)
        self.log.info(f"Deriving the detector readnoise from a list of {nfiles} inputs.")

        input_models = [datamodels.open(f) for f in _input_list_tmp]
        datamodel0 = input_models[0]
        nreads, ny, nx = datamodel0.data.shape

        detgain, _ = self.get_calibration(datamodel0, 'detgain')
        detgain_model = datamodels.open(detgain)
        detgain_arr = detgain_model.gain

        if self.init_RN is None:
            rn_cal_init, _ = self.get_calibration(datamodel0, 'rn_init')
            rn_model_init = datamodels.open(rn_cal_init)
            _init_center_RN = rn_model_init.RN
            self.log.info(f"Using {rn_cal_init} as initial read noise values with median value {np.nanmedian(_init_center_RN)} DN.")
        elif isinstance(self.init_RN, float):
            _init_center_RN = self.init_RN
            self.log.info(f"Using user-defined initial read noise values: {self.init_RN}.")

        delta_readtimes = datamodel0.get_delta_readtimes()

        if self.method == 'quad':
            _init_delta_RNs = np.array([-self.quad_delta_RN,0,+self.quad_delta_RN])
        elif self.method == 'grid' or self.method == 'both':
            if self.init_delta_RNs is None:
                _init_delta_RNs = np.linspace(-10,10, 51)
            else:
                _init_delta_RNs = self.init_delta_RNs

        if self.mode.lower() == 'full_detector':
            self.log.info(
                f"Running in full_detector mode. All detector pixels will be fitted jointly and a single gain and/or readnoise value will be returned for the entire chip.")
        elif self.mode.lower() == 'per_pixel':
            self.log.info(
                f"Running in single_pixel mode. Each detector pixel will be fitted independently and a map of gain and/or readnoise will be returned.")

        # Define output model from inputs
        output_detrn = self.get_output_model(input_models, "ReadNoiseModel")

        # Make all ramps from all files into a big cube
        ramp_arr = []
        dq_raw_arr = []
        for opened_input in input_models:
            if nreads != opened_input.data.shape[0]:
                raise ValueError("All input ramps must have the same number of reads.")
            ramp_arr.append(opened_input.data)
            dq_raw_arr.append(opened_input.dq_raw)
        ramp_arr = np.array(ramp_arr)
        dq_raw_arr = np.array(dq_raw_arr)

        _out = fit_gain_and_readnoise(delta_readtimes, ramp_arr, dq_raw_arr, detgain_arr, _init_center_RN,None,_init_delta_RNs,
                                      countratesidentical=self.countratesidentical,
                                      mode = self.mode, method= self.method, max_cores =self.max_cores)
        _, bestfit_rn, cov = _out

        # Output
        output_detrn.rn = bestfit_rn
        output_detrn.rn_err = np.sqrt(cov[1,1])
        output_detrn.dq = np.zeros(bestfit_rn.shape, dtype=np.uint32)

        return dict(output=output_detrn)
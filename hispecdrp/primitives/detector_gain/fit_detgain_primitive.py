from . import fit_gain_Brandt
from ..ramp_fitting import fitramp_Brandt
from ..base_primitive import BasePrimitive
from ... import datamodels
from ...calibrations import DetectorGainSelector,ReadNoiseSelector
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence
from ...utils.math import get_err_from_posterior
from ...datamodels.dqflags import DQ_FLAGS

import multiprocessing as mp
import ctypes
from ...utils.parallelization_utils import _arraytonumpy, numba_thread_scope
from multiprocessing.sharedctypes import RawArray
from tqdm import tqdm


__all__ = ['FitDetectorGainPrimitive']

SATURATED_FLAG_uint8 =  np.uint8(DQ_FLAGS["SATURATED"])
DO_NOT_USE_FLAG_uint8 = np.uint8(DQ_FLAGS["DO_NOT_USE"])
JUMP_DET_FLAG_uint8 = np.uint8(DQ_FLAGS["JUMP_DET"])

class FitDetectorGainPrimitive(BasePrimitive):

    ARGUMENTS = {
        'suffix' : 'detgain',
        'max_cores' : 1,
        'init_gain' : None, # None | int, Define the initial gain value used for the maximum likelihood grid search.
        # If None, use calibration database, otherwise specify a np.array (eg, np.linspace(0.1, 10, 101)).
        'init_RN' : None,  # None | in,  Define the initial readnoise value used for the maximum likelihood grid search.
        # If None, use calibration database, otherwise specify a np.array (eg, np.linspace(1, 20, 101)).
        'mode': 'per_pixel', # str, 'per_pixel' | 'channel' | 'full_detector', mode to fit the parameters.
        # 'per_pixel' compute gain for each pixel independently,
        # 'channel' compute gain for each readout channel independently,
        # 'full_detector' compute gain for the full detector at once. This can be used when a single exposure is available.
        'method': 'quad',# str, 'quad' | 'grid' | 'both', method to fit the parameters.
        # 'quad' uses quadratic approximation,
        # 'grid' uses maximum likelihood grid search.
        # 'both' start with grid search and then apply quadratic approximation.
        'init_delta_gains' : None, # np.array, the grid sampling is defined as init_gain + init_delta_gains for the maximum likelihood grid search.
        # eg, np.linspace(0.1, 10, 101))
        'init_delta_RNs' : None,  # np.array, the grid sampling is defined as init_RN + init_delta_RNs for the maximum likelihood grid search.
        # specify a np.array (eg, np.linspace(1, 20, 101)).
        'quad_delta_gain' : 0.2, # float, used only in 'quad' mode, defines the initial delta gain around init_gain for the quadratic approximation.
        'quad_delta_RN' : 0.5, # float, used only in 'quad' mode, defines the initial delta read noise around init_RN for the quadratic approximation.
        'fix_RN' : False, # bool, if True, fix the read noise to init_RN value and only fit for gain.
        'countratesidentical': False, # bool, if True, assume all ramps have identical count rates.
    }

    CALIBRATIONS = {
        'detgain_init': {
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
            raise NotImplementedError(f"Only 'full_detector' and 'per_pixel' modes are implemented currently. Not {self.mode}")

        if isinstance(self.input, (datamodels.DataModel, str)):
            _input_list_tmp = [self.input]
        if  isinstance(self.input, Sequence):
            _input_list_tmp = self.input


        # Make sure to open all input models in case some files are passed as strings
        _input_list = []
        for _input in _input_list_tmp:
            _input_list.append(datamodels.open(_input))
        nreads,ny,nx = _input_list[0].data.shape
        nfiles = len(_input_list_tmp)
        self.log.info(f"Deriving the detector gain from a list of {len(_input_list)} inputs.")

        if self.init_gain is None:
            detgain_init, _ = self.get_calibration(_input_list[0], 'detgain_init')
            detgain_init_model = datamodels.open(detgain_init)
            detgain_init_arr = detgain_init_model.gain
            self.log.info(f"Using {detgain_init} as initial detector gain values with median value {np.nanmedian(detgain_init_arr)}.")
        elif isinstance(self.init_gain, float):
            detgain_init_arr = np.full((ny,nx),self.init_gain)
            self.log.info(f"Using user-defined initial detector gain values: {self.init_gain}.")
        if self.init_RN is None:
            rn_cal_init, _ = self.get_calibration(_input_list[0], 'rn_init')
            rn_model_init = datamodels.open(rn_cal_init)
            rn_init_arr = rn_model_init.RN
            self.log.info(f"Using {rn_cal_init} as initial read noise values with median value {np.nanmedian(rn_init_arr)} DN.")
        elif isinstance(self.init_RN, float):
            rn_init_arr = np.full((ny,nx),self.init_RN)
            self.log.info(f"Using user-defined initial read noise values: {self.init_RN}.")


        delta_readtimes = _input_list[0].get_delta_readtimes()


        # if delta_readtimes[1] - delta_readtimes[0] != 1:
        #     raise ValueError("Delta readtimes must be unity currently until the bug is fixed in Tim's code?")


        center_gain = detgain_init_arr[0, 0]  # single pixel for now
        center_rn = rn_init_arr[0, 0]  # single pixel for now

        if self.method == 'quad':
            gains_init = center_gain + np.array([-self.quad_delta_gain,0,+self.quad_delta_gain])
            if self.fix_RN:
                rns_init = np.array([center_rn])
            else:
                rns_init = center_rn + np.array([-self.quad_delta_RN,0,+self.quad_delta_RN])
        elif self.method == 'grid' or self.method == 'both':
            if self.init_delta_gains is None:
                self.init_delta_gains = np.linspace(0.1*center_gain, 10*center_gain, 101)
            if self.init_delta_RNs is None:
                self.init_delta_RNs = np.linspace(0.1*center_rn, 10*center_rn, 101)
            gains_init = center_gain + self.init_delta_gains
            if self.fix_RN:
                rns_init = np.array([center_rn])
            else:
                rns_init = center_rn + self.init_delta_RNs
            gains_init = gains_init[gains_init > 0]
            rns_init = rns_init[rns_init > 0]


        if self.mode.lower() == 'full_detector':
            self.log.info(f"Running in full_detector mode. All detector pixels will be fitted jointly and a single gain and/or readnoise value will be returned for the entire chip.")
        elif self.mode.lower() == 'per_pixel':
            self.log.info(f"Running in single_pixel mode. Each detector pixel will be fitted independently and a map of gain and/or readnoise will be returned.")
            
        # Define output model from inputs
        output_detgain = self.get_output_model(_input_list,"DetectorGainModel")


        ramp_arr = []
        dq_raw_arr = []
        for _input in _input_list:
            if nreads != _input.data.shape[0]:
                raise ValueError("All input ramps must have the same number of reads.")
            ramp_arr.append(_input.data)
            dq_raw_arr.append(_input.dq_raw)
        ramp_arr = np.array(ramp_arr)
        dq_raw_arr = np.array(dq_raw_arr)

        _out = fit_gain_and_readnoise(delta_readtimes, ramp_arr, dq_raw_arr, gains_init, rns_init,
                                      countratesidentical=self.countratesidentical,
                                      mode = self.mode, method= self.method, max_cores =self.max_cores)
        bestfit_gain, bestfit_rn, cov = _out

        # Output
        output_detgain.gain = bestfit_gain
        output_detgain.rn = bestfit_rn
        output_detgain.cov = cov
        output_detgain.dq = np.zeros(bestfit_gain.shape, dtype=np.uint32)

        return dict(output=output_detgain)

def fit_gain_and_readnoise(
    readtimes : np.ndarray, ramps : np.ndarray, dq_raw : np.ndarray,
    gains_init : np.ndarray, rns_init : np.ndarray,
    countratesidentical: bool = False, mode : str = "per_pixel", method: str = 'quad',
    max_cores : int = 1,
):
    """
    Fit for the detector gain and/or read noise.

    Parameters
    ----------
    readtimes : np.ndarray
        1D array of read times.
    ramps : np.ndarray
        2D array of input rates. Size n_files x nreads x ny x nx.
    dq_raw : np.ndarray
        2D array of input UTR data quality flags. Size n_files x nreads x ny x nx.
    gains_init : np.ndarray
        Initial guess for gain (e-/ADU). If 1D array, shoudl be three elements (left, middle, right).
        If singleton, only fit for read noise while fixing gain to this value.
    rns_init : float
        Initial guess for read noise (in DN). Should be a 1D array with three elements (left, middle, right).
        If singleton, only fit for gain while fixing read noise to this value.
    countratesidentical: bool
        If True, assume all ramps have identical count rates.
    mode : str
        'full_detector' to fit a single gain and read noise for the entire detector,
        'per_pixel' to fit gain and read noise for each pixel independently.
    method : str
        Method to use for fitting. Options are 'quad' for quadratic approximation, 'grid' for maximum likelihood grid search, 'both' for grid search followed by quadratic approximation.
    max_cores : int, optional
        Number of cores for parallelization. Default is 1 (no parallelization).
        Not used for mode == 'full_detector'.

    Returns
    -------
    gain : np.ndarray
        Best fit Gain (e-/ADU).
        Singleton if mode is 'full_detector', 2D array if mode is 'per_pixel'.
    rn : np.ndarray
        Best fit read noise (in DN).
        Singleton if mode is 'full_detector', 2D array if mode is 'per_pixel'.
    cov : np.ndarray
        Covariance matrix of the fit.
        2D matrix if mode is 'full_detector', 4D array if mode is 'per_pixel'.
    """
    if mode.lower() == 'full_detector':
        C = fitramp_Brandt.Covar(readtimes)

        if method == 'quad':
            _out = fit_single_gain_readnoise_quad_approx(readtimes, ramps, dq_raw, C,
                                                             gains_init, rns_init, do_once=False,
                                                             countratesidentical=countratesidentical)
        elif method == 'grid' or method == 'both':
            if self.method == 'grid':
                error_computation = 'posterior'
            elif self.method == 'both':
                error_computation = 'quad'
            _out = fit_single_gain_readnoise(readtimes, ramps, dq_raw, C, gains_init,
                                                 rns_init, error_computation=error_computation,
                                                 countratesidentical=countratesidentical)

        bestfit_gain, bestfit_rn, cov = np.array([_out[0]]), np.array([_out[1]]), _out[2]

    elif mode.lower() == 'per_pixel':
        _out = fit_gain_and_readnoise_per_pixel(readtimes, ramps, dq_raw,
                                                gains_init, rns_init,
                                                countratesidentical=countratesidentical,
                                                method= method,max_cores =max_cores)
        bestfit_gain, bestfit_rn, cov = _out

    return bestfit_gain, bestfit_rn, cov


def fit_gain_and_readnoise_per_pixel(
    readtimes : np.ndarray, ramps : np.ndarray, dq_raw : np.ndarray,
    gains_init : np.ndarray, rns_init : np.ndarray,
    countratesidentical: bool = False,  method: str = 'quad',
    max_cores : int = 1,
):
    """
    Fit the gain and read noise for each pixel on the detector using either quadratic approximation or maximum likelihood grid search.

    Parameters
    ----------
    readtimes : np.ndarray
        1D array of read times.
    ramps : np.ndarray
        2D array of input rates. Size n_files x nreads x ny x nx.
    dq_raw : np.ndarray
        2D array of input UTR data quality flags. Size n_files x nreads x ny x nx.
    gains_init : np.ndarray
        Initial guess for gain (e-/ADU). If 1D array, shoudl be three elements (left, middle, right).
        If singleton, only fit for read noise while fixing gain to this value.
    rns_init : float
        Initial guess for read noise (in DN). Should be a 1D array with three elements (left, middle, right).
        If singleton, only fit for gain while fixing read noise to this value.
    countratesidentical: bool
        If True, assume all ramps have identical count rates.
    method : str
        Method to use for fitting. Options are 'quad' for quadratic approximation, 'grid' for maximum likelihood grid search, 'both' for grid search followed by quadratic approximation.
    max_cores : int, optional
        Number of cores for parallelization. Default is 1 (no parallelization).

    Returns
    -------
    gain : float
        Best fit Gain (e-/ADU)
    rn : float
        Best fit read noise (in DN)
    cov : np.ndarray
        Covariance matrix of the fit.
    """
    n_files, n_reads, ny, nx = ramps.shape

    if ramps.dtype is np.uint16:
        mp_data_type = ctypes.c_ushort
    else:
        mp_data_type = ctypes.c_float
    mp_float_type = ctypes.c_float
    mp_dq_raw_type = ctypes.c_uint8
    mp_dq_type = ctypes.c_uint
    types_tuple = (mp_data_type, mp_float_type, mp_dq_raw_type, mp_dq_type)

    gain_mp = RawArray(mp_float_type, nx * ny)
    gain_shape = (ny, nx)
    gain_np = _arraytonumpy(gain_mp, gain_shape, dtype=mp_float_type)
    gain_np[:] = np.zeros(gain_shape, dtype=np.float32)

    readnoise_mp = RawArray(mp_float_type, nx * ny)
    readnoise_shape = (ny, nx)
    readnoise_np = _arraytonumpy(readnoise_mp, readnoise_shape, dtype=mp_float_type)
    readnoise_np[:] = np.zeros(readnoise_shape, dtype=np.float32)

    cov_mp = RawArray(mp_float_type, 2 * 2 * nx * ny)
    cov_shape = (2,2,ny, nx)
    cov_np = _arraytonumpy(cov_mp, cov_shape, dtype=mp_float_type)
    cov_np[:] = np.zeros(cov_shape, dtype=np.float32)

    if max_cores > 1:
        ramps_mp = RawArray(mp_data_type, nx * ny * n_reads * n_files)
        ramps_shape = (n_files,n_reads, ny, nx)
        ramps_np = _arraytonumpy(ramps_mp, ramps_shape, dtype=mp_data_type)
        ramps_np[:] = ramps  # copy original array into shared array
        ramps = ramps_np  # free the original array from memory to avoid duplicates

        dq_raw_mp = RawArray(mp_dq_raw_type, nx * ny * n_reads * n_files)
        dq_raw_shape = (n_files,n_reads, ny, nx)
        dq_raw_np = _arraytonumpy(dq_raw_mp, dq_raw_shape, dtype=mp_dq_raw_type)
        if dq_raw is None:
            dq_raw_np[:] = np.zeros(dq_raw_np, dtype=np.uint8)
        else:
            dq_raw_np[:] = dq_raw  # copy original array into shared array
            dq_raw = dq_raw_np  # free the original array from memory to avoid duplicates


        # Prepare blocks
        block_size = max(1, ny // (4*max_cores)) if max_cores > 1 else ny

        _init_args = (
            ramps_mp, ramps_shape,
            dq_raw_mp, dq_raw_shape,
            gain_mp, gain_shape,
            readnoise_mp, readnoise_shape,
            cov_mp, cov_shape,
        )

        ctx = mp.get_context("spawn")  # avoids unsafe fork after OpenMP init
        tpool = ctx.Pool(
            processes=max_cores,
            initializer=_tpool_init_fit_gain_readnoise,
            initargs=_init_args,
            maxtasksperchild=50
        )

        args_list = []
        for block_start in range(0, ny, block_size):
            block_end = min(block_start + block_size, ny)

            args = (block_start, block_end, readtimes,gains_init,rns_init, countratesidentical, method, types_tuple)
            args_list.append(args)

        try:
            with numba_thread_scope(max_cores):
                tasks = [
                    tpool.apply_async(_fit_gain_and_readnoise_worker, args=(args_tuple,))
                    for args_tuple in args_list
                ]

            for t in tqdm(tasks, desc="Processing blocks"):
                t.wait()
        finally:
            tpool.close()
            tpool.join()
    else:
        # If not parallelizing, ie max_cores=1, just run directly without shared memory
        results = []

        for row_id in np.arange(ny):
            print(f"Fitting detector gain and/or readnoise: Processing row {row_id + 1} of {ny}.")
            args = (
                row_id,
                readtimes,
                ramps[:, :, [row_id], :],
                dq_raw[:, :, [row_id], :],
                gains_init,
                rns_init,
                countratesidentical,
                method
            )

            results.append(_fit_gain_and_readnoise_row(args))

        for row_element in results:
            row_id, out_bestfit_gain, out_bestfit_rn, out_cov = row_element

            gain_np[row_id, :] = out_bestfit_gain
            readnoise_np[row_id, :] = out_bestfit_rn
            cov_np[:,:,row_id, :] = out_cov


    return gain_np, readnoise_np, cov_np


def _tpool_init_fit_gain_readnoise(
    data : np.ndarray, data_shape : tuple,
    dq_raw : np.ndarray, dq_raw_shape : tuple,
    gain : np.ndarray, gain_shape : tuple,
    readnoise : np.ndarray, readnoise_shape : tuple,
    cov : np.ndarray, cov_shape : tuple,
):
    """
    Initialize shared global variables for the multiprocessing pool.
    """

    global shared_data, shared_data_shape, \
        shared_dq_raw, shared_dq_raw_shape, \
        shared_gain, shared_gain_shape, \
        shared_readnoise, shared_readnoise_shape, \
        shared_cov, shared_cov_shape

    shared_data = data
    shared_data_shape = data_shape

    shared_dq_raw = dq_raw
    shared_dq_raw_shape = dq_raw_shape

    shared_gain = gain
    shared_gain_shape = gain_shape

    shared_readnoise = readnoise
    shared_readnoise_shape = readnoise_shape

    shared_cov = cov
    shared_cov_shape = cov_shape

def _fit_gain_and_readnoise_row(indata : tuple) -> tuple:
    """
    Helper function for parallelization fitting the gain and readnoise per detector row.
    """
    row_id, delta_readtimes, ramp_row, dq_raw_row, gains_init, rns_init, countratesidentical,method  = indata
    n_files, n_reads, _, nx = ramp_row.shape
    # print(row_id, ramp_row.shape, ramp_row[0,:,0,0])

    C = fitramp_Brandt.Covar(delta_readtimes)

    if method == 'grid':
        error_computation = 'posterior'
    elif method == 'both':
        error_computation = 'quad'

    out_bestfit_gain = np.zeros((nx,), dtype=np.float32)
    out_bestfit_rn = np.zeros((nx,), dtype=np.float32)
    out_cov = np.zeros((2,2,nx), dtype=np.float32)

    for i in range(nx):
        if method == 'quad':
            _out = fit_single_gain_readnoise_quad_approx(delta_readtimes, ramp_row[:,:,0,i], dq_raw_row[:,:,0,i], C,
                                                         gains_init, rns_init, do_once=False,
                                                         countratesidentical=countratesidentical)
        elif method == 'grid' or method == 'both':
            _out = fit_single_gain_readnoise(delta_readtimes, ramp_row[:,:,0,i], dq_raw_row[:,:,0,i], C, gains_init,
                                             rns_init, error_computation=error_computation,
                                             countratesidentical=countratesidentical)

        bestfit_gain, bestfit_rn, cov = _out

        out_bestfit_gain[i] = bestfit_gain
        out_bestfit_rn[i] = bestfit_rn
        out_cov[:,:,i] = cov

    return (row_id, out_bestfit_gain, out_bestfit_rn, out_cov)

def _fit_gain_and_readnoise_worker(args : tuple) -> tuple:
    """
    Worker function for multiprocessing. Runs _fit_gain_and_readnoise_row on a block of rows using shared arrays.

    Parameters
    ----------
    args : tuple
        Tuple containing: (block_start, block_end, delta_readtimes,gains_init,rns_init, countratesidentical, method, types_tuple)

    Returns
    -------
    block_start : int
        Starting index of the block of rows processed.
    block_end : int
        Ending index of the block of rows processed.
    """
    block_start, block_end, delta_readtimes,gains_init,rns_init, countratesidentical, method, types_tuple = args
    mp_data_type, mp_float_type, mp_dq_raw_type, mp_dq_type = types_tuple

    ramps_np = _arraytonumpy(shared_data, shared_data_shape, dtype=mp_data_type)
    dq_raw_np = _arraytonumpy(shared_dq_raw, shared_dq_raw_shape, dtype=mp_dq_raw_type)

    gain_np = _arraytonumpy(shared_gain, shared_gain_shape, dtype=mp_float_type)
    readnoise_np = _arraytonumpy(shared_readnoise, shared_readnoise_shape, dtype=mp_float_type)
    cov_np = _arraytonumpy(shared_cov, shared_cov_shape, dtype=mp_float_type)

    results = []
    for row_id in range(block_start, block_end):
        args_row = (
            row_id,
            delta_readtimes,
            ramps_np[:,:, [row_id], :],
            dq_raw_np[:,:, [row_id], :],
            gains_init,
            rns_init,
            countratesidentical,
            method

        )
        results.append(_fit_gain_and_readnoise_row(args_row))

    for row_element in results:
        row_id, out_bestfit_gain, out_bestfit_rn, out_cov = row_element

        gain_np[row_id, :] = out_bestfit_gain
        readnoise_np[row_id, :] = out_bestfit_rn
        cov_np[:,:,row_id, :] = out_cov

    return (block_start, block_end)

def fit_single_gain_readnoise_quad_approx(
    readtimes : np.ndarray, ramps : np.ndarray, dq_raw : np.ndarray, covar : fitramp_Brandt.Covar,
    gains_init : np.ndarray, rns_init : np.ndarray, do_once : bool = False,
        countratesidentical: bool = False
):
    """
    Fit the gain and read noise for each pixel on the detector using a quadratic approximation.
    The fit is actually performed twice to refine the best guess.
    The first iteration uses gains_init and rns_init.
    The second iteration recenters gains_init and rns_init around the best fit from the first iteration and use the covariance for scaling.

    Parameters
    ----------
    readtimes : np.ndarray
        1D array of read times.
    ramps : np.ndarray
        2D array of input rates. Size nramps x nreads (x ny x nx). 
        Extra y and x dimensions are flattened with nramps if they exist.
    dq_raw : np.ndarray
        2D array of input UTR data quality flags. Size nramps x nreads (x ny x nx).
        Extra y and x dimensions are flattened with nramps if they exist.
    covar : fitramp_Brandt.Covar class
           Should be appropriate for diffs_lin
    gains_init : np.ndarray
        Initial guess for gain (e-/ADU). If 1D array, shoudl be three elements (left, middle, right).
        If singleton, only fit for read noise while fixing gain to this value.
    rns_init : float
        Initial guess for read noise (in DN). Should be a 1D array with three elements (left, middle, right).
        If singleton, only fit for gain while fixing read noise to this value.
    do_once : bool
        If True, perform the fit only once (no refinement).
    countratesidentical: bool
        If True, assume all ramps have identical count rates.

    Returns
    -------
    gain : float
        Best fit Gain (e-/ADU)
    rn : float
        Best fit read noise (in DN)
    cov : np.ndarray
        Covariance matrix of the fit.
    """
    nreads = readtimes.shape[0]
    _ramps = np.array(ramps).swapaxes(0, 1) # n_reads, n_files, ny, nx
    _dq_raw = np.array(dq_raw).swapaxes(0, 1) # n_reads, n_files, ny, nx
    _ramps = _ramps.reshape(nreads, -1)  # shape (n_reads, nfiles*ny*nx)
    _dq_raw = _dq_raw.reshape(nreads, -1)  # shape (n_reads, nfiles*ny*nx)
    
    y0 = 0
    diff_times = np.diff(readtimes)[:, None]
    diffs_lin = np.diff(_ramps, axis=0).astype(np.float64)/diff_times
    diffs_quad = np.diff((_ramps - y0) ** 2, axis=0).astype(np.float64)/diff_times

    n_reads,n_ramps = _dq_raw.shape
    mask = ((_dq_raw & DO_NOT_USE_FLAG_uint8) | (_dq_raw & SATURATED_FLAG_uint8)).astype(np.bool)
    mask2 = (_dq_raw & JUMP_DET_FLAG_uint8).astype(np.bool)
    diffs2use = ~(mask[1:n_reads] | mask[0:n_reads - 1] | mask2[1:n_reads])
    # diffs2use = np.full((n_reads-1,n_ramps),1,dtype=np.uint8)
    # diffs2use[5,:] = 0
    # diffs2use[10,:] = 0
    # diffs2use[20,:] = 0
    diffs2use = diffs2use.astype(np.uint8)

    # The quadratic form fit, followed by the maximum value that the quadratic
    # form takes and the maximum value seen on the dense grid.  Run the
    # quadratric form fit twice to make sure we get very close to the maximum
    # of the likelihood.
    if do_once:
        n_iter = 1
    else:
        n_iter = 2
    for i in range(n_iter):
        # three points each in read noise and gain to fit a quadratic form
        if i == 0:
            gains_short = gains_init.astype(np.float64)
            sigs_short = rns_init.astype(np.float64)
        else:
            if len(gains_short) > 1:
                bestfit_gain_err = np.sqrt(cov[0, 0])
                gains_short = [bestfit_gain- bestfit_gain_err,bestfit_gain, bestfit_gain + bestfit_gain_err]
            if len(sigs_short) > 1:
                bestfit_rn_err = np.sqrt(cov[1,1])
                sigs_short = [bestfit_rn- bestfit_rn_err,bestfit_rn, bestfit_rn + bestfit_rn_err]

        bestfit_gain, bestfit_rn, cov = fitquad_gain_rn(diffs_lin, diffs_quad, covar,sigs_short, gains_short,
                                                        d2use=diffs2use,countratesidentical=countratesidentical)

    return bestfit_gain, bestfit_rn, cov


def fitquad_gain_rn(
    diffs_lin,
    diffs_quad,
    covar,
    sigs_short,
    gains_short,
    d2use=None,
    countratesidentical=False,
):
    """
    Estimate best-fit (gain, read-noise) and an approximate covariance matrix.

    Logic:
      - If both grids have >1 point: call fit_gain_Brandt.fitquad(...)
      - If only sigs_short has 1 point: compute logL vs gain, fit a quadratic,
        take maximum at -b/(2a), and estimate sigma_gain ~ sqrt(-1/(2a)).
      - If only gains_short has 1 point: compute logL vs rn, fit a quadratic,
        take maximum at -b/(2a), and estimate sigma_rn ~ sqrt(-1/(2a)).

    Returns
    -------
    bestfit_gain : float
    bestfit_rn   : float
    cov          : (2,2) ndarray
        cov[0,0] ~ var_gain (as in your snippet), cov[1,1] ~ var_rn;
        off-diagonals are NaN unless fitquad returns something else.
    """

    if len(sigs_short) > 1 and len(gains_short) > 1:
        (bestfit_gain, bestfit_rn), cov, coefs = fit_gain_Brandt.fitquad(diffs_lin, diffs_quad, covar,
                                                sigs_short, gains_short, log=False,d2use=d2use,
                                                                         countratesidentical=countratesidentical)
    elif len(sigs_short) == 1:
        logL = fit_gain_Brandt.gen_likelihood_map(diffs_lin, diffs_quad, covar,
                                  sigs_short, gains_short, [0],
                                  countratesidentical=countratesidentical,d2use=d2use)
        a,b,c = quad_1d_through_3pts(gains_short, logL[0,:,0])
        bestfit_gain = -b/(2*a)
        bestfit_rn = sigs_short[0]
        cov = np.full((2,2),np.nan)
        cov[0,0] = -1/(2*a)
    elif len(gains_short) == 1:
        logL = fit_gain_Brandt.gen_likelihood_map(diffs_lin, diffs_quad, covar,
                                  sigs_short, gains_short, [0],
                                  countratesidentical=countratesidentical,d2use=d2use)
        a,b,c = quad_1d_through_3pts(sigs_short, logL[:,0,0])
        bestfit_gain = gains_short[0]
        bestfit_rn = -b/(2*a)
        cov = np.full((2,2),np.nan)
        cov[1,1] = -1/(2*a)

    return bestfit_gain, bestfit_rn, cov

def fit_single_gain_readnoise(
        readtimes: np.ndarray, ramps: np.ndarray, dq_raw: np.ndarray, covar: fitramp_Brandt.Covar,
        gains_init: np.ndarray, rns_init: np.ndarray, error_computation: str = 'quad',
        countratesidentical: bool = False, return_likelihood: bool = False
):
    """
    Fit the gain and read noise using a grid search maximum likelihood.
    The grid search is defined by gains_init and rns_init.

    Parameters
    ----------
    readtimes : np.ndarray
        1D array of read times.
    ramps : np.ndarray
        2D array of input rates. Size nramps x nreads.
    dq_raw : np.ndarray
        2D array of input UTR data quality flags. Size nramps x nreads.
    covar : fitramp_Brandt.Covar class
           Should be appropriate for diffs_lin
    gains_init : np.ndarray
        Grid sampling for gain (e-/ADU). Elements should be all positive.
    rns_init : float
        Grid sampling for read noise (in DN). Elements should be all positive.
    error_computation : str
        Mode to compute the errors. Options are 'quad' for quadratic approximation, 'posterior' for posterior marginalization.
    countratesidentical: bool
        If True, assume all ramps have identical count rates.

    Returns
    -------
    gain : float
        Best fit Gain (e-/ADU)
    gain_err : float
        Best fit Gain error (e-/ADU)
    rn : float
        Best fit read noise (in DN)
    rn_err : float
        Best fit read noise error (in DN)
    """
    nreads = readtimes.shape[0]
    _ramps = np.array(ramps).swapaxes(0, 1) # n_reads, n_files, ny, nx
    _dq_raw = np.array(dq_raw).swapaxes(0, 1) # n_reads, n_files, ny, nx
    _ramps = _ramps.reshape(nreads, -1)  # shape (n_reads, nfiles *ny*nx)
    _dq_raw = _dq_raw.reshape(nreads, -1)  # shape (n_reads, nfiles*ny*nx)

    y0 = 0
    diff_times = np.diff(readtimes)[:, None]
    diffs_lin = np.diff(_ramps, axis=0).astype(np.float64)/diff_times
    diffs_quad = np.diff((_ramps - y0) ** 2, axis=0).astype(np.float64)/diff_times

    n_reads,n_ramps = _dq_raw.shape
    mask = ((_dq_raw & DO_NOT_USE_FLAG_uint8) | (_dq_raw & SATURATED_FLAG_uint8)).astype(np.bool)
    mask2 = (_dq_raw & JUMP_DET_FLAG_uint8).astype(np.bool)
    diffs2use = ~(mask[1:n_reads] | mask[0:n_reads - 1] | mask2[1:n_reads])
    # diffs2use = np.full((n_reads-1,n_ramps),1,dtype=np.uint8)
    # diffs2use[5,:] = 0
    # diffs2use[10,:] = 0
    # diffs2use[20,:] = 0
    diffs2use = diffs2use.astype(np.uint8)

    _rns_init = rns_init.astype(np.float64)
    _gains_init = gains_init.astype(np.float64)

    likelihoods = fit_gain_Brandt.gen_likelihood_map(diffs_lin, diffs_quad, covar, _rns_init, _gains_init, [0],
                                                     d2use=diffs2use,countratesidentical=countratesidentical)[..., 0]

    maxL = np.amax(likelihoods)
    k, l = np.unravel_index(np.argmax(likelihoods), likelihoods.shape)
    # bestfit_gain, bestfit_rn = _gains_init[l], rns_init[k]

    if error_computation == 'posterior':

        gain_post = np.sum(np.exp(likelihoods - np.amax(likelihoods)), axis=0)
        gain_post /= np.sum(gain_post)
        gains_init_diffs = np.diff(_gains_init, prepend=_gains_init[0] - (_gains_init[1] - _gains_init[0]))
        gain_post /= gains_init_diffs

        bestfit_gain, gain_left_error, gain_right_error = get_err_from_posterior(_gains_init, gain_post)
        bestfit_gain_err = np.max([gain_left_error, gain_right_error])

        if np.size(_rns_init) > 3:
            rn_post = np.sum(np.exp(likelihoods - np.amax(likelihoods)), axis=1)
            rn_post /= np.sum(rn_post)
            rns_init_diffs = np.diff(_rns_init, prepend=_rns_init[0] - (_rns_init[1] - _rns_init[0]))
            rn_post /= rns_init_diffs

            bestfit_rn, rn_left_error, rn_right_error = get_err_from_posterior(_rns_init, rn_post)
            bestfit_rn_err = np.max([rn_left_error, rn_right_error])
        else:
            bestfit_rn, bestfit_rn_err = np.nan, np.nan

        cov = np.full((2,2),np.nan)
        cov[0,0] = bestfit_gain_err
        cov[1,1] = bestfit_rn

        if return_likelihood:
            other_outputs = {}
            other_outputs["rns_init"] = _rns_init
            other_outputs["gains_init"] = _gains_init
            other_outputs["likelihoods"] = likelihoods
            return bestfit_gain, bestfit_rn, cov, other_outputs
        else:
            return bestfit_gain, bestfit_rn, cov

    elif error_computation == 'quad':
        # three points each in read noise and gain to fit a quadratic form
        n_iter = 2
        for i in range(n_iter):
            # three points each in read noise and gain to fit a quadratic form
            if i == 0:
                delta_gain = (_gains_init[1] - _gains_init[0])/2
                gains_short = [_gains_init[l]-delta_gain,_gains_init[l], _gains_init[l]+delta_gain]
                delta_rn = (_rns_init[1] - _rns_init[0])/2
                sigs_short = [_rns_init[k]-delta_rn,_rns_init[k], _rns_init[k]+delta_rn]
            else:
                if len(gains_short) > 1:
                    bestfit_gain_err = np.sqrt(cov[0, 0])
                    gains_short = [bestfit_gain- bestfit_gain_err,bestfit_gain, bestfit_gain + bestfit_gain_err]
                if len(sigs_short) > 1:
                    bestfit_rn_err = np.sqrt(cov[1,1])
                    sigs_short = [bestfit_rn- bestfit_rn_err,bestfit_rn, bestfit_rn + bestfit_rn_err]
                
            bestfit_gain, bestfit_rn, cov = fitquad_gain_rn(diffs_lin, diffs_quad, covar,sigs_short, gains_short,
                                                            d2use=diffs2use,countratesidentical=countratesidentical)

        return bestfit_gain, bestfit_rn, cov



def quad_1d_through_3pts(x, y):
    """
    Fit y = a*x^2 + b*x + c through exactly three points (xi, yi),
    without explicit matrix inversion.

    x, y: array-like of length 3
    Returns: (a, b, c)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != (3,) or y.shape != (3,):
        raise ValueError("x and y must be length-3 arrays")
    if np.unique(x).size != 3:
        raise ValueError("x values must be distinct")

    x1, x2, x3 = x
    y1, y2, y3 = y

    d1 = (x1 - x2) * (x1 - x3)
    d2 = (x2 - x1) * (x2 - x3)
    d3 = (x3 - x1) * (x3 - x2)

    a = y1 / d1 + y2 / d2 + y3 / d3
    b = -(y1 * (x2 + x3) / d1 + y2 * (x1 + x3) / d2 + y3 * (x1 + x2) / d3)
    c = y1 * x2 * x3 / d1 + y2 * x1 * x3 / d2 + y3 * x1 * x2 / d3

    return a, b, c

def plot_2d_posterior(
    likelihoods,
    gains_init,
    rns_init,
    *,
    aspect=0.4,
    vmin=-20,
    vmax=0,
    cmap="bone_r",
    contour_deltas=(11.8/2, 6.2/2, 2.3/2),
    contour_color="r",
    contour_lw=2,
    cbar_label=r"$\log {\cal L}/{\cal L}_{\rm max}$",
    xlabel=r"Gain ($e^-$/DN)",
    ylabel=r"Read Noise (DN)",
    title_2d=None,
    title_1d=None,
    show=False,
):
    """
    Plot a 2D log-likelihood map (shifted by max) with confidence contours,
    and optionally a 1D marginalized gain posterior.

    Parameters
    ----------
    likelihoods : (Nr, Ng) array
        Log-likelihood values on the (rns_init, gains_init) grid.
        Assumes axis-0 corresponds to rns_init and axis-1 corresponds to gains_init.
    gains_init : (Ng,) array
        Gain grid (x-axis).
    rns_init : (Nr,) array
        Read-noise grid (y-axis).
    show : bool
        If True, calls plt.show() at the end.

    Returns
    -------
    figs : dict
        {"2d": fig2d, "1d": fig1d_or_None}
    axes : dict
        {"2d": ax2d, "1d": ax1d_or_None}
    """
    likelihoods = np.asarray(likelihoods)
    gains_init = np.asarray(gains_init)
    rns_init = np.asarray(rns_init)

    gain_post = np.sum(np.exp(likelihoods - np.amax(likelihoods)), axis=0)
    gain_post /= np.sum(gain_post)
    gains_init_diffs = np.diff(gains_init, prepend=gains_init[0] - (gains_init[1] - gains_init[0]))
    gain_post /= gains_init_diffs

    if likelihoods.ndim != 2:

        raise ValueError("likelihoods must be 2D (Nr, Ng).")
    if likelihoods.shape != (rns_init.size, gains_init.size):
        raise ValueError(
            f"Shape mismatch: likelihoods has shape {likelihoods.shape} but "
            f"expected ({rns_init.size}, {gains_init.size}) = (len(rns_init), len(gains_init))."
        )

    maxL = np.nanmax(likelihoods)
    extent = [gains_init[0], gains_init[-1], rns_init[0], rns_init[-1]]

    # --- 2D figure ---
    fig2d, ax2d = plt.subplots(figsize=(8, 6))
    im = ax2d.imshow(
        likelihoods - maxL,
        origin="lower",
        aspect=aspect,
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    cbar = fig2d.colorbar(im, ax=ax2d)
    cbar.set_label(cbar_label)

    ax2d.set_xlabel(xlabel)
    ax2d.set_ylabel(ylabel)
    if title_2d is not None:
        ax2d.set_title(title_2d)

    # Contours at maxL - delta
    levels = [maxL - d for d in contour_deltas]
    ax2d.contour(
        likelihoods,
        extent=extent,
        levels=levels,
        colors=[contour_color] * len(levels),
        linewidths=[contour_lw] * len(levels),
    )

    gain_post = np.asarray(gain_post)
    if gain_post.shape != (gains_init.size,):
        raise ValueError(
            f"gain_post has shape {gain_post.shape} but expected ({gains_init.size},)."
        )
    fig1d, ax1d = plt.subplots(figsize=(8, 6))
    ax1d.plot(gains_init, gain_post, label="Full Numerical Marginalization")
    ax1d.set_xlabel(xlabel)
    ax1d.set_ylabel("Posterior (arb.)")
    if title_1d is not None:
        ax1d.set_title(title_1d)
    ax1d.legend()

    if show:
        plt.show()

    return {"2d": fig2d, "1d": fig1d}, {"2d": ax2d, "1d": ax1d}

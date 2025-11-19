import os
import numpy as np

from ...datamodels.dqflags import DQ_FLAGS
from .utils import get_mid_delta_times
from . import fitramp_cython

from tqdm import tqdm

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


__all__ = ['fitramps_cython', '_create_diff_ramps']

SATURATED_FLAG = DQ_FLAGS["SATURATED"]
DO_NOT_USE_FLAG = DQ_FLAGS["DO_NOT_USE"]
JUMP_DET_FLAG = DQ_FLAGS["JUMP_DET"]
SATURATED_FLAG_uint8 =  np.uint8(DQ_FLAGS["SATURATED"])
DO_NOT_USE_FLAG_uint8 = np.uint8(DQ_FLAGS["DO_NOT_USE"])
JUMP_DET_FLAG_uint8 = np.uint8(DQ_FLAGS["JUMP_DET"])



def _flatten_detector_image(data: np.ndarray):
    if data.ndim == 3:
        reshaped = data.reshape(data.shape[0], -1, order='C')
    elif data.ndim == 2:
        reshaped = data.flatten(order='C')
    else:
        raise ValueError("Input data must be either 2D or 3D.")
    return reshaped

class Covar:

    """

    class Covar holding read and photon noise components of alpha and
    beta and the time intervals between the resultant midpoints

    """
    
    def __init__(self, read_times, pedestal=False):

        """

        Compute alpha and beta, the diagonal and off-diagonal elements of
        the covariance matrix of the resultant differences, and the time 
        intervals between the resultant midpoints.

        Arguments:
        1. readtimes [list of values or lists for the times of reads.  If 
                      a list of lists, times for reads that are averaged
                      together to produce a resultant.]

        Optional arguments:
        2. pedestal  [boolean: does the covariance matrix include the terms
                      for the first resultant?  This is needed if fitting
                      for the pedestal (i.e. the reset value).  Default
                      False. ]

        """

        mean_t = []   # mean time of the resultant as defined in the paper
        tau = []   # variance-weighted mean time of the resultant
        N = []  # Number of reads per resultant
        
        for times in read_times:
            mean_t += [np.mean(times)]
            
            if hasattr(times, "__len__"):
                N += [len(times)]
                k = np.arange(1, N[-1] + 1)
                tau += [1/N[-1]**2*np.sum((2*N[-1] + 1 - 2*k)*np.array(times))]
            else:
                tau += [times]
                N += [1]
            
        mean_t = np.array(mean_t)
        tau = np.array(tau)
        N = np.array(N)
        
        delta_t = mean_t[1:] - mean_t[:-1]

        self.pedestal = pedestal
        self.delta_t = delta_t
        self.mean_t = mean_t
        self.tau = tau
        self.Nreads = N
        
        self.alpha_readnoise = (1/N[:-1] + 1/N[1:])/delta_t**2
        self.beta_readnoise = -1/N[1:-1]/(delta_t[1:]*delta_t[:-1])
        
        self.alpha_phnoise = (tau[:-1] + tau[1:] - 2*mean_t[:-1])/delta_t**2
        self.beta_phnoise = (mean_t[1:-1] - tau[1:-1])/(delta_t[1:]*delta_t[:-1])

        # If we want the reset value we need to include the first
        # resultant.  These are the components of the variance and
        # covariance for the first resultant.
        
        if pedestal:
            self.alpha_readnoise = np.array([1/N[0]/mean_t[0]**2]
                                            + list(self.alpha_readnoise))
            self.beta_readnoise = np.array([-1/N[0]/mean_t[0]/delta_t[0]]
                                           + list(self.beta_readnoise))
            self.alpha_phnoise = np.array([tau[0]/mean_t[0]**2]
                                          + list(self.alpha_phnoise))
            self.beta_phnoise = np.array([(mean_t[0] - tau[0])/mean_t[0]/delta_t[0]]
                                         + list(self.beta_phnoise))
            
def fitramps_cython(delta_readtimes : np.ndarray,
    ramps : np.ndarray,
    dq_raw : np.ndarray | None = None,
    dq : np.ndarray | None = None,
    rdnoise : np.ndarray | None = None,
    gain : np.ndarray | None = None,
    dark_current : np.ndarray | None = None,
    start_times : np.ndarray | None = None,
    num_coadd : int = 1,
) -> dict[str, np.ndarray]:

    n_reads, ny, nx = ramps.shape
    n_pix = ny*nx
    # Create remaining dummy variables if input arrays are not provided
    if dq_raw is None:
        dq_raw = np.zeros((n_reads, ny, nx), dtype=np.uint8)
    if dq is None:
        dq = np.zeros((ny, nx), dtype=np.uint32)
    if rdnoise is None:
        rdnoise = np.zeros((ny, nx), dtype=np.float32)
    if gain is None:
        gain = np.ones((ny, nx), dtype=np.float32)
    if dark_current is None:
        dark_current = np.zeros((ny, nx), dtype=np.float32)
    if start_times is None:
        start_times = np.zeros((ny, nx), dtype=np.float64)

    flat_times = _flatten_detector_image(delta_readtimes)
    flat_ramps = _flatten_detector_image(ramps)
    flat_dq_raw = _flatten_detector_image(dq_raw)
    flat_dq = _flatten_detector_image(dq)
    flat_rdnoise = _flatten_detector_image(rdnoise)
    flat_gain = _flatten_detector_image(gain)
    flat_dark_current = _flatten_detector_image(dark_current)
    flat_start_times = _flatten_detector_image(start_times)


    # convert delta times (seconds) → days
    delta_days = flat_times[:, None] / 86400.0    # shape (n_reads, 1)

    # broadcast-add
    actual_read_times = flat_start_times[None, :] + delta_days
    # shape (n_reads, npix)

    covar = Covar(actual_read_times)
    

    r = fitramp_cython.fit_ramps(
        diffs_lin, flat_dq_raw,
        covar.alpha_phnoise/gain, covar.beta_phnoise/gain,
        covar.alpha_readnoise, covar.beta_readnoise,
        flat_rdnoise, cguess=None, n1=n_reads, n0=n_pix
        )
    
    logdetC, A, B, C, countrate, chisq, uncert = r
   
    return A, B, C
    # return slope_image, slope_err, var_rnoise, var_poisson, new_dq, mid_time, b, b_err
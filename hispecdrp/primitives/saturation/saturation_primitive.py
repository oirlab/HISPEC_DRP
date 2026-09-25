from ..base_primitive import BasePrimitive
from ...calibrations import SaturationSelector
from ... import datamodels
from ...datamodels.dqflags import DQ_FLAGS
from ...utils.parallelization_utils import numba_thread_scope
from numba import njit, prange
import numpy as np
import logging

logger = logging.getLogger(__name__)

__all__ = ['CheckSaturationPrimitive', 'check_saturation']

SATURATED_FLAG =  DQ_FLAGS["SATURATED"]
DO_NOT_USE_FLAG = DQ_FLAGS["DO_NOT_USE"]
SATURATED_FLAG_uint8 =  np.uint8(DQ_FLAGS["SATURATED"])
DO_NOT_USE_FLAG_uint8 = np.uint8(DQ_FLAGS["DO_NOT_USE"])

class CheckSaturationPrimitive(BasePrimitive):
    """
    Flag saturated reads in the input data.

    This primitive identifies pixels that exceed their per-pixel saturation
    threshold, as defined in the saturation calibration reference file. For
    any read in which a pixel is saturated, the corresponding ``SATURATED``
    flag is set in the data quality (DQ) array.

    Parameters
    ----------
    dataset : str or RampModel
        Input file path or data model to process. Currently, this is expected
        to be a raw (ramp) data product.
    suffix : str or None, optional
        Suffix to append to the output filename. Default is ``'satcheck'``.
    max_cores : int, optional
        Maximum number of CPU cores to use for processing. Default is 1 (no parallelization).

    Returns
    -------
    output : DataModel
        Data model with the ``SATURATED`` DQ flags set for affected pixels and
        reads.

    Calibrations
    ------------
    saturation : SaturationModel
        - selector: :class:`~hispecdrp.calibrations.selectors.SaturationSelector`

    - **saturation**: SaturationModel
        The saturation calibration model used for flagging.
    """

    PARAMETERS = {
        'suffix' : 'satcheck',
    }

    CALIBRATIONS = {
        'saturation': {
            'selector': SaturationSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):

        # Load the input model
        input_model = datamodels.open(self.dataset)

        # Get the saturation threshold model
        sat, _ = self.get_calibration(input_model, 'saturation')
        saturation_model = datamodels.open(sat)

        # Output
        result = self.get_output_model(input_model)

        # Log
        logger.info(f"Applying saturation check with {saturation_model} to {input_model}")

        if not hasattr(input_model, "DQ_RAW"):
            raise AttributeError("Input model has no DQ_RAW array to flag saturation.")
        if not hasattr(input_model, "DQ") or input_model.dq is None:
            input_model.dq = np.zeros(input_model.data.shape[1:], dtype=np.uint32)

        # Perform saturation check
        max_cores = self.max_cores or 1
        check_saturation(
            result.data, result.dq_raw, result.dq,
            saturation_model.sat_thresh, saturation_model.dq,
            max_cores=max_cores
        )

        # Output
        return dict(output=result, saturation=sat)


def check_saturation(
    input_data : np.ndarray, input_dq_raw : np.ndarray, input_dq : np.ndarray,
    sat_thresh : np.ndarray, sat_dq : np.ndarray,
    max_cores : int = 1
):
    """
    Flag saturated pixels in ramps.

    Parameters
    ----------
    input_data : np.ndarray
        The input ramp data array of shape (Nreads, Ny, Nx).
    input_dq_raw : np.ndarray
        The input DQ_RAW array of shape (Nreads, Ny, Nx).
    input_dq : np.ndarray
        The input DQ array of shape (Ny, Nx).
    sat_thresh : np.ndarray
        The saturation threshold map of shape (Ny, Nx).
    sat_dq : np.ndarray
        The DQ map from the saturation model of shape (Ny, Nx).
    max_cores : int
        The maximum number of CPU cores to use.

    Returns
    -------
    input_dq_raw : np.ndarray
        The updated DQ_RAW array with saturation flags applied.
    input_dq : np.ndarray
        The updated DQ array with saturation flags applied.
    """
    with numba_thread_scope(max_cores):
        _check_saturation_numba(
            input_data, input_dq_raw, input_dq,
            sat_thresh, sat_dq,
        )
    return input_dq_raw, input_dq

@njit(nogil=True, parallel=True, cache=True)
def _check_saturation_numba(data, dq_raw, dq, sat_thresh, sat_dq):
    n_read, ny, nx = data.shape
    for j in prange(ny):  # parallelize over spatial rows
        for i in range(nx):
            thresh = sat_thresh[j, i]
            dq[j, i] |= sat_dq[j, i]
            for r in range(n_read):
                if data[r, j, i] >= thresh:
                    dq_raw[r::, j, i] |= (SATURATED_FLAG_uint8 | DO_NOT_USE_FLAG_uint8)
                    break
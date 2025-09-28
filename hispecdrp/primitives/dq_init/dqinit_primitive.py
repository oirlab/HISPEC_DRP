from ..base_primitive import BasePrimitive
from ...calibrations import DQSelector
from ... import datamodels
from numba import njit, prange
from numba import get_num_threads, set_num_threads
import numpy as np

__all__ = ['DQInitPrimitive']


class DQInitPrimitive(BasePrimitive):
    
    ARGUMENTS = {
        'suffix' : 'dqinit',
        'output_model' : 'input',
        'max_cores' : 1
    }

    CALIBRATIONS = {
        'dq': {
            'selector': DQSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):
        set_num_threads(self.max_cores)

        # Load the input model
        input_model = datamodels.open(self.input)

        # Get the dark model
        dq, _ = self.get_calibration(input_model, 'dq')
        dq_model = datamodels.open(dq)

        # Output
        result = self.get_output_model(input_model)

        # Log
        self.log.info(f"Applying DQ initialization with {dq} to {input_model}")
        self.log.info(f"Note: UTR data type is {result.data.dtype}")

        if not hasattr(input_model, 'DQ_RAW') or  input_model.dq_raw is None:
            result.dq_raw = np.zeros_like(result.data, dtype=np.uint8)
        # if hasattr(input_model, 'DQ_RAW'):
            # result.dq_raw = apply_dq_mask(input_model.dq_raw, dq_model.dq)
        if hasattr(input_model, 'DQ') and input_model.dq is not None:
            result.dq |= dq_model.dq
        else:
            result.dq = dq_model.dq

        # Output
        return dict(output=result, dq=dq)

@njit(nogil=True, parallel=True, cache=True)
def apply_dq_mask(dq_raw, dq_model):
    """
    Apply dq_model (2D) to dq_raw (3D: n_reads x ny x nx) in parallel along ny.

    Args:
        dq_raw (np.ndarray): shape (n_read, ny, nx), dtype=uint32
        dq_model (np.ndarray): shape (ny, nx), dtype=uint32

    Returns:
        np.ndarray: updated dq_raw with dq_model OR-ed in
    """
    n_read, ny, nx = dq_raw.shape

    for j in prange(ny):  # parallelize over ny
            for i in range(nx):
                dq_raw[:, j, i] = dq_raw[:, j, i] | dq_model[j, i]

    return dq_raw
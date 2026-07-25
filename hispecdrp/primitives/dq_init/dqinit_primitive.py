from ..base_primitive import BasePrimitive
from ...calibrations import DQSelector
from ... import datamodels
from ...utils.parallelization_utils import numba_thread_scope
from numba import njit, prange
import numpy as np
import logging

logger = logging.getLogger(__name__)

__all__ = ['DQInitPrimitive']


class DQInitPrimitive(BasePrimitive):
    """
    Initialize the data quality (DQ) array with known pixel flags.

    Definition of DQ flags can be found in :data:`~hispecdrp.datamodels.dqflags.DQ_FLAGS`.

    This primitive populates the DQ array using a reference DQ calibration
    file that encodes known pixel-level flags.

    Parameters
    ----------
    input : str or RampModel
        Input file path or data model to process. Currently, this is expected
        to be a raw (ramp) data product.
    suffix : str or None, optional
        Suffix to append to the output filename. Default is ``'dqinit'``.
    max_cores : int, optional
        Maximum number of CPU cores to use for parallel processing. Default is 1 (no parallelization).

    Calibrations
    ------------
    dq : DQModel
        - selector: :class:`~hispecdrp.calibrations.selectors.DQSelector`

    Returns
    -------
    output : DataModel
        Data model with the DQ array initialized from the DQ reference file.
    """

    ARGUMENTS = {
        'suffix' : 'dqinit',
        'max_cores' : 1
    }

    CALIBRATIONS = {
        'dq': {
            'selector': DQSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):

        # Load the input model
        input_model = datamodels.open(self.input)

        # Get the dark model
        dq, _ = self.get_calibration(input_model, 'dq')
        dq_model = datamodels.open(dq)

        # Output
        result = self.get_output_model(input_model)

        # Log (use input_model.data if result.data is not set; avoid "or" so numpy arrays are not used in boolean context)
        _r_data = getattr(result, "data", None)
        data_ref = _r_data if _r_data is not None else getattr(input_model, "data", None)
        logger.info(f"Applying DQ initialization with {dq} to {input_model}")
        if data_ref is not None:
            logger.info(f"Note: UTR data type is {data_ref.dtype}")

        if not hasattr(input_model, 'DQ_RAW') or  input_model.dq_raw is None:
            if data_ref is None:
                raise ValueError("Cannot initialize DQ_RAW: result and input have no data array.")
            result.dq_raw = np.zeros_like(data_ref, dtype=np.uint8)
        # if hasattr(input_model, 'DQ_RAW'):
            # result.dq_raw = _apply_dq_mask(input_model.dq_raw, dq_model.dq)
        if hasattr(input_model, 'DQ') and input_model.dq is not None:
            result.dq |= dq_model.dq
        else:
            result.dq = dq_model.dq

        # Output
        return dict(output=result, dq=dq)

@njit(nogil=True, parallel=True, cache=True)
def _apply_dq_mask(dq_raw, dq_model):
    n_read, ny, nx = dq_raw.shape
    for j in prange(ny):
        for i in range(nx):
            dq_raw[:, j, i] = dq_raw[:, j, i] | dq_model[j, i]
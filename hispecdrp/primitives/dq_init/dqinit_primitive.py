from ..base_primitive import BasePrimitive
from ...calibrations import DQSelector
from ... import datamodels
from ...utils.parallelization_utils import numba_thread_scope
from numba import njit, prange
import numpy as np

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
    output_model : str or type[DataModel] or None, optional
        The DataModel class or instance to return. ``'input'`` overwrite the input (default), while ``'input_copy'`` returns a copy.

    Returns
    -------
    output : DataModel
        Data model with the DQ array initialized from the DQ reference file.

    Notes
    -----
    Additional outputs accessible as ``primitive._output[<key>]``

    - **dq**: DQModel
        The DQ calibration model used for initialization.
    """

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
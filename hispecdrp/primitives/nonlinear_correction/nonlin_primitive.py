from ..base_primitive import BasePrimitive
from ...calibrations import NonlinearCorrectionSelector
from ... import datamodels
import numpy as np

from .nonlincorr_numba import correct_nonlinearity


__all__ = ['NonlinearCorrectionPrimitive']


class NonlinearCorrectionPrimitive(BasePrimitive):
    """
    Correct the detector non-linear response in the input data.

    This primitive applies a polynomial transformation to convert the
    measured non-linear detector signal into a linearized signal.

    The non-linearity correction is applied using the following polynomial
    equation:
        P(DN) = a0 + a1*DN + a2*DN^2 + ... + an*DN^n
        where P(DN) is the corrected signal, DN is the measured signal, and
    [a0, a1, ..., an] are the polynomial coefficients provided in the
    non-linearity calibration file.
    Note that the ordering of the coefficients in the calibration file is from lowest
    order to highest order, which is the reverse of the usual numpy.polyval ordering.

    Data quality (DQ) flags are propagated to the output using a bitwise OR. No
    error or variance attributes are modified or generated at this stage of
    processing.

    Parameters
    ----------
    input : str or RampModel
        Input file path or data model to process. This is expected to be a
        raw up-the-ramp (UTR) ramp data product.
    suffix : str or None, optional
        Suffix to append to the output filename. Default is ``'nonlincorr'``.
    max_cores : int, optional
        Maximum number of CPU cores to use for parallel processing. Default is 1.

    Returns
    -------
    output : DataModel
        Data model with the detector non-linearity corrected and updated DQ
        flags.

    Notes
    -----
    Additional outputs accessible as ``primitive._output[<key>]``

    - **nonlin**: NonlinearityModel
        The non-linearity calibration model used for correction.
    """

    ARGUMENTS = {
        'suffix' : 'nonlincorr',
        'ouput_model' : 'input',
        'max_cores' : 1,
    }

    CALIBRATIONS = {
        'nonlin': {
            'selector': NonlinearCorrectionSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):

        input_model = datamodels.open(self.input)

        nonlin, _ = self.get_calibration(input_model, 'nonlin')
        nonlin_model = datamodels.open(nonlin)

        result = self.get_output_model(input_model)
        self.log.info(f"Applying linearity correction with {nonlin_model} to {input_model}")

        if result.data.dtype != np.float32:
            self.log.info(f"Converting UTR data from {result.data.dtype} to float32")
            result.data = result.data.astype(np.float32)
        self.log.info(f"Polynomial coeffs dtype: {nonlin_model.coeffs.dtype}.")

        correct_nonlinearity(
            result.data,
            result.dq_raw,
            result.dq,
            nonlin_model.coeffs,
            nonlin_model.nonlin_max,
            nonlin_model.dq,
            max_cores=self.max_cores
        )

        return dict(output=result, nonlin=nonlin)
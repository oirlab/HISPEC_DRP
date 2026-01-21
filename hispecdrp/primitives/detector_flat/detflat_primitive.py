
from ..base_primitive import BasePrimitive
from ... import datamodels
from ...calibrations import DetectorFlatSelector
from ...utils.parallelization_utils import numba_thread_scope
import numpy as np
from numba import njit, prange

__all__ = ['DetectorFlatCorrectionPrimitive', 'detector_flat_correction']


class DetectorFlatCorrectionPrimitive(BasePrimitive):
    """
    Apply detector flat-field correction to the input data.

    This primitive corrects for pixel-to-pixel variations in detector
    response by dividing the input rate map by the appropriate master
    detector flat calibration. Uncertainties are propagated by combining
    the input errors with the detector flat errors in quadrature using the
    appropriate scaling. Data quality (DQ) flags from the flat calibration
    are merged into the output using a bitwise OR.

    Parameters
    ----------
    input : str or L0Model
        Input file path or data model to process. This is expected to be a
        level 0 (rate map) data product.
    suffix : str or None, optional
        Suffix to append to the output filename. Default is
        ``'detflat'``.
    output_model : str or type[DataModel] or None, optional
        The DataModel class or instance to return. ``'input'`` overwrite the input (default), while ``'input_copy'`` returns a copy.

    Returns
    -------
    output : DataModel
        Flat-field-corrected data model with updated error and data quality
        arrays.

    Notes
    -----
    Additional outputs accessible as ``primitive._output[<key>]``

    - **detflat**: DetectorFlatModel
        The detector flat calibration model used for correction.
    """

    ARGUMENTS = {
        'suffix' : 'detflat',
        'output_model' : 'input',
        'max_cores' : 1,
    }

    CALIBRATIONS = {
        'detflat': {
            'selector': DetectorFlatSelector,
            'selector_kwargs': {}
        }
    }

    def _perform(self):

        # Load the input model
        input_model = datamodels.open(self.input)

        # Get the detflat model
        detflat, _ = self.get_calibration(input_model, 'detflat')
        detflat_model = datamodels.open(detflat)

        # Output
        result = self.get_output_model(input_model)

        if not hasattr(result, 'RATE'):
            raise Exception("RATE extension not found in input model for detector flat correction.")
        if not hasattr(result, 'RATE_ERR'):
            raise Exception("RATE_ERR extension not found in input model for detector flat correction.")
        if not hasattr(result, 'DQ'):
            raise Exception("DQ extension not found in input model for detector flat correction.")
        if not hasattr(detflat_model, 'FLAT'):
            raise Exception("FLAT extension not found in flat model for dark subtraction.")
        if not hasattr(detflat_model, 'FLAT_ERR'):
            raise Exception("FLAT_ERR extension not found in flat model for dark subtraction.")
        if not hasattr(detflat_model, 'DQ'):
            raise Exception("DQ extension not found in flat model for dark subtraction.")

        # Log
        self.log.info(f"Applying detector flat correction with {detflat} to {input_model}")

        # Perform detector flat correction in place
        detector_flat_correction(
            result.rate, result.rate_err, result.dq,
            detflat_model.flat, detflat_model.flat_err, detflat_model.dq,
            max_cores=self.max_cores
        )

        # Output
        return dict(output=result, detflat=detflat)


def detector_flat_correction(
    input_rate : np.ndarray, input_err : np.ndarray, input_dq : np.ndarray,
    detflat : np.ndarray, detflat_err : np.ndarray, detflat_dq : np.ndarray,
    max_cores : int = 1
):
    """
    Correct the detector level flat field.

    Parameters
    ----------
    input_rate : ndarray
        The input rate image to be corrected.
    input_err : ndarray
        The error associated with the input rate image.
    input_dq : ndarray
        The data quality array associated with the input rate image.
    detflat : ndarray
        The detector flat field image.
    detflat_err : ndarray
        The error associated with the detector flat field image.
    detflat_dq : ndarray
        The data quality array associated with the detector flat field image.
    max_cores : int
        The number of CPU cores to use for processing.
    
    Returns
    -------
    input_rate : ndarray
        Input rate image (DN/s) with flat-field correction applied.
    input_err : ndarray
        Input error image (DN/s) with flat-field correction applied.
    input_dq : ndarray
        Input data quality image with flat-field correction applied.
    """
    with numba_thread_scope(max_cores):
        _detector_flat_correction_numba(
            input_rate, input_err, input_dq,
            detflat, detflat_err, detflat_dq
        )
    return input_rate, input_err, input_dq


@njit(nogil=True, parallel=True, cache=True)
def _detector_flat_correction_numba(
    input_rate : np.ndarray, input_err : np.ndarray, input_dq : np.ndarray,
    flat : np.ndarray, flat_err : np.ndarray, flat_dq : np.ndarray
):
    ny, nx = input_rate.shape
    for j in prange(ny):
        for i in range(nx):
            f = flat[j, i]
            r = input_rate[j, i]
            input_rate[j, i] = r / f
            z = np.abs(input_rate[j, i])
            _input_err = input_err[j, i]
            _flat_err = flat_err[j, i]

            # approximate error propagation for division of random variables
            # https://en.wikipedia.org/wiki/Propagation_of_uncertainty
            input_err[j, i] = z * np.sqrt((_input_err / r) ** 2 + (_flat_err / f) ** 2)

            # Update DQ
            input_dq[j, i] |= flat_dq[j, i]
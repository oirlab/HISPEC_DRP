from ..base_primitive import BasePrimitive
from ... import datamodels
import logging

from .jump_detection_jwst import jump_detection_jwst
from .jump_detection_difference import jump_detection_difference
from ...calibrations import GainSelector, ReadNoiseSelector

logger = logging.getLogger(__name__)

__all__ = ['JumpDetectionPrimitive']


class JumpDetectionPrimitive(BasePrimitive):
    """
    Detect jumps in up-the-ramp (UTR) data and update DQ flags.

    This primitive looks for large jumps between consecutive reads that are larger than the typical slope.
    These jumps can be caused by cosmic rays or telegraph noise.
    Jumps are flagged in the data quality (DQ) array.

    todo: the slope will also be variable in time due to clouds, variable seeing and AO performance.
    This might make real jump detection more difficult.
    This UTR variability on the ground remains to be characterized,
    but it could also be leveraged to partially mitigate the effect of the variable seeing.

    JWST algorithm reference:
    https://jwst-pipeline.readthedocs.io/en/stable/jwst/jump/description.html


    Parameters
    ----------
    input : str or RampModel
        Input file path or data model to process. This must be a raw
        up-the-ramp (UTR) ramp data product in which jumps are to be detected.
    suffix : str or None, optional
        Suffix to append to the output filename. Default is ``'jump_detec'``.
    method : {'tpd', 'jwst'}, optional
        Jump detection algorithm to use. ``'tpd'`` selects a simple
        two-point-difference-based method. ``'jwst'`` uses the two-point
        difference jump detection algorithm implemented in ``stcal``.
    max_cores : int, optional
        Maximum number of CPU cores to use for parallel processing. Default
        is 1 (no parallel execution).
    rejection_threshold : float, optional
        Rejection threshold (approximately in signal-to-noise) above which
        discontinuities are flagged as jumps.

        
    Calibrations
    ------------
    gain : GainModel
        - selector: :class:`~hispecdrp.calibrations.selectors.GainSelector`
    rn : ReadNoiseModel
        - selector: :class:`~hispecdrp.calibrations.selectors.ReadNoiseSelector`

    Returns
    -------
    output : DataModel
        Ramp data model with updated DQ flags marking detected jumps.
    """

    ARGUMENTS = {
        'suffix' : 'jumpdetec',
        'method': 'tpd',
        'max_cores': 1,
        'rejection_threshold': 6.0,
    }

    CALIBRATIONS = {
        'gain': {
            'selector': GainSelector,
            'selector_kwargs': {}
        },
        'rn': {
            'selector': ReadNoiseSelector,
            'selector_kwargs': {}
        }
    }
    def _perform(self):
        # Input and output data model should be RampData similar to ramp fitting
        input_model = datamodels.open(self.input, copy=False)
        logger.info(f"Finding jumps in {input_model} with method={self.method.lower()}")

        gain_cal, _ = self.get_calibration(input_model, 'gain')
        gain_model = datamodels.open(gain_cal)
        rn_cal, _ = self.get_calibration(input_model, 'rn')
        rn_model = datamodels.open(rn_cal)

        # Output
        result = self.get_output_model(input_model)

        n_reads = result.data.shape[0]
        if n_reads <= 2:
            logger.warning("Not enough reads for jump detection, skipping.")
            self.status = 'SKIPPED'
            return dict(output=result, method=self.method.lower())

        # Perform jump detection - this only modifies the DQ mask
        if self.method.lower() == 'jwst':
            logger.info(
                "JWST jump detection is not optimized with shared memory here.",
                "For a shared memory execution, "
                "set the flag jwst_jump_detection=True in the ramp fitting primitive and run jump detection from there."
                "Although shared memory only works with JWST ramp fitting methods."
            )
            jump_detection_jwst(
                result.data,
                result.dq_raw,
                result.dq,
                rdnoise=rn_model.rn,
                gain=gain_model.gain,
                max_cores=self.max_cores,
                rejection_threshold=self.rejection_threshold
            )
        elif self.method.lower() == 'tpd':
            jump_detection_difference(
                result.data,
                result.dq_raw,
                result.dq,
                rdnoise=rn_model.rn,
                gain=gain_model.gain,
                max_cores=self.max_cores,
                rejection_threshold=self.rejection_threshold
            )
        else:
            logger.error(f"Invalid jump detection method {self.method.lower()}")
            raise ValueError(f"Invalid jump detection method {self.method.lower()}")

        # Output
        return dict(output=result, method=self.method.lower())

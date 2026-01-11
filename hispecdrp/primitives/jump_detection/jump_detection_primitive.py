from ..base_primitive import BasePrimitive
from ... import datamodels

from .jump_detection_jwst import jump_detection_jwst
from .jump_detection_difference import jump_detection_difference
from ...calibrations import GainSelector, ReadNoiseSelector

__all__ = ['JumpDetectionPrimitive']


class JumpDetectionPrimitive(BasePrimitive):

    ARGUMENTS = {
        'suffix' : 'jumpdetec',
        'ouput_model' : 'input',
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
        self.log.info(f"Finding jumps in {input_model} with method={self.method.lower()}")

        gain_cal, _ = self.get_calibration(input_model, 'gain')
        gain_model = datamodels.open(gain_cal)
        rn_cal, _ = self.get_calibration(input_model, 'rn')
        rn_model = datamodels.open(rn_cal)

        # Output
        result = self.get_output_model(input_model)

        n_reads = result.data.shape[0]
        if n_reads <= 2:
            self.log.warning("Not enough reads for jump detection, skipping.")
            self.status = 'SKIPPED'
            return dict(output=result, method=self.method.lower())

        # Perform jump detection - this only modifies the DQ mask
        if self.method.lower() == 'jwst':
            self.log.info(
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
                gain=detgain_model.gain,
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
            self.log.error(f"Invalid jump detection method {self.method.lower()}")
            raise ValueError(f"Invalid jump detection method {self.method.lower()}")

        # Output
        return dict(output=result, method=self.method.lower())

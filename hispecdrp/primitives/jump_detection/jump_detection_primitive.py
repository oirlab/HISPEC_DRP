from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np

from .jump_detection_jwst import jump_detection_jwst
from .jump_detection_difference import jump_detection_difference
from ...calibrations import DetectorGainSelector,ReadNoiseSelector

__all__ = ['JumpDetectionPrimitive']


class JumpDetectionPrimitive(BasePrimitive):

    ARGUMENTS = {
        'suffix' : 'jumpdetec',
        'ouput_model' : 'input',
        'method': 'tpd',
        'max_cores': 1,
        'rejection_threshold': 4.0,
    }

    CALIBRATIONS = {
        'detgain': {
            'selector': DetectorGainSelector,
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

        detgain_cal, _ = self.get_calibration(input_model, 'detgain')
        detgain_model = datamodels.open(detgain_cal)
        rn_cal, _ = self.get_calibration(input_model, 'rn')
        rn_model = datamodels.open(rn_cal)

        # Output
        result = self.get_output_model(input_model)

        # Perform jump detection - this only modifies the DQ mask
        if self.method.lower()=='jwst':
            raise NotImplementedError("JWST jump detection method should not be run here, but from within the ramp-"
                                      "fitting primitive because of shared memory parallelization needs.")
            jump_detection_jwst(result.data, result.dq_raw, max_cores=self.max_cores, rejection_threshold=self.rejection_threshold)
        elif self.method.lower()=='tpd':
            jump_detection_difference(result.data,
                                      result.dq_raw,
                                      result.dq,
                                      rdnoise=rn_model.rn,
                                      gain=detgain_model.gain,
                                      max_cores=self.max_cores,
                                      rejection_threshold=self.rejection_threshold)
        else:
            self.log.error(f"Invalid jump detection method {self.method.lower()}")
            raise ValueError(f"Invalid jump detection method {self.method.lower()}")


        # Output
        return dict(output=result, method=self.method.lower())

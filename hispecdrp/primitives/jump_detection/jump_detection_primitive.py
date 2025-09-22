from ..base_primitive import BasePrimitive
from ... import datamodels
import numpy as np

from .jump_detection_jwst import jump_detection_jwst
from .jump_detection_difference import jump_detection_difference

__all__ = ['JumpDetectionPrimitive']


class JumpDetectionPrimitive(BasePrimitive):

    ARGUMENTS = {
        'suffix' : 'jumpdetection',
        'ouput_model' : 'input',
        'method': 'jwst',
        'max_cores': 1,
        'rejection_threshold': 4.0,
    }

    def _perform(self):
        # Input and output data model should be RampData similar to ramp fitting
        input_model = datamodels.open(self.input, copy=False)
        self.log.info(f"Finding jumps in {input_model} with method={self.method.lower()}")

        # Output
        result = self.get_output_model(input_model)

        # Input UTR data
        data = result.data.astype(np.float32)
    
        # Input DQ flags from previous stages
        dq = result.dq_raw.astype(np.uint32)

        # Perform jump detection - this only modifies the DQ mask
        if self.method.lower()=='jwst':
            dq_out = jump_detection_jwst(data, dq, max_cores=self.max_cores, rejection_threshold=self.rejection_threshold)
        elif self.method.lower()=='tpd':
            dq_out = jump_detection_difference(data, dq, max_cores=self.max_cores, rejection_threshold=self.rejection_threshold)
        else:
            self.log.error(f"Invalid jump detection method {self.method.lower()}")
            raise ValueError(f"Invalid jump detection method {self.method.lower()}")

        # Apply output DQ flags from jumps to the result
        result.dq |= dq_out

        # Output
        return dict(output=result, method=method)

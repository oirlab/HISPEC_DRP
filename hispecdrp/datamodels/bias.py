import numpy as np

from .base_model import CalibrationModel
from .extensions import IMAGE_HDU, DQ_L0

__all__ = ['BiasModel']


class BiasModel(CalibrationModel):
    """

    Extensions
    ----------

    HDU Name   HDU Type   Data Type   Dimensions     Units   Description
    --------   --------   ---------   ----------     -----   -----------
    BIAS       Image      Float32     4096 x 4160    DN      Bias level.
    KTC        Image      Float32     4096 x 4160    DN      Bias kTC noise level.
    DQ         Image      UInt32      4096 x 4160    None    Data quality.
    """
    EXTENSIONS = {
        f'BIAS': IMAGE_HDU(np.float32, ndim=2, default=0),
        f'KTC': IMAGE_HDU(np.float32, ndim=2, default=0),
        **DQ_L0
    }
    CAL_TYPE = 'bias'
    _USES_DETECTOR_SIZE = True

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'DARK')
        self.meta.set_default('target_name', self.meta.exposure_type)
        for fiber in range(1, self.num_fibers+1):
            self.meta.set_default(f'object_fiber{fiber}', 'BLOCKED')
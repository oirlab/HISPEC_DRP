import numpy as np

from .base_model import CalibrationModel
from .extensions import IMAGE_HDU, DQ_L0

__all__ = ['ReadNoiseModel']


class ReadNoiseModel(CalibrationModel):
    """
    The read noise specifies the read noise associated with each pixel.

    HDU Name   HDU Type   Data Type   Dimensions    Units   Description
    --------   --------   ---------   ----------    -----   -----------
    RN         Image      Float32     4096 x 4160   e-      Read noise data.
    RN_ERR     Image      Float32     4096 x 4160   e-      Read noise error.
    DQ         Image      UInt32      4096 x 4160   None    Data quality.
    """
    EXTENSIONS = {
        f'RN': IMAGE_HDU(np.float32, ndim=2, default=0),
        f'RN_ERR': IMAGE_HDU(np.float32, ndim=2, default=0),
        **DQ_L0
    }
    CAL_TYPE = 'rn'
    _USES_DETECTOR_SIZE = True

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'DARK')
        self.meta.set_default('target_name', self.meta.exposure_type)
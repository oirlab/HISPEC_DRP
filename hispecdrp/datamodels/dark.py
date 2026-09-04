import numpy as np

from .base_model import CalibrationModel
from .extensions import IMAGE_HDU, DQ_L0

__all__ = ['DarkModel']


class DarkModel(CalibrationModel):
    """
    The dark current (and any other contribution to the dark) is assumed to be linear with time and is stored as a rate map product (e-/s).

    Extensions
    ----------

    HDU Name           HDU Type   Data Type   Dimensions     Units   Description
    --------           --------   ---------   ----------     -----   -----------
    RATE               Image      Float32     4096 x 4160    DN      Dark current data.
    RATE_ERR           Image      Float32     4096 x 4160    DN      Dark current error (all sources).
    RATE_VAR_RNOISE    Image      Float32     4096 x 4160    DN^2    Dark current variance (read noise).
    RATE_VAR_POISSON   Image      Float32     4096 x 4160    DN^2    Dark current variance (Poisson).
    DQ                 Image      UInt32      4096 x 4160    None    Data quality.
    """
    EXTENSIONS = {
        f'RATE': IMAGE_HDU(np.float32, ndim=2, default=0),
        f'RATE_ERR': IMAGE_HDU(np.float32, ndim=2, default=0),
        f'RATE_VAR_RNOISE': IMAGE_HDU(np.float32, ndim=2, default=0),
        f'RATE_VAR_POISSON': IMAGE_HDU(np.float32, ndim=2, default=0),
        **DQ_L0
    }
    CAL_TYPE = 'dark'
    _USES_DETECTOR_SIZE = True

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'DARK')
        self.meta.set_default('target_name', self.meta.exposure_type)
        for fiber in range(1, self.num_fibers+1):
            self.meta.set_default(f'object_fiber{fiber}', 'BLOCKED')
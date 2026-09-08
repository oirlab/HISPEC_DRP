import numpy as np

from .base_model import CalibrationModel
from .extensions import IMAGE_HDU, DQ_L0

__all__ = ['DarkModel']


class DarkModel(CalibrationModel):
    """
    The dark current (and any other contribution to the dark) is assumed to be linear with time and is stored as a rate map product (e-/s).
    """
    EXTENSIONS = {
        f'RATE': IMAGE_HDU(np.float32, shape=('DETECTOR_SIZE_Y', 'DETECTOR_SIZE_X'), default=0, title='Dark current data.', units='DN'),
        f'RATE_ERR': IMAGE_HDU(np.float32, default=0, title='Dark current error (all sources).', units='DN'),
        f'RATE_VAR_RNOISE': IMAGE_HDU(np.float32, default=0, title='Dark current variance (read noise).', units='DN^2'),
        f'RATE_VAR_POISSON': IMAGE_HDU(np.float32, default=0, title='Dark current variance (Poisson).', units='DN^2'),
        **DQ_L0
    }
    CAL_TYPE = 'dark'

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'DARK')
        self.meta.set_default('target_name', self.meta.exposure_type)
        for fiber in range(1, self.num_fibers+1):
            self.meta.set_default(f'object_fiber{fiber}', 'BLOCKED')
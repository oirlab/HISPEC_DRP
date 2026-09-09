import numpy as np

from .base_model import CalibrationModel
from .extensions import IMAGE_HDU, DQ_L0

__all__ = ['BiasModel']


class BiasModel(CalibrationModel):
    """
    """
    EXTENSIONS = {
        f'BIAS': IMAGE_HDU(
            dtype=np.float32,
            shape=('DETECTOR_SIZE_Y', 'DETECTOR_SIZE_X'),
            default=0,
            title='Bias level.',
            units='DN'
        ),
        f'KTC': IMAGE_HDU(
            dtype=np.float32,
            default=0,
            title='Bias kTC noise level.',
            units='DN'
        ),
        **DQ_L0
    }
    CAL_TYPE = 'bias'

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'DARK')
        self.meta.set_default('target_name', self.meta.exposure_type)
        for fiber in range(1, self.num_fibers+1):
            self.meta.set_default(f'object_fiber{fiber}', 'BLOCKED')
import numpy as np

from .base_model import CalibrationModel
from .extensions import IMAGE_HDU, DQ_L0

__all__ = ['ReadNoiseModel']


class ReadNoiseModel(CalibrationModel):
    """
    The read noise specifies the read noise associated with each pixel.
    """
    EXTENSIONS = {
        f'RN': IMAGE_HDU(
            dtype=np.float32,
            shape=('DETECTOR_SIZE_Y', 'DETECTOR_SIZE_X'),
            default=0,
            title='Read noise data.',
            units='DN'
        ),
        f'RN_ERR': IMAGE_HDU(
            dtype=np.float32,
            default=0,
            title='Read noise error.',
            units='DN'
        ),
        **DQ_L0
    }
    CAL_TYPE = 'rn'

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'DARK')
        self.meta.set_default('target_name', self.meta.exposure_type)
import numpy as np

from .base_model import CalibrationModel

from .extensions import IMAGE_HDU, DQ_L0

__all__ = ['GainModel']


class GainModel(CalibrationModel):
    """
    The detector gain specifies the conversion factor between electrons and data numbers (DN) for each pixel in the detector.
    """
    EXTENSIONS = {
        f'GAIN': IMAGE_HDU(np.float32, shape=('DETECTOR_SIZE_Y', 'DETECTOR_SIZE_X'), default=0, title='Gain map.', units='DN'),
        f'RN': IMAGE_HDU(np.float32, default=0, title='Read noise map.', units='DN'),
        f'COV': IMAGE_HDU(np.float32, default=0, title='Covariance map.', units='DN^2'),
        **DQ_L0
    }
    CAL_TYPE = 'gain'

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'DETFLAT')
        self.meta.set_default('target_name', 'WHITELIGHT')
        for fiber in range(1, self.num_fibers+1):
            self.meta.set_default(f'object_fiber{fiber}', 'BLOCKED')

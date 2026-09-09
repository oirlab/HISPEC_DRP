import numpy as np

from .base_model import CalibrationModel
from .extensions import IMAGE_HDU, DQ_L0

__all__ = ['SaturationModel']


class SaturationModel(CalibrationModel):
    """
    The saturation specifies the saturation level associated with each pixel.
    """
    EXTENSIONS = {
        "SAT_THRESH": IMAGE_HDU(
            np.uint16, shape=('DETECTOR_SIZE_Y', 'DETECTOR_SIZE_X'), default=np.iinfo(np.uint16).max,
            title='Saturation threshold.', units='DN',
        ),
        **DQ_L0,
    }
    CAL_TYPE = 'saturation'

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'DETFLAT')
        self.meta.set_default('target_name', 'WHITELIGHT')
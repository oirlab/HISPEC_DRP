import numpy as np

from .base_model import CalibrationModel
from .extensions import IMAGE_HDU, DQ_L0

__all__ = ['SaturationModel']


class SaturationModel(CalibrationModel):
    """
    The saturation specifies the saturation level associated with each pixel.

    Extensions
    ----------

    HDU Name     HDU Type   Data Type   Dimensions    Units   Description
    --------     --------   ---------   ----------    -----   -----------
    SAT_THRESH   Image      UInt16      4096 x 4160   DN      Saturation threshold.
    DQ           Image      UInt32      4096 x 4160   None    Data quality.
    """
    EXTENSIONS = {"SAT_THRESH": IMAGE_HDU(np.uint16, ndim=2, default=np.iinfo(np.uint16).max), **DQ_L0}
    CAL_TYPE = 'saturation'
    _USES_DETECTOR_SIZE = True

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'DETFLAT')
        self.meta.set_default('target_name', 'WHITELIGHT')
from .base_model import CalibrationModel
from .extensions import IMAGE_UINT16_HDU, DQ_L0

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
    EXTENSIONS = {"SAT_THRESH": IMAGE_UINT16_HDU, **DQ_L0}
    CAL_TYPE = 'saturation'
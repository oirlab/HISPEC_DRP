from .base_model import CalibrationModel

from .extensions import IMAGE_FLOAT32_HDU, DQ_L0

__all__ = ['GainModel']


class GainModel(CalibrationModel):
    """
    The detector gain specifies the conversion factor between electrons and data numbers (DN) for each pixel in the detector.

    Extensions
    ----------

    HDU Name   HDU Type   Data Type   Dimensions     Units   Description
    --------   --------   ---------   ----------     -----   -----------
    GAIN       Image      Float32     4096 x 4160    DN      Gain map.
    RN         Image      Float32     4096 x 4160    DN      Read noise map.
    COV        Image      Float32     4096 x 4160    DN^2    Covariance map.
    DQ         Image      UInt32      4096 x 4160    None    Data quality.
    """
    EXTENSIONS = {
        f'GAIN': IMAGE_FLOAT32_HDU,
        f'RN': IMAGE_FLOAT32_HDU,
        f'COV': IMAGE_FLOAT32_HDU,
        **DQ_L0
    }
    CAL_TYPE = 'gain'
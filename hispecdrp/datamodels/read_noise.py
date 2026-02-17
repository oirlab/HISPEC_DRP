from .base_model import CalibrationModel
from .extensions import IMAGE_FLOAT32_HDU, DQ_L0

__all__ = ['ReadNoiseModel']


class ReadNoiseModel(CalibrationModel):
    """
    The read noise specifies the read noise associated with each pixel.

    .. csv-table::
        :header: "Hdu Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

        RN, Image, Float32, "4096 x 4160", "e-", "Read noise data."
        RN_ERR, Image, Float32, "4096 x 4160", "e-", "Read noise error"
        DQ, Image, UInt32, "4096 x 4160", None, "Data quality."
    """
    EXTENSIONS = {
        f'RN': IMAGE_FLOAT32_HDU,
        f'RN_ERR': IMAGE_FLOAT32_HDU,
        **DQ_L0
    }
    CAL_TYPE = 'rn'
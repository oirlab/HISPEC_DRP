from .base_model import CalibrationModel
from .extensions import IMAGE_FLOAT32_HDU, DQ_L0

__all__ = ['BiasModel']


class BiasModel(CalibrationModel):
    """
    The bias is (to be updated)

    # NOTE: Also change image sizes to include ref channel for raw and level 0.

    .. csv-table::
        :header: "Hdu Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

        BIAS, Image, Float32, "4096 x 4160", "DN", "Bias level."
        BIAS_ERR, Image, Float32, "4096 x 4160", "DN", "Bias error."
        BIAS_KTC, Image, Float32, "4096 x 4160", "DN", "Bias kTC noise level."
        DQ, Image, UInt32, "4096 x 4160", None, "Data quality."
    """
    EXTENSIONS = {
        f'BIAS': IMAGE_FLOAT32_HDU,
        f'KTC': IMAGE_FLOAT32_HDU,
        **DQ_L0
    }
    CAL_TYPE = 'bias'
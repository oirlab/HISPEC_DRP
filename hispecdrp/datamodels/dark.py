from .base_model import CalibrationModel
from .extensions import IMAGE_FLOAT32_HDU,DQ_L0

__all__ = ['DarkModel']


class DarkModel(CalibrationModel):
    """
    The dark current (and any other contribution to the dark) is assumed to be linear with time and is stored as a rate map product (e-/s).

    .. csv-table::
        :header: "Hdu Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

        DARK_RATE, Image, Float32, "4096 x 4160", "DN", "Dark current data."
        DARK_ERR, Image, Float32, "4096 x 4160", "DN", "Dark current error (all sources)."
        DARK_VAR_POISSON, Image, Float32, "4096 x 4160", "DN^2", "Dark current variance (Poisson)."
        DARK_VAR_RNOISE, Image, Float32, "4096 x 4160", "DN^2", "Dark current variance (read noise)."
        DQ, Image, UInt32, "4096 x 4160", None, "Data quality."
    """
    EXTENSIONS = {
        f'RATE': IMAGE_FLOAT32_HDU,
        f'RATE_ERR': IMAGE_FLOAT32_HDU,
        f'RATE_VAR_RNOISE': IMAGE_FLOAT32_HDU,
        f'RATE_VAR_POISSON': IMAGE_FLOAT32_HDU,
        **DQ_L0
    }
    CAL_TYPE = 'dark'
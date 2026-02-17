from .base_model import CalibrationModel

from .extensions import IMAGE_FLOAT32_HDU, DQ_L0

__all__ = ['GainModel']


class GainModel(CalibrationModel):
    """
    The detector gain specifies the conversion factor between electrons and data numbers (DN) for each pixel in the detector.

    .. csv-table::
        :header: "Hdu Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

        GAIN, Image, Float32, "4096 x 4160", "e-/DN", "Gain data."
        GAIN_ERR, Image, Float32, "4096 x 4160", "e-/DN", "Gain error (all sources)."
        GAIN_VAR_POISSON, Image, Float32, "4096 x 4160", "(e-/DN)^2", "Gain variance (Poisson)."
        GAIN_VAR_RNOISE, Image, Float32, "4096 x 4160", "(e-/DN)^2", "Gain variance (read noise)."
        DQ, Image, UInt32, "4096 x 4160", None, "Data quality."
    """
    EXTENSIONS = {
        f'GAIN': IMAGE_FLOAT32_HDU,
        f'RN': IMAGE_FLOAT32_HDU,
        f'COV': IMAGE_FLOAT32_HDU,
        **DQ_L0
    }
    CAL_TYPE = 'gain'
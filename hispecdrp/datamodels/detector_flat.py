from .base_model import CalibrationModel

from .extensions import IMAGE_FLOAT32_HDU, DQ_L0

__all__ = ['DetectorFlatModel']


class DetectorFlatModel(CalibrationModel):
    """
    The HISPEC detector flat is a 2D image that corrects the intrinsic pixel-to-pixel sensitivity of the detector.

    .. csv-table::
        :header: "HDU Name", "HDU Type", "Data Type", "Dimensions", "Units", "Description"

        DATA, Image, Float32, "4096 x 4160", "None (normalized)", "Flat data."
        ERR, Image, Float32, "4096 x 4160", None, "Flat field error (all sources)."
        VAR_POISSON, Image, Float32, "4096 x 4160", None, "Flat field variance (Poisson)."
        VAR_RNOISE, Image, Float32, "4096 x 4160", None, "Flat field variance (read noise)."
        DQ, Image, UInt32, "4096 x 4160", None, "Data quality."
    """
    EXTENSIONS = {
        f'FLAT': IMAGE_FLOAT32_HDU,
        f'FLAT_ERR': IMAGE_FLOAT32_HDU,
        f'FLAT_VAR_RNOISE': IMAGE_FLOAT32_HDU,
        f'FLAT_VAR_POISSON': IMAGE_FLOAT32_HDU,
        **DQ_L0
    }
    CAL_TYPE = 'detflat'
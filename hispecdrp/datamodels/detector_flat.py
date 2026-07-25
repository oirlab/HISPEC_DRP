from .base_model import CalibrationModel

from .extensions import IMAGE_FLOAT32_HDU, DQ_L0

__all__ = ['DetectorFlatModel']


class DetectorFlatModel(CalibrationModel):
    """
    The HISPEC detector flat is a 2D image that corrects the intrinsic pixel-to-pixel sensitivity of the detector.

    Extensions
    ----------

    HDU Name            HDU Type   Data Type   Dimensions     Units   Description
    --------            --------   ---------   ----------     -----   -----------
    FLAT                Image      Float32     4096 x 4160    DN      Flat field data.
    FLAT_ERR            Image      Float32     4096 x 4160    DN      Flat field error.
    FLAT_VAR_RNOISE     Image      Float32     4096 x 4160    DN^2    Flat field variance (read noise).
    FLAT_VAR_POISSON    Image      Float32     4096 x 4160    DN^2    Flat field variance (Poisson).
    DQ                  Image      UInt32      4096 x 4160    None    Data quality.
    """
    EXTENSIONS = {
        f'FLAT': IMAGE_FLOAT32_HDU,
        f'FLAT_ERR': IMAGE_FLOAT32_HDU,
        f'FLAT_VAR_RNOISE': IMAGE_FLOAT32_HDU,
        f'FLAT_VAR_POISSON': IMAGE_FLOAT32_HDU,
        **DQ_L0
    }
    CAL_TYPE = 'detflat'

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'DETFLAT')
        self.meta.set_default('target_name', self.meta.exposure_type)
        for fiber in range(1, self.num_fibers+1):
            self.meta.set_default(f'object_fiber{fiber}', 'BLOCKED')

    @property
    def primary_extension_name(self):
        return 'flat'
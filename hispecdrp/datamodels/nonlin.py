import numpy as np

from .base_model import CalibrationModel

from .extensions import IMAGE_HDU, DQ_L0

__all__ = ['NonlinearCorrectionModel']

class NonlinearCorrectionModel(CalibrationModel):
    """
    Specifies the nonlinearity of the detector readout response with a unique polynomial for each pixel.

    Extensions
    ----------

    HDU Name         HDU Type   Data Type   Dimensions              Units   Description
    --------         --------   ---------   ----------              -----   -----------
    COEFFS           Image      Float64     4096 x 4160 x Ncoeff   None    Polynomial coeffs describing the nonlinearity.
    NONLIN_THRESH    Image      UInt16      4096 x 4160             None    TBD, from JWST.
    NONLIN_MAX       Image      UInt16      4096 x 4160             None    TBD, from JWST.
    """
    EXTENSIONS = {
        'COEFFS': IMAGE_HDU(np.float32, ndim=3),
        'NONLIN_THRESH' : IMAGE_HDU(np.uint16, ndim=2, default=0.9 * np.iinfo(np.uint16).max),
        'NONLIN_MAX' : IMAGE_HDU(np.uint16, ndim=2, default=np.iinfo(np.uint16).max),
        **DQ_L0
    }
    CAL_TYPE = 'nonlin'
    _USES_DETECTOR_SIZE = True

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'DETFLAT')
        self.meta.set_default('target_name', 'WHITELIGHT')
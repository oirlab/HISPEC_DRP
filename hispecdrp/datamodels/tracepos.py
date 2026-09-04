from .base_model import CalibrationModel
from .mixins import FiberAttributeProxyMixin
from .extensions import IMAGE_HDU

import numpy as np

__all__ = ['TracePosModel']


class TracePosModel(FiberAttributeProxyMixin, CalibrationModel):
    """
    The trace positions are determined from the fiber flat data. They are stored as polynomial coeffs and evaluated positions for each order/fiber.

    Extensions
    ----------

    HDU Name          HDU Type   Data Type   Dimensions       Units     Description
    --------          --------   ---------   ----------       -----     -----------
    COEFF_POS_FIB#    Image      Float64     Nord x Ncoeff    None      Polynomial coefficients for the trace positions of fiber #.
    COEFF_FWHM_FIB#   Image      Float64     Nord x Ncoeff    None      Polynomial coefficients for the trace widths (FWHM) of fiber #.
    POS_FIB#          Image      Float64     Nord x Ncoeff    Pixels    Evaluated positions for fiber #.
    FWHM_FIB#         Image      Float64     Nord x Ncoeff    Pixels    Evaluated fwhm for fiber #.
    """

    EXTENSIONS = {
        'COEFF_POS': IMAGE_HDU(np.float64, ndim=2, per_fiber=True),
        'COEFF_FWHM': IMAGE_HDU(np.float64, ndim=2, per_fiber=True),
        'POS': IMAGE_HDU(np.float64, ndim=2, per_fiber=True),
        'FWHM': IMAGE_HDU(np.float64, ndim=2, per_fiber=True),
    }

    CAL_TYPE = 'tracepos'

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'SPEC')
        self.meta.set_default('target_name', 'WHITELIGHT')

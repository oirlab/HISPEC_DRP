from .base_model import CalibrationModel
from .mixins import FiberAttributeProxyMixin
from .extensions import IMAGE_HDU

import numpy as np

__all__ = ['TracePosModel']


class TracePosModel(FiberAttributeProxyMixin, CalibrationModel):
    """
    The trace positions are determined from the fiber flat data. They are stored as polynomial coeffs and evaluated positions for each order/fiber.
    """

    EXTENSIONS = {
        'COEFF_POS': IMAGE_HDU(np.float64, shape=('N_ORDERS', 'ANY'), enumerate='FIB', title='Polynomial coefficients for the trace positions of fiber #.'),
        'COEFF_FWHM': IMAGE_HDU(np.float64, shape=('N_ORDERS', 'ANY'), enumerate='FIB', title='Polynomial coefficients for the trace widths (FWHM) of fiber #.'),
        'POS': IMAGE_HDU(np.float32, shape=('N_ORDERS', 'ANY'), enumerate='FIB', title='Evaluated positions for fiber #.', units='Pixels'),
        'FWHM': IMAGE_HDU(np.float32, shape=('N_ORDERS', 'ANY'), enumerate='FIB', title='Evaluated fwhm for fiber #.', units='Pixels'),
    }

    CAL_TYPE = 'tracepos'

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'SPEC')
        self.meta.set_default('target_name', 'WHITELIGHT')

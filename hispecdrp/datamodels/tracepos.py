from .base_model import CalibrationModel
from .mixins import FiberAttributeProxyMixin

from astropy.io import fits
import numpy as np

__all__ = ['TracePosModel']


def _create_tracepos_extensions(num_fibers: int):
    ext = {}
    for fiber in range(1, num_fibers + 1):
        ext[f'COEFF_POS_FIB{fiber}'] = {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        }
        ext[f'COEFF_FWHM_FIB{fiber}'] = {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        }
        ext[f'POS_FIB{fiber}'] = {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        }
        ext[f'FWHM_FIB{fiber}'] = {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        }
    return ext


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
    CAL_TYPE = 'tracepos'

    def _get_extensions(self):
        ext = self.EXTENSIONS.copy()
        ext.update(_create_tracepos_extensions(self.config.num_fibers))
        return ext
    
    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'SPEC')
        self.meta.set_default('target_name', 'WHITELIGHT')
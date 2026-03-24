from hispecdrp.datamodels.traceprof import _create_traceprof_extensions
from .base_model import CalibrationModel

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


class TracePosModel(CalibrationModel):
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
    def __init__(self, *args, **kwargs):
        config = kwargs.get('config', None)
        if config is None:
            config = self._get_instrument_config(
                meta=kwargs.get('meta', None),
                filepath=kwargs.get('filepath', None),
                instrument_name=kwargs.get('instrument_name', None),
            )
        self.EXTENSIONS = self.EXTENSIONS.copy()
        self.EXTENSIONS.update(_create_tracepos_extensions(config.num_fibers))
        super().__init__(*args, **kwargs)
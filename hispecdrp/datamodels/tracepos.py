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
    CAL_TYPE = 'tracepos'
    def __init__(self, *args, **kwargs):
        if 'config' in kwargs and kwargs['config'] is not None:
            self.EXTENSIONS.update(_create_tracepos_extensions(kwargs['config'].num_fibers))
        else:
            self.EXTENSIONS.update(_create_tracepos_extensions(self.DEFAULT_CONFIG.num_fibers))
        super().__init__(*args, **kwargs)
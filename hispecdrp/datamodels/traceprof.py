from .base_model import CalibrationModel

from astropy.io import fits
import numpy as np

__all__ = ['TraceProfModel']


def _create_traceprof_extensions(num_fibers: int):
    ext = {}
    for fiber in range(1, num_fibers + 1):
        ext[f'PROF_FIB{fiber}'] = {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        }
        ext[f'PROF_ERR_FIB{fiber}'] = {
            'hdu_type' : fits.ImageHDU,
            'dtype' : np.float64
        }
    return ext


class TraceProfModel(CalibrationModel):
    CAL_TYPE = 'traceprof'
    def __init__(self, *args, **kwargs):
        if 'config' in kwargs and kwargs['config'] is not None:
            self.EXTENSIONS.update(_create_traceprof_extensions(kwargs['config'].num_fibers))
        else:
            self.EXTENSIONS.update(_create_traceprof_extensions(self.DEFAULT_CONFIG.num_fibers))
        super().__init__(*args, **kwargs)
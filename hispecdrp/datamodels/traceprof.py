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
    """
    The trace profiles are determined from the fiber flat data. They are currently stored as-is with no modifications. The spectral extraction algorithm uses these weights appropriately in optimal extraction.

    Extensions
    ----------

    HDU Name        HDU Type   Data Type   Dimensions   Units   Description
    --------        --------   ---------   ----------   -----   -----------
    PROF_FIB#       Image      Float64     Ny x Nx      None    Profiles for fiber #.
    PROF_ERR_FIB#   Image      Float64     Ny x Nx      None    Profile errors for fiber #.
    """

    CAL_TYPE = 'traceprof'

    def _get_extensions(self):
        ext = self.EXTENSIONS.copy()
        ext.update(_create_traceprof_extensions(self.config.num_fibers))
        return ext
    
    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'SPEC')
        self.meta.set_default('target_name', 'WHITELIGHT')
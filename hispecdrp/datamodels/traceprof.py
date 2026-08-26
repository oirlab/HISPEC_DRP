from .base_model import CalibrationModel
from .mixins import FiberAttributeProxyMixin

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
        # Spectroperfection moments on the spectral-row grid (n_orders × n_pix).
        # SHEAR = ∂row/∂col of an iso-λ cut (from LFC spot tilt).
        # SIGMA_SPEC / SIGMA_SPAT = Gaussian 2nd moments (pixels).
        # MOM2_CROSS = flux-weighted covariance cov(row, col).
        ext[f'SHEAR_FIB{fiber}'] = {
            'hdu_type': fits.ImageHDU,
            'dtype': np.float64,
        }
        ext[f'SIGMA_SPEC_FIB{fiber}'] = {
            'hdu_type': fits.ImageHDU,
            'dtype': np.float64,
        }
        ext[f'SIGMA_SPAT_FIB{fiber}'] = {
            'hdu_type': fits.ImageHDU,
            'dtype': np.float64,
        }
        ext[f'MOM2_CROSS_FIB{fiber}'] = {
            'hdu_type': fits.ImageHDU,
            'dtype': np.float64,
        }
    return ext


class TraceProfModel(FiberAttributeProxyMixin, CalibrationModel):
    """
    The 2-D spatial profile comes from a fiberflat. Spectroperfection extraction
    also stores iso-λ shear and 2nd-moment LSF/spatial widths measured from
    LFC spots (plus fiberflat spatial moments).

    Extensions
    ----------

    HDU Name           HDU Type   Data Type   Dimensions        Units   Description
    --------           --------   ---------   ----------        -----   -----------
    PROF_FIB#          Image      Float64     Ny x Nx           None    2-D spatial profile (fiberflat).
    PROF_ERR_FIB#      Image      Float64     Ny x Nx           None    Profile errors for fiber #.
    SHEAR_FIB#         Image      Float64     n_orders x n_pix  None    Iso-λ shear ∂row/∂col (LFC).
    SIGMA_SPEC_FIB#    Image      Float64     n_orders x n_pix  None    Spectral Gaussian σ (pix).
    SIGMA_SPAT_FIB#    Image      Float64     n_orders x n_pix  None    Spatial Gaussian σ (pix).
    MOM2_CROSS_FIB#    Image      Float64     n_orders x n_pix  None    cov(row, col) (pix²).
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
from .base_model import CalibrationModel
from .mixins import FiberAttributeProxyMixin
from .extensions import IMAGE_HDU

import numpy as np

__all__ = ['TraceProfModel']


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

    # NOTE: PROF/PROF_ERR are (Ny x Nx) while the SHEAR/SIGMA_*/MOM2_CROSS family is
    # (n_orders x n_pix) -- same ndim, different actual shape -- so none of these
    # declare a `default` (no auto-initialization).
    EXTENSIONS = {
        'PROF': IMAGE_HDU(np.float64, ndim=2, per_fiber=True),
        'PROF_ERR': IMAGE_HDU(np.float64, ndim=2, per_fiber=True),
        'SHEAR': IMAGE_HDU(np.float64, ndim=2, per_fiber=True),
        'SIGMA_SPEC': IMAGE_HDU(np.float64, ndim=2, per_fiber=True),
        'SIGMA_SPAT': IMAGE_HDU(np.float64, ndim=2, per_fiber=True),
        'MOM2_CROSS': IMAGE_HDU(np.float64, ndim=2, per_fiber=True),
    }

    CAL_TYPE = 'traceprof'

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'SPEC')
        self.meta.set_default('target_name', 'WHITELIGHT')

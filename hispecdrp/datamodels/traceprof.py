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
    """

    # NOTE: PROF/PROF_ERR are (Ny x Nx), an ambiguous per-fiber stamp size with
    # no established symbol -> ANY for both axes. The SHEAR/SIGMA_*/MOM2_CROSS
    # family is (n_orders x n_pix), matching the same convention as the
    # spectral extensions (Nord x 4096) -> (N_ORDERS, DETECTOR_SIZE_Y). None of
    # these declare a `default` (no auto-initialization either way).
    EXTENSIONS = {
        'PROF': IMAGE_HDU(np.float32, shape=('DETECTOR_SIZE_Y', 'DETECTOR_SIZE_X'), enumerate='FIB', title='2-D spatial profile (fiberflat).', default=0),
        'PROF_ERR': IMAGE_HDU(np.float32, shape=('DETECTOR_SIZE_Y', 'DETECTOR_SIZE_X'), enumerate='FIB', title='Profile errors for fiber #.',  default=0),
        'SHEAR': IMAGE_HDU(np.float32, shape=('N_ORDERS', 'DETECTOR_SIZE_Y'), enumerate='FIB', title='Iso-λ shear ∂row/∂col (LFC).',  default=0),
        'SIGMA_SPEC': IMAGE_HDU(np.float32, shape=('N_ORDERS', 'DETECTOR_SIZE_Y'), enumerate='FIB', title='Spectral Gaussian σ (pix).', default=0),
        'SIGMA_SPAT': IMAGE_HDU(np.float32, shape=('N_ORDERS', 'DETECTOR_SIZE_Y'), enumerate='FIB', title='Spatial Gaussian σ (pix).', default=0),
        'MOM2_CROSS': IMAGE_HDU(np.float32, shape=('N_ORDERS', 'DETECTOR_SIZE_Y'), enumerate='FIB', title='cov(row, col) (pix²).',  default=0),
    }

    CAL_TYPE = 'traceprof'

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'SPEC')
        self.meta.set_default('target_name', 'WHITELIGHT')

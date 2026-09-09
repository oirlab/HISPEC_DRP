import numpy as np

from .base_model import CalibrationModel
from .mixins import FiberAttributeProxyMixin
from .extensions import IMAGE_HDU, create_dataspec_extensions, create_specdq_extensions, create_wave_extensions, create_chebyshev_extensions, create_lfcfitparam_extensions

__all__ = ['GaussianLSFModel']


class GaussianLSFModel(FiberAttributeProxyMixin, CalibrationModel):
    """
    The effective line spread function (LSF) is modeled with a Gaussian where the width is described with a 2D polynomial (pixel, order, width).
    """

    EXTENSIONS = {
        **create_dataspec_extensions(title='LFC spectrum.', units='e-/s'),
        **create_specdq_extensions(title='LFC spectrum data quality.'),
        **create_wave_extensions(title='LFC spectrum wavelength grid.', units='nm'),
        # No default given -> not auto-initialized (behavior unchanged), though
        # the shape (Nord x 4096 x 3: mu, sigma, offset) is now fully resolvable.
        'LSF_PARAMS': IMAGE_HDU(np.float32, shape=('N_ORDERS', 'DETECTOR_SIZE_Y', 3), enumerate='FIB', title='LFC spectrum Gaussian LSF parameters (mu, sigma, offset).'),
        **create_chebyshev_extensions(),
        **create_lfcfitparam_extensions(),
    }

    CAL_TYPE = 'lsf'

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'SPEC')

    @property
    def primary_extension_name(self):
        for fiber in range(1, self.config.num_fibers + 1):
            if self.has_extension_value(f'spec_fib{fiber}'):
                return f'spec_fib{fiber}'
        return None

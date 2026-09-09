from .base_model import CalibrationModel
from .mixins import FiberAttributeProxyMixin
from .extensions import create_dataspec_extensions, create_specdq_extensions, create_wave_extensions, create_midtime_extensions, create_trans_extensions

__all__ = ['FiberflatModel']


class FiberflatModel(FiberAttributeProxyMixin, CalibrationModel):
    """
    The fiber flat model contains the relative transmission of each fiber as a function of wavelength.
    """

    EXTENSIONS = {
        **create_dataspec_extensions(prefix='SPEC', title='Extracted fiber spectrum.', units='e-/s'),
        **create_specdq_extensions(),
        **create_wave_extensions(title='Wavelength solution.'),
        **create_midtime_extensions(),
        **create_trans_extensions(title='Normalized transmission.'),
    }

    CAL_TYPE = 'fiberflat'

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'SPEC')
        self.meta.set_default('target_name', 'WHITELIGHT')
        for fiber in range(1, self.num_fibers + 1):
            self.meta.set_default(f'object_fiber{fiber}', 'WHITELIGHT')

    @property
    def primary_extension_name(self):
        for fiber in range(1, self.config.num_fibers + 1):
            if self.has_extension_value(f'spec_fib{fiber}'):
                return f'spec_fib{fiber}'
        return None

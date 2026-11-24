from .base_model import CalibrationModel
from .mixins import FiberAttributeProxyMixin
from .extensions import create_dataspec_extensions, create_specdq_extensions, create_wave_extensions, create_chebyshev_extensions, create_uraniumfitparam_extensions

__all__ = ['UraniumWavesolModel']


class UraniumWavesolModel(FiberAttributeProxyMixin, CalibrationModel):
    """
    The Uranium (+Ar/Ne) lamp wavelength solution.
    """

    EXTENSIONS = {
        **create_dataspec_extensions(prefix='SPEC', title='Uranium spectrum.', units='e-/s'),
        **create_specdq_extensions(title='Uranium spectrum data quality.'),
        **create_wave_extensions(title='Uranium spectrum wavelength grid.'),
        **create_chebyshev_extensions(norm=False),
        **create_uraniumfitparam_extensions(),
    }

    CAL_TYPE = 'uranium_wavesol'

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'SPEC')
        if self.meta.target_name is None:
            spectrograph = self.meta.spectrograph
            if spectrograph is not None:
                self.meta.target_name = f'uranium-{spectrograph[0].upper()}'

    @property
    def primary_extension_name(self):
        for fiber in range(1, self.config.num_fibers + 1):
            if self.has_extension_value(f'spec_fib{fiber}'):
                return f'spec_fib{fiber}'
            if self.has_extension_value(f'wave_fib{fiber}'):
                return f'wave_fib{fiber}'
        return None

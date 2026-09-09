from .base_model import CalibrationModel
from .mixins import FiberAttributeProxyMixin

from .extensions import create_dataspec_extensions, create_specdq_extensions, create_wave_extensions, create_etalonfitparam_extensions, create_drift_extensions, create_barycorr_extensions

__all__ = ['ReferenceEtalonModel']


class ReferenceEtalonModel(FiberAttributeProxyMixin, CalibrationModel):
    """
    Contains the reference etalon and corresponding parameters for wavelength drift corrections.
    """

    EXTENSIONS = {
        **create_dataspec_extensions(prefix='SPEC', title='Etalon spectrum.', units='e-/s'),
        **create_specdq_extensions(title='Etalon spectrum data quality.'),
        **create_barycorr_extensions(),
        **create_wave_extensions(title='Etalon spectrum wavelength grid.', units='micron'),
        **create_etalonfitparam_extensions(),
        **create_drift_extensions(),
    }

    CAL_TYPE = 'ref_etalon'

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'SPEC')

    @property
    def primary_extension_name(self):
        for fiber in range(1, self.config.num_fibers + 1):
            if self.has_extension_value(f'spec_fib{fiber}'):
                return f'spec_fib{fiber}'
            if self.has_extension_value(f'wave_fib{fiber}'):
                return f'wave_fib{fiber}'
        return None

from .base_model import CalibrationModel
from .mixins import FiberAttributeProxyMixin
from .extensions import create_dataspec_extensions, create_specdq_extensions, create_wave_extensions, create_lfcfitparam_extensions

__all__ = ['LFCFitParamsModel']


class LFCFitParamsModel(FiberAttributeProxyMixin, CalibrationModel):
    """
    Contains the best fit Gaussian parameters for each LFC mode.
    """

    EXTENSIONS = {
        **create_dataspec_extensions(prefix='SPEC', title='Corresponding LFC spectrum.', units='e-/s'),
        **create_specdq_extensions(title='LFC spectrum data quality.'),
        **create_wave_extensions(title='LFC spectrum wavelength grid.', units='nm'),
        **create_lfcfitparam_extensions(),
    }

    CAL_TYPE = 'lfc_fitparams'

    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'SPEC')

    @property
    def primary_extension_name(self):
        for fiber in range(1, self.config.num_fibers + 1):
            if self.has_extension_value(f'spec_fib{fiber}'):
                return f'spec_fib{fiber}'
        return None

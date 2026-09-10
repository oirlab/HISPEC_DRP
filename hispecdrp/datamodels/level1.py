from .base_model import DataModel
from .mixins import FiberAttributeProxyMixin
from .extensions import create_dataspec_extensions, create_specdq_extensions, create_wave_extensions, create_midtime_extensions, create_barycorr_extensions

__all__ = ['L1Model']

class L1Model(FiberAttributeProxyMixin, DataModel):
    """
    HISPEC level 1 (L1) data products consist of 1D spectra immediately following spectral extraction. These spectra include a wavelength solution derived from the most recent wavelength calibration available in the calibration database (likely from the LFC). However, they do not yet incorporate the simultaneous etalon data to account for any wavelength drift or any additional corrections.
    """

    EXTENSIONS = {
        **create_dataspec_extensions(prefix='SPEC_UNCAL', title='Extracted 1D spectra.', units='e-/s'),
        **create_specdq_extensions(prefix='SPEC_UNCAL', title='Uncalibrated spectrum data-quality flags.'),
        **create_wave_extensions(title='Wavelength solution.'),
        **create_midtime_extensions(),
        **create_barycorr_extensions(),
    }

    @property
    def data_level_suffix(self):
        return 'specuncal'

    @property
    def primary_extension_name(self):
        for fiber in range(1, self.config.num_fibers + 1):
            if self.has_extension_value(f'spec_uncal_fib{fiber}'):
                return f'spec_uncal_fib{fiber}'
        return None

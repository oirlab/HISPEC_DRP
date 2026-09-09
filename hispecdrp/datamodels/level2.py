
from .base_model import DataModel
from .mixins import FiberAttributeProxyMixin
from .extensions import create_dataspec_extensions, create_specdq_extensions, create_wave_extensions, create_midtime_extensions, create_barycorr_extensions, create_starmodel_extensions, create_throughput_extensions, create_rv_extensions, create_sky_extensions, create_telluric_extensions, create_trans_extensions, create_drift_extensions, create_etalonfitparam_extensions, create_fluxabs_extensions

__all__ = ['L2Model']

class L2Model(FiberAttributeProxyMixin, DataModel):
    """
    HISPEC Level 2 (L2) data products are fully calibrated science-level data. These products retain the original Level 1 spectra to simplify tracking of data processing steps. Additionally, they include advanced calibrations such as sky subtraction, blaze correction, telluric correction, and stellar radial velocity (RV) measurements. Data products from BSPEC and RSPEC are still kept separate.
    """

    EXTENSIONS = {
        **create_dataspec_extensions(prefix='SPEC_UNCAL', title='L1 spectrum.', units='e-/s'),
        **create_specdq_extensions(prefix='SPEC_UNCAL', title='Uncalibrated spectrum data-quality flags.'),
        **create_dataspec_extensions(prefix='SPEC', title='Calibrated spectrum.', units='e-/s'),
        **create_specdq_extensions(title='Calibrated spectrum data-quality flags.'),
        **create_wave_extensions(title='Drift corrected wavelength solution in rest frame of the instrument.', units='micron'),
        **create_drift_extensions(),
        **create_midtime_extensions(title='Flux-weighted exposure mid-point times.'),
        **create_barycorr_extensions(),
        **create_starmodel_extensions(title='Stellar model spectrum Doppler shifted according to Barycenter OR RVs.', units='W/m^2/micron'),
        **create_throughput_extensions(title='End-to-end throughput derived from STAR_MOD.'),
        **create_rv_extensions(),
        **create_sky_extensions(title='Sky spectrum or model.', units='e-/s'),
        **create_telluric_extensions(),
        **create_trans_extensions(),
        **create_etalonfitparam_extensions(),
        **create_fluxabs_extensions(),
    }

    @property
    def data_level_suffix(self):
        return 'spec'

    @property
    def primary_extension_name(self):
        for fiber in range(1, self.config.num_fibers + 1):
            if self.has_extension_value(f'spec_fib{fiber}'):
                return f'spec_fib{fiber}'
            if self.has_extension_value(f'spec_uncal_fib{fiber}'):
                return f'spec_uncal_fib{fiber}'
        return None

    @classmethod
    def from_datamodels(cls, *args, **kwargs) -> 'DataModel':

        # Initialize the L2 model
        model_out = super().from_datamodels(*args, **kwargs)

        # Copy the uncal spectrum into the calibrated spectrum
        for fiber in range(1, model_out.num_fibers + 1):
            if model_out.has_extension_value(f'spec_uncal_fib{fiber}'):
                model_out[fiber].spec = model_out[fiber].spec_uncal.copy()
                model_out[fiber].spec_err = model_out[fiber].spec_uncal_err.copy()
                model_out[fiber].spec_var_rnoise = model_out[fiber].spec_uncal_var_rnoise.copy()
                model_out[fiber].spec_var_poisson = model_out[fiber].spec_uncal_var_poisson.copy()
                model_out[fiber].spec_dq = model_out[fiber].spec_uncal_dq.copy()

        return model_out
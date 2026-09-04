from .base_model import CalibrationModel
from .mixins import FiberAttributeProxyMixin
from .extensions import create_dataspec_extensions, create_specdq_extensions, create_wave_extensions, create_lfcfitparam_extensions

__all__ = ['LFCFitParamsModel']


class LFCFitParamsModel(FiberAttributeProxyMixin, CalibrationModel):
    """
    Contains the best fit Gaussian parameters for each LFC mode.

    Extensions
    ----------

    HDU Name                  HDU Type   Data Type   Dimensions      Units      Description
    --------                  --------   ---------   ----------      -----      -----------
    SPEC_FIB#                 Image      Float32     Nord x 4096     e-/s       Corresponding LFC spectrum.
    SPEC_ERR_FIB#             Image      Float32     Nord x 4096     e-/s       LFC spectrum error.
    SPEC_VAR_POISSON_FIB#     Image      Float32     Nord x 4096     e-/s       LFC spectrum Poisson variance.
    SPEC_VAR_RNOISE_FIB#      Image      Float32     Nord x 4096     e-/s       LFC spectrum read noise variance.
    SPEC_DQ_FIB#              Image      UInt32      Nord x 4096     None       LFC spectrum data quality.
    WAVE_FIB#                 Image      Float64     Nord x 4096     nm         LFC spectrum wavelength grid.
    LFC_FITPARAMS_FIB#        Table      See below   See below       See below  LFC fit parameters for each LFC mode, order, and fiber.
    """

    EXTENSIONS = {
        **create_dataspec_extensions(prefix='SPEC'),
        **create_specdq_extensions(),
        **create_wave_extensions(),
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

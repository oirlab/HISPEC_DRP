from .base_model import CalibrationModel
from .mixins import FiberAttributeProxyMixin
from .extensions import IMAGE_FLOAT32_HDU, create_dataspec_extensions, create_specdq_extensions, create_wave_extensions, create_chebyshev_extensions, create_lfcfitparam_extensions

__all__ = ['GaussianLSFModel']


def _create_gausslsf_extensions(num_fibers : int):
    ext = {}
    for fiber in range(1, num_fibers + 1):
        ext.update(create_dataspec_extensions(fiber=fiber))
        ext.update(create_specdq_extensions(fiber=fiber))
        ext.update(create_wave_extensions(fiber=fiber))
        ext[f'LSF_PARAMS_FIB{fiber}'] = IMAGE_FLOAT32_HDU # Actual LSF Parameters sampled on detector pixels. This might change.
        ext.update(create_chebyshev_extensions(fiber=fiber))
        ext.update(create_lfcfitparam_extensions(fiber=fiber))
    return ext


class GaussianLSFModel(FiberAttributeProxyMixin, CalibrationModel):
    """
    The effective line spread function (LSF) is modeled with a Gaussian where the width is described with a 2D polynomial (pixel, order, width).

    Extensions
    ----------

    HDU Name                 HDU Type   Data Type   Dimensions           Units   Description
    --------                 --------   ---------   ----------           -----   -----------
    SPEC_FIB#                Image      Float32     Nord x 4096          e-/s    Corresponding LFC spectrum
    SPEC_ERR_FIB#            Image      Float32     Nord x 4096          e-/s    LFC spectrum error.
    SPEC_VAR_POISSON_FIB#    Image      Float32     Nord x 4096          e-/s    LFC spectrum Poisson variance
    SPEC_VAR_RNOISE_FIB#     Image      Float32     Nord x 4096          e-/s    LFC spectrum read noise variance
    SPEC_DQ_FIB#             Image      UInt32      Nord x 4096          None    LFC spectrum data quality
    WAVE_FIB#                Image      Float64     Nord x 4096          nm      LFC spectrum wavelength grid
    LFC_FITPARAMS_FIB#       Table      See docs    See docs             See docs LFC fit parameters for each LFC mode, order, and fiber
    LSF_PARAMS_FIB#          Image      Float64     Nord x 4096 x 3      None    LFC spectrum Gaussian LSF parameters (mu, sigma, offset)
    """
    
    CAL_TYPE = 'lsf'

    def _get_extensions(self):
        ext = self.EXTENSIONS.copy()
        ext.update(_create_gausslsf_extensions(self.config.num_fibers))
        return ext
    
    def on_init(self):
        super().on_init()
        self.meta.set_default('exposure_type', 'SPEC')
        
    @property
    def primary_extension_name(self):
        for fiber in range(1, self.config.num_fibers + 1):
            if self.has_extension_value(f'spec_fib{fiber}'):
                return f'spec_fib{fiber}'
        return None
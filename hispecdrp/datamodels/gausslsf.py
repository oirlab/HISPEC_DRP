from astropy.io import fits
from .base_model import CalibrationModel

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


class GaussianLSFModel(CalibrationModel):
    def __init__(self, *args, **kwargs):
        if 'config' in kwargs and kwargs['config'] is not None:
            self.EXTENSIONS.update(_create_gausslsf_extensions(kwargs['config'].num_fibers))
        else:
            self.EXTENSIONS.update(_create_gausslsf_extensions(self.DEFAULT_CONFIG.num_fibers))
        super().__init__(*args, **kwargs)
    CAL_TYPE = 'lsf'
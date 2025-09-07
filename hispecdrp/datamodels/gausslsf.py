from astropy.io import fits
from .base_model import CalibrationModel
from .hispec_base import HISPECCalibrationModel
from .parvi_base import PARVICalibrationModel

from .extensions import IMAGE_FLOAT32_HDU, create_dataspec_extensions, create_specdq_extensions, create_wave_extensions, create_chebyshev_extensions, create_lfcfitparam_extensions

__all__ = ['HISPECGaussianLSFModel', 'PARVIGaussianLSFModel']


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
    CAL_TYPE = 'lsf'


class HISPECGaussianLSFModel(HISPECCalibrationModel, GaussianLSFModel):
    EXTENSIONS = _create_gausslsf_extensions(num_fibers=HISPECCalibrationModel.NUM_FIBERS)


class PARVIGaussianLSFModel(PARVICalibrationModel, GaussianLSFModel):
    EXTENSIONS = _create_gausslsf_extensions(num_fibers=PARVICalibrationModel.NUM_FIBERS)
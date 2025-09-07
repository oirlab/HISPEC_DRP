from .base_model import CalibrationModel
from .hispec_base import HISPECCalibrationModel
from .parvi_base import PARVICalibrationModel

from .extensions import create_dataspec_extensions, create_specdq_extensions, create_wave_extensions, create_lfcfitparam_extensions

__all__ = ['HISPECLFCFitParamsModel', 'PARVILFCFitParamsModel']


def _create_lfcfitparams_extensions(num_fibers: int):
    ext = {}
    for fiber in range(1, num_fibers + 1):
        ext.update(create_dataspec_extensions(fiber=fiber, prefix='SPEC'))
        ext.update(create_specdq_extensions(fiber=fiber))
        ext.update(create_wave_extensions(fiber=fiber))
        ext.update(create_lfcfitparam_extensions(fiber=fiber))
    return ext


class LFCFitParamsModel(CalibrationModel):
    CAL_TYPE = 'lfc_fitparams'


class HISPECLFCFitParamsModel(HISPECCalibrationModel, LFCFitParamsModel):
    EXTENSIONS = _create_lfcfitparams_extensions(num_fibers=HISPECCalibrationModel.NUM_FIBERS)


class PARVILFCFitParamsModel(PARVICalibrationModel, LFCFitParamsModel):
    EXTENSIONS = _create_lfcfitparams_extensions(num_fibers=PARVICalibrationModel.NUM_FIBERS)
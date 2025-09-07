from .base_model import CalibrationModel
from .hispec_base import HISPECCalibrationModel
from .parvi_base import PARVICalibrationModel

from .extensions import create_dataspec_extensions, create_specdq_extensions, create_wave_extensions, create_etalonfitparam_extensions, create_drift_extensions, create_barycorr_extensions

__all__ = ['ReferenceEtalonModel', 'HISPECReferenceEtalonModel', 'PARVIReferenceEtalonModel']

def _create_refetalon_extensions(num_fibers: int):
    ext = {}
    for fiber in range(1, num_fibers + 1):
        ext.update(create_dataspec_extensions(fiber=fiber, prefix='SPEC'))
        ext.update(create_specdq_extensions(fiber=fiber))
        ext.update(create_barycorr_extensions(fiber=fiber))
        ext.update(create_wave_extensions(fiber=fiber))
        ext.update(create_etalonfitparam_extensions(fiber=fiber))
        ext.update(create_drift_extensions(fiber=fiber))
    return ext


class ReferenceEtalonModel(CalibrationModel):
    CAL_TYPE = 'ref_etalon'


class HISPECReferenceEtalonModel(HISPECCalibrationModel, ReferenceEtalonModel):
    EXTENSIONS = _create_refetalon_extensions(num_fibers=HISPECCalibrationModel.NUM_FIBERS)


class PARVIReferenceEtalonModel(PARVICalibrationModel, ReferenceEtalonModel):
    EXTENSIONS = _create_refetalon_extensions(num_fibers=PARVICalibrationModel.NUM_FIBERS)

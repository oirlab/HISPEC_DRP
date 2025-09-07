from .base_model import CalibrationModel
from .hispec_base import HISPECCalibrationModel
from .parvi_base import PARVICalibrationModel

from .extensions import create_L0_extensions

__all__ = ['HISPECDetectorFlatModel', 'PARVIDetectorFlatModel']


class DetectorFlatModel(CalibrationModel):
    EXTENSIONS = create_L0_extensions(name='DATA')
    CAL_TYPE = 'detflat'


class HISPECDetectorFlatModel(HISPECCalibrationModel, DetectorFlatModel):
    pass


class PARVIDetectorFlatModel(PARVICalibrationModel, DetectorFlatModel):
    pass
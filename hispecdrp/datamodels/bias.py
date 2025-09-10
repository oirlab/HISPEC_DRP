from .base_model import CalibrationModel
from .hispec_base import HISPECCalibrationModel
from .parvi_base import PARVICalibrationModel

from .extensions import create_L0_extensions

__all__ = ['BiasModel', 'HISPECBiasModel', 'PARVIBiasModel']


class BiasModel(CalibrationModel):
    EXTENSIONS = create_L0_extensions(name='DATA')
    CAL_TYPE = 'bias'


class HISPECBiasModel(HISPECCalibrationModel, BiasModel):
    pass


class PARVIBiasModel(PARVICalibrationModel, BiasModel):
    pass
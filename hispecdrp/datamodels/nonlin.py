from .base_model import CalibrationModel
from .hispec_base import HISPECCalibrationModel
from .parvi_base import PARVICalibrationModel

from .extensions import IMAGE_FLOAT32_HDU, IMAGE_UINT16_HDU, DQ_L0

__all__ = ['NonlinearCorrectionModel', 'HISPECNonlinearCorrectionModel', 'PARVINonlinearCorrectionModel']

class NonlinearCorrectionModel(CalibrationModel):
    EXTENSIONS = {
        'COEFFS': IMAGE_FLOAT32_HDU,
        'NONLIN_THRESH' : IMAGE_UINT16_HDU,
        'NONLIN_MAX' : IMAGE_UINT16_HDU,
        **DQ_L0
    }
    CAL_TYPE = 'nonlin'


class HISPECNonlinearCorrectionModel(HISPECCalibrationModel, NonlinearCorrectionModel):
    pass


class PARVINonlinearCorrectionModel(PARVICalibrationModel, NonlinearCorrectionModel):
    pass
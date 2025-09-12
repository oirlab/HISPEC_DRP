from .base_model import CalibrationModel
from .hispec_base import HISPECCalibrationModel
from .parvi_base import PARVICalibrationModel

from .extensions import IMAGE_FLOAT32_HDU, IMAGE_UINT16_HDU

__all__ = ['NonlinearCorrectionModel', 'HISPECNonlinearCorrectionModel', 'PARVINonlinearCorrectionModel']

class NonlinearCorrectionModel(CalibrationModel):
    EXTENSIONS = {
        'COEFFS': IMAGE_FLOAT32_HDU,
        'THRESH' : IMAGE_UINT16_HDU,
        'MAX' : IMAGE_UINT16_HDU
    }
    CAL_TYPE = 'nonlin'


class HISPECNonlinearCorrectionModel(HISPECCalibrationModel, NonlinearCorrectionModel):
    pass


class PARVINonlinearCorrectionModel(PARVICalibrationModel, NonlinearCorrectionModel):
    pass
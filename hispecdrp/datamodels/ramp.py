from .base_model import DataModel
from .hispec_base import HISPECDataModel
from .parvi_base import PARVIDataModel
from .extensions import IMAGE_UINT16_HDU, DQ_RAW, READTIMES

__all__ = ['RampModel', 'HISPECRampModel', 'PARVIRampModel']


class RampModel(DataModel):
    EXTENSIONS = {'DATA': IMAGE_UINT16_HDU, **DQ_RAW, **READTIMES}

    @property
    def data_level_suffix(self) -> str:
        return "raw"
    

class HISPECRampModel(HISPECDataModel, RampModel):
    pass


class PARVIRampModel(PARVIDataModel, RampModel):
    pass
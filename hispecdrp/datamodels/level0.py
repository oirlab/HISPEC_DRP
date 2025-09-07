from .base_model import DataModel
from .hispec_base import HISPECDataModel
from .parvi_base import PARVIDataModel
from .extensions import create_L0_extensions

__all__ = ['HISPECL0Model', 'PARVIL0Model']

class L0Model(DataModel):

    EXTENSIONS = create_L0_extensions(name='DATA')

    @property
    def data_level_suffix(self):
        return 'rate'
    

class HISPECL0Model(HISPECDataModel, L0Model):
    pass

class PARVIL0Model(PARVIDataModel, L0Model):
    pass
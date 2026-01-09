from .base_model import DataModel
from .extensions import create_L0_extensions

__all__ = ['L0Model']

class L0Model(DataModel):

    EXTENSIONS = create_L0_extensions(name='RATE')

    @property
    def data_level_suffix(self):
        return 'rate'
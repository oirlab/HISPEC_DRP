from .meta_dict import MetaDict
from .base_model import DataModel

from .hispec_base import HISPECDataModel
from .hispec_rawread import HISPECRawModel
from .hispec_level0 import HISPECL0Model
from .hispec_level1 import HISPECL1Model
from .hispec_level2 import HISPECL2Model

from .utils import open

__all__ = [
    "open",
    "MetaDict",
    "DataModel",
    "HISPECDataModel", "HISPECRawModel", "HISPECL0Model", "HISPECL1Model", "HISPECL2Model"
]

_local_dict = locals()
_models = [
    "DataModel",
    "HISPECDataModel", "HISPECRawModel", "HISPECL0Model", "HISPECL1Model", "HISPECL2Model"
]
MODEL_REGISTER = {name : _local_dict[name] for name in _models}
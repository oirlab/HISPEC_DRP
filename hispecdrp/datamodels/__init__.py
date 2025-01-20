from .meta_dict import MetaDict
from .base_model import DataModel

from .hispec_base import HISPECDataModel
from .hispec_rawread import HISPECRawRead
from .hispec_level0 import HISPECL0
from .hispec_level1 import HISPECL1
from .hispec_level2 import HISPECL2

from .utils import open

__all__ = [
    "open",
    "MetaDict",
    "DataModel",
    "HISPECDataModel", "HISPECRawRead", "HISPECL0", "HISPECL1", "HISPECL2",
]

_local_dict = locals()
_models = [
    "DataModel",
    "HISPECDataModel", "HISPECRawRead", "HISPECL0", "HISPECL1", "HISPECL2"
]
MODEL_REGISTER = {name : _local_dict[name] for name in _models}
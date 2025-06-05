from .meta_dict import *
from .base_model import *
from .utils import *
from .create_ramp import *

from .hispec import *
from .parvi import *

_local_dict = locals()
MODEL_REGISTER = {name : _local_dict[name] for name in _local_dict if name.endswith('Model')}
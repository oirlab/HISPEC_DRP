# Metadata
from .meta_data import *
from .hispec_meta import *
from .parvi_meta import *

# DQ flags
from .dqflags import *

# Base model class
from .base_model import *

# Utils
from .utils import *
from .create_ramp import *

# Extensions
# from . import extensions

# Base instrument models
from .hispec_base import *
from .parvi_base import *

# Concrete models
from .ramp import *
from .level0 import *
from .level1 import *
from .level2 import *

# Calibration models
from .instrument_params import *
from .dark import *
from .detector_flat import *
from .nonlin import *
from .lfcfitparams import *
from .lfcwavesol import *
from .gausslsf import *
from .ref_etalon import *

# Registry of models
_local_dict = locals()
MODEL_REGISTER = {name : _local_dict[name] for name in _local_dict if name.endswith('Model')}

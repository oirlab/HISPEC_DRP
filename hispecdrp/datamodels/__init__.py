# Metadata
from .meta_data import *

# DQ flags
from .dqflags import *

# Base model class
from .base_model import *

# Utils
from .utils import *
from .create_ramp import *

# Extensions
# from . import extensions

# Concrete models
from .ramp import *
from .level0 import *
from .level1 import *
from .level2 import *

# Calibration models
from .instrument_params import *
from .dq import *
from .saturation import *
from .dark import *
from .bias import *
from .read_noise import *
from .detector_flat import *
from .detector_gain import *
from .nonlin import *
from .lfcfitparams import *
from .lfcwavesol import *
from .gausslsf import *
from .ref_etalon import *
from .tracepos import *
from .traceprof import *
from .fiberflat import *

# Instrument config
from .parvi_config import *
from .hispec_config import *
from .kpic_config import *
INSTRUMENT_CONFIGS = {
    'HISPEC' : HISPECConfig,
    'PARVI'  : PARVIConfig,
    'NIRSPAO' : KPICConfig
}

# Registry of models
_local_dict = locals()
MODEL_REGISTER = {name : _local_dict[name] for name in _local_dict if name.endswith('Model')}

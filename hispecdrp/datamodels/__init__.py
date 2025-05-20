from .meta_dict import MetaDict
from .base_model import DataModel


from .hispec_base import HISPECDataModel
from .hispec_ramp import HISPECRampModel
from .hispec_level0 import HISPECL0Model
from .hispec_level1 import HISPECL1Model
from .hispec_level2 import HISPECL2Model
from .hispec_calibration_base import HISPECCalibrationModel
from .hispec_dark import HISPECDarkModel
from .hispec_flat import HISPECFlatModel
from .hispec_nonlin import HISPECNonlinearCoeffModel
from .hispec_dqflags import DQ_FLAGS

from .utils import open

_local_dict = locals()
MODEL_REGISTER = {name : _local_dict[name] for name in _local_dict if name.endswith('Model')}

from .hispec_store import HISPECCalibrationStore
from .koa_sync import verify_koa_calibrations
from .selectors import *

from koa_middleware import CalibrationSelector

_local_dict = locals()
SELECTOR_REGISTER = {name : _local_dict[name] for name in _local_dict if isinstance(_local_dict[name], type) and issubclass(_local_dict[name], CalibrationSelector) and _local_dict[name] is not CalibrationSelector}
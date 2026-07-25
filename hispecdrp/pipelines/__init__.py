from .base_pipeline import BasePipeline
from .stage0 import *
from .stage1 import *
from .stage2 import *
from .calibration_pipeline_base import *
from .calibration import *

# Registry of pipelines
_local_dict = locals()
PIPELINE_REGISTER = {
    name : _local_dict[name]
    for name in _local_dict 
        if isinstance(_local_dict[name], type)
            and issubclass(_local_dict[name], BasePipeline)
            and _local_dict[name] not in (BasePipeline,)
}
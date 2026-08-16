from .base_primitive import BasePrimitive
from .dq_init import *
from .saturation import *
from .dark_subtraction import *
from .bias_subtraction import *
from .detector_flat import *
from .gain import *
from .readnoise import *
from .ramp_fitting import *
from .coadd import *
from .nonlinear_correction import *
from .trace_fitting import *
from .line_fitting import *
from .wavesol import *
from .lsf import *
from .register_calibration import *
from .jump_detection import *
from .spectral_extraction import *
from .blaze_correction import *
from .stellar_rv import *
from .refpix import *
from .barycenter_correction import *

# Registry of primitives
_local_dict = locals()
PRIMITIVE_REGISTER = {
    name : _local_dict[name]
    for name in _local_dict 
        if isinstance(_local_dict[name], type)
            and issubclass(_local_dict[name], BasePrimitive)
            and _local_dict[name] not in (BasePrimitive,)
}

# Exclude the ``utils`` submodule from ``from hispecdrp.primitives import *``
# so it cannot shadow ``hispecdrp.utils`` when re-exported from the top package.
__all__ = [name for name in list(globals()) if not name.startswith("_") and name != "utils"]

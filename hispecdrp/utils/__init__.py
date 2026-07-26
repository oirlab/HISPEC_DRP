from . import config
from . import logging
from .endian_utils import *
from .gdrive import *
from . import testing_data
from .parallelization_utils import *

# Spectral template helpers live in :mod:`hispecdrp.primitives.utils.spectral_utils`.
# Import them lazily so ``import hispecdrp.utils`` (used by datamodels during package
# bootstrap) does not pull in ``hispecdrp.primitives`` and create a circular import.


def __getattr__(name: str):
    if name in {
        "fetch_svo_2mass_filter",
        "get_spectrum_path",
        "load_btsettl_spectrum",
    }:
        from hispecdrp.primitives.utils.spectral_utils import (
            fetch_svo_2mass_filter,
            get_spectrum_path,
            load_btsettl_spectrum,
        )

        mapping = {
            "fetch_svo_2mass_filter": fetch_svo_2mass_filter,
            "get_spectrum_path": get_spectrum_path,
            "load_btsettl_spectrum": load_btsettl_spectrum,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

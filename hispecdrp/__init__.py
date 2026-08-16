from ._version import version as __version__

from .primitives import * # Import all primitives
from .pipelines import * # Import all pipelines
from . import datamodels
from . import calibrations
from . import keck

# ``from .primitives import *`` re-exports the ``utils`` submodule
# (``hispecdrp.primitives.utils``), which shadows this package's ``utils``.
# Drop the shadowed name so the real ``hispecdrp.utils`` package binds.
globals().pop("utils", None)
from . import utils

from .utils.logging import setup_default_logging
setup_default_logging()


def __getattr__(name: str):
    if name == "simulations":
        from . import simulations as _simulations
        return _simulations
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | {"simulations"})
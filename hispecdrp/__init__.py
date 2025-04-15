from .primitives import *
from .pipelines import *
from .datamodels import *
from .calibrations import *
from .keck import * # NOTE: Remove this import eventually


try:
    from setuptools_scm import get_version
    __version__ = get_version(root="..", relative_to=__file__)
except ImportError:
    __version__ = "unknown"
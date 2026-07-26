"""Compatibility shim for :mod:`hispecdrp.primitives.utils.spectral_utils`.

Template I/O and spectral conversion helpers now live in
:mod:`hispecdrp.primitives.utils.spectral_utils`. This module re-exports the
public API so older ``hispecdrp.utils.spectral_templates_utils`` imports keep
working.
"""

from hispecdrp.primitives.utils.spectral_utils import *  # noqa: F403
from hispecdrp.primitives.utils.spectral_utils import __all__  # noqa: F401

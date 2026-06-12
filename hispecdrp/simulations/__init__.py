from .utr import *

_LAZY_RATE_MAP_EXPORTS = frozenset(
    {
        "DEFAULT_RATE_MAP_FILENAME",
        "DEFAULT_PYECHELLE_FILENAME",
        "SimulateRateMap",
        "read_pyechelle_hdf",
    }
)


def __getattr__(name: str):
    if name in _LAZY_RATE_MAP_EXPORTS:
        from . import simulate_rate_map as _simulate_rate_map

        value = getattr(_simulate_rate_map, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _LAZY_RATE_MAP_EXPORTS)

"""
Shared helpers for parallel CPU work: shared-memory views, Numba thread limits, and
Numba decorators for the pipeline.

The HISPEC DRP stack depends on Numba at install time; this module imports it
eagerly. :func:`numba_thread_scope` and :func:`resolve_numba_threads` share logic
that probes Numba for the maximum allowed thread count (cached). On some CI runners
Numba caps parallel threads at 2 even when ``os.cpu_count()`` is higher; clamping
avoids ``ValueError`` from ``set_num_threads``.
"""
import ctypes
import os
import numpy as np
from contextlib import contextmanager

from numba import get_num_threads, njit, prange, set_num_threads

_CTYPES_TO_NP = {
    ctypes.c_float:   np.float32,
    ctypes.c_double:  np.float64,
    ctypes.c_int32:   np.int32,
    ctypes.c_uint32:  np.uint32,
    ctypes.c_int16:   np.int16,
    ctypes.c_uint16:  np.uint16,
    ctypes.c_int8:    np.int8,
    ctypes.c_uint8:   np.uint8,
}

__all__ = [
    "_arraytonumpy",
    "njit",
    "numba_thread_scope",
    "prange",
    "resolve_numba_threads",
]


def _arraytonumpy(shared_array, shape=None, dtype=None):
    """
    Covert a shared array to a numpy array

    Originally from pyklip Wang, J. J., Ruffio, J.-B., De Rosa, R. J., et al. 2015, ASCL, ascl:1506.001
    https://bitbucket.org/pyKLIP/pyklip/src/main/pyklip/parallelized.py

    Args:
        shared_array: a multiprocessing.Array array
        shape: a shape for the numpy array. otherwise, will assume a 1d array
        dtype: data type of the arrays. Should be either ctypes.c_float(default) or ctypes.c_double

    Returns:
        numpy_array: numpy array for vectorized operation. still points to the same memory!
                     returns None is shared_array is None
    """
    if dtype is None:
        dtype = ctypes.c_float
    np_dtype = _CTYPES_TO_NP.get(dtype)

    # if you passed in nothing you get nothing
    if shared_array is None:
        return None

    buf = shared_array.get_obj() if hasattr(shared_array, "get_obj") else shared_array
    numpy_array = np.frombuffer(buf, dtype=np_dtype)
    # numpy_array = np.frombuffer(shared_array.get_obj(), dtype=dtype)
    if shape is not None:
        numpy_array.shape = shape

    return numpy_array


# Cached cap: largest n accepted by set_num_threads on this build (CI often ≤ 2).
_NUMBA_MAX_PARALLEL_THREADS: int | None = None


def _numba_max_parallel_threads() -> int:
    """
    Probe and cache the maximum thread count Numba allows for ``prange`` / ``set_num_threads``.

    Some wheels and CI VMs reject ``set_num_threads(n)`` for n above a small cap (e.g. 2)
    even when ``os.cpu_count()`` is larger. Result is cached for the process.
    """
    global _NUMBA_MAX_PARALLEL_THREADS
    if _NUMBA_MAX_PARALLEL_THREADS is not None:
        return _NUMBA_MAX_PARALLEL_THREADS
    cpu = max(1, int(os.cpu_count() or 1))
    saved = int(get_num_threads())
    hi = min(cpu, 128)
    best = 1
    try:
        for candidate in range(hi, 0, -1):
            try:
                set_num_threads(candidate)
                best = candidate
                break
            except ValueError:
                continue
    finally:
        set_num_threads(saved)
    _NUMBA_MAX_PARALLEL_THREADS = max(1, best)
    return _NUMBA_MAX_PARALLEL_THREADS


def _effective_numba_threads(requested: int | None) -> int:
    """Map requested parallelism to a safe count (uses cached Numba max-thread probe)."""
    cap = _numba_max_parallel_threads()
    cpu = max(1, int(os.cpu_count() or 1))
    if requested is None or requested == 0:
        return max(1, min(cpu, cap))
    return max(1, min(int(requested), cap))


def resolve_numba_threads(max_cores: int | None) -> int:
    """
    Map a requested parallelism level to a thread count safe for this host and Numba build.

    Uses the same maximum-thread probe and clamping as :func:`numba_thread_scope`.

    - ``None`` or ``0``: use all logical CPUs, capped to Numba's maximum.
    - Positive ``max_cores``: use at most that many threads, capped the same way.

    This prevents ``ValueError`` from ``set_num_threads`` on runners where Numba allows
    fewer threads than ``os.cpu_count()`` (e.g. GitHub Actions).
    """
    return _effective_numba_threads(max_cores)


@contextmanager
def numba_thread_scope(num_threads: int | None = None):
    """
    Temporarily set Numba's thread pool, then restore the previous setting.

    On entry, this uses the same logic as :func:`resolve_numba_threads`: it determines
    the **maximum number of threads Numba allows** on this build (probed once and cached;
    often matches CPU count but may be **as low as 2 on CI**), then sets the active
    count to ``num_threads`` clamped to that maximum and to ``os.cpu_count()``. This
    addresses environments where ``set_num_threads(6)`` raises because Numba enforces
    ``Nthreads <= 2``.

    Parameters
    ----------
    num_threads : int or None, optional
        Explicit thread count, or ``None`` / ``0`` for all logical CPUs (capped). Default
        is ``None``.

    Examples
    --------
    >>> with numba_thread_scope(4):  # noqa: SIM117
    ...     ...  # Numba parallel regions here
    """
    effective = _effective_numba_threads(num_threads)
    original_num_threads = get_num_threads()
    set_num_threads(effective)
    try:
        yield
    finally:
        set_num_threads(original_num_threads)

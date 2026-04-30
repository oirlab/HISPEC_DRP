from contextlib import contextmanager
import time
from datetime import datetime
import psutil
import os
import threading
from astropy.table import Table
from hispecdrp import datamodels



@contextmanager
def benchmark_wrapper(**kwargs):
    records = {**kwargs}
    start = time.perf_counter()

    with _RSSSampler() as sampler:
        try:
            yield records
        except Exception:
            raise
        finally:
            t = time.perf_counter() - start

    # sampler.__exit__ has now fired — thread joined, max_rss_bytes is set
    peak_rss_bytes = sampler.max_rss_bytes
    peak_rss_gb = peak_rss_bytes / (1024**3) if peak_rss_bytes is not None else float("nan")

    print(f"{kwargs.get('msg', 'Runtime')}: {t:.4f} seconds")
    print(f"Peak RSS: {peak_rss_gb:.2f} GB")

    records["RUNTIME"] = t
    records["PEAK_MEM_GB"] = peak_rss_gb

class _RSSSampler:
    """
    Sample process RSS in a background thread to estimate peak RSS during an interval.

    Uses psutil when available; otherwise returns None for RSS metrics.
    """

    def __init__(self, *, poll_s: float = 0.05):
        self.poll_s = float(poll_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._max_rss_bytes: int | None = None
        self._proc = psutil.Process(os.getpid()) if psutil is not None else None

    def __enter__(self):
        if self._proc is None:
            return self

        def _run():
            max_rss = 0
            while not self._stop.is_set():
                try:
                    rss = int(self._proc.memory_info().rss)
                except Exception:
                    rss = 0
                if rss > max_rss:
                    max_rss = rss
                self._stop.wait(self.poll_s)
            self._max_rss_bytes = max_rss

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._proc is None:
            return False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return False

    @property
    def max_rss_bytes(self) -> int | None:
        return self._max_rss_bytes

def _get_primitive_runtime(receipt, i):
    prim_class = receipt[i]['CLASS']
    dt1 = datetime.fromisoformat(receipt[i]['PROC_START_TIME'])
    dt2 = datetime.fromisoformat(receipt[i]['PROC_END_TIME'])
    runtime = (dt2 - dt1).total_seconds()
    return prim_class, runtime


def _run_pipeline_perf(
    pipeline_class : type,
    input : datamodels.DataModel | str,
    config_file : str,
):

    stage0_pipe = pipeline_class(config=config_file)

    with benchmark_wrapper() as records:
        model_result = stage0_pipe.apply(input)

    receipt = model_result.receipt
    results = Table(names=("CLASS", "RUNTIME", "PEAK_MEM_GB"), dtype=("U32", "f8", "f8"))

    for i in range(len(receipt)):
        prim_class, prim_runtime = _get_primitive_runtime(receipt, i)
        prim_class = prim_class.split('.')[-1]
        results.add_row((prim_class, prim_runtime, records['PEAK_MEM_GB']))

    # Total runtime
    results.add_row((pipeline_class.__name__, records['RUNTIME'], records['PEAK_MEM_GB']))

    return results
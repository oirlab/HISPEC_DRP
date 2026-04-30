import numpy as np
from contextlib import contextmanager
import time
from datetime import datetime
import psutil
import os
from astropy.table import Table
from hispecdrp import datamodels



@contextmanager
def benchmark_wrapper(**kwargs):
    records = {**kwargs}

    proc = psutil.Process(os.getpid())
    peak_rss = 0

    start = time.perf_counter()

    try:
        yield records

    finally:
        # final measurement
        current_rss = proc.memory_info().rss
        peak_rss = max(peak_rss, current_rss)

        t = time.perf_counter() - start

        print(f"{kwargs.get('msg', 'Runtime')}: {t:.4f} seconds")
        print(f"Peak RSS: {peak_rss / (1024**3):.2f} GB")

        records["RUNTIME"] = t
        records["PEAK_MEM_GB"] = peak_rss / (1024**3)


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

    with benchmark_wrapper() as t:
        model_result = stage0_pipe.apply(input)

    receipt = model_result.receipt
    results = Table(names=("CLASS", "RUNTIME", "PEAK_MEM_GB"), dtype=("U32", "f8", "f8"))

    for i in range(len(receipt)):
        prim_class, prim_runtime = _get_primitive_runtime(receipt, i)
        prim_class = prim_class.split('.')[-1]
        results.add_row((prim_class, prim_runtime, t['PEAK_MEM_GB']))

    # Total runtime
    results.add_row((pipeline_class.__name__, t['RUNTIME'], t['PEAK_MEM_GB']))

    return results
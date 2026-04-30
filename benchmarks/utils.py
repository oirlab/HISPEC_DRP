import numpy as np
from contextlib import contextmanager
import time
from datetime import datetime
from hispecdrp import datamodels


def add_refchan_ramp(
    model : datamodels.RampModel,
    channel_size : int = 64,
    value: float = 0,
):
    """
    Extend ramp data arrays to include ref channel on right side.
    
    Parameters
    ----------
    model : datamodels.RampModel
        Input ramp data model.
    channel_size : int, optional
        Number of columns in each readout channel, by default 64.
    value : float, optional
        Constant value used to fill the reference channel, by default 0.
    """

    # DATA ext
    pad_width = ((0, 0), (0, 0), (0, channel_size))
    model.data = np.pad(model.data, pad_width, mode="constant", constant_values=value)

    # DQ RAW ext
    if model.dq_raw is not None:
        model.dq_raw = np.pad(model.dq_raw, pad_width, mode="constant", constant_values=0)



@contextmanager
def benchmark_wrapper(**kwargs):
    records = {**kwargs}
    start = time.perf_counter()
    try:
        yield records
    finally:
        t = time.perf_counter() - start
        print(f"{kwargs.get('msg', 'Runtime')}: {t:.4f} seconds")
        records["t"] = t


def _get_primitive_runtime(receipt, i):
    prim_class = receipt[i]['CLASS']
    dt1 = datetime.fromisoformat(receipt[i]['PROC_START_TIME'])
    dt2 = datetime.fromisoformat(receipt[i]['PROC_END_TIME'])
    runtime = (dt2 - dt1).total_seconds()
    return prim_class, runtime

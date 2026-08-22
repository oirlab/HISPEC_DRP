"""Unit tests for the DQ init primitive."""

import numpy as np
import pytest

from hispecdrp import datamodels
from hispecdrp.datamodels.dqflags import DQ_FLAGS
from hispecdrp.primitives.dq_init.dqinit_primitive import DQInitPrimitive


def test_dq_init_primitive():
    """DQInitPrimitive applies DQ calibration to ramp and sets dq_raw/dq."""
    n_reads, ny, nx = 4, 8, 8

    # Minimal ramp: data and dq_raw
    ramp_data = np.zeros((n_reads, ny, nx), dtype=np.uint16)
    ramp_data[:] = np.arange(1, n_reads + 1, dtype=np.uint16)[:, None, None]
    dq_raw = np.zeros((n_reads, ny, nx), dtype=np.uint8)
    meta = {
        "instrument_name": "HISPEC",
        "datetime_obs": "2024-01-01T00:00:00",
        "mjd_start": 60320.0,
        "n_channels": 1,
        "channels_are_vertical": True,
        "clock_rate": 1e6,
    }
    ramp_model = datamodels.RampModel(
        data=ramp_data,
        dq_raw=dq_raw,
        meta=meta,
    )
    ramp_model.meta.filename = "simulated_ramp.fits"

    # DQ calibration: mark row 2 as DO_NOT_USE
    dq_map = np.zeros((ny, nx), dtype=np.uint32)
    dq_map[2, :] |= DQ_FLAGS["DO_NOT_USE"]
    dq_model = datamodels.DQModel(dq=dq_map)

    out = DQInitPrimitive.call(ramp_model, dq=dq_model)
    assert out is not None
    assert out.data.shape == (n_reads, ny, nx)
    assert out.dq_raw.shape == (n_reads, ny, nx)
    assert out.dq.shape == (ny, nx)
    # DQ from calibration applied
    np.testing.assert_array_equal(out.dq, dq_map)
    # Row 2 should have DO_NOT_USE
    assert np.all((out.dq[2, :] & DQ_FLAGS["DO_NOT_USE"]) != 0)
    assert np.all((out.dq[0, :] & DQ_FLAGS["DO_NOT_USE"]) == 0)

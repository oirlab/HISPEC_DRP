"""Ramp mid-time must stay in MJD (days), not seconds added to MJD."""

import numpy as np

from hispecdrp import datamodels
from hispecdrp.primitives.ramp_fitting.utils import (
    SECONDS_PER_DAY,
    get_mid_delta_times_ramp,
    seconds_to_mjd_offset,
)


def test_seconds_to_mjd_offset():
    assert seconds_to_mjd_offset(86400.0) == 1.0
    assert seconds_to_mjd_offset(0.0) == 0.0


def test_ramp_first_read_mjd_readout_offset_is_sub_day():
    mjd_start = 61000.0
    n_reads, ny, nx = 10, 4096, 4160
    model = datamodels.RampModel(
        data=np.zeros((n_reads, ny, nx), dtype=np.uint16),
        dq_raw=np.zeros((n_reads, ny, nx), dtype=np.uint8),
        meta={
            "mjd_start": mjd_start,
            "clock_rate": 10023152.94117647,
            "n_channels": 1,
            "channels_are_vertical": True,
        },
    )
    first = model.get_first_read_mjd()
    max_offset_days = float(np.max(first) - mjd_start)
    assert max_offset_days < 10.0 / SECONDS_PER_DAY


def test_ramp_exposure_mid_delta_is_sub_day():
    mjd_start = 61209.79547434599
    n_reads, ny, nx = 10, 4096, 4160
    model = datamodels.RampModel(
        data=np.zeros((n_reads, ny, nx), dtype=np.uint16),
        dq_raw=np.zeros((n_reads, ny, nx), dtype=np.uint8),
        meta={
            "mjd_start": mjd_start,
            "clock_rate": 10023152.94117647,
            "n_channels": 1,
            "channels_are_vertical": True,
        },
    )
    times = model.get_delta_readtimes()
    mid_delta_sec = get_mid_delta_times_ramp(times, np.zeros(n_reads, dtype=np.uint8))
    mid_mjd = mjd_start + seconds_to_mjd_offset(mid_delta_sec)
    assert abs(mid_mjd - mjd_start) < 20.0 / SECONDS_PER_DAY

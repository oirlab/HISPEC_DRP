"""Unit tests for the saturation check primitive."""

import numpy as np
import pytest

from hispecdrp import datamodels
from hispecdrp.datamodels.dqflags import DQ_FLAGS
from hispecdrp.primitives.saturation.saturation_primitive import CheckSaturationPrimitive


def test_saturation_primitive():
    """CheckSaturationPrimitive flags pixels at or above saturation threshold."""
    n_reads, ny, nx = 4, 6, 6
    sat_thresh_value = 1000

    # Ramp: one pixel (1, 1) exceeds threshold at read 2 onward
    ramp_data = np.ones((n_reads, ny, nx), dtype=np.uint16) * 100
    ramp_data[2:, 1, 1] = sat_thresh_value + 10  # saturated from read 2
    dq_raw = np.zeros((n_reads, ny, nx), dtype=np.uint8)
    dq = np.zeros((ny, nx), dtype=np.uint32)
    meta = {
        "instrument_name": "HISPEC",
        "datetime_obs": "2024-01-01T00:00:00",
        "mjd_start": 60320.0,
    }
    ramp_model = datamodels.RampModel(
        data=ramp_data.copy(),
        dq_raw=dq_raw.copy(),
        dq=dq.copy(),
        meta=meta,
    )
    ramp_model.meta.filename = "simulated_ramp.fits"

    # Saturation calibration: uniform threshold (SAT_THRESH is UInt16)
    sat_map_u16 = np.full((ny, nx), sat_thresh_value, dtype=np.uint16)
    sat_dq = np.zeros((ny, nx), dtype=np.uint32)
    saturation_model = datamodels.SaturationModel(
        sat_thresh=sat_map_u16,
        dq=sat_dq,
    )

    primitive = CheckSaturationPrimitive()
    primitive.saturation = saturation_model

    result = primitive.apply(ramp_model)

    out = result  # apply() returns the main output model directly
    assert out is not None
    assert out.data.shape == (n_reads, ny, nx)
    assert out.dq_raw.shape == (n_reads, ny, nx)
    # Primitive sets SATURATED/DO_NOT_USE only in dq_raw (per-read), not in the 2D dq
    saturated_u8 = np.uint8(DQ_FLAGS["SATURATED"])
    do_not_use_u8 = np.uint8(DQ_FLAGS["DO_NOT_USE"])
    dq_raw_pixel = out.dq_raw[:, 1, 1]
    assert np.any((dq_raw_pixel & saturated_u8) != 0) or np.any(
        (dq_raw_pixel & do_not_use_u8) != 0
    ), "Pixel (1,1) should be flagged in dq_raw (saturated from read 2)"

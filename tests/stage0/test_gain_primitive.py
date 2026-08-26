"""Unit tests for the gain correction primitive."""

import numpy as np
import pytest

from hispecdrp import datamodels
from hispecdrp.primitives.gain.gain_primitive import GainCorrectionPrimitive


def test_gain_primitive():
    """GainCorrectionPrimitive converts rate from DN/s to e-/s using gain."""
    ny, nx = 4, 4
    gain_value = 2.0  # e-/DN

    # L0 rate map in DN/s (before gain correction)
    rate_dn_s = np.ones((ny, nx), dtype=np.float32) * 10.0
    rate_err_dn_s = np.ones((ny, nx), dtype=np.float32) * 0.5
    dq = np.zeros((ny, nx), dtype=np.uint32)
    meta = {
        "instrument_name": "HISPEC",
        "datetime_obs": "2024-01-01T00:00:00",
        "mjd_start": 60320.0,
    }
    l0_model = datamodels.L0Model(
        rate=rate_dn_s.copy(),
        rate_err=rate_err_dn_s.copy(),
        dq=dq.copy(),
        meta=meta,
    )
    l0_model.meta.filename = "rate.fits"

    # Gain calibration: uniform gain
    gain_map = np.full((ny, nx), gain_value, dtype=np.float32)
    cov = np.zeros((2, 2, ny, nx), dtype=np.float32)
    cov[0, 0] = 0.01  # gain variance
    gain_model = datamodels.GainModel(
        gain=gain_map,
        rn=np.zeros_like(gain_map),
        cov=cov,
        dq=np.zeros_like(gain_map, dtype=np.uint32),
    )

    out = GainCorrectionPrimitive.call(l0_model, gain=gain_model)
    assert out is not None
    # Rate should be rate_dn_s * gain -> e-/s
    expected_rate = rate_dn_s * gain_value
    np.testing.assert_allclose(out.rate, expected_rate, rtol=1e-5)
    # Error should be scaled (approximate)
    assert np.all(out.rate_err > 0)
    np.testing.assert_allclose(out.rate, 20.0, rtol=1e-5)  # 10 * 2

import numpy as np
import pytest
from astropy.io import fits

from hispecdrp import datamodels
from hispecdrp.datamodels.parvi_config import PARVIConfig


def test_get_delta_readtimes_non_parvi_unchanged():
    data = np.zeros((4, 2, 4), dtype=np.uint16)
    dq_raw = np.zeros_like(data, dtype=np.uint8)

    ramp = datamodels.RampModel(
        data=data,
        dq_raw=dq_raw,
        meta={
            "clock_rate": 2.0,
            "n_channels": 2,
            "channels_are_vertical": True,
            "mjd_start": 60000.0,
        },
    )

    delta = ramp.get_delta_readtimes()
    np.testing.assert_allclose(delta, np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32))


def test_get_delta_readtimes_parvi_uniform_respacing_from_timei00():
    data = np.zeros((4, 2, 4), dtype=np.uint16)
    dq_raw = np.zeros_like(data, dtype=np.uint8)

    base_mjd = 60000.0
    read_secs = np.array([0.0, 1.467, 4.401, 5.868], dtype=np.float64)
    timei00_mjd = base_mjd + read_secs / 86400.0

    ramp = datamodels.RampModel(
        data=data,
        dq_raw=dq_raw,
        meta={
            "clock_rate": 2.0,
            "n_channels": 2,
            "channels_are_vertical": True,
            "mjd_start": base_mjd,
            "parvi_timei00_mjd": timei00_mjd.tolist(),
            "parvi_readtime_sec": 1.467,
        },
        config=PARVIConfig,
    )

    delta = ramp.get_delta_readtimes()
    expected = np.linspace(0.0, 5.868, 4, dtype=np.float32)
    np.testing.assert_allclose(delta, expected, rtol=0, atol=1e-6)


def test_get_first_read_mjd_parvi_uses_timei00_anchor_and_pixel_offsets():
    data = np.zeros((4, 2, 4), dtype=np.uint16)
    dq_raw = np.zeros_like(data, dtype=np.uint8)

    base_mjd = 61000.0
    read_secs = np.array([0.0, 1.467, 2.934, 4.401], dtype=np.float64)
    timei00_mjd = base_mjd + read_secs / 86400.0

    ramp = datamodels.RampModel(
        data=data,
        dq_raw=dq_raw,
        meta={
            "clock_rate": 2.0,
            "n_channels": 2,
            "channels_are_vertical": True,
            "mjd_start": 12345.0,
            "parvi_timei00_mjd": timei00_mjd.tolist(),
            "parvi_readtime_sec": 1.467,
        },
        config=PARVIConfig,
    )

    first_mjd = ramp.get_first_read_mjd()

    channel = base_mjd + np.arange(4, dtype=np.float64).reshape(2, 2) / 2.0 / 86400.0
    expected = np.tile(channel, (1, 2))

    assert first_mjd.shape == (2, 4)
    np.testing.assert_allclose(first_mjd, expected, rtol=0, atol=1e-12)
    assert np.isclose(first_mjd[0, 0], base_mjd)


def test_get_delta_readtimes_parvi_invalid_timei00_length_raises():
    data = np.zeros((4, 2, 4), dtype=np.uint16)
    dq_raw = np.zeros_like(data, dtype=np.uint8)

    ramp = datamodels.RampModel(
        data=data,
        dq_raw=dq_raw,
        meta={
            "clock_rate": 2.0,
            "n_channels": 2,
            "channels_are_vertical": True,
            "mjd_start": 60000.0,
            "parvi_timei00_mjd": [60000.0, 60000.00001],
            "parvi_readtime_sec": 1.467,
        },
        config=PARVIConfig,
    )

    with pytest.raises(ValueError, match="does not match n_reads"):
        ramp.get_delta_readtimes()


def test_parvi_unix_ns_meta_is_auto_converted_to_mjd():
    data = np.zeros((4, 2, 4), dtype=np.uint16)
    dq_raw = np.zeros_like(data, dtype=np.uint8)

    unix_ns = np.array([
        1_700_000_000_000_000_000,
        1_700_000_001_467_000_000,
        1_700_000_002_934_000_000,
        1_700_000_004_401_000_000,
    ], dtype=np.float64)
    expected_mjd = unix_ns / 1e9 / 86400.0 + 40587.0

    ramp = datamodels.RampModel(
        data=data,
        dq_raw=dq_raw,
        meta={
            "clock_rate": 2.0,
            "n_channels": 2,
            "channels_are_vertical": True,
            "parvi_timei00_unix_ns": unix_ns.tolist(),
            "parvi_readtime_sec": 1.467,
        },
        config=PARVIConfig,
    )

    np.testing.assert_allclose(np.asarray(ramp.meta.parvi_timei00_mjd), expected_mjd, rtol=0, atol=1e-12)
    assert np.isclose(ramp.meta.mjd_start, expected_mjd[0])


def test_parvi_timei_header_is_auto_loaded_and_converted(tmp_path):
    data = np.zeros((4, 2, 4), dtype=np.uint16)
    dq_raw = np.zeros_like(data, dtype=np.uint8)
    dq = np.zeros((2, 4), dtype=np.uint32)

    unix_ns = [
        1_700_000_000_000_000_000,
        1_700_000_001_467_000_000,
        1_700_000_002_934_000_000,
        1_700_000_004_401_000_000,
    ]
    expected_mjd = np.asarray(unix_ns, dtype=np.float64) / 1e9 / 86400.0 + 40587.0

    primary = fits.PrimaryHDU()
    primary.header["TIMEI00"] = float(unix_ns[0])
    primary.header["TIMEI01"] = float(unix_ns[1])
    primary.header["TIMEI02"] = float(unix_ns[2])
    primary.header["TIMEI03"] = float(unix_ns[3])

    hdul = fits.HDUList([
        primary,
        fits.ImageHDU(data=data, name="DATA"),
        fits.ImageHDU(data=dq_raw, name="DQ_RAW"),
        fits.ImageHDU(data=dq, name="DQ"),
    ])

    filepath = tmp_path / "parvi_ramp_auto_timei.fits"
    hdul.writeto(filepath, overwrite=True)

    ramp = datamodels.RampModel(
        filepath=str(filepath),
        config=PARVIConfig,
    )

    np.testing.assert_allclose(np.asarray(ramp.meta.parvi_timei00_mjd), expected_mjd, rtol=0, atol=1e-12)
    assert np.isclose(ramp.meta.mjd_start, expected_mjd[0])

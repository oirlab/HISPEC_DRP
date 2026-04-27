import numpy as np
from astropy.io import fits

from hispecdrp.datamodels import load_parvi_utr_cube
from hispecdrp.datamodels.ramp import RampModel


def _payload_to_model(payload):
    model = RampModel(**payload.as_ramp_kwargs())
    model.meta.filename = payload.filename
    return model


def _write_parvi_read(filepath, frame, timei00_ns):
    hdu = fits.PrimaryHDU(data=frame.astype(np.uint16))
    hdu.header["TIMEI00"] = float(timei00_ns)
    hdu.writeto(filepath, overwrite=True)


def _write_parvi_read_with_vectors(filepath, frame, timei00_ns, timei_vec=None, timef_vec=None):
    hdu = fits.PrimaryHDU(data=frame.astype(np.uint16))
    hdu.header["TIMEI00"] = float(timei00_ns)
    if timei_vec is not None:
        for i, v in enumerate(timei_vec):
            hdu.header[f"TIMEI{i:02d}"] = float(v)
    if timef_vec is not None:
        for i, v in enumerate(timef_vec):
            hdu.header[f"TIMEF{i:02d}"] = float(v)
    hdu.writeto(filepath, overwrite=True)


def test_load_parvi_utr_cube_from_list(tmp_path):
    f1 = tmp_path / "H2RG_R01_M01_N01.fits"
    f2 = tmp_path / "H2RG_R01_M02_N01.fits"
    f3 = tmp_path / "H2RG_R01_M03_N01.fits"

    _write_parvi_read(f1, np.full((4, 5), 11, dtype=np.uint16), 1_700_000_000_000_000_000)
    _write_parvi_read(f2, np.full((4, 5), 22, dtype=np.uint16), 1_700_000_001_467_000_000)
    _write_parvi_read(f3, np.full((4, 5), 33, dtype=np.uint16), 1_700_000_002_934_000_000)

    payload = load_parvi_utr_cube([str(f1), str(f2), str(f3)], clock_rate=88802)
    model = _payload_to_model(payload)

    assert model.data.shape == (3, 4, 5)
    assert model.dq_raw.shape == (3, 4, 5)
    assert model.dq.shape == (4, 5)
    assert np.all(model.data[0] == 11)
    assert np.all(model.data[1] == 22)
    assert np.all(model.data[2] == 33)
    assert model.meta.instrument_name == "PARVI"
    assert model.meta.parvi_timei00_mjd is not None
    assert len(model.meta.parvi_timei00_mjd) == 3


def test_load_parvi_utr_cube_from_pattern(tmp_path):
    pattern = tmp_path / "H2RG_R01_M{}_N01.fits"

    _write_parvi_read(str(pattern).format("01"), np.full((2, 3), 1, dtype=np.uint16), 1_700_000_000_000_000_000)
    _write_parvi_read(str(pattern).format("02"), np.full((2, 3), 2, dtype=np.uint16), 1_700_000_001_467_000_000)

    payload = load_parvi_utr_cube(str(pattern), n_reads=2, read_id_width=2)
    model = _payload_to_model(payload)

    assert model.data.shape == (2, 2, 3)
    assert np.all(model.data[0] == 1)
    assert np.all(model.data[1] == 2)


def test_load_parvi_utr_cube_missing_timei00_raises(tmp_path):
    fp = tmp_path / "H2RG_R01_M01_N01.fits"
    fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.uint16)).writeto(fp, overwrite=True)

    try:
        load_parvi_utr_cube([str(fp)])
        assert False, "Expected ValueError for missing TIMEI00"
    except ValueError as err:
        assert "TIMEI00" in str(err)


def test_load_parvi_utr_cube_sorts_by_timei00(tmp_path):
    f1 = tmp_path / "H2RG_R01_M01_N01.fits"
    f2 = tmp_path / "H2RG_R01_M02_N01.fits"
    f3 = tmp_path / "H2RG_R01_M03_N01.fits"

    # Intentionally out of order in file sequence.
    _write_parvi_read(f1, np.full((2, 2), 11, dtype=np.uint16), 1_700_000_002_934_000_000)
    _write_parvi_read(f2, np.full((2, 2), 22, dtype=np.uint16), 1_700_000_000_000_000_000)
    _write_parvi_read(f3, np.full((2, 2), 33, dtype=np.uint16), 1_700_000_001_467_000_000)

    payload = load_parvi_utr_cube([str(f1), str(f2), str(f3)], sort_by_timei00=True)
    model = _payload_to_model(payload)

    # After sorting by TIMEI00: M02, M03, M01 based on the timestamps above.
    assert np.all(model.data[0] == 22)
    assert np.all(model.data[1] == 33)
    assert np.all(model.data[2] == 11)


def test_load_parvi_utr_cube_two_placeholder_pattern(tmp_path):
    pattern = tmp_path / "H2RG_R{}_M{}_N01.fits"

    _write_parvi_read(str(pattern).format("03", "01"), np.full((2, 3), 101, dtype=np.uint16), 1_700_000_000_000_000_000)
    _write_parvi_read(str(pattern).format("03", "02"), np.full((2, 3), 202, dtype=np.uint16), 1_700_000_001_467_000_000)

    payload = load_parvi_utr_cube(
        str(pattern),
        n_reads=2,
        exposure_index=3,
        exposure_id_width=2,
        read_id_width=2,
    )
    model = _payload_to_model(payload)

    assert model.data.shape == (2, 2, 3)
    assert np.all(model.data[0] == 101)
    assert np.all(model.data[1] == 202)


def test_load_parvi_utr_cube_repairs_non_monotonic_timei00(tmp_path):
    f1 = tmp_path / "H2RG_R01_M01_N01.fits"
    f2 = tmp_path / "H2RG_R01_M02_N01.fits"
    f3 = tmp_path / "H2RG_R01_M03_N01.fits"

    # Identical timestamps (observed in some real files due header precision limits).
    _write_parvi_read(f1, np.full((2, 2), 1, dtype=np.uint16), 1_700_000_000_000_000_000)
    _write_parvi_read(f2, np.full((2, 2), 2, dtype=np.uint16), 1_700_000_000_000_000_000)
    _write_parvi_read(f3, np.full((2, 2), 3, dtype=np.uint16), 1_700_000_000_000_000_000)

    payload = load_parvi_utr_cube([str(f1), str(f2), str(f3)], nominal_readtime_sec=1.467)
    model = _payload_to_model(payload)
    dt = np.diff(np.asarray(model.meta.parvi_timei00_mjd, dtype=np.float64)) * 86400.0
    np.testing.assert_allclose(dt, np.array([1.467, 1.467], dtype=np.float64), rtol=0, atol=1e-6)


def test_load_parvi_utr_cube_non_monotonic_strict_raises(tmp_path):
    f1 = tmp_path / "H2RG_R01_M01_N01.fits"
    f2 = tmp_path / "H2RG_R01_M02_N01.fits"

    _write_parvi_read(f1, np.full((2, 2), 1, dtype=np.uint16), 1_700_000_000_000_000_000)
    _write_parvi_read(f2, np.full((2, 2), 2, dtype=np.uint16), 1_700_000_000_000_000_000)

    try:
        load_parvi_utr_cube([str(f1), str(f2)], repair_non_monotonic_timei00=False)
        assert False, "Expected ValueError for non-monotonic TIMEI00"
    except ValueError as err:
        assert "Non-monotonic TIMEI00" in str(err)


def test_load_parvi_utr_cube_uses_time_vector_header_from_last_read(tmp_path):
    f1 = tmp_path / "H2RG_R01_M01_N01.fits"
    f2 = tmp_path / "H2RG_R01_M02_N01.fits"
    f3 = tmp_path / "H2RG_R01_M03_N01.fits"

    # Per-file TIMEI00 is repeated/non-useful.
    repeated = 1_700_000_000_000_000_000
    # Vector timing carried in the last-read file (M03 in this small test).
    timei_vec = [
        1_700_000_000_000_000_000,
        1_700_000_001_467_000_000,
        1_700_000_002_934_000_000,
    ]
    timef_vec = [
        1_700_000_000_733_500_000,
        1_700_000_002_200_500_000,
        1_700_000_003_667_500_000,
    ]

    _write_parvi_read(str(f1), np.full((2, 2), 1, dtype=np.uint16), repeated)
    _write_parvi_read(str(f2), np.full((2, 2), 2, dtype=np.uint16), repeated)
    _write_parvi_read_with_vectors(str(f3), np.full((2, 2), 3, dtype=np.uint16), repeated, timei_vec, timef_vec)

    payload = load_parvi_utr_cube([str(f1), str(f2), str(f3)], use_timing_vector_header=True)
    model = _payload_to_model(payload)
    dt = np.diff(np.asarray(model.meta.parvi_timei00_mjd, dtype=np.float64)) * 86400.0
    np.testing.assert_allclose(dt, np.array([1.467, 1.467], dtype=np.float64), rtol=0, atol=1e-6)
    assert model.meta.parvi_timef_mjd is not None
    assert len(model.meta.parvi_timef_mjd) == 3


def test_load_parvi_utr_cube_timing_reference_mid(tmp_path):
    f1 = tmp_path / "H2RG_R01_M01_N01.fits"
    f2 = tmp_path / "H2RG_R01_M02_N01.fits"

    timei_vec = [1_700_000_000_000_000_000, 1_700_000_001_467_000_000]
    timef_vec = [1_700_000_000_500_000_000, 1_700_000_001_967_000_000]

    _write_parvi_read(str(f1), np.full((2, 2), 1, dtype=np.uint16), timei_vec[0])
    _write_parvi_read_with_vectors(str(f2), np.full((2, 2), 2, dtype=np.uint16), timei_vec[1], timei_vec, timef_vec)

    payload = load_parvi_utr_cube([str(f1), str(f2)], timing_reference="mid")
    model = _payload_to_model(payload)
    mids = (np.asarray(timei_vec, dtype=np.float64) + np.asarray(timef_vec, dtype=np.float64)) / 2.0
    mjd_expected = mids / 1e9 / 86400.0 + 40587.0
    np.testing.assert_allclose(np.asarray(model.meta.parvi_timei00_mjd, dtype=np.float64), mjd_expected, rtol=0, atol=1e-12)

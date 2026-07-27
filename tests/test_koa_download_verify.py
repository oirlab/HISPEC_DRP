"""Tests for KOA download on-disk verification helpers."""

from __future__ import annotations

import os

import pytest

from hispecdrp.calibrations.download_verify import (
    MIN_CALIBRATION_BYTES,
    ensure_calibration_on_disk,
    normalize_downloaded_calibration_path,
)


def test_ensure_calibration_on_disk_missing_raises(tmp_path):
    missing = tmp_path / "nope.fits"
    with pytest.raises(FileNotFoundError, match="does not exist on disk"):
        ensure_calibration_on_disk(str(missing), expected_filename="nope.fits")


def test_ensure_calibration_on_disk_rejects_tiny_file(tmp_path):
    path = tmp_path / "tiny.fits"
    path.write_bytes(b"not-a-fits")
    with pytest.raises(RuntimeError, match="only .* bytes"):
        ensure_calibration_on_disk(str(path), expected_filename="tiny.fits")


def test_ensure_calibration_on_disk_ok(tmp_path):
    path = tmp_path / "ok.fits"
    path.write_bytes(b"0" * MIN_CALIBRATION_BYTES)
    out = ensure_calibration_on_disk(str(path), expected_filename="ok.fits")
    assert out == os.path.abspath(str(path))


def test_normalize_moves_zip_member_to_expected_name(tmp_path):
    unexpected = tmp_path / "member_from_zip.fits"
    unexpected.write_bytes(b"0" * MIN_CALIBRATION_BYTES)
    expected = tmp_path / "HB.20260623.00000.00.mastercal-dq.001.fits"
    out = normalize_downloaded_calibration_path(str(unexpected), str(expected))
    assert out == os.path.abspath(str(expected))
    assert expected.is_file()
    assert not unexpected.exists()
    ensure_calibration_on_disk(out, expected_filename=expected.name)

"""Fast unit tests for HISPECCalibrationSelector filter helpers."""

from __future__ import annotations

import pytest

from hispecdrp.calibrations.selectors import BiasSelector
from hispecdrp.calibrations.utils import get_calibration_selector_type


def _row(origin="LOCAL", cal_version="002", master_cal=1, datetime_obs="2024-01-01"):
    return {
        "origin": origin,
        "cal_version": cal_version,
        "master_cal": master_cal,
        "datetime_obs": datetime_obs,
    }


def test_get_calibration_selector_type_known_and_unknown():
    assert get_calibration_selector_type("BiasSelector") is BiasSelector
    with pytest.raises(ValueError, match="not found"):
        get_calibration_selector_type("NotARealSelector")


def test_filter_origin_local_remote_any_and_invalid():
    rows = [_row(origin="LOCAL"), _row(origin="REMOTE"), _row(origin="LOCAL")]
    sel = BiasSelector(origin="ANY")

    assert len(sel._filter_origin(rows, origin="ANY")) == 3
    assert [r["origin"] for r in sel._filter_origin(rows, origin="LOCAL")] == [
        "LOCAL",
        "LOCAL",
    ]
    assert [r["origin"] for r in sel._filter_origin(rows, origin="REMOTE")] == [
        "REMOTE",
    ]
    with pytest.raises(ValueError, match="Invalid origin"):
        sel._filter_origin(rows, origin="CLOUD")


def test_filter_version_latest_specific_and_rollback():
    rows = [
        _row(cal_version="001"),
        _row(cal_version="003"),
        _row(cal_version="002"),
        _row(cal_version=None),
    ]
    sel = BiasSelector(cal_version="LATEST")

    latest = sel._filter_version(rows, cal_version="LATEST")
    assert [r["cal_version"] for r in latest] == ["003"]

    specific = sel._filter_version(rows, cal_version="002")
    assert [r["cal_version"] for r in specific] == ["002"]

    prev = sel._filter_version(rows, cal_version="-1")
    assert [r["cal_version"] for r in prev] == ["002"]

    too_far = sel._filter_version(rows, cal_version="-9")
    assert too_far == []


def test_filter_master_true_false_prefer_and_any():
    rows = [
        _row(master_cal=1),
        _row(master_cal=0),
        _row(master_cal=0),
    ]
    sel = BiasSelector(master_cal="ANY")

    assert len(sel._filter_master(rows, master_cal=True)) == 1
    assert len(sel._filter_master(rows, master_cal=False)) == 2
    assert len(sel._filter_master(rows, master_cal="ANY")) == 3

    preferred = sel._filter_master(rows, master_cal="PREFER")
    assert [r["master_cal"] for r in preferred] == [1]

    non_master_only = [_row(master_cal=0), _row(master_cal=0)]
    fallback = sel._filter_master(non_master_only, master_cal="PREFER")
    assert len(fallback) == 2
    assert all(r["master_cal"] == 0 for r in fallback)

    with pytest.raises(ValueError, match="Invalid master_cal"):
        sel._filter_master(rows, master_cal="SOMETIMES")

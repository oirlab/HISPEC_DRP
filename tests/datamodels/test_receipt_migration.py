"""Tests for RECEIPT schema migration (``ARGUMENTS`` → ``PARAMETERS``)."""

from __future__ import annotations

import numpy as np
from astropy.table import Column, Table

from hispecdrp.datamodels.extensions import RECEIPT, migrate_receipt_table


def _legacy_empty_receipt() -> Table:
    """Empty RECEIPT shaped like pre-config-files products (ARGUMENTS, no CONFIG_FILE)."""
    return Table(
        [
            Column(name="CLASS", data=np.empty(0, dtype=str)),
            Column(name="PROC_START_TIME", data=np.empty(0, dtype=str)),
            Column(name="PROC_END_TIME", data=np.empty(0, dtype=str)),
            Column(name="FILENAME", data=np.empty(0, dtype=str)),
            Column(name="ARGUMENTS", data=np.empty(0, dtype=str)),
            Column(name="DRP_VERSION", data=np.empty(0, dtype=str)),
            Column(name="DRP_GIT_COMMIT_ID", data=np.empty(0, dtype=str)),
            Column(name="STATUS", data=np.empty(0, dtype=str)),
        ]
    )


def test_migrate_receipt_table_none():
    assert migrate_receipt_table(None) is None


def test_migrate_receipt_empty_legacy_rebuilds_current_schema():
    migrated = migrate_receipt_table(_legacy_empty_receipt())
    assert migrated is not None
    assert list(migrated.colnames) == list(RECEIPT["RECEIPT"]["columns"].keys())
    assert "ARGUMENTS" not in migrated.colnames
    assert "PARAMETERS" in migrated.colnames
    assert "CONFIG_FILE" in migrated.colnames
    assert len(migrated) == 0


def test_migrate_receipt_renames_arguments_and_adds_config_file():
    table = Table(
        {
            "CLASS": ["hispecdrp.primitives.Foo"],
            "PROC_START_TIME": ["2020-01-01T00:00:00.000"],
            "PROC_END_TIME": ["2020-01-01T00:00:01.000"],
            "FILENAME": ["x.fits"],
            "ARGUMENTS": ['{"run": true}'],
            "DRP_VERSION": ["0.0.0"],
            "DRP_GIT_COMMIT_ID": ["abc"],
            "STATUS": ["COMPLETED"],
        }
    )
    migrated = migrate_receipt_table(table)
    assert "ARGUMENTS" not in migrated.colnames
    assert migrated["PARAMETERS"][0] == '{"run": true}'
    assert "CONFIG_FILE" in migrated.colnames
    assert migrated["CONFIG_FILE"][0] == ""
    assert list(migrated.colnames) == list(RECEIPT["RECEIPT"]["columns"].keys())


def test_migrate_receipt_idempotent_on_current_schema():
    cols = []
    for name, dtype in RECEIPT["RECEIPT"]["columns"].items():
        cols.append(Column(name=name, data=np.array(["x"], dtype=dtype)))
    table = Table(cols)
    migrated = migrate_receipt_table(table)
    assert list(migrated.colnames) == list(table.colnames)
    assert migrated["CLASS"][0] == "x"

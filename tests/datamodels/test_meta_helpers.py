"""Fast unit tests for datamodel meta / instrument alias helpers."""

from __future__ import annotations

import pytest

from hispecdrp.datamodels.meta_data import _merge_fiber_state
from hispecdrp.datamodels.utils import get_instrument_alias


def test_merge_fiber_state_priority():
    assert _merge_fiber_state([]) == "NONE"
    assert _merge_fiber_state([None, None]) == "NONE"
    assert _merge_fiber_state(["NONE", "BLOCKED"]) == "BLOCKED"
    assert _merge_fiber_state(["BLOCKED", "SCIENCE", "NONE"]) == "SCIENCE"
    assert _merge_fiber_state(["SKY", "SCIENCE"]) == "SKY"


def test_get_instrument_alias_hispec():
    assert get_instrument_alias("HISPEC", spectrograph="BSPEC") == "HB"
    assert get_instrument_alias("HISPEC", spectrograph="rspec") == "HR"

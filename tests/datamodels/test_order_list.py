"""Echelle order lists from the instrument configs."""

import pytest

from hispecdrp.datamodels import HISPECConfig, PARVIConfig


def test_hispec_order_list_and_subrange():
    assert HISPECConfig.get_order_list("RSPEC") == list(range(60, 98))
    assert HISPECConfig.get_order_list("BSPEC", 130, 132) == [130, 131, 132]
    assert HISPECConfig.get_order_list("BSPEC", order_end=111) == [110, 111]
    assert HISPECConfig.get_order_list("XYZ") == []


def test_parvi_order_list_decreases():
    orders = PARVIConfig.get_order_list("JH")
    assert orders[0] == 128 and orders[-1] == 84 and len(orders) == 45
    assert PARVIConfig.get_order_list("JH", 100, 98) == [100, 99, 98]


@pytest.mark.parametrize("config, args", [
    (HISPECConfig, ("BSPEC", 10, None)),     # not an order of the spectrograph
    (HISPECConfig, ("BSPEC", 140, 130)),     # start after end
    (PARVIConfig, ("JH", 98, 100)),          # start after end (decreasing orders)
])
def test_order_list_bad_bounds_raise(config, args):
    with pytest.raises(ValueError):
        config.get_order_list(*args)

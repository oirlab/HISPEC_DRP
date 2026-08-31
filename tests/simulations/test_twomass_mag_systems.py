"""2MASS Vega/AB magnitude helpers used by template normalization."""

import pytest

from hispecdrp.primitives.utils.spectral_utils import (
    TWOMASS_VEGA_TO_AB,
    resolve_twomass_mag_ab,
    twomass_ab_to_vega,
    twomass_vega_to_ab,
)


@pytest.mark.parametrize(
    "band,offset",
    [
        ("J", 0.910),
        ("H", 1.390),
        ("K", 1.850),
        ("Ks", 1.850),
        ("KS", 1.850),
    ],
)
def test_twomass_vega_ab_roundtrip(band, offset):
    mag_vega = 5.0
    mag_ab = twomass_vega_to_ab(mag_vega, band)
    assert mag_ab == pytest.approx(mag_vega + offset)
    assert twomass_ab_to_vega(mag_ab, band) == pytest.approx(mag_vega)
    assert TWOMASS_VEGA_TO_AB["J"] == 0.910


def test_resolve_mag_vega_overrides_mag_ab():
    mag_ab, band = resolve_twomass_mag_ab(
        mag_ab=3.05,
        mag_vega=2.14,
        filter_band="J",
    )
    assert band == "J"
    assert mag_ab == pytest.approx(2.14 + 0.910)


def test_resolve_mag_ab_only():
    mag_ab, band = resolve_twomass_mag_ab(mag_ab=3.05, filter_band="J")
    assert (mag_ab, band) == (3.05, "J")


def test_resolve_none():
    assert resolve_twomass_mag_ab() == (None, None)


def test_resolve_requires_band_with_magnitude():
    with pytest.raises(ValueError, match="filter_band is required"):
        resolve_twomass_mag_ab(mag_vega=5.0)
    with pytest.raises(ValueError, match="filter_band is required"):
        resolve_twomass_mag_ab(mag_ab=5.0)

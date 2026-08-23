"""Unit tests for E2E Gaia target RV / BERV frame conventions."""

from hispecdrp.simulations.e2e_gaia import E2EGaiaTarget


def test_topocentric_doppler_uses_barycorrpy_sign():
    """``v_obs = v_bary - BERV`` (barycorrpy: ``v_bary = v_obs + BERV``)."""
    target = E2EGaiaTarget(
        source_id="test",
        star_data={},
        target_meta={},
        gaia_rv_km_s=34.42,
        berv_km_s=-2.741,
        injected_rv_km_s=34.42,
        mjd_mid=60577.0,
    )
    # v_obs = 34.42 - (-2.741) = 37.161 km/s
    assert abs(target.topocentric_doppler_rv_km_s - 37.161) < 1e-9
    assert abs(target.stellar_rv_km_s - 34.42) < 1e-9


def test_topocentric_doppler_includes_extra_rv():
    target = E2EGaiaTarget(
        source_id="test",
        star_data={},
        target_meta={},
        gaia_rv_km_s=10.0,
        berv_km_s=1.0,
        injected_rv_km_s=10.0,
        extra_rv_km_s=0.5,
        mjd_mid=60577.0,
    )
    assert abs(target.topocentric_doppler_rv_km_s - (10.0 - 1.0 + 0.5)) < 1e-9

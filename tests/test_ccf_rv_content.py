"""Tests for photon-limited RV content and Doppler wavelength mapping."""
import numpy as np

from hispecdrp.primitives.utils.ccf.ccf_rv_content import (
    compute_rv_content,
    doppler_shift_from_barycenter,
)
from hispecdrp.utils.physical_constants import C_M_S


def test_doppler_shift_from_barycenter_observed_wavelength():
    lam_rest = np.array([0.65, 0.70, 0.75], dtype=np.float64)
    vel = 12_345.0
    zb = 0.0001
    expected = lam_rest * (1.0 + vel / C_M_S) / (1.0 + zb)
    got = doppler_shift_from_barycenter(lam_rest, vel, zb)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_doppler_shift_handles_nonfinite_zb():
    lam = np.array([1.0], dtype=np.float64)
    got = doppler_shift_from_barycenter(lam, 0.0, float("nan"))
    np.testing.assert_allclose(got, lam)


def test_compute_rv_content_linear_spectrum_inverse_quadrature():
    wave = np.linspace(0.6, 0.8, 50, dtype=np.float64)
    spec = 2.0 * wave + 1.0
    spec_err = np.full_like(spec, 0.05)
    spec_dq = np.zeros_like(spec, dtype=np.int32)

    rvc_per_pix, rvc_tot = compute_rv_content(wave, spec, spec_err, spec_dq)

    slope = 2.0
    expected_per_pix = C_M_S * spec_err / (wave * abs(slope))
    np.testing.assert_allclose(rvc_per_pix, expected_per_pix, rtol=1e-10)
    expected_tot = float(np.sum(1.0 / expected_per_pix**2) ** (-0.5))
    assert np.isfinite(rvc_tot)
    np.testing.assert_allclose(rvc_tot, expected_tot, rtol=1e-10)


def test_compute_rv_content_excludes_bad_dq_pixels():
    wave = np.linspace(0.6, 0.8, 20, dtype=np.float64)
    spec = wave.copy()
    spec_err = np.full_like(spec, 0.01)
    spec_dq = np.zeros_like(spec, dtype=np.int32)
    spec_dq[5] = 1

    rvc_per_pix, _ = compute_rv_content(wave, spec, spec_err, spec_dq)
    assert np.isnan(rvc_per_pix[5])
    assert np.isfinite(rvc_per_pix[4])
    assert np.isfinite(rvc_per_pix[6])


def test_compute_rv_content_all_bad_returns_nan_total():
    wave = np.linspace(0.6, 0.8, 10, dtype=np.float64)
    spec = wave.copy()
    spec_err = np.full_like(spec, 0.01)
    spec_dq = np.ones_like(spec, dtype=np.int32)

    _, rvc_tot = compute_rv_content(wave, spec, spec_err, spec_dq)
    assert not np.isfinite(rvc_tot)

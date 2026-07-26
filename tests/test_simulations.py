"""
Unit tests for simulations forward models: rate map, UTR ramp, and noise helpers.
"""
import os
import time

import numpy as np
import astropy.units as u
import pytest

import hispecdrp.simulations as simulations
from hispecdrp.utils.spectral_templates_utils import get_spectrum_path, load_btsettl_spectrum


def test_simulate_rate_map():
    """SimulateRateMap creates a native rate map with expected shape and type."""
    start_time = time.time()
    srm = simulations.SimulateRateMap()
    fiber_1_wavelengths = np.arange(0.9, 2.5, 2.5e-6) * u.micron
    fiber_1_flux = np.ones_like(fiber_1_wavelengths.value) * 1e-14 * u.erg / u.s / u.cm**2 / u.micron
    fiber_23_wavelengths = np.arange(0.9, 2.5, 2.5e-3) * u.micron
    fiber_23_flux = np.ones_like(fiber_23_wavelengths.value) * 1e-15 * u.erg / u.s / u.cm**2 / u.micron

    srm.set_fiber_spectrum(fiber_number=1, spectrum_wavelengths=fiber_1_wavelengths, spectrum_flux=fiber_1_flux, continuous=True)
    srm.set_fiber_spectrum(fiber_number=2, spectrum_wavelengths=fiber_23_wavelengths, spectrum_flux=fiber_23_flux, continuous=False)
    srm.set_fiber_spectrum(fiber_number=3, spectrum_wavelengths=fiber_23_wavelengths, spectrum_flux=fiber_23_flux, continuous=False)

    spectrograph = "BSPEC"
    srm.create_native_rate_map(spectrograph=spectrograph, method="prf_convolution", save=False)
    elapsed = time.time() - start_time

    assert isinstance(srm.rate_map, np.ndarray)
    assert srm.rate_map.ndim == 2
    # BSPEC native rate map can be (4096, 4096) or (4096, 4160) depending on detector layout
    assert srm.rate_map.shape[0] in (4096, 4160) and srm.rate_map.shape[1] in (4096, 4160)
    assert elapsed < 120, "Rate map took too long"


def test_create_ramp_shape_and_keys(tmp_path):
    """create_ramp returns dict with data, dq_raw, meta and correct shapes."""
    ny, nx = 64, 64
    source = np.ones((ny, nx), dtype=np.float64) * 10.0  # e-/s/pixel
    out = simulations.create_ramp(
        source,
        readtime=0.1,
        n_reads=5,
        poisson_noise=False,
        convert_to_uint16=False,
        max_cores=1,
    )
    assert "data" in out
    assert "dq_raw" in out
    assert "meta" in out
    assert out["data"].shape == (5, ny, nx)
    assert out["dq_raw"].shape == (5, ny, nx)
    assert np.all(np.isfinite(out["data"]))


def test_create_ramp_uint16(tmp_path):
    """create_ramp with convert_to_uint16 produces uint16 or float32 data in uint16 range."""
    source = np.ones((32, 32), dtype=np.float64) * 5.0
    out = simulations.create_ramp(
        source,
        readtime=0.1,
        n_reads=4,
        poisson_noise=False,
        convert_to_uint16=True,
        max_cores=1,
    )
    # Implementation may return float32 buffer with uint16-range values or actual uint16
    assert out["data"].dtype in (np.uint16, np.float32)
    assert np.all(out["data"] >= 0) and np.all(out["data"] <= np.iinfo(np.uint16).max)


def test_generate_1overf_noise():
    """generate_1overf_noise returns 1D array with requested length and approximate std."""
    n = 1000
    noise = simulations.generate_1overf_noise(n, dt=0.01, std=2.0, seed=42)
    assert noise.shape == (n,)
    assert np.isclose(np.std(noise), 2.0, rtol=0.5)
    noise2, t = simulations.generate_1overf_noise(n, dt=0.01, std=1.0, seed=42, return_time=True)
    assert t.shape == (n,)
    assert np.allclose(t, np.arange(n) * 0.01)


def test_generate_1overf_noise_map():
    """generate_1overf_noise_map returns 3D array of given shape."""
    shape_3d = (5, 64, 64)
    out = simulations.generate_1overf_noise_map(shape_3d, n_channels=1, vertical=True, std_1overf=1.0, seed=1)
    assert out.shape == shape_3d


def test_generate_1overf_noise_map_vertical_channels():
    """generate_1overf_noise_map with n_channels divides x dimension."""
    shape_3d = (4, 32, 64)
    out = simulations.generate_1overf_noise_map(shape_3d, n_channels=2, vertical=True, std_1overf=0.5, seed=2)
    assert out.shape == shape_3d


def test_utr_variable_rate_cube():
    """utr_variable_rate_cube returns rate cube and scaling with correct shapes."""
    source = np.ones((8, 8)) * 100.0
    nt = 20
    flux_scaling = np.ones(nt) * 1.0
    flux_scaling_times = np.linspace(0, 10, nt)
    rate_cube, scaling = simulations.utr_variable_rate_cube(
        source, flux_scaling, flux_scaling_times, n_reads=5, readtime=1.0
    )
    assert rate_cube.shape == (4, 8, 8)
    assert scaling.shape == (4,)
    assert np.allclose(np.mean(rate_cube, axis=(1, 2)), 100.0 * scaling, rtol=1e-10)


def test_get_spectrum_path():
    """get_spectrum_path resolves relative path under hispecdrp templates dir."""
    rel = "lte025.0-5.5-0.0a+0.0.BT-Settl.spec.7.h5"
    out = get_spectrum_path(rel)
    assert "templates" in out
    assert "data" in out
    assert out.endswith(rel)
    abs_path = "/absolute/path/to/file.h5"
    assert get_spectrum_path(abs_path) == abs_path


@pytest.mark.skipif(
    not os.path.isfile(get_spectrum_path("lte025.0-5.5-0.0a+0.0.BT-Settl.spec.7.h5")),
    reason="BT-Settl spectrum file not present",
)
def test_load_btsettl_spectrum():
    """load_btsettl_spectrum loads wavelength and flux from bundled .h5 if present."""
    path = get_spectrum_path("lte025.0-5.5-0.0a+0.0.BT-Settl.spec.7.h5")
    wl, flux = load_btsettl_spectrum(path, wl_min_um=1.0, wl_max_um=2.0)
    assert wl is not None
    assert flux is not None
    assert len(wl) == len(flux)
    assert np.all(wl >= 1.0 * u.micron)
    assert np.all(wl <= 2.0 * u.micron)


def test_fetch_svo_2mass_filter():
    """fetch_svo_2mass_filter returns wavelength and transmission arrays (requires network)."""
    pytest.importorskip("urllib.request")
    try:
        from hispecdrp.utils import fetch_svo_2mass_filter
        wl_A, trans = fetch_svo_2mass_filter("J")
    except OSError:
        pytest.skip("SVO filter service unreachable (network or server)")
    assert wl_A is not None
    assert trans is not None
    assert len(wl_A) == len(trans)
    assert np.all(np.isfinite(wl_A))
    assert np.all(np.isfinite(trans))

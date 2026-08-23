"""Smoke tests for packaged sim-HDF line-ZP wavesol."""

from __future__ import annotations

import numpy as np
import pytest

from hispecdrp.simulations.sim_hdf_wavesol_packaging import (
    PACKAGED_SIM_HDF_LINE_ZP_NOTE,
    load_packaged_sim_hdf_line_zp_wavesol,
    packaged_sim_hdf_wavesol_path,
)


@pytest.mark.skipif(
    not packaged_sim_hdf_wavesol_path("BSPEC").is_file(),
    reason="packaged BSPEC line-ZP wavesol not present",
)
def test_load_packaged_bspec_line_zp_wavesol():
    ws = load_packaged_sim_hdf_line_zp_wavesol("BSPEC")
    assert PACKAGED_SIM_HDF_LINE_ZP_NOTE in str(getattr(ws.meta, "note", "") or "")
    wave = np.asarray(ws.wave_fib2, dtype=np.float64)
    assert wave.ndim == 2
    assert wave.shape[1] == 4096
    assert np.isfinite(wave).sum() > 1000

    # Packaged WAVE is deposit-inverse + absorption-line ZP / poly rescale, so it
    # must differ from a freshly built geometric HDF WAVE at a small Doppler scale.
    # Absolute median |dv| can be tens–hundreds of m/s (order-dependent poly) and
    # can drift with scipy/interp across platforms (~1 spectral pixel ≈ few×100 m/s);
    # keep a wide band that still rejects identical WAVE and unit mismatches.
    from hispecdrp.simulations.e2e_pipeline import create_wavelength_solution_model_from_sim_hdf
    import scipy.constants

    hdf = create_wavelength_solution_model_from_sim_hdf("BSPEC")
    hdf_wave = np.asarray(hdf.wave_fib2, dtype=np.float64)
    assert wave.shape == hdf_wave.shape
    assert not np.allclose(wave, hdf_wave, rtol=0.0, atol=0.0, equal_nan=True)

    ratio = wave / hdf_wave
    dv = (1.0 - ratio) * float(scipy.constants.c)
    med = float(np.nanmedian(dv))
    assert np.isfinite(med)
    assert 1.0 < abs(med) < 5_000.0

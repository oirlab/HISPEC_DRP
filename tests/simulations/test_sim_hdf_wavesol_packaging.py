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
    # Line-ZP should shift WAVE slightly vs geometric deposit-inverse (~tens of m/s).
    from hispecdrp.simulations.e2e_pipeline import create_wavelength_solution_model_from_sim_hdf
    import scipy.constants

    hdf = create_wavelength_solution_model_from_sim_hdf("BSPEC")
    ratio = wave / np.asarray(hdf.wave_fib2, dtype=np.float64)
    dv = (1.0 - ratio) * float(scipy.constants.c)
    med = float(np.nanmedian(dv))
    assert 20.0 < abs(med) < 120.0

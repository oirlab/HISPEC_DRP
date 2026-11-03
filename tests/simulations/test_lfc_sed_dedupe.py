"""Overlap modes shared by two orders must enter the LFC SED only once."""

from __future__ import annotations

import numpy as np

from hispecdrp.simulations import e2e_pipeline


def _fake_grids():
    wave = np.linspace(1.29, 1.33, 4096)
    return {2: {110: {"wavelength": wave}, 111: {"wavelength": wave}}}


def test_sed_dedupes_modes_shared_between_orders(monkeypatch):
    monkeypatch.setattr(
        e2e_pipeline, "get_sim_hdf_trace_grids", lambda spec: _fake_grids()
    )
    line_list = {
        110: np.array([1.300, 1.310]),
        111: np.array([1.310, 1.320]),  # 1.310 is in both orders' spans
    }
    wl, flux, info = e2e_pipeline.build_gaussian_lfc_sed(
        line_list, spectrograph="BSPEC", fiber=2, amplitude=10.0
    )
    assert info["n_lines"] == 3

    def peak_at(lam):
        return float(flux[np.argmin(np.abs(wl - lam))])

    # Before the fix the shared mode summed two amplitude-10 Gaussians.
    assert peak_at(1.310) < 1.2 * peak_at(1.300)
    assert peak_at(1.310) > 0.8 * peak_at(1.300)

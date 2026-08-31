"""Tests for continuum-normalized spectrum fidelity helpers."""

from __future__ import annotations

import numpy as np

from hispecdrp.simulations.plotting.spectrum_fidelity import (
    compare_continuum_normalized_spectra,
    continuum_normalize_orders,
)


def test_continuum_normalize_orders_recovers_flat_continuum():
    rng = np.random.default_rng(0)
    n_ord, n_pix = 3, 200
    wave = np.linspace(1.0, 1.1, n_pix)
    wave_2d = np.vstack([wave, wave + 0.01, wave + 0.02])
    # Blazed continuum × shallow lines
    x = np.arange(n_pix, dtype=np.float64)
    blaze = 1.0 + 0.3 * np.sin(x / 40.0)
    lines = 1.0 - 0.2 * np.exp(-0.5 * ((x - 100) / 3.0) ** 2)
    flux = np.vstack([blaze * lines for _ in range(n_ord)])
    norm = continuum_normalize_orders(flux, size=51)
    # Continuum regions away from the line should be near 1.
    core = np.abs(x - 100) > 20
    assert abs(float(np.median(norm[0, core])) - 1.0) < 0.05


def test_compare_identical_spectra_near_zero_rms():
    n_ord, n_pix = 2, 256
    wave = np.linspace(1.0, 1.05, n_pix)
    wave_2d = np.vstack([wave, wave + 0.002])
    x = np.arange(n_pix, dtype=np.float64)
    flux = np.vstack(
        [
            1.0 - 0.3 * np.exp(-0.5 * ((x - 80) / 2.5) ** 2),
            1.0 - 0.25 * np.exp(-0.5 * ((x - 120) / 2.0) ** 2),
        ]
    )
    # Add mild blaze — continuum norm should remove it.
    flux = flux * (1.0 + 0.2 * np.sin(x / 30.0))[None, :]
    out = compare_continuum_normalized_spectra(
        wave_2d, flux, wave_2d, flux, label="self", edge_trim=10
    )
    assert out["n_pix"] > 100
    assert out["norm_rms_pct"] < 0.5
    assert out["corr"] > 0.99

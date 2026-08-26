"""Tests for LSF-matched build_l2 depth vs native paint."""

from __future__ import annotations

import numpy as np
import pytest

from hispecdrp.primitives.lsf.lsf_convolve import convolve_order_spectrum_gaussian_lsf


def test_lsf_matched_line_shallower_than_native_synthetic():
    """Proxy for build_l2 apply_lsf without loading full BT-Settl + HDF paint."""
    wave = np.linspace(1.2, 1.21, 400)
    x = np.arange(wave.size, dtype=np.float64)
    native = 1.0 - 0.6 * np.exp(-0.5 * ((x - 200) / 0.7) ** 2)
    sigma = np.full_like(wave, 3.0)
    lsf = convolve_order_spectrum_gaussian_lsf(wave, native, sigma)
    depth_n = 1.0 - float(np.min(native))
    depth_l = 1.0 - float(np.min(lsf))
    assert depth_l < 0.9 * depth_n


@pytest.mark.slow
def test_build_l2_apply_lsf_runs_smoke():
    from hispecdrp.simulations.spectra import build_l2

    # Two-order smoke (HDF LSF load). Skipped in default CI via -m "not slow".
    model = build_l2(
        0.0,
        spectrograph="BSPEC",
        template="lte040.0-4.5-0.0a+0.0.BT-Settl.spec.7.h5",
        fiber=2,
        order_start=110,
        order_end=111,
        apply_lsf=True,
    )
    flux = np.asarray(getattr(model, "spec_fib2"), dtype=np.float64)
    assert flux.shape[0] == 2
    assert np.isfinite(flux).any()

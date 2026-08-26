"""Machine-precision round-trip for PSF-deposit spectroperfectionism."""

from __future__ import annotations

import numpy as np

from hispecdrp.primitives.spectral_extraction.extract_spectroperfectionism import (
    extract_spectrum_spectroperfectionism,
    forward_psf_deposit_oriented,
)


def _gaussian_stamp(h: int = 17, w: int = 21, sy: float = 1.8, sx: float = 2.2) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = 0.5 * (h - 1), 0.5 * (w - 1)
    return np.exp(-0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))


def test_spectroperfectionism_noiseless_round_trip():
    """``d = A f`` then extract recovers ``f`` near machine precision."""
    nx = 96
    n_spec = 64
    oversample = 3.0
    stamp = _gaussian_stamp()
    # Linear native x so oriented row = (nx-1) - x tracks spectral index.
    # Place trace so oriented rows 16..79 map to spectral samples 0..63.
    row0 = 16
    wave = np.linspace(1.0, 1.01, n_spec)

    def fx_native(lam: float) -> float:
        # Inverse of row = (nx-1) - x  with row = row0 + index
        # Find nearest sample index from lam
        idx = float(np.interp(lam, wave, np.arange(n_spec, dtype=np.float64)))
        row = row0 + idx
        return float((nx - 1) - row)

    def fy_native(lam: float) -> float:
        # Constant cross-disp → oriented col near center
        col = nx / 2.0
        return float((nx - 1) - col)

    psfw = np.array([1.005], dtype=np.float64)
    stamps = [stamp]

    x = np.arange(n_spec, dtype=np.float64)
    f_true = 20.0 * (1.0 + 0.15 * np.sin(x / 7.0))
    f_true *= 1.0 - 0.4 * np.exp(-0.5 * ((x - 32) / 2.0) ** 2)

    det = forward_psf_deposit_oriented(
        f_true,
        wave,
        fx_native=fx_native,
        fy_native=fy_native,
        psfw=psfw,
        psf_stamps=stamps,
        oversample=oversample,
        nx_native=nx,
        det_shape=(nx, nx),
    )
    rate_var = np.ones_like(det)

    rec = extract_spectrum_spectroperfectionism(
        det,
        rate_var,
        wave,
        fx_native=fx_native,
        fy_native=fy_native,
        psfw=psfw,
        psf_stamps=stamps,
        oversample=oversample,
        nx_native=nx,
        spectral_bandwidth=n_spec,  # no bandwidth truncation for exactness
    )
    f_hat = rec["spec"]
    good = np.isfinite(f_hat) & np.isfinite(f_true)
    # Edge samples may be weakly constrained; compare interior.
    good[0:4] = False
    good[-4:] = False
    assert int(np.sum(good)) > n_spec // 2
    rel = np.max(np.abs(f_hat[good] - f_true[good]) / np.maximum(np.abs(f_true[good]), 1.0))
    assert rel < 1.0e-8, f"max relative error {rel}"

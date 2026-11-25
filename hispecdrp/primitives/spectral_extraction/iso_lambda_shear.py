"""Iso-λ shear from HDF PSF stamps for sheared Horne extraction."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from ...datamodels import hispec_config


def psf_stamp_shear_dx_dy(
    stamp: np.ndarray,
    *,
    oversample: float,
    min_row_flux_frac: float = 1.0e-3,
) -> float:
    """
    Measure native-detector shear ``dx/dy`` from one oversampled PSF stamp.

    Stamp layout matches :func:`_deposit_oversampled_psf_numba`: axis 0 is
    cross-dispersion (``y``), axis 1 is spectral (``x``). Returns detector-pixel
    shear (spectral shift per cross-dispersion pixel).
    """
    psf = np.asarray(stamp, dtype=np.float64)
    if psf.ndim != 2:
        raise ValueError(f"PSF stamp must be 2-D; got shape {psf.shape}")
    h, w = psf.shape
    total = float(np.sum(psf))
    if not np.isfinite(total) or total <= 0.0:
        return float("nan")

    j = np.arange(h, dtype=np.float64)
    i = np.arange(w, dtype=np.float64)
    row_flux = np.sum(psf, axis=1)
    thresh = float(min_row_flux_frac) * float(np.max(row_flux))
    good = row_flux > thresh
    if int(np.sum(good)) < 3:
        return float("nan")

    # First moment along spectral axis for each cross-disp row.
    mom = np.sum(psf * i[None, :], axis=1) / np.maximum(row_flux, 1.0e-30)
    dy = (j[good] - 0.5 * (h - 1)) / float(oversample)
    dx = (mom[good] - 0.5 * (w - 1)) / float(oversample)
    # Robust linear fit dx = s * dy + b (both axes flip under orient → s unchanged).
    A = np.column_stack([dy, np.ones(dy.size, dtype=np.float64)])
    coef, _, _, _ = np.linalg.lstsq(A, dx, rcond=None)
    return float(coef[0])


def create_trace_shear_from_sim_hdf(
    spectrograph: str,
    *,
    fiber: int = 2,
    n_pix: int = 4096,
) -> np.ndarray:
    """
    Per-order iso-λ shear ``∂row/∂col`` on the oriented spectral-row grid.

    Shape matches ``TracePosModel.pos_fib*``: ``(n_orders, n_pix)``, with
    ``shear[k, x]`` the spectral-row shift per cross-dispersion pixel at
    spectral sample ``x`` for echelle order index ``k``.
    """
    # Local import avoids circular import with e2e_pipeline helpers.
    from ...simulations.e2e_pipeline import (
        _load_hdf_model_data,
        _order_list,
        get_sim_hdf_trace_grids,
    )

    spectrograph = str(spectrograph).upper()
    fiber = int(fiber)
    if fiber not in (1, 2, 3):
        raise ValueError(f"fiber must be 1, 2, or 3; got {fiber}")

    _, model_data = _load_hdf_model_data(spectrograph)
    order_list = _order_list(spectrograph)
    grids = get_sim_hdf_trace_grids(spectrograph, n_pix=n_pix)
    pixelsize = float(model_data.get("pixelsize", 1.0))

    n_orders = len(order_list)
    shear = np.full((n_orders, int(n_pix)), np.nan, dtype=np.float64)
    fiber_key = f"fiber_{fiber}"
    fiber_block = model_data.get(fiber_key, {})

    for k, od in enumerate(order_list):
        if od not in fiber_block:
            continue
        block = fiber_block[od]
        if "psf_stamps" not in block or "psf_wavelengths" not in block:
            continue
        stamps = block["psf_stamps"]
        psfw = np.asarray(block["psf_wavelengths"], dtype=np.float64)
        spacing = float(block.get("psf_data_spacing", pixelsize))
        oversample = float(pixelsize) / float(spacing) if spacing > 0 else 1.0

        s_vals = np.full(psfw.size, np.nan, dtype=np.float64)
        for i in range(psfw.size):
            s_vals[i] = psf_stamp_shear_dx_dy(stamps[i], oversample=oversample)
        ok_s = np.isfinite(s_vals) & np.isfinite(psfw)
        if int(np.sum(ok_s)) < 1:
            continue

        wave_row = grids.get(fiber, {}).get(od, {}).get("wavelength")
        if wave_row is None:
            # Constant shear if WAVE grid missing.
            shear[k, :] = float(np.nanmedian(s_vals[ok_s]))
            continue

        wave_row = np.asarray(wave_row, dtype=np.float64)
        lam_s = psfw[ok_s]
        shear_s = s_vals[ok_s]
        order = np.argsort(lam_s)
        lam_s = lam_s[order]
        shear_s = shear_s[order]
        if lam_s.size == 1:
            shear[k, :] = float(shear_s[0])
            continue
        # Deduplicate wavelengths for PCHIP.
        ulam, inv = np.unique(lam_s, return_inverse=True)
        if ulam.size == 1:
            shear[k, :] = float(np.nanmean(shear_s))
            continue
        ushear = np.bincount(inv, weights=shear_s) / np.maximum(
            np.bincount(inv), 1
        )
        f = PchipInterpolator(ulam, ushear, extrapolate=True)
        good_w = np.isfinite(wave_row)
        out = np.full(int(n_pix), np.nan, dtype=np.float64)
        out[good_w] = np.asarray(f(wave_row[good_w]), dtype=np.float64)
        # Fill gaps with median shear.
        med = float(np.nanmedian(out))
        if np.isfinite(med):
            out[~np.isfinite(out)] = med
        shear[k, :] = out

    return shear


def order_list_for_spectrograph(spectrograph: str) -> list[int]:
    """Echelle order numbers for ``spectrograph`` (public helper)."""
    return list(hispec_config.HISPECConfig.get_order_list(str(spectrograph).upper()))

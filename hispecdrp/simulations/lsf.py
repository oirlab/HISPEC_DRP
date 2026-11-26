"""HDF / pyechelle LSF helpers used by simulated L2 paint.

Generic convolution and PSF-stamp collapse live in
:mod:`hispecdrp.primitives.lsf.lsf_convolve`. This module only loads the
instrument HDF model and maps it onto those primitives. Empirical LFC LSFs
(:class:`~hispecdrp.datamodels.gausslsf.GaussianLSFModel`) are applied via
:func:`convolve_l2_with_gaussian_lsf_model`.
"""

from __future__ import annotations

import numpy as np

from ..datamodels import hispec_config
from ..primitives.lsf.lsf_convolve import (
    convolve_l2_with_gaussian_lsf,
    lsf_library_from_psf_fiber_block,
    sigma_pix_from_lsf_library,
)

_SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _fwhm_wave_from_lsf_params(lsf_params) -> np.ndarray:
    """Per-pixel FWHM in wavelength; accept 2-D FWHM or 3-D ``(..., 3)`` layout."""
    arr = np.asarray(lsf_params, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr[..., 1] * _SIGMA_TO_FWHM
    return arr


def _slice_lsf_fwhm_to_wave(
    fwhm_wave: np.ndarray,
    wave_2d: np.ndarray,
    *,
    fiber: int,
    order_start: int | None = None,
    lsf_order_start: int | None = None,
    spectrograph: str | None = None,
) -> np.ndarray:
    """Align full-band LSF FWHM rows to a (possibly order-subset) ``wave_2d``."""
    fwhm_wave = np.asarray(fwhm_wave, dtype=np.float64)
    wave_2d = np.asarray(wave_2d, dtype=np.float64)
    if fwhm_wave.shape == wave_2d.shape:
        return fwhm_wave
    if fwhm_wave.ndim != 2 or wave_2d.ndim != 2:
        raise ValueError(
            f"lsf_params_fib{fiber} shape {fwhm_wave.shape} does not match "
            f"wave_2d shape {wave_2d.shape}"
        )
    if fwhm_wave.shape[1] != wave_2d.shape[1]:
        raise ValueError(
            f"lsf_params_fib{fiber} n_pix={fwhm_wave.shape[1]} does not match "
            f"wave_2d n_pix={wave_2d.shape[1]}"
        )
    n_ord = int(wave_2d.shape[0])
    if order_start is None:
        raise ValueError(
            f"lsf_params_fib{fiber} has {fwhm_wave.shape[0]} orders but wave_2d "
            f"has {n_ord}; pass order_start to select the matching LSF rows"
        )
    if lsf_order_start is None and spectrograph is not None:
        sp = str(spectrograph).upper()
        lsf_order_start = int(hispec_config.HISPECConfig.echelle_orders[sp]["order_start"])
    if lsf_order_start is None:
        raise ValueError(
            "lsf_order_start (or spectrograph) is required when slicing a "
            "full-band LSF onto a subset of echelle orders"
        )
    i0 = int(order_start) - int(lsf_order_start)
    i1 = i0 + n_ord
    if i0 < 0 or i1 > fwhm_wave.shape[0]:
        raise ValueError(
            f"LSF row slice [{i0}:{i1}] for orders "
            f"{int(order_start)}..{int(order_start) + n_ord - 1} is outside "
            f"lsf_params_fib{fiber} with {fwhm_wave.shape[0]} rows "
            f"(lsf_order_start={int(lsf_order_start)})"
        )
    return fwhm_wave[i0:i1]


def sigma_pix_from_gaussian_lsf_model(
    lsf_model,
    wave_2d: np.ndarray,
    *,
    fiber: int,
    order_start: int | None = None,
    lsf_order_start: int | None = None,
    spectrograph: str | None = None,
) -> np.ndarray:
    """Map :class:`~hispecdrp.datamodels.gausslsf.GaussianLSFModel` FWHM→σ in pixels.

    When ``wave_2d`` covers a subset of the LSF orders, pass ``order_start`` (first
    echelle order in ``wave_2d``) and either ``lsf_order_start`` or ``spectrograph``
    so the matching LSF rows are selected.
    """
    fiber = int(fiber)
    wave_2d = np.asarray(wave_2d, dtype=np.float64)
    if lsf_order_start is None and hasattr(lsf_model, "meta"):
        meta_ost = getattr(lsf_model.meta, "order_start", None)
        if meta_ost is not None:
            lsf_order_start = int(meta_ost)
    fwhm_wave = _slice_lsf_fwhm_to_wave(
        _fwhm_wave_from_lsf_params(getattr(lsf_model, f"lsf_params_fib{fiber}")),
        wave_2d,
        fiber=fiber,
        order_start=order_start,
        lsf_order_start=lsf_order_start,
        spectrograph=spectrograph,
    )
    # Floor dispersion with a fraction of the per-order median |dλ/dpix| so
    # flat / NaN WAVE edge pixels do not explode σ into terabyte kernels.
    sigma_pix = np.empty_like(wave_2d)
    for i in range(wave_2d.shape[0]):
        dw = np.abs(np.gradient(wave_2d[i]))
        med_dw = float(np.nanmedian(dw[np.isfinite(dw) & (dw > 0.0)]))
        if not np.isfinite(med_dw) or med_dw <= 0.0:
            med_dw = 1.0e-6
        dw = np.where(np.isfinite(dw) & (dw > 0.05 * med_dw), dw, med_dw)
        sigma_pix[i] = (fwhm_wave[i] / _SIGMA_TO_FWHM) / dw
        bad = ~np.isfinite(sigma_pix[i]) | (sigma_pix[i] <= 0.0)
        if np.any(bad):
            fill = float(np.nanmedian(sigma_pix[i][~bad])) if np.any(~bad) else 1.0
            sigma_pix[i] = np.where(bad, fill, sigma_pix[i])
        # Detector-pixel LSF widths are O(1); clamp pathological outliers.
        sigma_pix[i] = np.clip(sigma_pix[i], 0.25, 20.0)
    return sigma_pix


def convolve_l2_with_gaussian_lsf_model(
    wave_2d: np.ndarray,
    flux_2d: np.ndarray,
    lsf_model,
    *,
    fiber: int,
    order_start: int | None = None,
    lsf_order_start: int | None = None,
    spectrograph: str | None = None,
) -> np.ndarray:
    """Convolve a 2-D spectrum with an empirical ProcessLFC ``GaussianLSFModel``."""
    sigma_pix = sigma_pix_from_gaussian_lsf_model(
        lsf_model,
        wave_2d,
        fiber=int(fiber),
        order_start=order_start,
        lsf_order_start=lsf_order_start,
        spectrograph=spectrograph,
    )
    return convolve_l2_with_gaussian_lsf(
        np.asarray(wave_2d, dtype=np.float64),
        np.asarray(flux_2d, dtype=np.float64),
        sigma_pix,
    )


def hdf_psf_lsf_library(
    spectrograph: str,
    fiber: int = 2,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Per-order library of 1-D LSFs from the simulated HDF PSF stamps.

    Returns ``{order: {"wavelengths", "offsets_list", "kernels"}}``.
    """
    from .e2e_pipeline import _load_hdf_model_data

    spectrograph = str(spectrograph).upper()
    fiber = int(fiber)
    _, model_data = _load_hdf_model_data(spectrograph)
    pixelsize = float(model_data.get("pixelsize", 1.0))
    fiber_block = model_data.get(f"fiber_{fiber}", {})
    return lsf_library_from_psf_fiber_block(fiber_block, pixelsize=pixelsize)


def sigma_pix_from_hdf_lsf(
    spectrograph: str,
    fiber: int,
    order: int,
    wave: np.ndarray,
) -> np.ndarray:
    """
    Effective Gaussian σ (detector pixels) vs spectral sample from HDF PSF LSF.
    """
    lib = hdf_psf_lsf_library(spectrograph, fiber)
    return sigma_pix_from_lsf_library(lib, order, wave)


def convolve_l2_with_hdf_lsf(
    wave_2d: np.ndarray,
    flux_2d: np.ndarray,
    *,
    spectrograph: str,
    fiber: int,
    order_start: int | None = None,
) -> np.ndarray:
    """
    Apply per-order HDF-derived Gaussian LSF to a 2-D (order, pix) spectrum.
    """
    spectrograph = str(spectrograph).upper()
    fiber = int(fiber)
    wave_2d = np.asarray(wave_2d, dtype=np.float64)
    flux_2d = np.asarray(flux_2d, dtype=np.float64)
    n_ord = wave_2d.shape[0]
    if order_start is None:
        order_start = int(
            hispec_config.HISPECConfig.echelle_orders[spectrograph]["order_start"]
        )
    lib = hdf_psf_lsf_library(spectrograph, fiber)
    sigma_2d = np.empty_like(wave_2d)
    for i in range(n_ord):
        od = int(order_start) + i
        sigma_2d[i] = sigma_pix_from_lsf_library(lib, od, wave_2d[i])
    return convolve_l2_with_gaussian_lsf(wave_2d, flux_2d, sigma_2d)
